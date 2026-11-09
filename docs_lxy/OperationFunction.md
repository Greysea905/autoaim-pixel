# 处理线程 OperationFunction 逐步讲解

> 位置：`main.cpp`。这是整个程序的核心主循环（处理线程），按顺序拆解每一步，函数后括号标注所在文件。

## 主循环整体（`main.cpp`）

```cpp
void* OperationFunction(void* arg) {           // main.cpp
    SerialPort* serialPort = static_cast<SerialPort*>(arg);
    START_WEB_SERVER(web_port);                // Web UI 也在这个线程

    int empty_frame_count = 0;
    while (!shutdown_requested.load(std::memory_order_relaxed)) {
        const auto process_start = std::chrono::steady_clock::now();

        // ① 拷贝最新控制器帧
        if (!copyLatestFrame(translator, received_at)) { sleep; continue; }

        // ② 校验（过期 / 状态 / 有限值）
        if (!frameIsFresh(received_at) || !isAutoAimStatus(status) ||
            !hasFiniteControlInput(msg)) { setSafeOutput; writeFrame; continue; }

        // ③ 提取云台
        const GimbalState gimbal = gimbalStateFromMessage(translator.message);

        // ④ 取流（失败>3次退出）
        const bool got_pic = get_pic(pic);
        if (!got_pic || pic.empty()) { /* 黑帧兜底 or 退出 */ }

        // ⑤⑥⑦ 检测 → PnP → EKF
        detections = armor_detect(pic, enemyColorFromStatus(status));
        observations = armor_solve(detections, camera, gimbal);
        prediction  = ekf_predict(observations, gimbal, time_stamp);
        bool prediction_valid = applyPrediction(translator, prediction);   // ⑧

        // 处理超时 → 降级安全帧
        if (using_fallback_frame || !frameIsFresh(received_at)) {
            setSafeOutput(translator); prediction_valid = false;
        }

        writeFrame(*serialPort, translator);   // ⑨ 写回串口
    }
    STOP_WEB_SERVER();
}
```

---

## ① copyLatestFrame（`main.cpp`）

```cpp
bool copyLatestFrame(Translator& destination,
                     std::chrono::steady_clock::time_point& received_at) {
    std::lock_guard<std::mutex> lock(message_mutex);   // 加锁
    if (!message_available) return false;
    destination = temp;               // 拷贝读线程写入的最新帧
    received_at = message_received_at;
    return true;
}
```

**讲解**：唯一跨线程共享的就是 `temp`（读线程写、这里读），用 `message_mutex` 保护。拷贝到 `translator` 后，后面的处理完全在本地副本上进行，不再碰锁——「一把锁护一个共享变量」，后面 8 步全部无锁。

## ② 校验（`main.cpp` + `src/AppSupport.cpp`）

```cpp
// main.cpp
bool frameIsFresh(const std::chrono::steady_clock::time_point& received_at) {
    return std::chrono::steady_clock::now() - received_at <= kMessageMaxAge; // 500ms
}
bool hasFiniteControlInput(const MessData_AutoAim& message) noexcept {
    return std::isfinite(message.yaw) && std::isfinite(message.pitch) &&
           std::isfinite(message.roll) && std::isfinite(message.bias);
}

// src/AppSupport.cpp
bool isAutoAimStatus(std::uint8_t status) noexcept { return status == 0U || status == 5U; }
```

**讲解**：三道闸——①帧没超过 500ms 旧；② status 是 0(红)/5(蓝)；③ yaw/pitch/roll/bias 都是有限数（防 NaN/Inf）。任何一道不过 → `setSafeOutput` 写安全帧，直接 `continue`。

## ③ gimbalStateFromMessage（`src/AppSupport.cpp`）

```cpp
GimbalState gimbalStateFromMessage(const MessData_AutoAim& message) noexcept {
    return {message.yaw, message.pitch, message.roll, message.bias};
}
```

**讲解**：从 64 字节帧抽出电控给的 4 个量（云台 yaw/pitch/roll + 预测偏置 bias），打包成 `GimbalState`，用于 PnP 坐标变换和 Planner 预测时域。

## ④ get_pic（`student/src/StudentTasks.cpp`）

```cpp
bool get_pic(cv::Mat& pic) {
    if (!ensureCameraStarted()) { pic.release(); return false; }  // 惰性开相机
    static rmv::camera::Frame frame;
    if (!cameraInstance().grab(frame) || frame.data.empty()) { pic.release(); return false; }
    pic = cv::Mat(frame.height, frame.width, CV_8UC3, frame.data.data()).clone();
    cv::flip(pic, pic, -1);   // 机上相机倒装，转 180°
    return true;
}
```

**讲解**：相机对象用函数内 `static` 单例（进程只开一次）；`clone()` 把 16 字节对齐的相机缓冲拷到 64 字节对齐的 Mat（避免 OpenCV AVX2 resize 段错误）。失败 `release()` 返回 false，主循环黑帧兜底，连续 3 次退出。

## ⑤ armor_detect（`student/src/StudentTasks.cpp`）

```cpp
std::vector<ArmorDetection> armor_detect(const cv::Mat& pic, TeamColor enemy_color) {
    static rmv::detect::Detector detector("detect/models/armor_detector.onnx");
    static rmv::detect::ArmorClassifier classifier("detect/models/armor_classifier.onnx");
    const auto dets = detector.detect(pic);
    const int target_class = (enemy_color == TeamColor::Blue) ? 2 : 1;  // 敌方颜色→类别
    for (const auto& d : dets) {
        if (d.class_id != target_class) continue;
        const auto armor = rmv::detect::extractArmor(pic, d.box, d.class_id); // 灯条提取+配对
        if (!armor.valid) continue;
        // 分类出 target_id，只保留底盘类 2/3/4/6/7
        ...
        results.push_back(ad);
    }
    return results;
}
```

**讲解**：检测器、分类器都是函数内 `static`（进程只加载一次 onnx）。`enemy_color` 由 `enemyColorFromStatus(status)`（AppSupport.cpp）反推：我方红(0)→打蓝(留 class 2)、我方蓝(5)→打红(留 class 1)。框内跑 `extractArmor`（灯条提取+配对），分类器读数字得 `target_id`。

## ⑥ armor_solve（`student/src/StudentTasks.cpp`）

```cpp
std::vector<ArmorPose> armor_solve(detections, camera, gimbal) {
    g_camera_cache = camera;                       // 存给 ekf_predict 用
    for (const auto& det : detections) {
        // 按板尺寸建 3D 点 → solvePnP(IPPE) → 重投影误差>10 拒掉
        cv::solvePnP(object_points, image_points, camera_matrix, dist, rvec, tvec, false, IPPE);
        // 相机系 → 云台系 → 世界系
        const cv::Vec3d p_gimbal = camera.rotation_camera_to_gimbal * tvec + camera.translation_camera_to_gimbal;
        const cv::Vec3d p_world  = gimbal_to_world(p_gimbal, gimbal);
        // 法向 yaw（重投影搜索，锚定云台朝向+π）
        const double normal_yaw = optimizeArmorYaw(object_points, image_points, camera, gimbal, p_world);
        pose.position_camera_m = p_cam;
        pose.position_gimbal_m = p_world;          // 字段名历史遗留，实际存世界系
        pose.armor_normal_yaw  = normal_yaw;
        ...
    }
}
```

**讲解**：每个装甲板 → PnP 求位姿 → 坐标变换链（相机→云台→世界）→ 法向 yaw 用重投影搜索（不取会翻面的 PnP 直接法向）。输出 `ArmorPose` 喂给 EKF。

## ⑦ ekf_predict（`student/src/StudentTasks.cpp`）

```cpp
PredictionResult ekf_predict(observations, gimbal, timestamp_seconds) {
    static rmv::tracker::EKFEstimator estimator;   // 单例，状态跨帧保持
    static rmv::tracker::Planner planner;
    estimator.update(observations, timestamp_seconds, gimbal, g_camera_cache);
    PredictionResult result = planner.plan(estimator.state(), estimator.trackingState(),
                                           gimbal, timestamp_seconds, estimator.targetId());
    result.target_obs_index = estimator.associatedObservationIndex(result.best_slot);
    return result;
}
```

**讲解**：EKF 和 Planner 都是 `static` 单例——「状态跨帧」的关键，车体状态存在 estimator 里下一帧接着用。`estimator.update` 做预测+关联+更新+状态机；`planner.plan` 做前馈外推+选板+弹道。

## ⑧ applyPrediction（`src/AppSupport.cpp`）

```cpp
bool applyPrediction(Translator& translator, const PredictionResult& prediction) noexcept {
    if (prediction.state == TrackingState::NoTarget || !finitePrediction(prediction)) {
        setSafeOutput(translator); return false;   // 无目标/非有限 → 安全帧
    }
    auto& message = translator.message;
    message.armor_flag     = prediction.target_id;
    message.distance       = prediction.distance_m;
    message.coo_x          = prediction.center_x_m * 1000.0F;  // 米→毫米
    message.coo_y          = prediction.center_y_m * 1000.0F;
    message.target_yaw     = prediction.target_yaw;
    message.target_pitch   = prediction.target_pitch;
    message.yaw_vel        = prediction.target_yaw_velocity;
    message.pitch_vel      = prediction.target_pitch_velocity;
    message.crc            = static_cast<uint16_t>(prediction.state);  // 0/1/2
    message.fire_allowance = prediction.state == TrackingState::Stable ? 1U : 0U;
    return true;
}
```

**讲解**：把内部单位（米/弧度）映射到 64 字节协议。关键安全逻辑：`fire_allowance` 只在 `Stable` 给 1，其余全 0。`coo_x/coo_y` 米转毫米。`crc` 存 EKF 状态。

## ⑨ writeFrame（`main.cpp`）

```cpp
bool writeFrame(SerialPort& serialPort, Translator& frame) {
    frame.message.head = kFrameHead;   // 0x71
    frame.message.tail = kFrameTail;   // 0x4C
    const int written = serialPort.Write(frame.data, kCommunicationFrameSize); // 64 字节
    if (written != kCommunicationFrameSize) {
        std::cerr << "串口写入失败\n";
        shutdown_requested.store(true);   // 写失败 → 触发退出
        return false;
    }
    return true;
}
```

**讲解**：填帧头帧尾，一次性写满 64 字节。写失败（短写）说明串口挂了，置 `shutdown_requested` 干净退出。

---

## 总结

```
读线程:  串口 ──Read()──> temp ──────────┐
                                      (mutex)
处理线程: copyLatestFrame ──> 校验 ──> gimbalState
         ──> get_pic ──> armor_detect ──> armor_solve ──> ekf_predict
         ──> applyPrediction ──> writeFrame ──> 串口
                                          ▲
                             (每个 static 单例存状态跨帧)
```
