# RMVtraining 装甲板自瞄

> (此处有这样那样的想法和一段想要进行的不重要发言)

流程：串口收云台数据 → 相机取流 → 装甲板检测和分类/灯条匹配 → PnP 位姿解算 → 10 维整车 EKF 预测 → 弹道 + 前馈预测 → 回写 yaw/pitch。

## 数据流（双线程 + 双向串口闭环）

```
                      读线程 ReadFunction
电控 ──64字节(云台 yaw/pitch/roll/status/bias)──▶  串口Read → 找帧边界 → temp
                                                          │  (message_mutex)
                                                          ▼
                                                translator（当前云台状态）
                                                          │
                      处理线程 OperationFunction
                                                          │
            ┌─────────────────────────────────────────────┘
            ▼
        ① copyLatestFrame ── 拷贝最新云台帧
        ② get_pic ────────── 相机 → 图像(BGR)                  ┐
        ③ armor_detect ───── 图像 → 检测框+四角点+target_id     │ 阶段一
        ④ armor_solve ────── 四角点 → 世界坐标+法向yaw(ArmorPose)┘
        ⑤ ekf_predict ────── ArmorPose → target_yaw/pitch      ┐ 阶段二
        ⑥ applyPrediction ── 结果 → 64字节结构体                 │
        ⑦ writeFrame ─────── 64字节 → 串口                      ┘
            │
            ▼
电控 ◀──64字节(target_yaw/pitch/vel/crc/fire)── 处理线程写回
```

## 项目结构（按处理流程顺序组织）

```
rmv_autoaim_lxy/
├── main.cpp                  # 入口：双线程（读线程 + 处理线程）
│
├── serialPort/               # ① 串口：收云台数据（读线程）
│   ├── Communication.hpp     #    64 字节结构体 MessData_AutoAim
│   └── SerialPort.*          #    串口读写
│
├── camera/                   # ② 相机取流
│   └── MvsCamera.*           #    海康 MVS SDK 封装
│
├── detect/                   # ③ 识别
│   ├── Detector.*            #    YOLO 装甲板检测（OpenVINO）
│   ├── LightBarExtractor.*   #    灯条提取 + 配对
│   ├── ArmorClassifier.*     #    装甲板分类
│   └── models/               #    检测 / 分类 onnx 权重
│
├── student/                  # ④ 四个任务函数
│   └── StudentTasks.cpp      #    get_pic / armor_detect / armor_solve / ekf_predict
│
├── tracker/                  # ⑤ EKF + 选板 + 弹道 + 前馈
│   ├── EKFEstimator.*        #    10 维整车 EKF（球坐标观测 + NIS + 状态机）
│   ├── Planner.*             #    前馈预测 + 选板 + 弹道解算
│   ├── BulletModel.*         #    弹道模型
│   └── TrackState.hpp        #    状态定义（cx/vx/cy/vy/z0/z1/r0/r1/yaw/w）
│
├── include/ + src/           # 辅助：AppSupport（通信映射）+ AutoAimTypes（类型）
├── config/                   # 配置：相机标定 + EKF 调参 yaml
├── calib/ + calib_images/    # 标定脚本 calibrate.py + 标定图片（20 张）
├── web_debug/                # Web UI 数值曲线
├── tests/                    # test_ekf（EKF 回归测试）
├── EKF_video/                # 调参录像（blue.avi / red.avi）
│
├── ekf_video_debug.cpp       # 离线视频调参工具（可视化 + Web UI）
├── ekf_live_debug.cpp        # 实车调参工具（只读串口）
├── ekf_video_save.cpp        # 保存可视化
├── lightbar_debug.cpp        # 灯条提取调参工具（灰度剖面 + trackbar）
└── virtual_gimbal.py         # 虚拟云台（无硬件联调）

```

## 构建 / 运行

```bash
# 构建（需 OpenCV + Eigen + OpenVINO(conda task5) + MVS SDK）
cmake -S . -B build -DMVS_ROOT=/opt/MVS
cmake --build build -j

# 正式跑（写回串口）
./build/rmv_training /dev/ttyACM0 115200 config/camera_gimbal.yaml 8080

# 实车调参（只读串口，不写回）
./build/ekf_live_debug /dev/ttyACM0 115200 config/camera_gimbal.yaml 8080 blue

# 离线视频调参
./build/ekf_video_debug EKF_video/blue.avi config/camera_gimbal.yaml 8080 blue
```

## 关键工具

| 工具 | 用途 |
|---|---|
| `rmv_training` | 生产主程序（双线程，写回串口，无图像窗口、只 Web UI） |
| `ekf_live_debug` | 实车调参（相机 + 只读串口 + 可视化 + Web UI） |
| `ekf_video_debug` | 离线视频调参（收敛 / 切换 / 遮挡恢复可视化） |
| `ekf_video_save` | 把离线调参可视化保存成视频 |
| `lightbar_debug` | 灯条提取调参（二值化 / fitLine / 灰度剖面 + trackbar） |
| `test_ekf` | EKF 回归测试（期望 ALL EKF TESTS PASSED） |


