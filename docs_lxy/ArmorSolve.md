# ArmorSolve 逐步讲解（PnP 位姿解算）

> 文件：`student/src/StudentTasks.cpp` 的 `armor_solve`。
> 输入：识别出的装甲板（4 角点 + size + target_id）+ 相机参数 + 云台姿态；输出：每块板的 `ArmorPose`（世界坐标 + 法向 yaw）。

```
armor_solve
   ├─ ① 建 3D 物点
   ├─ ② 2D 角点转 double
   ├─ ③ solvePnP (IPPE) → 相机系 tvec
   ├─ ④ 深度非法拒绝
   ├─ ⑤ 重投影误差拒绝
   ├─ ⑥ 相机系 → 云台系（固定外参）
   ├─ ⑦ 云台系 → 世界系（云台当前姿态）→ p_world
   ├─ ⑧ optimizeArmorYaw(...) ← 法向 yaw 搜索
   └─ ⑨ 组装 ArmorPose
```

## ① 建 3D 物点

```cpp
const double width = (det.size == ArmorSize::Large) ? 0.230 : 0.135;  // 大板宽230mm / 小板135mm
const double hw = width * 0.5, hh = 0.057 * 0.5;                       // 高 57mm
const std::vector<cv::Point3d> object_points = {
    {-hw, -hh, 0.0}, {hw, -hh, 0.0}, {hw, hh, 0.0}, {-hw, hh, 0.0}};
//  左上              右上            右下          左下
```

**作用**：按装甲板真实尺寸（米）在板平面上建 4 个 3D 角点，顺序和检测到的 2D 角点 `corners` 严格一致（左上/右上/右下/左下）。PnP 靠这 4 对「3D 世界点 ↔ 2D 图像点」解位姿。

## ② 2D 角点转 double

```cpp
std::vector<cv::Point2d> image_points;
for (const auto& c : det.corners) image_points.emplace_back(c.x, c.y);  // Point2f → Point2d
```

**作用**：`det.corners` 是 `Point2f`（float），`solvePnP` 要 `Point2d`（double）。统一转 double，和内参（CV_64F）、物点（Point3d）一致，否则类型混用会触发断言崩溃。

## ③ solvePnP (IPPE)

```cpp
cv::Mat rvec, tvec;
if (!cv::solvePnP(object_points, image_points, camera.camera_matrix,
                  camera.distortion_coefficients, rvec, tvec, false,
                  cv::SOLVEPNP_IPPE)) {
    continue;   // 解算失败，丢弃这块板
}
```

**作用**：核心一步。用 4 对点对应 + 内参 + 畸变，解出板在**相机系**的位姿（`rvec` 旋转 + `tvec` 平移）。`SOLVEPNP_IPPE` 是平面目标专用算法（装甲板共面 4 点），比通用 ITERATIVE 对平面退化更稳。

## ④ 深度非法拒绝

```cpp
if (tvec.at<double>(2) <= 0.0) continue;   // z<=0 在相机后方
```

**作用**：相机系 z 轴朝前（光轴方向），`tvec[2]` 是板在光轴方向的距离。`z <= 0` 说明解出来板在相机**背后**——物理不可能，是 PnP 的伪解，丢弃。

## ⑤ 重投影误差拒绝

```cpp
std::vector<cv::Point2d> projected;
cv::projectPoints(object_points, rvec, tvec, camera.camera_matrix,
                  camera.distortion_coefficients, projected);
double reproj = 0.0;
for (int i = 0; i < 4; ++i) reproj += cv::norm(projected[i] - image_points[i]);
reproj /= 4.0;
if (reproj > 10.0) continue;   // 平均像素误差 >10px 丢弃
```

**作用**：把 3D 点用解出的位姿**投回图像**，和检测到的角点比平均像素距离。误差大说明「解出来的位姿和实际看到的对不上」——解错了，丢弃，避免错误位姿污染 EKF。

## ⑥ 相机系 → 云台系（固定外参）

```cpp
const cv::Mat p_gimbal_mat = camera.rotation_camera_to_gimbal * tvec +
                             camera.translation_camera_to_gimbal;
const cv::Vec3d p_cam(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
const cv::Vec3d p_gimbal(p_gimbal_mat.at<double>(0), p_gimbal_mat.at<double>(1),
                         p_gimbal_mat.at<double>(2));
```

**作用**：把相机系位置 `tvec` 转到云台系。`rotation_camera_to_gimbal` 把 OpenCV 相机坐标（右/下/前）拧成云台坐标（前/左/上），`translation_camera_to_gimbal` 是相机在云台上的安装偏移。这俩是**标定死的固定外参**，每帧不变。

## ⑦ 云台系 → 世界系（云台当前姿态）

```cpp
const cv::Vec3d p_world = gimbal_to_world(p_gimbal, gimbal);   // R_yaw·R_pitch·R_roll
```

**作用**：云台会转（yaw/pitch/roll 每帧变），用**电控发来的当前云台姿态**把云台坐标转到固定不动的世界系。转完 `p_world` 就是车体所在的固定世界系坐标，EKF 才能用「匀速平移+旋转」模型跟踪。

## ⑧ optimizeArmorYaw（法向 yaw 搜索）

```cpp
const double normal_yaw = optimizeArmorYaw(object_points, image_points, camera, gimbal, p_world);
```

**作用**：位置已由 ③~⑦ 求好、固定不动，这一步只求「板面朝哪」（法向 yaw）。因为 PnP 对平面目标有 ±π 法向歧义（会翻面），所以不取 PnP 直接法向，而是把 yaw 当唯一自由度、锚定「云台朝向+π」在 ±70° 内搜重投影误差最小的那个。结果是 `armor_normal_yaw`，作 EKF 第 4 维观测。

## ⑨ 组装 ArmorPose

```cpp
ArmorPose pose;
pose.detection          = det;           // 识别信息（含 target_id）
pose.position_camera_m  = p_cam;         // 相机系位置
pose.position_world_m  = p_world;        // 世界系坐标（前/左/上）
pose.armor_normal_yaw   = normal_yaw;    // 法向方位角（EKF 第4维观测）
pose.armor_tilt         = 0.2618;        // 板倾角固定 15°
pose.reprojection_error = reproj;        // 重投影误差
results.push_back(pose);
```

**作用**：把所有结果打包成一个 `ArmorPose`，作为这块板的观测输出。整帧所有板的 `vector<ArmorPose>` 返回后，喂给 `ekf_predict`。

---

## 总结

①建 3D 点 ②转 double → ③ PnP 解相机系位姿 → ④⑤ 拒绝非法/离群 → ⑥⑦ 相机→云台→世界坐标变换 → ⑧ 搜法向 yaw → ⑨ 打包输出。
