# autoaim-pixel 项目说明

像素直滤 EKF（pixel-direct EKF）实验项目，从 `rmv_autoaim_Greysea905` 拷贝而来。目标：在 RoboMaster 自瞄（只打底盘装甲板，不打能量机关）里探索「用原始像素角点代替 PnP 位姿作为 EKF 观测」是否更鲁棒。

## 项目定位与关系

- **本项目**：实验田，验证像素直滤 EKF 的思路与效果，跑通后再考虑迁移到实际项目。
- **父项目** `rmv_autoaim_Greysea905`（`/home/niki/Greysea/RM2627_tasks/rmv_autoaim_Greysea905`）：训练项目，本项目的代码来源（EKFEstimator、Planner、armor_solve、detect 等）。**已回退干净（git 到 HEAD），不再改动**，只有零拷贝一处相机改动留在那。
- **去年实际项目** `aimbot_26`：传统「PnP + 球坐标 EKF」，13 维整车模型，15° 板倾角在输出端（calc_armor_back/AimAuto）固定加上。
- **今年实际项目** `aimbot_lxy`（会话工作目录）：计划从本项目迁移改进。

## 目录结构（关键部分）

- `PixelEKF/` —— 核心：像素直滤 EKF（`include/PixelEKF.hpp` + `src/PixelEKF.cpp`）
- `pixelekf_video_debug.cpp` —— 视频离线调参工具（跳过 PnP，角点直喂 PixelEKF，带 Web UI 曲线）
- `config/pixel_ekf.yaml` —— PixelEKF 调参（改完重跑即可，无需重编译）
- `tracker/` —— 旧 EKFEstimator（4 维球坐标，对照组）+ Planner + TrackState
- `student/src/StudentTasks.cpp` —— armor_detect / armor_solve / optimizeArmorYaw 等
- `tests/test_pixel_ekf.cpp` —— A/B 对比测试（**注意：还是旧 ArmorDetection 版，未改成 ArmorPose，暂不能编译**）
- `EKF_video/` —— blue.avi / red.avi 实机录制视频（无真值）

## 核心设计：像素直滤 EKF（9 维混合观测）

**动机**：传统方案把 PnP 位姿（球坐标 yaw/pitch/dist）直接当观测，但 PnP 误差非高斯、强相关、距离二次恶化，违背 EKF 高斯假设 → 调参难、难维护。

**方案**：观测 = 8 像素角点 + 1 板法向 yaw（9 维）；位姿（距离/yaw）成为隐藏状态，由投影模型 + 像素观测推断。

**状态（13 维世界系）**：`[cx,vx,cy,vy,z0,z1,r0,r1,yaw,w,aw,mag,azi]`
- 平移 CV + 旋转 CT（匀速自转）
- z0/z1：对侧板共享高度（槽 0/2 用 z0，槽 1/3 用 z1）；r0/r1：对侧板共享半径
- mag/azi：底盘倾斜（准静态，轮腿车旋转轴不垂直地面）——mag=倾斜角（与 +z 夹角）、azi=倾斜方向（旋转轴在 xoy 投影朝向）
- aw：角加速度（IMM 机动模型用；q_aw=0 稳态 w 恒定，q_aw>0 机动 w 可 ramp）

**观测（9 维）**：`[u0,v0,u1,v1,u2,v2,u3,v3, yaw]`
- 8 像素 = 4 角点（左上/右上/右下/左下）针孔投影
- yaw = `armor_normal_yaw`，来自 `optimizeArmorYaw`（1D 重投影搜索、锚定「云台+π」、避免 IPPE ± 翻面）

**观测方程 h**：板位置/法向绕倾斜轴 `n=(sinφcosψ, sinφsinψ, cosφ)` 转、再相对倾斜轴施加 15° 板安装角（竖直边 up 后仰 15°），然后世界→相机（R_w2c）→ 针孔投影（含深度 clamp `min_depth=0.1` 防 Z→0 NaN）；`h[8] = wrapPi(x(YAW)+slot·π/2)`。

**关键设计点**：
- 距离不再是直接测量，而是隐藏状态；投影雅可比 `∂u/∂Z=-fx·X/Z²` 自动编码「越远距离越弱可观」，无需动态距离 R。
- χ² 门限按 9 维：`mahalanobis_gate2 = nis_threshold = 16.92`（χ²(9) 95%），`nis_stable_threshold = 9.0`（χ²(9) 均值）。
- 相机零拷贝：`Frame.data` 从 std::vector 改成 cv::Mat（64 字节对齐），`get_pic` 去掉 `.clone()`。
- `initialize()` 用 `position_world_m`（armor_solve 已提供），不内部跑 PnP。

## 调参状态（当前能收敛）

`config/pixel_ekf.yaml` 里用户手调出的收敛配置：
- `q_pos = 100`、`q_yaw = 100`（低过程噪声 → 平滑）
- `sigma_pixel_u/v = 20`（高测量噪声 → 信任模型）
- `sigma_yaw_rad = 0.4`
- reset 门限已放宽过一轮后调回默认（低 Q + 高 R 收敛后不再需要放宽）

**代价**：低 Q + 高 R = 机动滞后（旋转反向时状态摆动+滞后），这是当前已知问题，下一步用 IMM 解决。

## 踩过的坑（别重犯）

1. **观测维度改变必须同步改 NIS/χ² 门限**：4 维球坐标 → 9 维，χ² 9.49→16.92，nis_stable_threshold 4.0→9.0。漏改会让自适应 Q 的平滑档永不触发。

2. **π 偏移假说（已证伪、已撤销）**：曾怀疑 `armor_normal_yaw` 与状态 yaw 差 π，给 h_8 加 +π，但实测没用（真实管线里旧 EKF 的 w 是对的），已撤销回 `wrapPi(x(YAW)+slot·π/2)`。约定本身是对的，别再动。

3. **「旋转 reset」根因是 yaw 可观测性不足**：像素角点在正对视角下 yaw 一阶不可观（foreshortening 对 yaw 导数≈0），纯像素 EKF 的 w 弱可观 → 板切换时漂移 → reset。解药是加 yaw 第 9 维观测（optimizeArmorYaw）。

4. **动态 R 的取舍**：旧球坐标的动态 R（距离/法向随距离放大），在像素方案里距离那维由 H 自动接管（不用），但 yaw 那维是独立测量、噪声随距离涨，理想应保留动态（当前是常数 0.4，未动态化也未调准）。

5. **reset 门限治不了「机动滞后」**：reset 是「要不要丢弃估计」，滞后是「跟得多快」（Q/R 决定），两者正交。滞后只能靠 IMM 或自适应 Q 解决。

## 当前状态（已完成的改动）

- **tilt 状态（mag/azi）**：已实现（12 维状态）。板绕倾斜轴 n 转，板高度 = center_z - mag·r·cos(yaw+off-azi)。**结果：依然收敛，但暂时没看出精度优化**——15° 只纠正 ~3.4% 距离偏置（3m 处约 10cm），且视频中靶车倾斜幅度未知，需看 `ekf_mag` 是否收敛到非零来判断 tilt 是否真正起作用。
- **15° 板安装角**：已实现。组合 R_tilt·R_15°·R_yaw（不硬编码 +z），竖直边 up 从 `(0,0,1)` 改为 `(-cos a·sin15°, -sin a·sin15°, cos15°)`。
- **Web UI**：观测向量只保留 `obs_yaw`；指标按管线顺序（相机→检测→观测→EKF→电控）排列；加了 `ekf_mag/ekf_azi`、`imm_prob_maneuver` 曲线。
- **IMM（2 模型）**：已实现（`PixelEKF/include/IMM.hpp` + `src/IMM.cpp`）。稳态 CV+CAR（q_aw=0）+ 机动 CV+CAA（q_aw>0，yaml 可调），统一 13 维状态，update 做 mixing→滤波→概率→融合。`imm_prob_maneuver` 基线 ≈0.25（Markov 稳态解，非 bug），反向时冲到 0~0.5（软切换，非 1）。

## 下一步计划

1. **IMM 调参验证**：调 `q_aw`（现 5.0，可试 20~50），看反向时 `imm_prob_maneuver` 是否冲高、`ekf_w` 是否更平滑。
2. **若旋转不够**：扩成 2×2 网格（加平移加速度 ax/ay、CA 模型），但先等"平移加速导致滞后"的证据。
3. **学长反馈待验证**：打旋转目标时 target_yaw 前半段有平段，疑似云台 bias 使「相对 w」非恒定（看相对 w 是否在 ramp）。

## 构建运行

```bash
cd /home/niki/Greysea/RMV27/autoaim-pixel
cmake -S . -B build
cmake --build build --target pixelekf_video_debug -j4
./build/pixelekf_video_debug EKF_video/blue.avi config/camera.yaml 8080 blue
```

- 相机标定：`config/camera.yaml`（或 `config/camera_gimbal.yaml`，机上相机）。
- 调参：改 `config/pixel_ekf.yaml` 后重跑，无需重编译。
- Web UI（8080）关键曲线：`obs_yaw`（观测 yaw）vs `ekf_yaw_rad`（估计 yaw，直接可比）、`ekf_yaw_rate_rad_s`（角速度 w）、`ekf_mag/ekf_azi`（tilt 是否收敛）、`ekf_z0_m/z1_m`（高度）。
