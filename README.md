# autoaim-pixel
> 在原项目 `rmv_autoaim_Greysea905` 基础上做优化尝试。内容暂由 ai 整理，有待梳理。

## 1. PixelEKF 像素级卡尔曼滤波

**改动原因**：传统自瞄把 PnP 解出的位姿（球坐标 yaw/pitch/dist）直接当观测喂 EKF，但 PnP 误差**非高斯、强相关、距离二次恶化**，违背 EKF 的高斯假设，导致调参难、难维护。

**实现思路**：把观测从「PnP 位姿（4 维）」换成「8 像素角点 + 1 板法向 yaw（9 维混合观测）」，位姿（距离/yaw）变成**隐藏状态**，由投影模型 + 像素观测推断。关键点：

- 距离不再直接测量，而是隐藏状态；投影雅可比 `∂u/∂Z = -fx·X/Z²` 自动编码「越远距离越弱可观」，无需动态距离 R。
- 板法向 yaw 用 `optimizeArmorYaw`（1D 重投影搜索、锚定云台+π）直通观测，解决「正对视角下 yaw 弱可观导致 w 漂移」的问题。

## 2. 15° 板安装角

**改动原因**：装甲板不是竖直的，而是**后仰 15°**（法向相对水平面上仰 15°）。旧模型假设板竖直，导致板高被压短 `cos15°≈0.966`，距离被**高估 ~3.4%**（3m 处约 10cm）。

**实现思路**：在 `plateCornersWorld` 里，把竖直边 `up` 从 `(0,0,1)` 改成后仰 15° 的方向：

```
up = (-cos a·sin15°, -sin a·sin15°, cos15°)
```

板的法向随之变为 `(cos a·cos15°, sin a·cos15°, sin15°)`（上仰 15°），距离无偏。**注意**：这个 15° 是相对**底盘竖直轴**的，和下面的 tilt 组合成 `R_tilt·R_15°·R_yaw`。

## 3. tilt（mag/azi）底盘倾斜

**改动原因**：轮腿车（高低腿小陀螺）的**旋转轴不垂直地面**，板绕倾斜轴转，导致板高度随 yaw 振荡、板法向倾斜。当前模型假设绕竖直轴转，对倾斜车是错的。

**实现思路**：加两个**准静态状态**：

- `mag`（倾斜角，与 +z 夹角）
- `azi`（倾斜方向，旋转轴在 xoy 投影朝向）

板绕倾斜轴 `n = (sinφcosψ, sinφsinψ, cosφ)` 转，板高度 = `center_z - mag·r·cos(yaw+off-azi)`。tilt 的可观测性：**旋转时**通过板高度振荡可观测；静止时不可观测（但静止时 tilt 也不重要）。

## 4. IMM（交互多模型）

**改动原因**：单模型在「旋转反向」时有**机动滞后**——低 Q 平滑但跟不上反向，高 Q 快但稳态抖。调参只能在两端之间移动，不能同时兼顾。

**实现思路**：用 2 模型 IMM，把「稳态」和「机动」分开：

- **稳态模型**：CV+CAR（`q_aw=0`，w 恒定）
- **机动模型**：CV+CAA（`q_aw>0`，w += aw·dt，w 能 ramp）

两者共用 13 维状态（含 `aw` 角加速度），差异只在 `q_aw`。每帧做 **mixing → 各自滤波 → 概率更新 → 融合**，Markov 概率软切换：反向时机动模型概率冲高、快速跟上，反向结束后回到稳态。

## 状态与观测

- **状态（13 维世界系）**：`[cx,vx,cy,vy,z0,z1,r0,r1,yaw,w,aw,mag,azi]`
- **观测（9 维）**：`[u0,v0,u1,v1,u2,v2,u3,v3, yaw]`

## 构建运行

```bash
cd /home/niki/Greysea/RMV27/autoaim-pixel
cmake -S . -B build
cmake --build build --target pixelekf_video_debug -j4
./build/pixelekf_video_debug EKF_video/blue.avi config/camera.yaml 8080 blue
```

- 视频：`EKF_video/blue.avi` / `red.avi`
- 相机标定：`config/camera.yaml`（或 `camera_gimbal.yaml`）
- Web UI：`http://localhost:8080`，改 `config/pixel_ekf.yaml` 后重跑即可，无需重编译

## 目录结构

```
PixelEKF/            核心：PixelEKF（像素直滤 EKF）+ IMM（交互多模型）
pixelekf_video_debug.cpp   视频离线调参工具（Web UI）
config/pixel_ekf.yaml      调参配置
tracker/              旧 EKFEstimator（4 维球坐标，对照组）+ Planner
student/src/           armor_detect / armor_solve / optimizeArmorYaw
web_debug/             Web UI
EKF_video/             blue.avi / red.avi 实机录像
```

## 相关项目

- `rmv_autoaim_Greysea905`：训练项目，本项目的代码来源
- `aimbot_26`：去年实际自瞄（传统 PnP + 球坐标 EKF）
- `aimbot_lxy`：今年实际自瞄，计划从本项目迁移改进
