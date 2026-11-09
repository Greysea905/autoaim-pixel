# Tracker 逐步讲解（EKF 旋转车体跟踪）

> 文件：`tracker/include/EKFEstimator.hpp` / `tracker/src/EKFEstimator.cpp`（+ `Planner.*`、`BulletModel.*`、`TrackState.hpp`）。
> 输入：每帧的 `ArmorPose` 观测（相机系球坐标 + 世界系法向）；输出：旋转车体的状态估计 + Planner 的瞄准结果。

## 1. 一段话讲解

EKF 负责把每帧的装甲板观测融合成对**旋转车体**的连续估计。难点在目标不是单个静止物体，而是四块板绕中心旋转的整车——板会轮流对着你、随时切换。所以状态是 10 维整车（旋转中心 + 速度 + 两对板高度/半径 + 车体朝向 yaw + 角速度 w），观测用「球坐标 + 世界系法向」。EKF 每帧做「预测（匀速平移+旋转）→ 关联（最近邻 + 马氏门限）→ 更新 → 状态机（收敛/发散/丢失）」，输出稳定的车体状态，供 Planner 做前馈预测和瞄准。

## 2. 10 维状态向量（世界系）

$$\mathbf x = [cx,\ vx,\ cy,\ vy,\ z_0,\ z_1,\ r_0,\ r_1,\ yaw,\ w]^\top$$

| 下标 | 符号 | 含义 | 说明 |
|---|---|---|---|
| 0 | `cx` | 旋转中心 x | 世界系（前） |
| 1 | `vx` | 中心速度 x | 匀速 |
| 2 | `cy` | 旋转中心 y | 世界系（左） |
| 3 | `vy` | 中心速度 y | 匀速 |
| 4 | `z0` | 板 0/2 高度 | 对侧板等高 |
| 5 | `z1` | 板 1/3 高度 | 对侧板等高 |
| 6 | `r0` | 板 0/2 半径 | 物理常数 |
| 7 | `r1` | 板 1/3 半径 | 物理常数 |
| 8 | `yaw` | 板 0 外法向方位 | 车体朝向 |
| 9 | `w` | 角速度 | 匀速旋转 |

## 3. 4 维观测向量（混合系）

$$\mathbf z = [cam\_yaw,\ cam\_pitch,\ cam\_distance,\ world\_normal\_yaw]^\top$$

| 下标 | 符号 | 坐标系 | 含义 |
|---|---|---|---|
| 0 | `cam_yaw` | 相机系 | 板中心方位角（0=前，+z 光轴） |
| 1 | `cam_pitch` | 相机系 | 板中心俯仰角 |
| 2 | `cam_distance` | 相机系 | 板中心距离 |
| 3 | `world_normal_yaw` | 世界系 | 板法向方位角（= yaw+off） |

## 4. 状态转移方程（过程模型）

匀速平移 + 匀速旋转（CV 模型，无加速度）：

$$\begin{cases}
cx_{k+1} = cx_k + vx_k\,\Delta t \\[2pt]
cy_{k+1} = cy_k + vy_k\,\Delta t \\[2pt]
yaw_{k+1} = \mathrm{wrapPi}(yaw_k + w_k\,\Delta t)
\end{cases}$$

其余 7 维（`vx, vy, z0, z1, r0, r1, w`）保持不变。

状态转移矩阵 F（10×10）：

```
F = I，除了：
F(CX, VX) = Δt
F(CY, VY) = Δt
F(YAW, W) = Δt
```

协方差传播：`P ← F·P·Fᵀ + Q`（Q 是过程噪声，NIS 驱动自适应缩放）。

## 5. 观测方程（非线性）

给定状态 x 和槽位 slot（`off = slot·π/2`，`r = r0/r1`，`z = z0/z1`）：

板世界位置：
$$p_w = \begin{pmatrix} cx + r\cos(yaw+off) \\ cy + r\sin(yaw+off) \\ z \end{pmatrix}$$

世界系 → 相机系：
$$p_c = R_{w2c}\, p_w + t_{w2c}$$

球坐标：
$$cam\_yaw = \operatorname{atan2}(p_c.x,\ p_c.z)$$
$$cam\_pitch = \operatorname{atan2}(p_c.y,\ \sqrt{p_c.x^2 + p_c.z^2})$$
$$cam\_distance = |p_c|$$
$$world\_normal\_yaw = \mathrm{wrapPi}(yaw + off)$$

> 观测雅可比用**数值差分**（中心差分，角度维 wrapPi 差分），不手推解析。

## 6. 各环节所在的坐标系

| 环节 | 坐标系 |
|---|---|
| 状态 x（10 维） | **世界系**（前/左/上，IMU 初始化） |
| 过程模型 predict | **世界系**（匀速平移 + 旋转） |
| 观测前 3 维（cam_yaw/cam_pitch/cam_distance） | **相机系**（球坐标，光心为原点） |
| 观测第 4 维（world_normal_yaw） | **世界系**（法向方位角） |
| 观测函数 h | 世界 → 相机（`R_w2c`），混合系输出 |
| 关联 / 更新 | 观测空间（混合系） |

关键：**状态在世界系（固定不动），观测前 3 维在相机系（随光轴锚定）**。球坐标观测让 R 天然对角、无需每帧旋转；第 4 维法向 yaw 在世界系，直接观测车体朝向。

## 7. 算法亮点

1. **整车建模（核心）**：不把每块板当独立目标，而是建「旋转中心 + yaw + w + 半径」的 10 维整车，板切换时靠 `yaw + slot·π/2` 统一表达。
2. **球坐标观测**：前 3 维相机系球坐标，R 天然对角、随光轴锚定，解决笛卡尔观测各向异性锚错方向的问题。
3. **数值雅可比**：观测雅可比用中心差分，免手推解析式，角度维 wrapPi 差分避免 ±π 伪导数。
4. **对侧板等高 + 半径/高度差物理常数**：z0/z1 只存两对板（对侧等高），半径/高度差跨 reset 保留（半径改为每次从预设重收敛，防塌缩死循环）。
5. **NIS 发散检测 + 状态机**：新息平方和做卡方检验判发散 → reset；`converge_frames` 收敛锁存 `stable_stage`，开火只看是否 Stable。
6. **动态 R（距离线性）**：距离和法向 yaw 的观测噪声随距离线性放大，远处观测自动降权。
7. **NIS 驱动 Q**：上一帧最小 NIS 低 → Q 收缩平滑，NIS 飙高 → Q 全量加速收敛。

---

## 一句话串起来

状态在世界系（10 维整车）→ predict 匀速平移+旋转 → 观测 h 世界转相机球坐标 + 世界法向 → 最近邻关联 → correct 更新 → 状态机（收敛/发散/丢失）→ 输出车体状态给 Planner。

---

## 8. 观测方程详解（逐步拆解 + 代码）

观测方程回答一个问题：**「如果车体处于状态 x，那第 slot 块板在相机眼里会是什么样？」** 由 `h(x, slot)` → `worldToObs(pw, world_yaw)` 两个函数组成。

### 8.1 状态转移方程（简单，对照 predict）

```cpp
// predict(dt) 完整代码（tracker/src/EKFEstimator.cpp）：
void EKFEstimator::predict(double dt) {
    if (dt <= 0.0) return;

    // ① 状态更新：匀速平移 + 匀速旋转（只有 cx/cy/yaw 三个量变）
    x_(CX) += x_(VX) * dt;                  // cx' = cx + vx·dt
    x_(CY) += x_(VY) * dt;                  // cy' = cy + vy·dt
    x_(YAW) = wrapPi(x_(YAW) + x_(W) * dt); // yaw' = yaw + w·dt

    // ② 状态转移矩阵 F（单位矩阵 + 三个 dt 项）
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(kDim, kDim);
    F(CX, VX) = dt;
    F(CY, VY) = dt;
    F(YAW, W) = dt;
    P_ = F * P_ * F.transpose();            // 协方差传播

    // ③ 过程噪声 Q（NIS 驱动自适应缩放 + 二重积分器离散化）
    const double q_scale = (last_nis_ < cfg_.nis_stable_threshold) ? cfg_.q_stable_scale : 1.0;
    const double qp = cfg_.q_pos * q_scale;   // 平移加速度噪声
    const double qy = cfg_.q_yaw * q_scale;   // 旋转角加速度噪声
    const double t2 = dt * dt, t3 = t2 * dt, t4 = t2 * t2;
    P_(CX, CX) += qp * t4 / 4.0; P_(CX, VX) += qp * t3 / 2.0;   // 位置-速度块
    P_(VX, CX) += qp * t3 / 2.0; P_(VX, VX) += qp * t2;
    P_(CY, CY) += qp * t4 / 4.0; P_(CY, VY) += qp * t3 / 2.0;
    P_(VY, CY) += qp * t3 / 2.0; P_(VY, VY) += qp * t2;
    P_(YAW, YAW) += qy * t4 / 4.0; P_(YAW, W) += qy * t3 / 2.0;  // yaw-w 块
    P_(W, YAW) += qy * t3 / 2.0; P_(W, W) += qy * t2;
    // 高度差恒定：z0/z1 共享同一随机游走（正相关），差 z0-z1 不漂
    P_(Z0, Z0) += cfg_.q_z * dt; P_(Z1, Z1) += cfg_.q_z * dt;
    P_(Z0, Z1) += cfg_.q_z * dt; P_(Z1, Z0) += cfg_.q_z * dt;
    P_(R0, R0) += cfg_.q_r * dt; P_(R1, R1) += cfg_.q_r * dt;     // 半径随机游走
}
```

「匀速平移 + 匀速旋转」，只有 3 个量在变，其余 7 维（速度/高度/半径/角速度）是常数。

### 8.2 第 1 步：状态 → 板的世界位置（+r 约定）

```cpp
// h(x, slot) 完整代码（tracker/src/EKFEstimator.cpp）：
Eigen::Vector4d EKFEstimator::h(const Eigen::VectorXd& x, int slot) const {
    const double off = static_cast<double>(slot) * kHalfPi;    // off = slot·π/2
    const int r_idx  = (slot % 2 == 0) ? R0 : R1;
    const double r   = x(r_idx);                                // r = r0(slot偶) / r1(slot奇)
    const double a   = x(YAW) + off;                            // a = yaw + off（板外法向方位角）
    return worldToObs(Eigen::Vector3d(x(CX) + r * std::cos(a),  // p_w = (cx + r·cos a, cy + r·sin a, z)
                                      x(CY) + r * std::sin(a),
                                      x(slot % 2 == 0 ? Z0 : Z1)),
                      a);                                        // world_yaw = a
}
```

**直观**：车体绕中心转，4 块板在「中心 + 半径」的圆周上、各差 90°。`yaw` 是板 0 朝向，第 slot 块板朝向 = `yaw + slot·π/2`，板位置 = 旋转中心 + 半径沿该方向伸出去。

$$p_w = \begin{pmatrix} cx + r\cos(yaw+off) \\ cy + r\sin(yaw+off) \\ z \end{pmatrix}$$

### 8.3 第 2 步：世界位置 → 相机系

```cpp
// worldToObs(pw, world_yaw) 完整代码（tracker/src/EKFEstimator.cpp）：
Eigen::Vector4d EKFEstimator::worldToObs(const Eigen::Vector3d& pw, double world_yaw) const {
    const Eigen::Vector3d pc = R_w2c_ * pw + t_w2c_;   // p_c = R_w2c · p_w + t_w2c
    const double d = pc.norm();                         // d = |p_c|
    return Eigen::Vector4d(std::atan2(pc.x(), pc.z()),              // cam_yaw = atan2(p_c.x, p_c.z)
                           std::atan2(pc.y(), std::hypot(pc.x(), pc.z())),  // cam_pitch = atan2(p_c.y, √(p_c.x²+p_c.z²))
                           d,                                        // cam_distance = |p_c|
                           wrapPi(world_yaw));                       // world_normal_yaw = wrapPi(yaw+off)
}
```

`R_w2c_`/`t_w2c_` 是「世界→相机」的旋转/平移，每帧 `update` 开头由相机外参 + 云台当前姿态算好缓存。把板从固定世界系转到「光心为原点」的相机系。

$$p_c = R_{w2c}\, p_w + t_{w2c}$$

### 8.4 第 3 步：相机系 → 球坐标（前 3 维）

```cpp
const double d = pc.norm();
return Eigen::Vector4d(std::atan2(pc.x(), pc.z()),              // cam_yaw
                       std::atan2(pc.y(), std::hypot(pc.x(), pc.z())),  // cam_pitch
                       d,                                        // cam_distance
                       wrapPi(world_yaw));                       // world_normal_yaw
```

相机系是 OpenCV 约定（+x 右、+y 下、+z 前/光轴），转球坐标：

| 维度 | 数学 | 含义 |
|---|---|---|
| `cam_yaw` | $\operatorname{atan2}(p_c.x,\ p_c.z)$ | 水平方位角（0=正前方） |
| `cam_pitch` | $\operatorname{atan2}(p_c.y,\ \sqrt{p_c.x^2+p_c.z^2})$ | 垂直俯仰角（0=水平） |
| `cam_distance` | $\|p_c\|$ | 到光心直线距离 |

这三个量是「相机怎么看到这块板」的自然描述，R 噪声天然对角（三个方向独立），不用每帧旋转 R。

### 8.5 第 4 维：世界法向（不做相机变换）

```cpp
wrapPi(world_yaw)   // = wrapPi(a) = wrapPi(yaw + off)
```

第 4 维**不做相机变换**，直接取板外法向在世界系的方位角 = `yaw + off`。因为这一维要**直接观测车体朝向 yaw**，而相机系随云台转、表达不了固定世界系里的朝向。

### 8.6 完整链路拼起来

```
状态 x（世界系）
  └─ 第1步：+r 约定 → 板世界位置 pw = (cx+r·cos a, cy+r·sin a, z)
       └─ 第2步：R_w2c → 相机系 pc
            └─ 第3步：atan2/norm → cam_yaw, cam_pitch, cam_distance（相机系）
  └─ 第4维：wrapPi(yaw+off) → world_normal_yaw（世界系）
```

**为什么混合系**：状态在世界系（车体好建模）；前 3 维观测在相机系（光轴锚定、R 对角）；第 4 维在世界系（直接观测车体朝向）。
