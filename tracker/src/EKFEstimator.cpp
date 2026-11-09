#include "EKFEstimator.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>

namespace rmv::tracker {

namespace {

constexpr double kHalfPi = kPi * 0.5;

// 云台 → 世界 旋转（R_yaw·R_pitch·R_roll，与 StudentTasks 的 gimbal_to_world 同款）。
Eigen::Matrix3d gimbalToWorldRot(const GimbalState& g) {
    const double cy = std::cos(g.yaw), sy = std::sin(g.yaw);
    const double cp = std::cos(g.pitch), sp = std::sin(g.pitch);
    const double cr = std::cos(g.roll), sr = std::sin(g.roll);
    Eigen::Matrix3d R;
    R << cy * cp, -sy * cr - cy * sp * sr, sy * sr - cy * sp * cr,
         sy * cp, cy * cr - sy * sp * sr, -cy * sr - sy * sp * cr,
         sp, cp * sr, cp * cr;
    return R;
}

// 把 3x3 / 3x1 的 cv::Mat(CV_64F) 拷成 Eigen。
Eigen::Matrix3d mat33(const cv::Mat& m) {
    Eigen::Matrix3d r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) r(i, j) = m.at<double>(i, j);
    return r;
}
Eigen::Vector3d vec3(const cv::Mat& m) {
    return Eigen::Vector3d(m.at<double>(0, 0), m.at<double>(1, 0), m.at<double>(2, 0));
}

// ArmorPose（相机系 3D 位置 + 世界系法向 yaw）→ 观测向量。
Eigen::Vector4d poseToObs(const ArmorPose& o) {
    const cv::Vec3d& p = o.position_camera_m;
    const double d = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    return Eigen::Vector4d(std::atan2(p[0], p[2]),          // θ：相机方位角（0=前，+z 光轴）
                           std::atan2(p[1], std::hypot(p[0], p[2])),  // φ：相机俯仰角
                           d,                                // d：距离
                           o.armor_normal_yaw);              // 世界系法向 yaw
}

}  // namespace

EKFEstimator::EKFEstimator(const EKFConfig& cfg) : cfg_(cfg) { reset(); }

void EKFEstimator::reset() {
    // 跨 reset 保留物理常数（半径 + 高度差，同一辆车不变）。
    if (initialized_) {
        saved_r0_ = x_(R0);
        saved_r1_ = x_(R1);
        saved_dz_ = x_(Z0) - x_(Z1);
        has_saved_ = true;
    }
    initialized_ = false;
    converged_frames_ = 0;
    last_timestamp_ = 0.0;
    last_obs_timestamp_ = 0.0;
    last_target_id_ = 0;
    nis_fail_frames_ = 0;
    last_nis_ = 1e18;   // reset 后首帧 predict 用全量 Q（加速重新收敛）
    state_ = TrackingState::NoTarget;
    x_ = Eigen::VectorXd::Zero(kDim);
    P_ = Eigen::MatrixXd::Zero(kDim, kDim);
}

double EKFEstimator::wrapPi(double a) {
    while (a > kPi) a -= 2.0 * kPi;
    while (a <= -kPi) a += 2.0 * kPi;
    return a;
}

TrackState EKFEstimator::state() const {
    TrackState s;
    s.cx = x_(CX); s.vx = x_(VX);
    s.cy = x_(CY); s.vy = x_(VY);
    s.z0 = x_(Z0); s.z1 = x_(Z1);
    s.r0 = x_(R0); s.r1 = x_(R1);
    s.yaw = x_(YAW); s.w = x_(W);
    return s;
}

TrackingState EKFEstimator::trackingState() const { return state_; }

int EKFEstimator::stableStage() const { return stable_stage_; }

std::uint8_t EKFEstimator::targetId() const { return last_target_id_; }

Eigen::Vector4d EKFEstimator::worldToObs(const Eigen::Vector3d& pw, double world_yaw) const {
    const Eigen::Vector3d pc = R_w2c_ * pw + t_w2c_;
    const double d = pc.norm();
    return Eigen::Vector4d(std::atan2(pc.x(), pc.z()),
                           std::atan2(pc.y(), std::hypot(pc.x(), pc.z())),
                           d, wrapPi(world_yaw));
}

Eigen::Vector4d EKFEstimator::h(const Eigen::VectorXd& x, int slot) const {
    const double off = static_cast<double>(slot) * kHalfPi;
    const int r_idx = (slot % 2 == 0) ? R0 : R1;
    const double r = x(r_idx);
    const double a = x(YAW) + off;
    // 板世界位置（+r 约定：板位置 = 中心 + r·(cos a, sin a)），再走世界→相机→球坐标。
    return worldToObs(Eigen::Vector3d(x(CX) + r * std::cos(a),
                                      x(CY) + r * std::sin(a),
                                      x(slot % 2 == 0 ? Z0 : Z1)),
                      a);
}

Eigen::MatrixXd EKFEstimator::numericalJacobian(const Eigen::VectorXd& x, int slot) const {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(4, kDim);
    const double eps = 1e-6;
    for (int col = 0; col < kDim; ++col) {
        Eigen::VectorXd xp = x, xm = x;
        xp(col) += eps;
        xm(col) -= eps;
        const Eigen::Vector4d hp = h(xp, slot);
        const Eigen::Vector4d hm = h(xm, slot);
        for (int row = 0; row < 4; ++row) {
            double diff = hp(row) - hm(row);
            // 角度维（θ/φ/yaw）跨 ±π 时用 wrap 差分，避免伪导数；距离 d 直接减。
            if (row == 0 || row == 1 || row == 3) diff = wrapPi(diff);
            H(row, col) = diff / (2.0 * eps);
        }
    }
    return H;
}

void EKFEstimator::predict(double dt) {
    if (dt <= 0.0) return;

    // 线性过程模型：匀速(平移) + 匀速(旋转)。
    x_(CX) += x_(VX) * dt;
    x_(CY) += x_(VY) * dt;
    x_(YAW) = wrapPi(x_(YAW) + x_(W) * dt);

    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(kDim, kDim);
    F(CX, VX) = dt;
    F(CY, VY) = dt;
    F(YAW, W) = dt;
    P_ = F * P_ * F.transpose();

    // 过程噪声 Q：位置-速度/yaw-角速度白噪声 + 高度/半径随机游走。
    // 自适应 Q（NIS 驱动）：上一帧最小 NIS 低=模型拟合好→Q 小（平滑压波动）；
    //   高=模型失配(运动变了)→Q 全量（快速收敛）。比按 Stable 锁存更能即时响应运动变化。
    const double q_scale = (last_nis_ < cfg_.nis_stable_threshold) ? cfg_.q_stable_scale : 1.0;
    const double qp = cfg_.q_pos * q_scale;
    const double qy = cfg_.q_yaw * q_scale;
    const double t2 = dt * dt, t3 = t2 * dt, t4 = t2 * t2;
    P_(CX, CX) += qp * t4 / 4.0; P_(CX, VX) += qp * t3 / 2.0;
    P_(VX, CX) += qp * t3 / 2.0; P_(VX, VX) += qp * t2;
    P_(CY, CY) += qp * t4 / 4.0; P_(CY, VY) += qp * t3 / 2.0;
    P_(VY, CY) += qp * t3 / 2.0; P_(VY, VY) += qp * t2;
    P_(YAW, YAW) += qy * t4 / 4.0; P_(YAW, W) += qy * t3 / 2.0;
    P_(W, YAW) += qy * t3 / 2.0; P_(W, W) += qy * t2;
    // 高度差恒定：z0/z1 共享同一随机游走（正相关过程噪声），差 z0-z1 不漂。
    P_(Z0, Z0) += cfg_.q_z * dt;
    P_(Z1, Z1) += cfg_.q_z * dt;
    P_(Z0, Z1) += cfg_.q_z * dt;
    P_(Z1, Z0) += cfg_.q_z * dt;
    P_(R0, R0) += cfg_.q_r * dt;
    P_(R1, R1) += cfg_.q_r * dt;
}

double EKFEstimator::mahalanobis2(const Eigen::Vector4d& z, int slot) const {
    Eigen::Vector4d y = z - h(x_, slot);
    y(0) = wrapPi(y(0));
    y(1) = wrapPi(y(1));
    y(3) = wrapPi(y(3));
    const Eigen::MatrixXd H = numericalJacobian(x_, slot);
    Eigen::Matrix4d S = H * P_ * H.transpose();
    // 动态 R：distance 和 world_normal_yaw 随距离线性放大，yaw/pitch 固定
    const double d = z(2);
    const double over = std::max(0.0, d - cfg_.sigma_scale_dref);
    const double sigma_dist = cfg_.sigma_cam_distance + cfg_.sigma_dist_slope * over;
    const double sigma_norm = cfg_.sigma_world_normal_yaw + cfg_.sigma_norm_slope * over;
    S(0, 0) += cfg_.sigma_cam_yaw * cfg_.sigma_cam_yaw;
    S(1, 1) += cfg_.sigma_cam_pitch * cfg_.sigma_cam_pitch;
    S(2, 2) += sigma_dist * sigma_dist;
    S(3, 3) += sigma_norm * sigma_norm;
    return y.transpose() * S.inverse() * y;
}

void EKFEstimator::correct(const Eigen::Vector4d& obs, int slot) {
    Eigen::Vector4d y = obs - h(x_, slot);
    y(0) = wrapPi(y(0));
    y(1) = wrapPi(y(1));
    y(3) = wrapPi(y(3));
    const Eigen::MatrixXd H = numericalJacobian(x_, slot);

    Eigen::Matrix4d S = H * P_ * H.transpose();
    // 动态 R：distance 和 world_normal_yaw 随距离线性放大，yaw/pitch 固定
    const double d = obs(2);
    const double over = std::max(0.0, d - cfg_.sigma_scale_dref);
    const double sigma_dist = cfg_.sigma_cam_distance + cfg_.sigma_dist_slope * over;
    const double sigma_norm = cfg_.sigma_world_normal_yaw + cfg_.sigma_norm_slope * over;
    S(0, 0) += cfg_.sigma_cam_yaw * cfg_.sigma_cam_yaw;
    S(1, 1) += cfg_.sigma_cam_pitch * cfg_.sigma_cam_pitch;
    S(2, 2) += sigma_dist * sigma_dist;
    S(3, 3) += sigma_norm * sigma_norm;

    const Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();  // kDim×4
    x_ += K * y;
    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(kDim, kDim);
    P_ = (I - K * H) * P_;
    P_ = 0.5 * (P_ + P_.transpose());  // 保持对称半正定

    x_(YAW) = wrapPi(x_(YAW));
}

bool EKFEstimator::diverged() const {
    const bool r_ok = x_(R0) > cfg_.radius_min && x_(R0) < cfg_.radius_max &&
                      x_(R1) > cfg_.radius_min && x_(R1) < cfg_.radius_max;
    const bool w_ok = std::fabs(x_(W)) < cfg_.w_max;
    return !(r_ok && w_ok);
}

void EKFEstimator::initialize(const std::vector<ArmorPose>& obs, double ts) {
    // 取离原点(云台)最近的观测作为参考板（近侧板，相机实际看到的那块）。
    const ArmorPose* best = &obs.front();
    double best_norm = 1e18;
    for (const auto& o : obs) {
        const double n = std::hypot(o.position_world_m[0], o.position_world_m[1]);
        if (n < best_norm) { best_norm = n; best = &o; }
    }

    const double px = best->position_world_m[0];
    const double py = best->position_world_m[1];
    const double pz = best->position_world_m[2];
    const double theta = std::atan2(py, px);
    const double r_avg = 0.5 * (cfg_.r0_init + cfg_.r1_init);

    // 初值中心：单板时用「板 + 预设半径」外推。
    double cx0 = px + r_avg * std::cos(theta);
    double cy0 = py + r_avg * std::sin(theta);
    // 几何法：两块相邻板法向延长线交点 = 旋转中心（不依赖预设半径，更准）。
    if (obs.size() >= 2) {
        std::vector<const ArmorPose*> sorted;
        for (const auto& o : obs) sorted.push_back(&o);
        std::sort(sorted.begin(), sorted.end(), [](const ArmorPose* a, const ArmorPose* b) {
            return std::hypot(a->position_world_m[0], a->position_world_m[1]) <
                   std::hypot(b->position_world_m[0], b->position_world_m[1]);
        });
        const ArmorPose* a = sorted[0];
        const ArmorPose* b = sorted[1];
        const double na_x = std::cos(a->armor_normal_yaw), na_y = std::sin(a->armor_normal_yaw);
        const double nb_x = std::cos(b->armor_normal_yaw), nb_y = std::sin(b->armor_normal_yaw);
        const double det = na_x * nb_y - na_y * nb_x;  // 两法向叉积
        if (std::fabs(det) > 1e-6) {
            const double dx = a->position_world_m[0] - b->position_world_m[0];
            const double dy = a->position_world_m[1] - b->position_world_m[1];
            const double s = (dx * nb_y - dy * nb_x) / det;  // 板 a 的半径
            cx0 = a->position_world_m[0] - s * na_x;
            cy0 = a->position_world_m[1] - s * na_y;
        }
    }

    x_ = Eigen::VectorXd::Zero(kDim);
    x_(CX) = cx0;
    x_(CY) = cy0;
    x_(R0) = cfg_.r0_init;   // 每次从预设半径开始，不恢复可能塌缩的旧值（防 reset 死循环，见 §10）
    x_(R1) = cfg_.r1_init;
    x_(YAW) = wrapPi(theta + kPi);  // 近侧板在 yaw = θ+π（+r 约定）
    // 高度差 dz 物理常数跨 reset 保留；公共高度用新观测 pz 初始化。
    x_(Z0) = pz + (has_saved_ ? saved_dz_ * 0.5 : 0.0);
    x_(Z1) = pz - (has_saved_ ? saved_dz_ * 0.5 : 0.0);

    // 初始协方差：位置/高度/半径中等，速度/角速度/角度大（表示不确定）。
    P_ = Eigen::MatrixXd::Zero(kDim, kDim);
    P_(CX, CX) = P_(CY, CY) = 0.01;   // 0.1m
    P_(VX, VX) = P_(VY, VY) = 25.0;   // 5m/s
    P_(Z0, Z0) = P_(Z1, Z1) = 0.01;
    P_(R0, R0) = P_(R1, R1) = 0.0225;   // 15cm 先验：每次 init 半径更大自由度，更快收敛到真值
    P_(YAW, YAW) = 4.0;               // 2rad
    P_(W, W) = 25.0;                  // 5rad/s

    initialized_ = true;
    converged_frames_ = 0;
    last_obs_timestamp_ = ts;
    state_ = TrackingState::Unstable;
}

void EKFEstimator::update(const std::vector<ArmorPose>& observations, double timestamp,
                          const GimbalState& gimbal, const CameraParameters& camera) {
    // 0. 缓存本帧世界→相机变换（h/数值雅可比用）。
    const Eigen::Matrix3d R_c2g = mat33(camera.rotation_camera_to_gimbal);
    const Eigen::Matrix3d R_g2w = gimbalToWorldRot(gimbal);
    R_w2c_ = R_c2g.transpose() * R_g2w.transpose();
    t_w2c_ = -R_c2g.transpose() * vec3(camera.translation_camera_to_gimbal);

    // 1. 帧间隔
    double dt = 0.0;
    if (last_timestamp_ > 0.0 && std::isfinite(timestamp)) {
        dt = timestamp - last_timestamp_;
    }
    last_timestamp_ = timestamp;
    if (!std::isfinite(dt) || dt < 0.0) dt = 0.0;
    if (dt > 0.5) dt = 0.5;  // 长时间停顿夹紧，避免预测发散

    // 2. 未初始化：首帧有观测即初始化（取近侧板）。
    if (!initialized_) {
        if (!observations.empty()) {
            initialize(observations, timestamp);
        } else {
            state_ = TrackingState::NoTarget;
        }
        return;
    }

    // 3. 预测
    predict(dt);

    // 4. 关联（贪婪最近邻）：每观测独立选马氏距离最小且低于门控的槽位；
    //    同槽冲突时保留代价更小者。槽位 90° 分离，贪婪与全局最优等价。
    const ArmorPose* slot_obs[4] = {nullptr, nullptr, nullptr, nullptr};
    const double kInf = 1e18;
    double slot_cost[4] = {kInf, kInf, kInf, kInf};
    for (int s = 0; s < 4; ++s) slot_obs_idx_[s] = -1;  // 重置关联下标
    const int n = static_cast<int>(observations.size());
    bool matched = false;
    double frame_min_nis = kInf;  // 本帧所有观测的最佳匹配 NIS（发散检测用）
    if (n > 0) {
        for (int i = 0; i < n; ++i) {
            const Eigen::Vector4d z = poseToObs(observations[i]);
            int best_j = -1;
            double best_d2 = kInf;
            double all_min_d2 = kInf;  // 所有槽的最小马氏距离（不设门控，发散检测用）
            for (int j = 0; j < 4; ++j) {
                const double d2 = mahalanobis2(z, j);
                if (d2 < all_min_d2) all_min_d2 = d2;
                if (d2 < cfg_.mahalanobis_gate2 && d2 < best_d2) { best_d2 = d2; best_j = j; }
            }
            if (all_min_d2 < frame_min_nis) frame_min_nis = all_min_d2;
            if (best_j < 0) continue;  // 无槽位低于门控 → 丢弃该观测
            if (slot_obs[best_j] == nullptr || best_d2 < slot_cost[best_j]) {
                slot_obs[best_j] = &observations[i];
                slot_cost[best_j] = best_d2;
                slot_obs_idx_[best_j] = i;
            }
        }
        for (int s = 0; s < 4; ++s) {
            if (slot_obs[s] != nullptr) matched = true;
        }

        // 目标车编号：直接取第一个观测到的非零分类 id（不做迟滞）。
        for (int s = 0; s < 4; ++s) {
            if (slot_obs[s] == nullptr) continue;
            const std::uint8_t tid = static_cast<std::uint8_t>(slot_obs[s]->detection.target_id);
            if (tid != 0) { last_target_id_ = tid; break; }
        }
    }

    // 记录本帧最小 NIS，供下一帧 predict 决定 Q 缩放（低→平滑压波动，高→全量加速收敛）。
    last_nis_ = (n > 0) ? frame_min_nis : kInf;

    // 4.5 correction：只 correction 被观测到的槽（不合成不可见板）。
    for (int s = 0; s < 4; ++s) {
        if (slot_obs[s] == nullptr) continue;
        correct(poseToObs(*slot_obs[s]), s);
    }

    // 4.6 发散检测：物理越界（半径/角速度）→ 立即重置；新息不自洽 → 连续多帧重置。
    if (diverged()) { reset(); return; }
    if (n > 0 && frame_min_nis > cfg_.nis_threshold) ++nis_fail_frames_;
    else nis_fail_frames_ = 0;
    if (nis_fail_frames_ >= cfg_.nis_fail_frames) { reset(); return; }

    // 5. 状态机
    if (matched) {
        last_obs_timestamp_ = timestamp;
        if (converged_frames_ < cfg_.converge_frames) ++converged_frames_;
    }
    const double lost_s = timestamp - last_obs_timestamp_;

    if (lost_s > cfg_.reset_lost_s) {  // 超时重置
        reset();
        return;
    }
    const bool stable = converged_frames_ >= cfg_.converge_frames;  // 收敛即锁存，直到 reset
    if (stable && state_ != TrackingState::Stable) ++stable_stage_;  // 进入新的稳定期
    state_ = stable ? TrackingState::Stable : TrackingState::Unstable;
}

EKFConfig loadEKFConfigOrDefault(const std::string& path) {
    EKFConfig cfg;  // 先取默认值，YAML 里出现的键覆盖默认。
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "[ekf] 打不开 " << path << "，使用默认参数（如需调参请创建该文件）\n";
        return cfg;
    }
    auto rd = [&](const char* key, double& v) {
        cv::FileNode n = fs[key];
        if (!n.empty()) v = static_cast<double>(n);
    };
    auto ri = [&](const char* key, int& v) {
        cv::FileNode n = fs[key];
        if (!n.empty()) v = static_cast<int>(n);
    };
    rd("q_pos", cfg.q_pos);
    rd("q_yaw", cfg.q_yaw);
    rd("q_z", cfg.q_z);
    rd("q_r", cfg.q_r);
    rd("sigma_cam_yaw", cfg.sigma_cam_yaw);
    rd("sigma_cam_pitch", cfg.sigma_cam_pitch);
    rd("sigma_cam_distance", cfg.sigma_cam_distance);
    rd("sigma_dist_slope", cfg.sigma_dist_slope);
    rd("sigma_world_normal_yaw", cfg.sigma_world_normal_yaw);
    rd("sigma_norm_slope", cfg.sigma_norm_slope);
    rd("sigma_scale_dref", cfg.sigma_scale_dref);
    rd("mahalanobis_gate2", cfg.mahalanobis_gate2);
    ri("converge_frames", cfg.converge_frames);
    rd("reset_lost_s", cfg.reset_lost_s);
    rd("r0_init", cfg.r0_init);
    rd("r1_init", cfg.r1_init);
    rd("radius_min", cfg.radius_min);
    rd("radius_max", cfg.radius_max);
    rd("w_max", cfg.w_max);
    rd("nis_threshold", cfg.nis_threshold);
    ri("nis_fail_frames", cfg.nis_fail_frames);
    rd("nis_stable_threshold", cfg.nis_stable_threshold);
    rd("q_stable_scale", cfg.q_stable_scale);
    return cfg;
}

}  // namespace rmv::tracker
