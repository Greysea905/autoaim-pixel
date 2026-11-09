#include "PixelEKF.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>

namespace rmv::tracker {

namespace {

constexpr double kHalfPi = kPi * 0.5;

// 云台 → 世界 旋转（R_yaw·R_pitch·R_roll，与 EKFEstimator 同款）。
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

// ArmorPose → 9 维观测 [u0,v0,u1,v1,u2,v2,u3,v3, yaw]。
Eigen::VectorXd poseToObs(const ArmorPose& p) {
    Eigen::VectorXd z(9);
    for (int k = 0; k < 4; ++k) {
        z(2 * k) = p.detection.corners[k].x;
        z(2 * k + 1) = p.detection.corners[k].y;
    }
    z(8) = p.armor_normal_yaw;
    return z;
}

}  // namespace

PixelEKF::PixelEKF(const PixelEKFConfig& cfg) : cfg_(cfg) { reset(); }

void PixelEKF::reset() {
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
    last_nis_ = 1e18;
    state_ = TrackingState::NoTarget;
    x_ = Eigen::VectorXd::Zero(kDim);
    P_ = Eigen::MatrixXd::Zero(kDim, kDim);
}

double PixelEKF::wrapPi(double a) {
    while (a > kPi) a -= 2.0 * kPi;
    while (a <= -kPi) a += 2.0 * kPi;
    return a;
}

TrackState PixelEKF::state() const {
    TrackState s;
    s.cx = x_(CX); s.vx = x_(VX);
    s.cy = x_(CY); s.vy = x_(VY);
    s.z0 = x_(Z0); s.z1 = x_(Z1);
    s.r0 = x_(R0); s.r1 = x_(R1);
    s.yaw = x_(YAW); s.w = x_(W);
    return s;
}

TrackingState PixelEKF::trackingState() const { return state_; }

int PixelEKF::stableStage() const { return stable_stage_; }

std::uint8_t PixelEKF::targetId() const { return last_target_id_; }

Eigen::Vector2d PixelEKF::worldCornerToPixel(const Eigen::Vector3d& pw) const {
    const Eigen::Vector3d pc = R_w2c_ * pw + t_w2c_;
    double z = pc.z();
    if (z < cfg_.min_depth) z = cfg_.min_depth;   // 深度保护：防 NaN/Inf
    const double u = fx_ * pc.x() / z + cx_;
    const double v = fy_ * pc.y() / z + cy_;
    return Eigen::Vector2d(u, v);
}

std::array<Eigen::Vector3d, 4> PixelEKF::plateCornersWorld(const Eigen::VectorXd& x, int slot) const {
    const double off = static_cast<double>(slot) * kHalfPi;
    const double a = x(YAW) + off;
    const int r_idx = (slot % 2 == 0) ? R0 : R1;
    const double r = x(r_idx);
    const double hz = (slot % 2 == 0) ? x(Z0) : x(Z1);

    const Eigen::Vector3d center(x(CX) + r * std::cos(a),
                                 x(CY) + r * std::sin(a),
                                 hz);
    const Eigen::Vector3d t(-std::sin(a), std::cos(a), 0.0);
    const Eigen::Vector3d up(0.0, 0.0, 1.0);
    const double hw = cfg_.plate_half_width;
    const double hh = cfg_.plate_half_height;
    return {center - hw * t + hh * up,
            center + hw * t + hh * up,
            center + hw * t - hh * up,
            center - hw * t - hh * up};
}

Eigen::VectorXd PixelEKF::h(const Eigen::VectorXd& x, int slot) const {
    const auto corners = plateCornersWorld(x, slot);
    Eigen::VectorXd z(9);
    for (int k = 0; k < 4; ++k) {
        const Eigen::Vector2d px = worldCornerToPixel(corners[k]);
        z(2 * k) = px.x();
        z(2 * k + 1) = px.y();
    }
    z(8) = wrapPi(x(YAW) + static_cast<double>(slot) * kHalfPi);  // 板法向 yaw
    return z;
}

Eigen::MatrixXd PixelEKF::pixelR() const {
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(9, 9);
    for (int k = 0; k < 4; ++k) {
        R(2 * k, 2 * k) = cfg_.sigma_pixel_u * cfg_.sigma_pixel_u;
        R(2 * k + 1, 2 * k + 1) = cfg_.sigma_pixel_v * cfg_.sigma_pixel_v;
    }
    R(8, 8) = cfg_.sigma_yaw_rad * cfg_.sigma_yaw_rad;
    return R;
}

Eigen::MatrixXd PixelEKF::numericalJacobian(const Eigen::VectorXd& x, int slot) const {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(9, kDim);
    const double eps = 1e-6;
    for (int col = 0; col < kDim; ++col) {
        Eigen::VectorXd xp = x, xm = x;
        xp(col) += eps;
        xm(col) -= eps;
        const Eigen::VectorXd hp = h(xp, slot);
        const Eigen::VectorXd hm = h(xm, slot);
        for (int row = 0; row < 9; ++row) {
            double diff = hp(row) - hm(row);
            if (row == 8) diff = wrapPi(diff);  // yaw 行跨 ±π 用 wrap 差分
            H(row, col) = diff / (2.0 * eps);
        }
    }
    return H;
}

void PixelEKF::predict(double dt) {
    if (dt <= 0.0) return;

    x_(CX) += x_(VX) * dt;
    x_(CY) += x_(VY) * dt;
    x_(YAW) = wrapPi(x_(YAW) + x_(W) * dt);

    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(kDim, kDim);
    F(CX, VX) = dt;
    F(CY, VY) = dt;
    F(YAW, W) = dt;
    P_ = F * P_ * F.transpose();

    // 自适应 Q（NIS 驱动）：模型拟合好→Q 小（平滑压波动）；失配→Q 全量（加速收敛）。
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
    P_(Z0, Z0) += cfg_.q_z * dt;
    P_(Z1, Z1) += cfg_.q_z * dt;
    P_(Z0, Z1) += cfg_.q_z * dt;
    P_(Z1, Z0) += cfg_.q_z * dt;
    P_(R0, R0) += cfg_.q_r * dt;
    P_(R1, R1) += cfg_.q_r * dt;
}

double PixelEKF::mahalanobis2(const Eigen::VectorXd& z, int slot) const {
    Eigen::VectorXd y = z - h(x_, slot);
    y(8) = wrapPi(y(8));   // yaw 残差归一化
    const Eigen::MatrixXd H = numericalJacobian(x_, slot);
    Eigen::MatrixXd S = H * P_ * H.transpose() + pixelR();
    return y.transpose() * S.inverse() * y;
}

void PixelEKF::correct(const Eigen::VectorXd& obs, int slot) {
    Eigen::VectorXd y = obs - h(x_, slot);
    y(8) = wrapPi(y(8));   // yaw 残差归一化
    const Eigen::MatrixXd H = numericalJacobian(x_, slot);
    Eigen::MatrixXd S = H * P_ * H.transpose() + pixelR();
    const Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();   // kDim×9
    x_ += K * y;
    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(kDim, kDim);
    P_ = (I - K * H) * P_;
    P_ = 0.5 * (P_ + P_.transpose());  // 保持对称半正定
    x_(YAW) = wrapPi(x_(YAW));
}

bool PixelEKF::diverged() const {
    const bool r_ok = x_(R0) > cfg_.radius_min && x_(R0) < cfg_.radius_max &&
                      x_(R1) > cfg_.radius_min && x_(R1) < cfg_.radius_max;
    const bool w_ok = std::fabs(x_(W)) < cfg_.w_max;
    return !(r_ok && w_ok);
}

void PixelEKF::initialize(const std::vector<ArmorPose>& observations, double ts) {
    // 取离原点(云台)最近的观测作参考板（近侧板）。position_world_m 已由 armor_solve 提供。
    const ArmorPose* best = nullptr;
    double best_norm = 1e18;
    for (const auto& o : observations) {
        const double n = std::hypot(o.position_world_m[0], o.position_world_m[1]);
        if (n < best_norm) { best_norm = n; best = &o; }
    }
    if (!best) return;  // 无观测 → 保持未初始化

    const double px = best->position_world_m[0];
    const double py = best->position_world_m[1];
    const double pz = best->position_world_m[2];
    const double theta = std::atan2(py, px);
    const double r_avg = 0.5 * (cfg_.r0_init + cfg_.r1_init);

    x_ = Eigen::VectorXd::Zero(kDim);
    x_(CX) = px + r_avg * std::cos(theta);
    x_(CY) = py + r_avg * std::sin(theta);
    x_(R0) = cfg_.r0_init;
    x_(R1) = cfg_.r1_init;
    x_(YAW) = wrapPi(theta + kPi);  // 近侧板在 yaw = θ+π（+r 约定）
    x_(Z0) = pz + (has_saved_ ? saved_dz_ * 0.5 : 0.0);
    x_(Z1) = pz - (has_saved_ ? saved_dz_ * 0.5 : 0.0);

    P_ = Eigen::MatrixXd::Zero(kDim, kDim);
    P_(CX, CX) = P_(CY, CY) = 0.01;   // 0.1m
    P_(VX, VX) = P_(VY, VY) = 25.0;   // 5m/s
    P_(Z0, Z0) = P_(Z1, Z1) = 0.01;
    P_(R0, R0) = P_(R1, R1) = 0.0225; // 15cm 先验
    P_(YAW, YAW) = 4.0;               // 2rad
    P_(W, W) = 25.0;                  // 5rad/s

    initialized_ = true;
    converged_frames_ = 0;
    last_obs_timestamp_ = ts;
    last_target_id_ = best->detection.target_id;
    state_ = TrackingState::Unstable;
}

void PixelEKF::update(const std::vector<ArmorPose>& observations, double timestamp,
                      const GimbalState& gimbal, const CameraParameters& camera) {
    // 0. 缓存本帧世界→相机变换 + 内参（h/数值雅可比用）。
    const Eigen::Matrix3d R_c2g = mat33(camera.rotation_camera_to_gimbal);
    const Eigen::Matrix3d R_g2w = gimbalToWorldRot(gimbal);
    R_w2c_ = R_c2g.transpose() * R_g2w.transpose();
    t_w2c_ = -R_c2g.transpose() * vec3(camera.translation_camera_to_gimbal);
    if (camera.camera_matrix.rows == 3 && camera.camera_matrix.cols == 3) {
        fx_ = camera.camera_matrix.at<double>(0, 0);
        fy_ = camera.camera_matrix.at<double>(1, 1);
        cx_ = camera.camera_matrix.at<double>(0, 2);
        cy_ = camera.camera_matrix.at<double>(1, 2);
    }

    // 1. 帧间隔
    double dt = 0.0;
    if (last_timestamp_ > 0.0 && std::isfinite(timestamp)) {
        dt = timestamp - last_timestamp_;
    }
    last_timestamp_ = timestamp;
    if (!std::isfinite(dt) || dt < 0.0) dt = 0.0;
    if (dt > 0.5) dt = 0.5;  // 长时间停顿夹紧，避免预测发散

    // 2. 未初始化：首帧有观测即初始化。
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

    // 4. 关联（贪婪最近邻）：像素域 + yaw 混合马氏距离。
    const ArmorPose* slot_obs[4] = {nullptr, nullptr, nullptr, nullptr};
    const double kInf = 1e18;
    double slot_cost[4] = {kInf, kInf, kInf, kInf};
    for (int s = 0; s < 4; ++s) slot_obs_idx_[s] = -1;
    const int n = static_cast<int>(observations.size());
    bool matched = false;
    double frame_min_nis = kInf;
    if (n > 0) {
        for (int i = 0; i < n; ++i) {
            const Eigen::VectorXd z = poseToObs(observations[i]);
            int best_j = -1;
            double best_d2 = kInf;
            double all_min_d2 = kInf;
            for (int j = 0; j < 4; ++j) {
                const double d2 = mahalanobis2(z, j);
                if (d2 < all_min_d2) all_min_d2 = d2;
                if (d2 < cfg_.mahalanobis_gate2 && d2 < best_d2) { best_d2 = d2; best_j = j; }
            }
            if (all_min_d2 < frame_min_nis) frame_min_nis = all_min_d2;
            if (best_j < 0) continue;
            if (slot_obs[best_j] == nullptr || best_d2 < slot_cost[best_j]) {
                slot_obs[best_j] = &observations[i];
                slot_cost[best_j] = best_d2;
                slot_obs_idx_[best_j] = i;
            }
        }
        for (int s = 0; s < 4; ++s) {
            if (slot_obs[s] != nullptr) matched = true;
        }
        for (int s = 0; s < 4; ++s) {
            if (slot_obs[s] == nullptr) continue;
            const std::uint8_t tid = static_cast<std::uint8_t>(slot_obs[s]->detection.target_id);
            if (tid != 0) { last_target_id_ = tid; break; }
        }
    }

    last_nis_ = (n > 0) ? frame_min_nis : kInf;

    // 4.5 只 correction 被观测到的槽。
    for (int s = 0; s < 4; ++s) {
        if (slot_obs[s] == nullptr) continue;
        correct(poseToObs(*slot_obs[s]), s);
    }

    // 4.6 发散检测：物理越界 → 立即重置；新息不自洽 → 连续多帧重置。
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
    if (lost_s > cfg_.reset_lost_s) { reset(); return; }
    const bool stable = converged_frames_ >= cfg_.converge_frames;
    if (stable && state_ != TrackingState::Stable) ++stable_stage_;
    state_ = stable ? TrackingState::Stable : TrackingState::Unstable;
}

PixelEKFConfig loadPixelEKFConfigOrDefault(const std::string& path) {
    PixelEKFConfig cfg;
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "[PixelEKF] 打不开 " << path << "，使用默认参数\n";
        return cfg;
    }
    auto rd = [&](const char* key, double& v) {
        cv::FileNode node = fs[key];
        if (!node.empty()) v = static_cast<double>(node);
    };
    auto ri = [&](const char* key, int& v) {
        cv::FileNode node = fs[key];
        if (!node.empty()) v = static_cast<int>(node);
    };
    rd("q_pos", cfg.q_pos);
    rd("q_yaw", cfg.q_yaw);
    rd("q_z", cfg.q_z);
    rd("q_r", cfg.q_r);
    rd("sigma_pixel_u", cfg.sigma_pixel_u);
    rd("sigma_pixel_v", cfg.sigma_pixel_v);
    rd("sigma_yaw_rad", cfg.sigma_yaw_rad);
    rd("min_depth", cfg.min_depth);
    rd("plate_half_width", cfg.plate_half_width);
    rd("plate_half_height", cfg.plate_half_height);
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
