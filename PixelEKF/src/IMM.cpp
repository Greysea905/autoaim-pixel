#include "IMM.hpp"

#include <cmath>

namespace rmv::tracker {

IMM::IMM(const PixelEKFConfig& steady_cfg, const PixelEKFConfig& maneuver_cfg)
    : steady_(steady_cfg), maneuver_(maneuver_cfg) {
    mu_ << 0.9, 0.1;   // 初始：稳态概率高
    Pi_ << 0.95, 0.05, // 稳态→稳态 0.95，稳态→机动 0.05
           0.15, 0.85; // 机动→稳态 0.15，机动→机动 0.85
    x_fused_ = Eigen::VectorXd::Zero(PixelEKF::kDim);
    P_fused_ = Eigen::MatrixXd::Zero(PixelEKF::kDim, PixelEKF::kDim);
}

void IMM::reset() {
    steady_.reset();
    maneuver_.reset();
    mu_ << 0.9, 0.1;
    x_fused_ = Eigen::VectorXd::Zero(PixelEKF::kDim);
    P_fused_ = Eigen::MatrixXd::Zero(PixelEKF::kDim, PixelEKF::kDim);
}

void IMM::update(const std::vector<ArmorPose>& observations, double timestamp,
                 const GimbalState& gimbal, const CameraParameters& camera) {
    // 1. 交互混合：上一帧后验按 Markov 转移概率混合成各模型的先验
    const double c0 = mu_(0) * Pi_(0, 0) + mu_(1) * Pi_(1, 0);  // 转移到模型0的预测概率
    const double c1 = mu_(0) * Pi_(0, 1) + mu_(1) * Pi_(1, 1);  // 转移到模型1的预测概率

    const Eigen::VectorXd xs = steady_.x();
    const Eigen::VectorXd xm = maneuver_.x();
    const Eigen::MatrixXd Ps = steady_.P();
    const Eigen::MatrixXd Pm = maneuver_.P();

    // 混合到模型0（稳态）的先验
    const double u00 = (c0 > 0.0) ? mu_(0) * Pi_(0, 0) / c0 : 0.0;
    const double u10 = (c0 > 0.0) ? mu_(1) * Pi_(1, 0) / c0 : 0.0;
    const Eigen::VectorXd x0 = u00 * xs + u10 * xm;
    const Eigen::MatrixXd P0 = u00 * (Ps + (xs - x0) * (xs - x0).transpose())
                             + u10 * (Pm + (xm - x0) * (xm - x0).transpose());

    // 混合到模型1（机动）的先验
    const double u01 = (c1 > 0.0) ? mu_(0) * Pi_(0, 1) / c1 : 0.0;
    const double u11 = (c1 > 0.0) ? mu_(1) * Pi_(1, 1) / c1 : 0.0;
    const Eigen::VectorXd x1 = u01 * xs + u11 * xm;
    const Eigen::MatrixXd P1 = u01 * (Ps + (xs - x1) * (xs - x1).transpose())
                             + u11 * (Pm + (xm - x1) * (xm - x1).transpose());

    steady_.setX(x0);
    steady_.setP(P0);
    maneuver_.setX(x1);
    maneuver_.setP(P1);

    // 2. 各自滤波
    steady_.update(observations, timestamp, gimbal, camera);
    maneuver_.update(observations, timestamp, gimbal, camera);

    // 3. 概率更新：μ_j ∝ Λ_j · c_j
    const double L0 = steady_.lastLikelihood();
    const double L1 = maneuver_.lastLikelihood();
    const double p0 = L0 * c0;
    const double p1 = L1 * c1;
    const double psum = p0 + p1;
    if (psum > 0.0) {
        mu_(0) = p0 / psum;
        mu_(1) = p1 / psum;
    }
    // psum == 0（两模型都无观测）时保持 μ 不变

    // 4. 融合输出
    const Eigen::VectorXd xf = mu_(0) * steady_.x() + mu_(1) * maneuver_.x();
    x_fused_ = xf;
    P_fused_ = mu_(0) * (steady_.P() + (steady_.x() - xf) * (steady_.x() - xf).transpose())
             + mu_(1) * (maneuver_.P() + (maneuver_.x() - xf) * (maneuver_.x() - xf).transpose());
}

TrackState IMM::toTrackState(const Eigen::VectorXd& x) {
    TrackState s;
    s.cx = x(PixelEKF::CX); s.vx = x(PixelEKF::VX);
    s.cy = x(PixelEKF::CY); s.vy = x(PixelEKF::VY);
    s.z0 = x(PixelEKF::Z0); s.z1 = x(PixelEKF::Z1);
    s.r0 = x(PixelEKF::R0); s.r1 = x(PixelEKF::R1);
    s.yaw = x(PixelEKF::YAW); s.w = x(PixelEKF::W);
    return s;
}

TrackState IMM::state() const { return toTrackState(x_fused_); }

TrackingState IMM::trackingState() const {
    // 用概率更高的模型的跟踪状态
    return (mu_(0) >= mu_(1)) ? steady_.trackingState() : maneuver_.trackingState();
}

std::uint8_t IMM::targetId() const {
    return (mu_(0) >= mu_(1)) ? steady_.targetId() : maneuver_.targetId();
}

int IMM::stableStage() const {
    return (mu_(0) >= mu_(1)) ? steady_.stableStage() : maneuver_.stableStage();
}

int IMM::associatedObservationIndex(int slot) const {
    return (mu_(0) >= mu_(1)) ? steady_.associatedObservationIndex(slot)
                              : maneuver_.associatedObservationIndex(slot);
}

}  // namespace rmv::tracker
