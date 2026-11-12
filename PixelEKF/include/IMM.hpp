#pragma once

#include "PixelEKF.hpp"

#include <Eigen/Dense>

namespace rmv::tracker {

// 2 模型 IMM（交互多模型）：稳态 CV+CAR（q_aw=0）+ 机动 CV+CAA（q_aw>0）。
// 解决单模型在「旋转反向」时的机动滞后：稳态时平滑模型主导，机动时切到快模型。
class IMM {
public:
    IMM(const PixelEKFConfig& steady_cfg, const PixelEKFConfig& maneuver_cfg);

    // 每帧调用一次：mixing → 各自滤波 → 概率更新 → 融合。
    void update(const std::vector<ArmorPose>& observations, double timestamp,
                const GimbalState& gimbal, const CameraParameters& camera);

    TrackState state() const;
    TrackingState trackingState() const;
    std::uint8_t targetId() const;
    void reset();

    // 模型概率（调试用）：0=稳态，1=机动。
    double modelProb(int i) const { return mu_(i); }
    int stableStage() const;
    int associatedObservationIndex(int slot) const;
    double mag() const { return x_fused_(PixelEKF::MAG); }
    double azi() const { return x_fused_(PixelEKF::AZI); }

private:
    PixelEKF steady_;    // 模型1：稳态（q_aw=0，w 恒定）
    PixelEKF maneuver_;  // 模型2：机动（q_aw>0，w 可 ramp）
    Eigen::Vector2d mu_;  // 模型概率 [μ1, μ2]
    Eigen::Matrix2d Pi_;  // Markov 转移矩阵 Π

    Eigen::VectorXd x_fused_;  // 融合状态（13 维）
    Eigen::MatrixXd P_fused_;  // 融合协方差

    static TrackState toTrackState(const Eigen::VectorXd& x);
};

}  // namespace rmv::tracker
