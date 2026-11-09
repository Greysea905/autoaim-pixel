#pragma once

#include "AutoAimTypes.hpp"
#include "TrackState.hpp"

#include <cstdint>

namespace rmv::tracker {

// 弹道 + 瞄准：EKF 状态 → 云台指令 PredictionResult。
// 职责：外推状态、选板、算 target_yaw/pitch、角速度前馈、开火判据。
// （弹道重力补偿暂未接入，target_pitch 为纯几何。）
class Planner {
public:
    Planner();

    // state: EKF 状态（世界系）；tracking: EKF 稳定状态；
    // gimbal: 用 prediction_bias_s 与当前 yaw；timestamp: 帧时间（算处理延迟）；
    // target_id: 目标板编号（armor_flag 用，来自 EKF 的分类结果）。
    PredictionResult plan(const TrackState& state, TrackingState tracking,
                          const GimbalState& gimbal, double timestamp,
                          std::uint8_t target_id);

private:
    static double wrapPi(double a);
};

}  // namespace rmv::tracker
