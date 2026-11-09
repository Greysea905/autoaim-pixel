#include "Planner.hpp"
#include "BulletModel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace rmv::tracker {

namespace {

constexpr double kHalfPi = kPi * 0.5;

// ---- 可调参数（初值，上机调）----
constexpr double kLambda = 0.3;        // 选板代价里「板面正对」的次级权重

}  // namespace

Planner::Planner() = default;

double Planner::wrapPi(double a) {
    while (a > kPi) a -= 2.0 * kPi;
    while (a <= -kPi) a += 2.0 * kPi;
    return a;
}

PredictionResult Planner::plan(const TrackState& state, TrackingState tracking,
                               const GimbalState& gimbal, double timestamp,
                               std::uint8_t target_id) {
    PredictionResult result;
    result.state = tracking;
    result.target_id = target_id;

    if (tracking == TrackingState::NoTarget) {
        return result;  // 无目标：其余字段全 0，applyPrediction 会走安全帧
    }

    // 1. 处理延迟（拍到→输出）+ 电控预测偏置（串口+云台响应，电控下发）。
    const double now = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    const double latency = std::clamp(now - timestamp, 0.0, 0.5);
    const double bias = std::clamp(static_cast<double>(gimbal.prediction_bias_s), 0.0, 1.0);

    // 2. 迭代解「弹道飞行时间 ↔ 外推距离」：飞行时间由距离决定，距离又随外推变化。
    //    严格参考 RMVtraining Planner::getTargetYawPitch：trackLine(t+bias) → bullet_iteration_cal 迭代。
    //    预测时域 = latency + bias + flight_time（flight_time 是弹道模型算出的子弹到靶延迟）。
    double flight_time = 0.0;
    double cx_T = state.cx, cy_T = state.cy, yaw_T = state.yaw;
    int best = 0;
    Eigen::Vector3d aim = Eigen::Vector3d::Zero();
    double beta = 0.0;
    for (int it = 0; it < 4; ++it) {
        const double lead = latency + bias + flight_time;
        cx_T = state.cx + state.vx * lead;
        cy_T = state.cy + state.vy * lead;
        yaw_T = wrapPi(state.yaw + state.w * lead);

        // 选板：第一项用「车体中心方位角 - gimbal.yaw」（bearing=中心，四板共用 → 常数），
        //    实际由「板面正对」（λ·|yaw2|）决定。每帧独立取最优，不做迟滞。
        double costs[4], yaw2s[4];
        Eigen::Vector3d aims[4];
        const double bearing = std::atan2(cy_T, cx_T);   // 车体中心方位角（从云台看，四板共用）
        for (int i = 0; i < 4; ++i) {
            const double off = static_cast<double>(i) * kHalfPi;
            const double r = (i % 2 == 0) ? state.r0 : state.r1;
            const double a = yaw_T + off;
            const double px = cx_T + r * std::cos(a);
            const double py = cy_T + r * std::sin(a);

            yaw2s[i] = wrapPi(a - std::atan2(-cy_T, -cx_T));  // 板相对视线（中心处）
            costs[i] = std::fabs(wrapPi(bearing - gimbal.yaw)) +
                       kLambda * std::fabs(yaw2s[i]);
            aims[i] = Eigen::Vector3d(px, py, state.height(i));
        }
        best = 0;
        for (int i = 1; i < 4; ++i) if (costs[i] < costs[best]) best = i;
        aim = aims[best];

        // 弹道模型：算出子弹飞行时间 + 发射角 beta（含重力 + 空气阻力）。
        Bullet_Model bullet(static_cast<float>(std::hypot(aim.x(), aim.y())),
                            static_cast<float>(aim.z()));
        flight_time = static_cast<double>(bullet.bullet_iteration_cal());
        beta = bullet.beta;
    }
    result.best_slot = best;

    // 3. 目标角度与距离（用收敛后的 aim）。
    result.distance_m = static_cast<float>(aim.norm());
    result.target_yaw = static_cast<float>(std::atan2(aim.y(), aim.x()));
    result.target_pitch = static_cast<float>(beta);

    // 5. 角速度前馈：板方位角的变化率。
    const double px = aim.x(), py = aim.y();
    const double vx_P = state.vx - state.w * (py - cy_T);
    const double vy_P = state.vy + state.w * (px - cx_T);
    const double d2 = px * px + py * py;
    result.target_yaw_velocity =
        (d2 > 1e-6) ? static_cast<float>((px * vy_P - py * vx_P) / d2) : 0.0F;
    result.target_pitch_velocity = 0.0F;

    // 6. 回填 EKF 状态（web_debug 显示用）。
    result.center_x_m = static_cast<float>(state.cx);
    result.center_y_m = static_cast<float>(state.cy);
    result.center_velocity_x_m_s = static_cast<float>(state.vx);
    result.center_velocity_y_m_s = static_cast<float>(state.vy);
    result.body_yaw = static_cast<float>(state.yaw);
    result.target_plate_yaw = static_cast<float>(wrapPi(state.yaw + best * kHalfPi));
    result.body_yaw_velocity = static_cast<float>(state.w);
    result.radius_1_m = static_cast<float>(state.r0);
    result.radius_2_m = static_cast<float>(state.r1);
    // 预测目标板（外推后）的位置 + 朝向，供可视化画开火框。
    result.aim_x = static_cast<float>(aim.x());
    result.aim_y = static_cast<float>(aim.y());
    result.aim_z = static_cast<float>(aim.z());
    result.aim_yaw = static_cast<float>(wrapPi(yaw_T + best * kHalfPi));

    return result;
}

}  // namespace rmv::tracker
