// EKF 仿真测试（离线，不需要相机/串口）。
// 作用：
//   阶段一：合成「旋转车体」轨迹，每帧观测 2 块近侧板，验证中心/半径/角速度收敛。
//   阶段二：每帧只观测 1 块近侧板（随旋转切换槽位），验证装甲板切换与跟踪保持。
//   阶段三：持续丢观测，验证超时重置为无目标。
// 退出码 0 = 通过，非 0 = 失败。

#include "EKFEstimator.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

// 角度归一化到 (-π, π]（与 EKFEstimator::wrapPi 一致）。
double wrapPi(double a) {
    while (a > kPi) a -= 2.0 * kPi;
    while (a <= -kPi) a += 2.0 * kPi;
    return a;
}

// ---- 合成车体真值 ----
constexpr double kCx = 3.0, kCy = 0.0;         // 旋转中心(m)
constexpr double kR0 = 0.19, kR1 = 0.23;       // 两对半径(m，实测底盘直径 38/46cm 的一半)
constexpr double kZ0 = 0.0, kZ1 = 0.02;  // 两对板高度：T0/2=0.0, T1/3=0.02
constexpr double kW = 1.0;                      // 车体角速度(rad/s)

// 板 slot 的真值位置（+r 约定，世界系）
Eigen::Vector3d groundTruth(double yaw, int slot) {
    const double off = static_cast<double>(slot) * kPi * 0.5;
    const double r = (slot % 2 == 0) ? kR0 : kR1;
    return Eigen::Vector3d(kCx + r * std::cos(yaw + off),
                           kCy + r * std::sin(yaw + off),
                           (slot % 2 == 0) ? kZ0 : kZ1);
}

ArmorPose makeObs(const Eigen::Vector3d& p, double car_yaw, int slot) {
    ArmorPose pose;
    pose.position_world_m = cv::Vec3d(p.x(), p.y(), p.z());  // 世界系（供 initialize 用）
    // 相机系：测试用恒等外参（R_c2g=I, t=0, gimbal=0），相机系与世界系坐标一致。
    pose.position_camera_m = cv::Vec3d(p.x(), p.y(), p.z());
    pose.armor_normal_yaw = wrapPi(car_yaw + slot * kPi * 0.5);  // 板法向 = 车体 yaw + 槽位偏移
    pose.detection.target_id = 1;
    return pose;
}

// 固定种子的高斯噪声，保证测试可复现。
std::mt19937& rng() {
    static std::mt19937 g(42);
    return g;
}
double noise(double sigma) {
    static std::normal_distribution<double> dist(0.0, 1.0);
    return dist(rng()) * sigma;
}

// 生成 k 个「离原点(云台)最近」的板观测（模拟相机只看到近侧板）。
std::vector<ArmorPose> observeNearest(double yaw, int k) {
    Eigen::Vector3d plates[4];
    for (int s = 0; s < 4; ++s) {
        plates[s] = groundTruth(yaw, s);
        plates[s].x() += noise(0.005);
        plates[s].y() += noise(0.005);
        plates[s].z() += noise(0.005);
    }
    std::vector<int> idx{0, 1, 2, 3};
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return plates[a].head<2>().norm() < plates[b].head<2>().norm();
    });
    std::vector<ArmorPose> obs;
    for (int j = 0; j < k; ++j) obs.push_back(makeObs(plates[idx[j]], yaw, idx[j]));
    return obs;
}

int g_failures = 0;
void check(bool ok, const char* msg) {
    if (!ok) {
        std::printf("  [FAIL] %s\n", msg);
        ++g_failures;
    }
}

}  // namespace

int main() {
    using rmv::tracker::EKFEstimator;

    const double dt = 0.02;
    const double start_ts = 10.0;

    EKFEstimator est;
    // 恒等外参 + 零位云台：相机系 = 世界系，简化测试（观测/预测共用同一变换）。
    CameraParameters camera;
    camera.calibrated = true;
    camera.rotation_camera_to_gimbal = cv::Mat::eye(3, 3, CV_64F);
    camera.translation_camera_to_gimbal = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 0.0);
    const GimbalState gimbal{};
    double t = 0.0;
    double ts = start_ts;

    // ---- 阶段一：每帧观测 2 块近侧板，验证收敛 ----
    for (int i = 0; i < 150; ++i) {  // 3s
        est.update(observeNearest(kW * t, 2), ts, gimbal, camera);
        t += dt;
        ts += dt;
    }
    {
        const auto st = est.state();
        std::printf("阶段一: state=%d cx=%.3f cy=%.3f r0=%.3f r1=%.3f w=%.3f\n",
                    static_cast<int>(est.trackingState()), st.cx, st.cy, st.r0, st.r1, st.w);
        check(est.trackingState() == TrackingState::Stable, "阶段一未达稳定");
        check(st.cx > 2.85 && st.cx < 3.15, "中心 x 未收敛");
        check(std::fabs(st.cy) < 0.15, "中心 y 未收敛");
        const double rmin = std::min(st.r0, st.r1);
        const double rmax = std::max(st.r0, st.r1);
        check(rmin > 0.15 && rmin < 0.25, "较小半径未收敛");
        check(rmax > 0.20 && rmax < 0.30, "较大半径未收敛");
        check(std::fabs(st.w) > 0.8 && std::fabs(st.w) < 1.2, "角速度未收敛");
    }

    // ---- 阶段二：每帧只观测 1 块近侧板，验证切换与保持 ----
    for (int i = 0; i < 150; ++i) {  // 3s
        est.update(observeNearest(kW * t, 1), ts, gimbal, camera);
        t += dt;
        ts += dt;
    }
    {
        const auto st = est.state();
        std::printf("阶段二: state=%d cx=%.3f cy=%.3f\n",
                    static_cast<int>(est.trackingState()), st.cx, st.cy);
        check(est.trackingState() == TrackingState::Stable, "阶段二丢跟踪");
        check(st.cx > 2.85 && st.cx < 3.15, "阶段二中心 x 漂移");
    }

    // ---- 阶段三：持续丢观测，验证超时重置 ----
    for (int i = 0; i < 60; ++i) {  // 1.2s > kResetLostS
        est.update({}, ts, gimbal, camera);
        t += dt;
        ts += dt;
    }
    {
        std::printf("阶段三: state=%d(应=0 无目标)\n", static_cast<int>(est.trackingState()));
        check(est.trackingState() == TrackingState::NoTarget, "超时未重置");
    }

    // 阶段四已移除：观测雅可比现为数值差分（numericalJacobian），无需再校验解析求导。

    if (g_failures == 0) {
        std::printf("ALL EKF TESTS PASSED\n");
        return 0;
    }
    std::printf("%d EKF TEST(S) FAILED\n", g_failures);
    return 1;
}
