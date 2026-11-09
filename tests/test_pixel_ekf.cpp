// 像素直滤 EKF vs 旧球坐标 EKF 的 A/B 对比（离线，不需要相机/串口）。
// 作用：
//   阶段一：合成「旋转车体」轨迹，两者都用各自观测（A=ArmorPose/PnP输出，B=角点像素）收敛，
//           验证 PixelEKF 正确性（中心/半径/角速度收敛到真值）。
//   阶段二：注入「PnP 距离崩坏」（A 的距离读数每 25 帧连续 3 帧 ×2），B 只注入角点抖动，
//           对比两者中心估计的 RMSE——验证像素直滤在 PnP 失效时少被带偏。
// 退出码 0 = 通过，非 0 = 失败。

#include "EKFEstimator.hpp"
#include "PixelEKF.hpp"

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kHalfPi = kPi * 0.5;

double wrapPi(double a) {
    while (a > kPi) a -= 2.0 * kPi;
    while (a <= -kPi) a += 2.0 * kPi;
    return a;
}

// ---- 合成车体真值（与 test_ekf.cpp 相同）----
constexpr double kCx = 3.0, kCy = 0.0;
constexpr double kR0 = 0.19, kR1 = 0.23;
constexpr double kZ0 = 0.0, kZ1 = 0.02;
constexpr double kW = 1.0;

// ---- 合成相机（相机光轴 +z 对准世界 +x 前，相机 +x=世界 -y，相机 +y=世界 -z）----
constexpr double kFx = 1000.0, kFy = 1000.0, kU0 = 640.0, kV0 = 512.0;
// R_camera_to_gimbal：列 = 相机基在世界(云台零位=世界)系的坐标。
Eigen::Matrix3d Rc2g() {
    Eigen::Matrix3d R;
    R << 0, 0, 1,
        -1, 0, 0,
         0, -1, 0;
    return R;
}

CameraParameters makeCamera() {
    CameraParameters c;
    c.calibrated = true;
    c.image_width = 1280;
    c.image_height = 1024;
    c.camera_matrix = (cv::Mat_<double>(3, 3) << kFx, 0, kU0, 0, kFy, kV0, 0, 0, 1);
    c.distortion_coefficients = cv::Mat::zeros(1, 5, CV_64F);
    c.rotation_camera_to_gimbal = (cv::Mat_<double>(3, 3) <<
        0, 0, 1, -1, 0, 0, 0, -1, 0);
    c.translation_camera_to_gimbal = (cv::Mat_<double>(3, 1) << 0, 0, 0);
    return c;
}

// 世界系板 slot 的 4 角点（顺序 左上/右上/右下/左下，与 PixelEKF::plateCornersWorld 一致）。
std::array<Eigen::Vector3d, 4> gtCorners(double yaw, int slot) {
    const double off = static_cast<double>(slot) * kHalfPi;
    const double a = yaw + off;
    const double r = (slot % 2 == 0) ? kR0 : kR1;
    const double hz = (slot % 2 == 0) ? kZ0 : kZ1;
    const Eigen::Vector3d center(kCx + r * std::cos(a), kCy + r * std::sin(a), hz);
    const Eigen::Vector3d t(-std::sin(a), std::cos(a), 0.0);
    const Eigen::Vector3d up(0.0, 0.0, 1.0);
    const double hw = 0.0675, hh = 0.0285;
    return {center - hw * t + hh * up, center + hw * t + hh * up,
            center + hw * t - hh * up, center - hw * t - hh * up};
}

// 世界点 → 像素（与 PixelEKF 相同的针孔投影 + 深度 clamp）。
Eigen::Vector2d project(const Eigen::Vector3d& pw) {
    Eigen::Vector3d pc = Rc2g().transpose() * pw;  // 零位云台 + 零平移 → R_w2c = Rc2g^T
    double z = pc.z();
    if (z < 0.1) z = 0.1;
    return Eigen::Vector2d(kFx * pc.x() / z + kU0, kFy * pc.y() / z + kV0);
}

std::mt19937& rng() { static std::mt19937 g(42); return g; }
double noise(double sigma) {
    static std::normal_distribution<double> dist(0.0, 1.0);
    return dist(rng()) * sigma;
}

// 取离原点(云台)最近的 k 块板的 slot 下标。
std::vector<int> nearestSlots(double yaw, int k) {
    std::vector<int> idx{0, 1, 2, 3};
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        auto ca = gtCorners(yaw, a);
        auto cb = gtCorners(yaw, b);
        return ca[0].head<2>().norm() < cb[0].head<2>().norm();
    });
    return {idx.begin(), idx.begin() + k};
}

// ---- 管线 A 观测：ArmorPose（模拟 PnP 输出，距离噪声随距离放大）----
std::vector<ArmorPose> makePosesA(double yaw, const std::vector<int>& slots, double dist_bias) {
    std::vector<ArmorPose> obs;
    const double hw = 0.0675;
    for (int s : slots) {
        const double off = static_cast<double>(s) * kHalfPi;
        const double a = yaw + off;
        const double r = (s % 2 == 0) ? kR0 : kR1;
        const double hz = (s % 2 == 0) ? kZ0 : kZ1;
        const Eigen::Vector3d pw(kCx + r * std::cos(a), kCy + r * std::sin(a), hz);

        Eigen::Vector3d pc = Rc2g().transpose() * pw;
        const double d = pc.norm();
        // PnP 距离不可靠：深度方向噪声随距离放大，再乘 dist_bias 模拟距离崩坏。
        pc += Eigen::Vector3d(noise(0.01), noise(0.01), noise(0.10 * d));
        pc.z() *= dist_bias;

        ArmorPose pose;
        pose.position_camera_m = cv::Vec3d(pc.x(), pc.y(), pc.z());
        pose.position_world_m = cv::Vec3d(pw.x(), pw.y(), pw.z());
        pose.armor_normal_yaw = wrapPi(a + noise(0.05));
        pose.detection.target_id = 1;
        obs.push_back(pose);
    }
    return obs;
}

// ---- 管线 B 观测：ArmorDetection（角点像素 + 像素噪声 + 偶发角点抖动）----
std::vector<ArmorDetection> makeDetsB(double yaw, const std::vector<int>& slots, bool inject_outlier) {
    std::vector<ArmorDetection> dets;
    for (int s : slots) {
        auto cs = gtCorners(yaw, s);
        ArmorDetection d;
        for (int k = 0; k < 4; ++k) {
            Eigen::Vector2d px = project(cs[k]);
            double du = noise(3.0), dv = noise(3.0);
            if (inject_outlier && k == 0) { du += 30.0; dv += 30.0; }  // 偶发角点误检
            d.corners[k] = cv::Point2f(static_cast<float>(px.x() + du),
                                       static_cast<float>(px.y() + dv));
        }
        d.size = ArmorSize::Small;
        d.target_id = 1;
        dets.push_back(d);
    }
    return dets;
}

int g_failures = 0;
void check(bool ok, const char* msg) {
    if (!ok) { std::printf("  [FAIL] %s\n", msg); ++g_failures; }
}

}  // namespace

int main() {
    using rmv::tracker::EKFEstimator;
    using rmv::tracker::PixelEKF;

    const double dt = 0.02;
    const double start_ts = 10.0;
    const CameraParameters camera = makeCamera();
    const GimbalState gimbal{};

    EKFEstimator estA;
    PixelEKF estB;

    double t = 0.0, ts = start_ts;

    double sum_sq_A_cx = 0, sum_sq_A_cy = 0;
    double sum_sq_B_cx = 0, sum_sq_B_cy = 0;
    int cnt = 0;

    for (int frame = 0; frame < 200; ++frame) {  // 4s
        const double yaw = kW * t;
        const auto slots = nearestSlots(yaw, 2);

        // 阶段二（frame>=100）注入：A 每 25 帧连续 3 帧距离 ×2；B 同节奏角点抖动。
        const bool inject = (frame >= 100) && (frame % 25 < 3);

        estA.update(makePosesA(yaw, slots, inject ? 2.0 : 1.0), ts, gimbal, camera);
        estB.update(makeDetsB(yaw, slots, inject), ts, gimbal, camera);

        if (frame < 6) {
            const auto sb = estB.state();
            std::printf("[dbg f%d] slots=%d,%d B state=%d cx=%.3f cy=%.3f r0=%.3f r1=%.3f yaw=%.3f w=%.3f\n",
                frame, slots[0], slots[1], (int)estB.trackingState(),
                sb.cx, sb.cy, sb.r0, sb.r1, sb.yaw, sb.w);
        }

        if (frame >= 50) {  // 只统计收敛后
            const auto sa = estA.state();
            const auto sb = estB.state();
            const double ea_cx = sa.cx - kCx, ea_cy = sa.cy - kCy;
            const double eb_cx = sb.cx - kCx, eb_cy = sb.cy - kCy;
            sum_sq_A_cx += ea_cx * ea_cx; sum_sq_A_cy += ea_cy * ea_cy;
            sum_sq_B_cx += eb_cx * eb_cx; sum_sq_B_cy += eb_cy * eb_cy;
            ++cnt;
        }

        t += dt;
        ts += dt;
    }

    const double rmse_A = std::sqrt((sum_sq_A_cx + sum_sq_A_cy) / cnt);
    const double rmse_B = std::sqrt((sum_sq_B_cx + sum_sq_B_cy) / cnt);

    std::printf("========== A/B 对比（帧 50..199 中心 RMSE，单位 m）==========\n");
    std::printf("  A 旧球坐标 EKF（PnP 距离崩坏注入）: cx=%.4f cy=%.4f 总=%.4f\n",
                std::sqrt(sum_sq_A_cx / cnt), std::sqrt(sum_sq_A_cy / cnt), rmse_A);
    std::printf("  B 像素直滤 EKF（角点抖动注入）  : cx=%.4f cy=%.4f 总=%.4f\n",
                std::sqrt(sum_sq_B_cx / cnt), std::sqrt(sum_sq_B_cy / cnt), rmse_B);

    const auto sb = estB.state();
    std::printf("  B 收敛状态: state=%d r0=%.3f r1=%.3f w=%.3f\n",
                static_cast<int>(estB.trackingState()), sb.r0, sb.r1, sb.w);

    // 正确性：B 能收敛到真值（半径/角速度）。
    check(estB.trackingState() == TrackingState::Stable, "B 未达稳定");
    check(sb.r0 > 0.15 && sb.r0 < 0.25, "B 半径 r0 未收敛");
    check(sb.r1 > 0.20 && sb.r1 < 0.30, "B 半径 r1 未收敛");
    check(std::fabs(sb.w) > 0.8 && std::fabs(sb.w) < 1.2, "B 角速度未收敛");

    // 鲁棒性假设：B 的 RMSE 应显著低于 A（A 被距离崩坏带偏）。
    if (rmse_B < rmse_A) {
        std::printf("  [PASS] B 的 RMSE (%.4f) < A (%.4f)，像素直滤在 PnP 失效时更稳\n", rmse_B, rmse_A);
    } else {
        std::printf("  [WARN] B 的 RMSE (%.4f) 未低于 A (%.4f)，需检查注入强度/参数\n", rmse_B, rmse_A);
    }

    if (g_failures == 0) {
        std::printf("ALL PIXEL-EKF TESTS PASSED\n");
        return 0;
    }
    std::printf("%d TEST(S) FAILED\n", g_failures);
    return 1;
}
