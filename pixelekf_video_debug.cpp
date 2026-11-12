// 像素直滤 EKF 视频离线调参工具：读录像 → 检测(角点) → PixelEKF → 可视化窗口 + Web UI。
// 与 ekf_video_debug 的区别：跳过 armor_solve(PnP)，角点直接喂给 PixelEKF。
//
// 用法：
//   ./build/pixelekf_video_debug <视频文件> <相机标定> [Web端口] [enemy=red|blue]
//   例：./build/pixelekf_video_debug EKF_video/blue.avi config/camera.yaml 8080 blue
// 空格 = 暂停/继续；q 或 Esc = 退出。
// 需在项目根目录运行（检测/分类模型是相对路径）。

#include "AppSupport.hpp"
#include "IMM.hpp"
#include "Planner.hpp"
#include "StudentTasks.hpp"
#include "WebLogger.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

std::atomic<bool> g_shutdown{false};
std::atomic<bool> g_paused{false};

cv::Mat g_frame;
std::mutex g_frame_mutex;
bool g_frame_ready = false;

void renderLoop(const std::string& win_name) {
    while (!g_shutdown.load(std::memory_order_relaxed)) {
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(g_frame_mutex);
            if (g_frame_ready) {
                frame = g_frame.clone();
                g_frame_ready = false;
            }
        }
        if (!frame.empty()) {
            cv::imshow(win_name, frame);
        }
        const int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) {
            g_shutdown.store(true, std::memory_order_relaxed);
        } else if (key == ' ') {
            g_paused.store(!g_paused.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 5) {
        std::cerr << "用法: " << argv[0]
                  << " <视频文件> <相机标定> [Web端口] [enemy=red|blue]\n";
        return 2;
    }
    const std::string video_path = argv[1];
    const std::string win_name = "pixelekf (空格=暂停, q=退出)";

    CameraParameters camera;
    std::string err;
    if (!loadCameraParameters(argv[2], camera, err)) {
        std::cerr << err << '\n';
        return 1;
    }
    const int web_port = (argc >= 4) ? std::atoi(argv[3]) : 8080;
    const TeamColor enemy =
        (argc >= 5 && std::string(argv[4]) == "blue") ? TeamColor::Blue : TeamColor::Red;

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "打开视频失败: " << video_path << '\n';
        return 1;
    }
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 1.0) fps = 109.0;
    const double dt = 1.0 / fps;
    const double frame_interval_ms = 1000.0 / fps;

    START_WEB_SERVER(web_port);
    std::thread renderer(renderLoop, win_name);

    rmv::tracker::PixelEKFConfig maneuver_cfg = rmv::tracker::loadPixelEKFConfigOrDefault();  // q_aw 取 yaml 值
    rmv::tracker::PixelEKFConfig steady_cfg = maneuver_cfg;
    steady_cfg.q_aw = 0.0;  // 稳态模型：w 恒定
    rmv::tracker::IMM estimator(steady_cfg, maneuver_cfg);
    rmv::tracker::Planner planner;
    const GimbalState gimbal{};  // 静止相机：云台角全 0（世界系=相机系）

    cv::Mat pic;
    double ts = 0.0;
    int frame_idx = 0;
    std::printf("开始处理 %s（enemy=%s, fps=%.1f, 空格=暂停 q=退出）\n",
                video_path.c_str(), enemy == TeamColor::Red ? "red" : "blue", fps);

    while (!g_shutdown.load(std::memory_order_relaxed)) {
        while (g_paused.load(std::memory_order_relaxed) &&
               !g_shutdown.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        const auto t_start = std::chrono::steady_clock::now();

        if (!cap.read(pic)) break;  // 视频结束
        cv::flip(pic, pic, -1);  // 云台相机倒置，录像转 180° 回正

        const auto detections = armor_detect(pic, enemy);
        // 混合观测：armor_solve 出 corners + armor_normal_yaw（position 仅用于初始化），
        // PixelEKF 用 8 像素 + 1 yaw 作观测，不吃 PnP 的距离/位置。
        const auto observations = armor_solve(detections, camera, gimbal);
        estimator.update(observations, ts, gimbal, camera);

        const double now_ts = std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        PredictionResult pred = planner.plan(estimator.state(), estimator.trackingState(),
                                             gimbal, now_ts, estimator.targetId());
        pred.target_obs_index = estimator.associatedObservationIndex(pred.best_slot);
        const auto st = estimator.state();

        // ---- 1. 相机取流 ----
        WEB_LOG("camera_fps", fps);

        // ---- 2. 装甲板检测 ----
        WEB_LOG("armor_count", static_cast<double>(detections.size()));

        // ---- 3. 灯条角点观测（观测向量，只保留 yaw）----
        if (!observations.empty()) {
            // 固定选最近的一块板（min 水平距离平方），避免多板时 front() 顺序跳变。
            const ArmorPose* ref = &observations.front();
            double best_d2 = 1e18;
            for (const auto& o : observations) {
                const double d2 = o.position_world_m[0] * o.position_world_m[0] +
                                  o.position_world_m[1] * o.position_world_m[1];
                if (d2 < best_d2) { best_d2 = d2; ref = &o; }
            }
            WEB_LOG("obs_yaw", ref->armor_normal_yaw);
        }

        // ---- 4. EKF（状态向量按顺序）----
        WEB_LOG("ekf_state", static_cast<double>(pred.state));
        WEB_LOG("stable_stage",
                (estimator.trackingState() == TrackingState::Stable)
                    ? static_cast<double>(estimator.stableStage()) : 0.0);
        WEB_LOG("ekf_x_m", pred.center_x_m);
        WEB_LOG("ekf_vx_m_s", pred.center_velocity_x_m_s);
        WEB_LOG("ekf_y_m", pred.center_y_m);
        WEB_LOG("ekf_vy_m_s", pred.center_velocity_y_m_s);
        WEB_LOG("ekf_z0_m", st.z0);
        WEB_LOG("ekf_z1_m", st.z1);
        WEB_LOG("ekf_radius_1_m", pred.radius_1_m);
        WEB_LOG("ekf_radius_2_m", pred.radius_2_m);
        WEB_LOG("ekf_yaw_rad", pred.target_plate_yaw);
        WEB_LOG("ekf_yaw_rate_rad_s", pred.body_yaw_velocity);
        WEB_LOG("ekf_mag", estimator.mag());   // 底盘倾斜角
        WEB_LOG("ekf_azi", estimator.azi());   // 底盘倾斜方向
        WEB_LOG("imm_prob_maneuver", estimator.modelProb(1));  // 机动模型概率（反向时冲高）
        WEB_LOG("imm_model", estimator.modelProb(1) > 0.5 ? 1.0 : 0.0);  // 0=稳态, 1=机动（哪个模型主导）

        // ---- 5. 发给电控 ----
        WEB_LOG("target_yaw", pred.target_yaw);
        WEB_LOG("target_pitch", pred.target_pitch);
        WEB_LOG("target_distance", pred.distance_m);

        // 目标板（Planner 选中的槽位对应的观测），画黄色框。
        const ArmorDetection* tracked = nullptr;
        if (pred.state != TrackingState::NoTarget && pred.target_obs_index >= 0 &&
            pred.target_obs_index < static_cast<int>(detections.size())) {
            tracked = &detections[pred.target_obs_index];
        }

        // 画框：先全画红色，被跟踪的板再覆盖一层黄色（更粗）。
        for (const auto& d : detections) {
            std::vector<cv::Point> pts;
            for (const auto& c : d.corners) {
                pts.emplace_back(static_cast<int>(c.x), static_cast<int>(c.y));
            }
            cv::polylines(pic, pts, true, cv::Scalar(0, 0, 255), 2);
        }
        if (tracked) {
            std::vector<cv::Point> pts;
            for (const auto& c : tracked->corners) {
                pts.emplace_back(static_cast<int>(c.x), static_cast<int>(c.y));
            }
            cv::polylines(pic, pts, true, cv::Scalar(0, 255, 255), 3);  // 黄色
        }

        // 画 EKF 底盘模型（蓝色）：旋转中心 + 4 板中心 + 半径连线 + 板框。
        if (pred.state != TrackingState::NoTarget) {
            constexpr double kPiL = 3.14159265358979323846;
            cv::Mat R_w2c = camera.rotation_camera_to_gimbal.t();
            cv::Mat t_w2c = -R_w2c * camera.translation_camera_to_gimbal;
            cv::Mat rvec_w2c;
            cv::Rodrigues(R_w2c, rvec_w2c);

            const double half_w = 0.135 * 0.5, half_h = 0.0285;
            const double center_z = 0.5 * (st.z0 + st.z1);
            std::vector<cv::Point3d> cpt{cv::Point3d(st.cx, st.cy, center_z)};
            std::vector<cv::Point2d> cimg;
            cv::projectPoints(cpt, rvec_w2c, t_w2c, camera.camera_matrix,
                              camera.distortion_coefficients, cimg);
            const cv::Point cc((int)cimg[0].x, (int)cimg[0].y);

            for (int i = 0; i < 4; ++i) {
                const double off = i * kPiL / 2.0;
                const double r = st.radius(i);
                const double ai = st.yaw + off;
                const double px = st.cx + r * std::cos(ai);
                const double py = st.cy + r * std::sin(ai);
                const double pz = st.height(i);

                std::vector<cv::Point3d> pc{cv::Point3d(px, py, pz)};
                std::vector<cv::Point2d> pi2;
                cv::projectPoints(pc, rvec_w2c, t_w2c, camera.camera_matrix,
                                  camera.distortion_coefficients, pi2);
                const cv::Point pp((int)pi2[0].x, (int)pi2[0].y);
                cv::line(pic, cc, pp, cv::Scalar(255, 0, 0), 1);

                const double wx = -std::sin(ai), wy = std::cos(ai);
                constexpr double kTilt = 0.26;
                const double sin_t = std::sin(kTilt), cos_t = std::cos(kTilt);
                const double dh = half_h * sin_t;
                const double dz = half_h * cos_t;
                const double nx = std::cos(ai), ny = std::sin(ai);
                std::vector<cv::Point3d> corners{
                    cv::Point3d(px - dh * nx - wx * half_w, py - dh * ny - wy * half_w, pz + dz),
                    cv::Point3d(px - dh * nx + wx * half_w, py - dh * ny + wy * half_w, pz + dz),
                    cv::Point3d(px + dh * nx + wx * half_w, py + dh * ny + wy * half_w, pz - dz),
                    cv::Point3d(px + dh * nx - wx * half_w, py + dh * ny - wy * half_w, pz - dz),
                };
                std::vector<cv::Point2d> ci;
                cv::projectPoints(corners, rvec_w2c, t_w2c, camera.camera_matrix,
                                  camera.distortion_coefficients, ci);
                std::vector<cv::Point> ci_pts;
                for (const auto& p : ci) ci_pts.emplace_back((int)p.x, (int)p.y);
                cv::polylines(pic, ci_pts, true, cv::Scalar(255, 0, 0), 2);
                cv::circle(pic, pp, 3, cv::Scalar(255, 0, 0), -1);
            }
            cv::circle(pic, cc, 4, cv::Scalar(0, 255, 0), -1);

            // 开火框（绿）：Planner 预测的目标板（外推后）
            const double ayaw = pred.aim_yaw;
            const double awx = -std::sin(ayaw), awy = std::cos(ayaw);
            std::vector<cv::Point3d> fcorners{
                cv::Point3d(pred.aim_x - awx * half_w, pred.aim_y - awy * half_w, pred.aim_z + half_h),
                cv::Point3d(pred.aim_x + awx * half_w, pred.aim_y + awy * half_w, pred.aim_z + half_h),
                cv::Point3d(pred.aim_x + awx * half_w, pred.aim_y + awy * half_w, pred.aim_z - half_h),
                cv::Point3d(pred.aim_x - awx * half_w, pred.aim_y - awy * half_w, pred.aim_z - half_h),
            };
            std::vector<cv::Point2d> fimg;
            cv::projectPoints(fcorners, rvec_w2c, t_w2c, camera.camera_matrix,
                              camera.distortion_coefficients, fimg);
            std::vector<cv::Point> fpts;
            for (const auto& p : fimg) fpts.emplace_back((int)p.x, (int)p.y);
            cv::polylines(pic, fpts, true, cv::Scalar(0, 255, 0), 2);
        }
        // ---- 俯视图 ----
        const int TV = 600;
        cv::Mat top(TV, TV, CV_8UC3, cv::Scalar(255, 255, 255));
        const double scl = 80.0;
        const int ox = TV / 2, oy = TV / 2;
        auto w2s = [&](double wx, double wy) -> cv::Point {
            return cv::Point((int)(ox - wy * scl), (int)(oy - wx * scl));
        };
        cv::line(top, cv::Point(ox, 0), cv::Point(ox, TV), cv::Scalar(200, 200, 200), 1);
        cv::line(top, cv::Point(0, oy), cv::Point(TV, oy), cv::Scalar(200, 200, 200), 1);
        cv::arrowedLine(top, cv::Point(ox, oy), cv::Point(ox, oy - 45), cv::Scalar(0, 0, 0), 2);
        cv::putText(top, "x(fwd)", cv::Point(ox + 4, oy - 50), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 0), 1);
        cv::arrowedLine(top, cv::Point(ox, oy), cv::Point(ox - 45, oy), cv::Scalar(0, 0, 0), 2);
        cv::putText(top, "y(left)", cv::Point(ox - 70, oy - 6), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 0), 1);
        const bool stable_now = (estimator.trackingState() == TrackingState::Stable);
        cv::putText(top,
                    stable_now ? "stable_stage:" + std::to_string(estimator.stableStage())
                               : "stable_stage: lost",
                    cv::Point(5, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    stable_now ? cv::Scalar(0, 128, 0) : cv::Scalar(0, 0, 255), 1);

        if (pred.state != TrackingState::NoTarget) {
            const cv::Point c2 = w2s(st.cx, st.cy);
            cv::circle(top, c2, 4, cv::Scalar(0, 255, 0), -1);
            for (int i = 0; i < 4; ++i) {
                const double ai = st.yaw + i * CV_PI / 2.0;
                const double r = st.radius(i);
                const cv::Point p2 = w2s(st.cx + r * std::cos(ai), st.cy + r * std::sin(ai));
                cv::line(top, c2, p2, cv::Scalar(255, 0, 0), 1);
                cv::circle(top, p2, 3, cv::Scalar(255, 0, 0), -1);
                cv::putText(top, std::to_string(i), cv::Point(p2.x + 6, p2.y - 6),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
            }
        }

        cv::Mat small, small_top;
        cv::resize(pic, small, cv::Size(), 0.5, 0.5);
        cv::resize(top, small_top, cv::Size(), 0.5, 0.5);
        const int cw = small.cols, ch = small.rows;
        const int tw = small_top.cols, th = small_top.rows;
        const int comb_h = (ch > th) ? ch : th;
        cv::Mat combined(comb_h, cw + tw, CV_8UC3, cv::Scalar(0, 0, 0));
        small.copyTo(combined(cv::Rect(0, (comb_h - ch) / 2, cw, ch)));
        small_top.copyTo(combined(cv::Rect(cw, (comb_h - th) / 2, tw, th)));
        {
            std::lock_guard<std::mutex> lock(g_frame_mutex);
            g_frame = combined;
            g_frame_ready = true;
        }

        ++frame_idx;
        ts += dt;

        const auto t_end = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(t_end - t_start).count();
        if (elapsed_ms < frame_interval_ms) {
            std::this_thread::sleep_for(std::chrono::milliseconds(
                static_cast<int>(frame_interval_ms - elapsed_ms)));
        }
    }

    g_shutdown.store(true, std::memory_order_relaxed);
    renderer.join();
    STOP_WEB_SERVER();
    std::printf("完成 %d 帧（%.1f 秒）\n", frame_idx, ts);
    return 0;
}
