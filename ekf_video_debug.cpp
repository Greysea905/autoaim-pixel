// 视频离线调参工具：读录像 → 检测+PnP+EKF → 可视化窗口 + Web UI。
// 与 ekf_debug 的区别：相机换成录制视频（无实车时替代实时取流）。
// 渲染在独立线程，不阻塞处理；处理按视频帧率节流，便于观察。
//
// 用法：
//   ./build/ekf_video_debug <视频文件> <相机标定> [Web端口] [enemy=red|blue]
//   例：./build/ekf_video_debug /path/video.avi config/camera.yaml 8080 blue
// 空格 = 暂停/继续；q 或 Esc = 退出。
// 需在项目根目录运行（检测/分类模型是相对路径）。

#include "AppSupport.hpp"
#include "EKFEstimator.hpp"
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

// 渲染线程共享的帧（处理线程写入，渲染线程读取）。
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
        if (key == 'q' || key == 27) {  // q 或 Esc 退出
            g_shutdown.store(true, std::memory_order_relaxed);
        } else if (key == ' ') {        // 空格 暂停/继续
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
    const std::string win_name = "video (空格=暂停, q=退出)";

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

    rmv::tracker::EKFEstimator estimator(rmv::tracker::loadEKFConfigOrDefault());
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
        const auto observations = armor_solve(detections, camera, gimbal);
        estimator.update(observations, ts, gimbal, camera);
        // Planner 用实时时间戳（latency≈0，不外推）；EKF 仍用视频时间。
        const double now_ts = std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        PredictionResult pred = planner.plan(estimator.state(), estimator.trackingState(),
                                             gimbal, now_ts, estimator.targetId());
        pred.target_obs_index = estimator.associatedObservationIndex(pred.best_slot);
        const auto st = estimator.state();

        // ---- Web UI 指标（按 12 维状态 + pnp + 帧率 排序）----
        WEB_LOG("camera_fps", fps);
        WEB_LOG("armor_count", static_cast<double>(detections.size()));
        WEB_LOG("ekf_state", static_cast<double>(pred.state));
        WEB_LOG("stable_stage",
                (estimator.trackingState() == TrackingState::Stable)
                    ? static_cast<double>(estimator.stableStage()) : 0.0);
        WEB_LOG("ekf_x_m", pred.center_x_m);               // cx
        WEB_LOG("ekf_vx_m_s", pred.center_velocity_x_m_s); // vx
        WEB_LOG("ekf_y_m", pred.center_y_m);               // cy
        WEB_LOG("ekf_vy_m_s", pred.center_velocity_y_m_s); // vy
        WEB_LOG("ekf_z0_m", st.z0);                        // 板0/2 高度
        WEB_LOG("ekf_z1_m", st.z1);                        // 板1/3 高度
        WEB_LOG("ekf_radius_1_m", pred.radius_1_m);        // r0
        WEB_LOG("ekf_radius_2_m", pred.radius_2_m);        // r1
        WEB_LOG("ekf_yaw_rad", pred.target_plate_yaw);     // 当前选中板朝向
        WEB_LOG("ekf_yaw_rate_rad_s", pred.body_yaw_velocity);  // w
        WEB_LOG("target_yaw", pred.target_yaw);
        WEB_LOG("target_pitch", pred.target_pitch);
        WEB_LOG("target_distance", pred.distance_m);
        if (!observations.empty()) {
            // 固定选最近的一块板（min 水平距离平方），避免多板时 front() 顺序跳变导致曲线乱跳。
            const ArmorPose* ref = &observations.front();
            double best_d2 = 1e18;
            for (const auto& o : observations) {
                const double d2 = o.position_world_m[0] * o.position_world_m[0] +
                                  o.position_world_m[1] * o.position_world_m[1];
                if (d2 < best_d2) { best_d2 = d2; ref = &o; }
            }
            const cv::Vec3d& p = ref->position_world_m;
            WEB_LOG("pnp_x_m", p[0]);
            WEB_LOG("pnp_y_m", p[1]);
            WEB_LOG("pnp_z_m", p[2]);
            WEB_LOG("pnp_yaw_rad", ref->armor_normal_yaw);
            WEB_LOG("pnp_tilt", ref->armor_tilt);
            WEB_LOG("pnp_distance_m", cv::norm(p));
            WEB_LOG("pnp_reprojection_error", ref->reprojection_error);
        }

        // 目标板（Planner 选中的槽位对应的观测），画黄色框。
        const ArmorPose* tracked = nullptr;
        if (pred.state != TrackingState::NoTarget && pred.target_obs_index >= 0 &&
            pred.target_obs_index < static_cast<int>(observations.size())) {
            tracked = &observations[pred.target_obs_index];
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
            for (const auto& c : tracked->detection.corners) {
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

            // 本任务只打底盘(小板 0.135m)，高 0.057m。
            const double half_w = 0.135 * 0.5, half_h = 0.0285;
            // 旋转中心高度 = 两对板高度均值。
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
                cv::line(pic, cc, pp, cv::Scalar(255, 0, 0), 1);  // 半径连线

                const double wx = -std::sin(ai), wy = std::cos(ai);  // 宽度方向（垂直于法向）
                // 装甲板向内倾斜（四棱锥侧边），物理常数 0.26 rad（实测值会受相机倾斜污染，故用固定值）。
                constexpr double kTilt = 0.26;
                const double sin_t = std::sin(kTilt), cos_t = std::cos(kTilt);
                const double dh = half_h * sin_t;  // 顶/底相对中心的水平内缩/外扩
                const double dz = half_h * cos_t;  // 顶/底相对中心的竖直偏移
                const double nx = std::cos(ai), ny = std::sin(ai);  // 法向（朝外）
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
                cv::polylines(pic, ci_pts, true, cv::Scalar(255, 0, 0), 2);  // 蓝色板框
                cv::circle(pic, pp, 3, cv::Scalar(255, 0, 0), -1);           // 板中心蓝点
            }
            cv::circle(pic, cc, 4, cv::Scalar(0, 255, 0), -1);  // 旋转中心绿点

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
        // ---- 俯视图（世界系 x=前, y=左；屏幕：前→上，左→左）----
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
        // 稳定期标号：处于稳定状态就标出 stable_stage(1,2,3,...)，一旦丧失稳定则不显示数字。
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
                // 板标号 0/1/2/3（标在 EKF 板中心附近，蓝色匹配车体模型）
                cv::putText(top, std::to_string(i), cv::Point(p2.x + 6, p2.y - 6),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
            }
        }
        for (const auto& o : observations) {
            const cv::Point p2 = w2s(o.position_world_m[0], o.position_world_m[1]);
            cv::circle(top, p2, 3, cv::Scalar(0, 0, 255), -1);
            const double yaw = o.armor_normal_yaw;
            const cv::Point tip = w2s(o.position_world_m[0] + 0.18 * std::cos(yaw),
                                      o.position_world_m[1] + 0.18 * std::sin(yaw));
            cv::arrowedLine(top, p2, tip, cv::Scalar(0, 0, 255), 1);
        }

        // 拼图：左相机视图（0.5×）+ 右俯视图（0.5×）
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

        // 节流到视频帧率，便于观察。
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
