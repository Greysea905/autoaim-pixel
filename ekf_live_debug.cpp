// 实车调试工具：接实时相机 + 串口（只读云台状态），跑检测+PnP+EKF，
// 弹窗口（左相机视图 + 右世界系俯视图）+ Web UI。不写回串口（纯观测，不干扰电控）。
//
// 用法: ./build/ekf_live_debug <串口设备> <波特率> <相机标定> [Web端口]
//   例: ./build/ekf_live_debug /dev/ttyUSB0 115200 config/camera_gimbal.yaml 8080
// 窗口：q 或 Esc 退出。

#include "AppSupport.hpp"
#include "EKFEstimator.hpp"
#include "Planner.hpp"
#include "SerialPort.hpp"
#include "StudentTasks.hpp"
#include "WebLogger.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr std::size_t kFrameSize = 64;
constexpr std::uint8_t kFrameHead = 0x71;
constexpr std::uint8_t kFrameTail = 0x4C;

bool hasFrameMarkers(const char* frame) {
    return static_cast<std::uint8_t>(frame[0]) == kFrameHead &&
           static_cast<std::uint8_t>(frame[kFrameSize - 1]) == kFrameTail;
}

// 从流缓冲里扫出完整帧，拿到一帧返回 true 并把内容写进 out。
bool extractFrame(std::vector<char>& buf, Translator& out) {
    while (buf.size() >= kFrameSize) {
        if (!hasFrameMarkers(buf.data())) {
            buf.erase(buf.begin());
            continue;
        }
        std::memcpy(out.data, buf.data(), kFrameSize);
        buf.erase(buf.begin(), buf.begin() + kFrameSize);
        return true;
    }
    return false;
}

// 云台系 → 世界系 的旋转矩阵（与 StudentTasks 里 gimbal_to_world 同款 R_yaw·R_pitch·R_roll）。
cv::Mat gimbalToWorldRotation(const GimbalState& gimbal) {
    const double cy = std::cos(gimbal.yaw), sy = std::sin(gimbal.yaw);
    const double cp = std::cos(gimbal.pitch), sp = std::sin(gimbal.pitch);
    const double cr = std::cos(gimbal.roll), sr = std::sin(gimbal.roll);
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
    R.at<double>(0, 0) = cy * cp; R.at<double>(0, 1) = -sy * cr - cy * sp * sr; R.at<double>(0, 2) = sy * sr - cy * sp * cr;
    R.at<double>(1, 0) = sy * cp; R.at<double>(1, 1) = cy * cr - sy * sp * sr;  R.at<double>(1, 2) = -cy * sr - sy * sp * cr;
    R.at<double>(2, 0) = sp;      R.at<double>(2, 1) = cp * sr;                 R.at<double>(2, 2) = cp * cr;
    return R;
}

// ---- 可视化：独立线程渲染，不阻塞处理流程（实际项目不需要可视化）----
struct RenderSnapshot {
    cv::Mat pic;
    std::vector<ArmorDetection> detections;
    PredictionResult pred;
    rmv::tracker::TrackState st;
    GimbalState gimbal;
    std::vector<ArmorPose> observations;
    bool stable_now;
    int stable_stage;
};

std::mutex g_render_mutex;
RenderSnapshot g_snapshot;
bool g_snapshot_ready = false;
std::atomic<bool> g_shutdown{false};

void renderLoop(const std::string& win_name, const CameraParameters& camera,
                const cv::Mat& R_c2g, const cv::Mat& t_w2c) {
    while (!g_shutdown.load(std::memory_order_relaxed)) {
        RenderSnapshot snap;
        bool ready = false;
        {
            std::lock_guard<std::mutex> lock(g_render_mutex);
            if (g_snapshot_ready) {
                snap = g_snapshot;
                g_snapshot_ready = false;
                ready = true;
            }
        }
        if (ready) {
            // 相机视图：检测框（红）
            for (const auto& d : snap.detections) {
                std::vector<cv::Point> pts;
                for (const auto& c : d.corners) pts.emplace_back((int)c.x, (int)c.y);
                cv::polylines(snap.pic, pts, true, cv::Scalar(0, 0, 255), 2);
            }
            // 目标板（Planner 选中的槽位对应的观测）→ 黄框
            if (snap.pred.state != TrackingState::NoTarget &&
                snap.pred.target_obs_index >= 0 &&
                snap.pred.target_obs_index < static_cast<int>(snap.observations.size())) {
                const ArmorPose& tracked = snap.observations[snap.pred.target_obs_index];
                std::vector<cv::Point> pts;
                for (const auto& c : tracked.detection.corners)
                    pts.emplace_back((int)c.x, (int)c.y);
                cv::polylines(snap.pic, pts, true, cv::Scalar(0, 255, 255), 3);
            }
            // EKF 模型投影到相机（蓝）
            if (snap.pred.state != TrackingState::NoTarget) {
                cv::Mat R_g2w = gimbalToWorldRotation(snap.gimbal);
                cv::Mat R_w2c = (R_g2w * R_c2g).t();
                cv::Mat rvec_w2c;
                cv::Rodrigues(R_w2c, rvec_w2c);
                const double center_z = 0.5 * (snap.st.z0 + snap.st.z1);
                std::vector<cv::Point3d> cpt{cv::Point3d(snap.st.cx, snap.st.cy, center_z)};
                std::vector<cv::Point2d> cimg;
                cv::projectPoints(cpt, rvec_w2c, t_w2c, camera.camera_matrix,
                                  camera.distortion_coefficients, cimg);
                const cv::Point cc((int)cimg[0].x, (int)cimg[0].y);
                for (int i = 0; i < 4; ++i) {
                    const double ai = snap.st.yaw + i * CV_PI / 2.0;
                    const double r = snap.st.radius(i);
                    const double px = snap.st.cx + r * std::cos(ai);
                    const double py = snap.st.cy + r * std::sin(ai);
                    const double pz = snap.st.height(i);
                    std::vector<cv::Point3d> pc{cv::Point3d(px, py, pz)};
                    std::vector<cv::Point2d> pi2;
                    cv::projectPoints(pc, rvec_w2c, t_w2c, camera.camera_matrix,
                                      camera.distortion_coefficients, pi2);
                    const cv::Point pp((int)pi2[0].x, (int)pi2[0].y);
                    cv::line(snap.pic, cc, pp, cv::Scalar(255, 0, 0), 1);

                    // 完整小板框（135×57mm，带固定倾斜 0.26 rad，照 ekf_video_debug）
                    constexpr double kHalfW = 0.135 * 0.5, kHalfH = 0.0285;
                    constexpr double kTilt = 0.26;
                    const double sin_t = std::sin(kTilt), cos_t = std::cos(kTilt);
                    const double dh = kHalfH * sin_t;
                    const double dz = kHalfH * cos_t;
                    const double nx = std::cos(ai), ny = std::sin(ai);   // 法向（朝外）
                    const double wx = -std::sin(ai), wy = std::cos(ai);  // 宽度方向
                    std::vector<cv::Point3d> corners{
                        cv::Point3d(px - dh * nx - wx * kHalfW, py - dh * ny - wy * kHalfW, pz + dz),
                        cv::Point3d(px - dh * nx + wx * kHalfW, py - dh * ny + wy * kHalfW, pz + dz),
                        cv::Point3d(px + dh * nx + wx * kHalfW, py + dh * ny + wy * kHalfW, pz - dz),
                        cv::Point3d(px + dh * nx - wx * kHalfW, py + dh * ny - wy * kHalfW, pz - dz),
                    };
                    std::vector<cv::Point2d> ci;
                    cv::projectPoints(corners, rvec_w2c, t_w2c, camera.camera_matrix,
                                      camera.distortion_coefficients, ci);
                    std::vector<cv::Point> ci_pts;
                    for (const auto& p : ci) ci_pts.emplace_back((int)p.x, (int)p.y);
                    cv::polylines(snap.pic, ci_pts, true, cv::Scalar(255, 0, 0), 2);
                    cv::circle(snap.pic, pp, 3, cv::Scalar(255, 0, 0), -1);
                }
                cv::circle(snap.pic, cc, 4, cv::Scalar(0, 255, 0), -1);

                // 开火框（绿）：Planner 预测的目标板（外推后）
                constexpr double kHalfW = 0.135 * 0.5, kHalfH = 0.0285;
                const double ayaw = snap.pred.aim_yaw;
                const double awx = -std::sin(ayaw), awy = std::cos(ayaw);
                std::vector<cv::Point3d> fcorners{
                    cv::Point3d(snap.pred.aim_x - awx * kHalfW, snap.pred.aim_y - awy * kHalfW, snap.pred.aim_z + kHalfH),
                    cv::Point3d(snap.pred.aim_x + awx * kHalfW, snap.pred.aim_y + awy * kHalfW, snap.pred.aim_z + kHalfH),
                    cv::Point3d(snap.pred.aim_x + awx * kHalfW, snap.pred.aim_y + awy * kHalfW, snap.pred.aim_z - kHalfH),
                    cv::Point3d(snap.pred.aim_x - awx * kHalfW, snap.pred.aim_y - awy * kHalfW, snap.pred.aim_z - kHalfH),
                };
                std::vector<cv::Point2d> fimg;
                cv::projectPoints(fcorners, rvec_w2c, t_w2c, camera.camera_matrix,
                                  camera.distortion_coefficients, fimg);
                std::vector<cv::Point> fpts;
                for (const auto& p : fimg) fpts.emplace_back((int)p.x, (int)p.y);
                cv::polylines(snap.pic, fpts, true, cv::Scalar(0, 255, 0), 2);
            }
            // 俯视图（世界系）
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
            cv::putText(top,
                        snap.stable_now ? "stable_stage:" + std::to_string(snap.stable_stage)
                                        : "stable_stage: lost",
                        cv::Point(5, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        snap.stable_now ? cv::Scalar(0, 128, 0) : cv::Scalar(0, 0, 255), 1);
            if (snap.pred.state != TrackingState::NoTarget) {
                const cv::Point c2 = w2s(snap.st.cx, snap.st.cy);
                cv::circle(top, c2, 4, cv::Scalar(0, 255, 0), -1);
                constexpr double kPlateHalfW = 0.135 * 0.5;  // 小板宽 0.135m 的一半
                for (int i = 0; i < 4; ++i) {
                    const double ai = snap.st.yaw + i * CV_PI / 2.0;
                    const double r = snap.st.radius(i);
                    const double px = snap.st.cx + r * std::cos(ai);
                    const double py = snap.st.cy + r * std::sin(ai);
                    const cv::Point p2 = w2s(px, py);
                    cv::line(top, c2, p2, cv::Scalar(255, 0, 0), 1);
                    // 板框：沿切向画 0.135m 宽的小板（俯视图里是一条线段）
                    const double tx = -std::sin(ai), ty = std::cos(ai);
                    const cv::Point pL = w2s(px - kPlateHalfW * tx, py - kPlateHalfW * ty);
                    const cv::Point pR = w2s(px + kPlateHalfW * tx, py + kPlateHalfW * ty);
                    cv::line(top, pL, pR, cv::Scalar(255, 0, 0), 3);
                    cv::putText(top, std::to_string(i), cv::Point(p2.x + 6, p2.y - 6),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
                }
            }
            for (const auto& o : snap.observations) {
                const cv::Point p2 = w2s(o.position_world_m[0], o.position_world_m[1]);
                cv::circle(top, p2, 3, cv::Scalar(0, 0, 255), -1);
                const double yaw = o.armor_normal_yaw;
                const cv::Point tip = w2s(o.position_world_m[0] + 0.18 * std::cos(yaw),
                                          o.position_world_m[1] + 0.18 * std::sin(yaw));
                cv::arrowedLine(top, p2, tip, cv::Scalar(0, 0, 255), 1);
            }
            // 拼图 + 显示
            cv::Mat small, small_top;
            cv::resize(snap.pic, small, cv::Size(), 0.5, 0.5);
            cv::resize(top, small_top, cv::Size(), 0.5, 0.5);
            const int cw = small.cols, ch = small.rows;
            const int tw = small_top.cols, th = small_top.rows;
            const int comb_h = (ch > th) ? ch : th;
            cv::Mat combined(comb_h, cw + tw, CV_8UC3, cv::Scalar(0, 0, 0));
            small.copyTo(combined(cv::Rect(0, (comb_h - ch) / 2, cw, ch)));
            small_top.copyTo(combined(cv::Rect(cw, (comb_h - th) / 2, tw, th)));
            cv::imshow(win_name, combined);
        }
        const int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) g_shutdown.store(true, std::memory_order_relaxed);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4 || argc > 6) {
        std::fprintf(stderr, "用法: %s <串口设备> <波特率> <相机标定> [Web端口] [enemy=red|blue]\n", argv[0]);
        return 2;
    }
    const std::string serial_device = argv[1];
    const int baud = std::atoi(argv[2]);

    CameraParameters camera;
    std::string err;
    if (!loadCameraParameters(argv[3], camera, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    const int web_port = (argc >= 5) ? std::atoi(argv[4]) : 8080;
    // 指定 enemy 后串口 status 不再覆盖（方便测蓝板时强制 blue）
    const bool enemy_forced = (argc >= 6);

    SerialPort serial(serial_device);
    if (!serial.isOpen()) {
        std::fprintf(stderr, "无法打开串口: %s\n", serial_device.c_str());
        return 1;
    }
    if (!serial.InitSerialPort(baud, 8, 1, 'N')) {
        std::fprintf(stderr, "串口配置失败\n");
        return 1;
    }
    serial.setReadTimeout(std::chrono::milliseconds(1));  // 近非阻塞

    START_WEB_SERVER(web_port);

    rmv::tracker::EKFEstimator estimator(rmv::tracker::loadEKFConfigOrDefault());
    rmv::tracker::Planner planner;

    GimbalState gimbal{};
    TeamColor enemy = (enemy_forced && std::string(argv[5]) == "blue")
        ? TeamColor::Blue : TeamColor::Red;
    std::vector<char> stream_buffer;
    stream_buffer.reserve(kFrameSize * 2);

    // 相机系 → 云台系（固定外参）
    const cv::Mat R_c2g = camera.rotation_camera_to_gimbal;
    const cv::Mat t_c2g = camera.translation_camera_to_gimbal;
    const cv::Mat t_w2c = -R_c2g.t() * t_c2g;  // 相机相对云台中心的平移，在相机系，恒定

    cv::Mat pic;
    const std::string win_name = "live (左=相机, 右=俯视图, q=退出)";
    // 无显示器（如 SSH 无 X 转发）时跳过可视化窗口，只跑 Web UI。
    const bool has_display = (std::getenv("DISPLAY") != nullptr && std::getenv("DISPLAY")[0] != '\0');
    if (!has_display) {
        std::fprintf(stderr, "[提示] 无显示器(DISPLAY 未设)，跳过可视化窗口，仅 Web UI；Ctrl+C 退出。\n");
    }
    auto last_frame_time = std::chrono::steady_clock::now();

    std::thread renderer;
    if (has_display) {
        renderer = std::thread(renderLoop, win_name, camera, R_c2g, t_w2c);
    }

    while (!g_shutdown.load(std::memory_order_relaxed)) {
        // 1. 读串口（只读，拿云台 yaw/pitch/roll + status）
        {
            std::array<char, kFrameSize> chunk{};
            const int len = serial.Read(chunk.data(), static_cast<int>(chunk.size()));
            if (len > 0) {
                stream_buffer.insert(stream_buffer.end(), chunk.begin(), chunk.begin() + len);
                Translator incoming{};
                while (extractFrame(stream_buffer, incoming)) {
                    const auto& m = incoming.message;
                    if (std::isfinite(m.yaw) && std::isfinite(m.pitch) && std::isfinite(m.roll)) {
                        gimbal = gimbalStateFromMessage(m);
                        if (!enemy_forced) enemy = enemyColorFromStatus(m.status);
                    }
                }
            }
        }

        // 2. 取流
        pic.release();
        if (!get_pic(pic) || pic.empty()) {
            continue;
        }

        // 3. 检测 + PnP + EKF + Planner
        const auto detections = armor_detect(pic, enemy);
        const auto observations = armor_solve(detections, camera, gimbal);
        const double now_ts = std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        estimator.update(observations, now_ts, gimbal, camera);
        auto pred = planner.plan(estimator.state(), estimator.trackingState(), gimbal,
                                 now_ts, estimator.targetId());
        pred.target_obs_index = estimator.associatedObservationIndex(pred.best_slot);
        const auto st = estimator.state();

        // 可视化快照：独立线程渲染，不阻塞处理流程
        if (has_display) {
            RenderSnapshot snap;
            snap.pic = pic;  // 浅拷贝（ref-counted）
            snap.detections = detections;
            snap.pred = pred;
            snap.st = st;
            snap.gimbal = gimbal;
            snap.observations = observations;
            snap.stable_now = (estimator.trackingState() == TrackingState::Stable);
            snap.stable_stage = estimator.stableStage();
            {
                std::lock_guard<std::mutex> lock(g_render_mutex);
                g_snapshot = std::move(snap);
                g_snapshot_ready = true;
            }
        }

        // ---- Web UI 指标 ----
        const auto now = std::chrono::steady_clock::now();
        const double frame_s =
            std::chrono::duration<double>(now - last_frame_time).count();
        last_frame_time = now;
        if (frame_s > 0.0) WEB_LOG("camera_fps", 1.0 / frame_s);
        WEB_LOG("armor_count", static_cast<double>(detections.size()));
        WEB_LOG("ekf_state", static_cast<double>(pred.state));
        WEB_LOG("ekf_x_m", pred.center_x_m);
        WEB_LOG("ekf_vx_m_s", pred.center_velocity_x_m_s);  // vcx（调 q_pos）
        WEB_LOG("ekf_y_m", pred.center_y_m);
        WEB_LOG("ekf_vy_m_s", pred.center_velocity_y_m_s);  // vcy（调 q_pos）
        WEB_LOG("ekf_z0_m", st.z0);
        WEB_LOG("ekf_z1_m", st.z1);
        WEB_LOG("ekf_radius_1_m", pred.radius_1_m);
        WEB_LOG("ekf_radius_2_m", pred.radius_2_m);
        WEB_LOG("ekf_yaw_rad", pred.target_plate_yaw);
        WEB_LOG("ekf_yaw_rate_rad_s", pred.body_yaw_velocity);
        WEB_LOG("target_yaw", pred.target_yaw);
        WEB_LOG("target_pitch", pred.target_pitch);
        WEB_LOG("target_distance", pred.distance_m);
        WEB_LOG("stable_stage",
                (estimator.trackingState() == TrackingState::Stable)
                    ? static_cast<double>(estimator.stableStage()) : 0.0);
        if (!observations.empty()) {
            const ArmorPose* ref = &observations.front();
            double best_d2 = 1e18;
            for (const auto& o : observations) {
                const double d2 = o.position_world_m[0] * o.position_world_m[0] +
                                  o.position_world_m[1] * o.position_world_m[1];
                if (d2 < best_d2) { best_d2 = d2; ref = &o; }
            }
            WEB_LOG("pnp_x_m", ref->position_world_m[0]);
            WEB_LOG("pnp_y_m", ref->position_world_m[1]);
            WEB_LOG("pnp_z_m", ref->position_world_m[2]);
            WEB_LOG("pnp_yaw_rad", ref->armor_normal_yaw);
            WEB_LOG("pnp_distance_m", cv::norm(ref->position_world_m));
        }
    }

    g_shutdown.store(true, std::memory_order_relaxed);
    if (renderer.joinable()) renderer.join();
    STOP_WEB_SERVER();
    return 0;
}
