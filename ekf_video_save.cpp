// 用我们自己的 EKF（12 维整车状态 + 4 槽观测）跑视频并保存可视化。
// 左：相机视图（红=检测框，蓝=EKF 4 板模型带倾斜，绿=旋转中心）。
// 右：世界系俯视图（绿=EKF 旋转中心，蓝=EKF 车体半径/板中心，红=PnP 板中心+箭头=pnp_yaw，标出 x=前/y=左 轴）。
// 用法: ./build/ekf_video_save <视频> <相机标定> <输出视频> [enemy=red|blue]

#include "AppSupport.hpp"
#include "EKFEstimator.hpp"
#include "Planner.hpp"
#include "StudentTasks.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    if (argc < 4 || argc > 5) {
        std::fprintf(stderr, "用法: %s <视频> <相机标定> <输出视频> [enemy=red|blue]\n", argv[0]);
        return 2;
    }
    const std::string video_path = argv[1];
    const std::string out_path = argv[3];
    const TeamColor enemy =
        (argc >= 5 && std::string(argv[4]) == "blue") ? TeamColor::Blue : TeamColor::Red;

    CameraParameters camera;
    std::string err;
    if (!loadCameraParameters(argv[2], camera, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::fprintf(stderr, "打开视频失败\n");
        return 1;
    }
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 1.0) fps = 109.0;
    const int W = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const int H = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    // 俯视图尺寸（世界系 x-y 平面）
    const int TV = 600;  // 600x600
    const int OW = W + TV;           // 拼图宽
    const int OH = (H > TV) ? H : TV;  // 拼图高
    cv::VideoWriter writer(out_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps,
                           cv::Size(OW, OH));
    if (!writer.isOpened()) {
        std::fprintf(stderr, "无法写输出视频\n");
        return 1;
    }

    rmv::tracker::EKFEstimator estimator(rmv::tracker::loadEKFConfigOrDefault());
    rmv::tracker::Planner planner;
    const GimbalState gimbal{};

    const cv::Mat R_w2c = camera.rotation_camera_to_gimbal.t();
    const cv::Mat t_w2c = -R_w2c * camera.translation_camera_to_gimbal;
    cv::Mat rvec_w2c;
    cv::Rodrigues(R_w2c, rvec_w2c);

    cv::Mat pic;
    double ts = 0.0;
    int frame_idx = 0;
    while (cap.read(pic)) {
        cv::flip(pic, pic, -1);  // 云台相机倒置，录像转 180° 回正
        const auto detections = armor_detect(pic, enemy);
        const auto observations = armor_solve(detections, camera, gimbal);
        estimator.update(observations, ts, gimbal, camera);
        const double now_ts = std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        const auto pred = planner.plan(estimator.state(), estimator.trackingState(), gimbal,
                                       now_ts, estimator.targetId());
        const auto st = estimator.state();

        // 红框：检测
        for (const auto& d : detections) {
            std::vector<cv::Point> pts;
            for (const auto& c : d.corners) pts.emplace_back((int)c.x, (int)c.y);
            cv::polylines(pic, pts, true, cv::Scalar(0, 0, 255), 2);
        }

        // 蓝：我们 EKF 的 4 板模型（+r 向外，带固定 0.26 rad 倾斜）
        if (pred.state != TrackingState::NoTarget) {
            const double half_w = 0.135 * 0.5, half_h = 0.0285;  // 本任务只打底盘(小板 0.135m)
            const double center_z = 0.5 * (st.z0 + st.z1);
            std::vector<cv::Point3d> cpt{cv::Point3d(st.cx, st.cy, center_z)};
            std::vector<cv::Point2d> cimg;
            cv::projectPoints(cpt, rvec_w2c, t_w2c, camera.camera_matrix,
                              camera.distortion_coefficients, cimg);
            const cv::Point cc((int)cimg[0].x, (int)cimg[0].y);

            for (int i = 0; i < 4; ++i) {
                const double ai = st.yaw + i * CV_PI / 2.0;
                const double r = st.radius(i);
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
                const double dh = half_h * sin_t, dz = half_h * cos_t;
                const double nx = std::cos(ai), ny = std::sin(ai);
                std::vector<cv::Point3d> corners{
                    cv::Point3d(px - dh * nx - wx * half_w, py - dh * ny - wy * half_w, pz + dz),
                    cv::Point3d(px - dh * nx + wx * half_w, py - dh * ny + wy * half_w, pz + dz),
                    cv::Point3d(px + dh * nx + wx * half_w, py + dh * ny + wy * half_w, pz - dz),
                    cv::Point3d(px + dh * nx - wx * half_w, py + dh * ny - wy * half_w, pz - dz)};
                std::vector<cv::Point2d> ci;
                cv::projectPoints(corners, rvec_w2c, t_w2c, camera.camera_matrix,
                                  camera.distortion_coefficients, ci);
                std::vector<cv::Point> ci_pts;
                for (const auto& p : ci) ci_pts.emplace_back((int)p.x, (int)p.y);
                cv::polylines(pic, ci_pts, true, cv::Scalar(255, 0, 0), 2);
                cv::circle(pic, pp, 3, cv::Scalar(255, 0, 0), -1);
            }
            cv::circle(pic, cc, 4, cv::Scalar(0, 255, 0), -1);
        }

        // 俯视图（世界系 x=前, y=左；屏幕：前→上，左→左）
        cv::Mat top(TV, TV, CV_8UC3, cv::Scalar(255, 255, 255));
        const double scl = 80.0;               // px/m
        const int ox = TV / 2, oy = TV / 2;    // 相机原点在中心
        auto w2s = [&](double wx, double wy) -> cv::Point {
            return cv::Point((int)(ox - wy * scl), (int)(oy - wx * scl));
        };
        // 坐标轴：x(前)→上，y(左)→左
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

        // EKF 车体（蓝半径+蓝板中心）+ 旋转中心（绿）
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
        // PnP 板中心（红点）+ pnp_yaw 箭头（红）
        for (const auto& o : observations) {
            const cv::Point p2 = w2s(o.position_world_m[0], o.position_world_m[1]);
            cv::circle(top, p2, 3, cv::Scalar(0, 0, 255), -1);
            const double yaw = o.armor_normal_yaw;
            const cv::Point tip = w2s(o.position_world_m[0] + 0.18 * std::cos(yaw),
                                      o.position_world_m[1] + 0.18 * std::sin(yaw));
            cv::arrowedLine(top, p2, tip, cv::Scalar(0, 0, 255), 1);
        }

        // 拼图：左相机视图 + 右俯视图
        cv::Mat combined(OH, OW, CV_8UC3, cv::Scalar(0, 0, 0));
        pic.copyTo(combined(cv::Rect(0, (OH - H) / 2, W, H)));
        top.copyTo(combined(cv::Rect(W, (OH - TV) / 2, TV, TV)));
        writer.write(combined);

        ++frame_idx;
        ts += 1.0 / fps;
    }

    std::fprintf(stderr, "完成 %d 帧，输出 %s\n", frame_idx, out_path.c_str());
    return 0;
}
