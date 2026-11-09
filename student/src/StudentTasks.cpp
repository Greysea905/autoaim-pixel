#include "StudentTasks.hpp"

#include "EKFEstimator.hpp"
#include "Planner.hpp"

#include "MvsCamera.hpp"
#include "Detector.hpp"
#include "LightBarExtractor.hpp"
#include "ArmorClassifier.hpp"

#include <cmath>
#include <iostream>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace {

// 全局变量：相机参数缓存：
// ekf_predict 的签名固定，缺少 camera 参数；
// 解决：调用 armor_solve 时将 camera 每帧先写入全局变量，则调用 ekf_predict 时可读取，传给观测函数 h()。
CameraParameters g_camera_cache;

// 相机对象用函数内 static 保持单例：整个进程只初始化/打开一次。
rmv::camera::MvsCamera& cameraInstance() {
    static rmv::camera::MvsCamera camera;
    return camera;
}

// 相机序列号：
// 留空 "" 则用 openFirst() 开第一台；填目标序列号则用 openBySerial() 
constexpr const char* kCameraSerial = "";

// 惰性初始化：首次调用时打开相机并开始取流。
bool ensureCameraStarted() {
    rmv::camera::MvsCamera& camera = cameraInstance();
    if (camera.isOpen()) {
        return true;
    }
    const bool ok = (kCameraSerial[0] != '\0')
        ? camera.openBySerial(kCameraSerial)   // 填了序列号 → openBySerial()
        : camera.openFirst();                  // 空 → openFirst()
    if (!ok) {
        std::cerr << "[camera] 打开相机失败: " << camera.lastError() << '\n';
        return false;
    }

    // 先设置参数、再开始取流：像素格式/曝光/增益/帧率必须在 StartGrabbing 之前设置。
    // 统一设成 BGR8(3 通道)：相机默认可能输出 Bayer，而 grab() 只认 BayerRG8
    if (!camera.setPixelFormat(static_cast<uint64_t>(PixelType_Gvsp_BGR8_Packed))) {
        std::cerr << "[camera] 设置像素格式失败: " << camera.lastError() << '\n';
    }
    // 固定短曝光：关掉自动曝光
    if (!camera.setExposureTime(9000.0F)) {
        std::cerr << "[camera] 设置曝光失败: " << camera.lastError() << '\n';
    }
    camera.setGain(0.0F);
    if (!camera.disableFrameRate()) {
        std::cerr << "[camera] 关闭帧率控制失败: " << camera.lastError() << '\n';
    }

    if (!camera.startGrabbing()) {
        std::cerr << "[camera] 开始取流失败: " << camera.lastError() << '\n';
        camera.close();
        return false;
    }

    // 诊断：打印实际曝光/帧率，便于观察。
    float exp_us = 0.0F, fps = 0.0F;
    camera.getExposureTime(exp_us);
    camera.getFrameRate(fps);
    std::cout << "[camera] 实际 曝光=" << exp_us << "us 实际最高帧率=" << fps << "FPS\n";
    return true;
}

// 云台 → 世界 的旋转矩阵 R_yaw·R_pitch·R_roll。
// m_yaw·m_pitch·m_roll（roll≈0）。
cv::Mat gimbalToWorldRotMat(const GimbalState& gimbal) {
    const double cy = std::cos(gimbal.yaw), sy = std::sin(gimbal.yaw);
    const double cp = std::cos(gimbal.pitch), sp = std::sin(gimbal.pitch);
    const double cr = std::cos(gimbal.roll), sr = std::sin(gimbal.roll);
    return (cv::Mat_<double>(3, 3) <<
        cy * cp, -sy * cr - cy * sp * sr,  sy * sr - cy * sp * cr,
        sy * cp,  cy * cr - sy * sp * sr, -cy * sr - sy * sp * cr,
        sp,       cp * sr,                 cp * cr);
}

// 云台系 → 世界系：复用上面的旋转矩阵，单个点转换。
cv::Vec3d gimbal_to_world(const cv::Vec3d& p, const GimbalState& gimbal) {
    const cv::Mat R = gimbalToWorldRotMat(gimbal);
    return cv::Vec3d(R.at<double>(0, 0) * p[0] + R.at<double>(0, 1) * p[1] + R.at<double>(0, 2) * p[2],
                     R.at<double>(1, 0) * p[0] + R.at<double>(1, 1) * p[1] + R.at<double>(1, 2) * p[2],
                     R.at<double>(2, 0) * p[0] + R.at<double>(2, 1) * p[1] + R.at<double>(2, 2) * p[2]);
}

// 装甲板局部 → 世界 的旋转矩阵。
// 局部系（按 object_points 的角点顺序，顺时针=左手系）：
//   +x = 板宽(切向)，+y = 朝下(从上到下)，+z = 法向(朝相机方向，法向相对水平下俯 15°)。
// yaw 就是「朝相机」的法向在 x-y 平面的方位角（与 armor_normal_yaw 一致）。
cv::Mat armorToWorldRotation(double yaw, double tilt) {
    const double cy = std::cos(yaw), sy = std::sin(yaw);
    const double ct = std::cos(tilt), st = std::sin(tilt);
    return (cv::Mat_<double>(3, 3) <<
        -sy,  st * cy, -cy * ct,
         cy,  st * sy, -sy * ct,
        0.0,     -ct,      -st);
}

// 在 yaw 上搜索、最小化重投影误差
// 固定位置 p_world、固定倾角 tilt=15°，yaw 为唯一自由度。
// 搜索中心：云台朝向 + π（外法向正方向）
double optimizeArmorYaw(const std::vector<cv::Point3d>& object_points,
                        const std::vector<cv::Point2d>& image_points,
                        const CameraParameters& camera, const GimbalState& gimbal,
                        const cv::Vec3d& p_world) {
    const cv::Mat R_w2c =
        camera.rotation_camera_to_gimbal.t() * gimbalToWorldRotMat(gimbal).t();
    const cv::Mat t_w2c =
        -camera.rotation_camera_to_gimbal.t() * camera.translation_camera_to_gimbal;
    const cv::Mat t_world = (cv::Mat_<double>(3, 1) << p_world[0], p_world[1], p_world[2]);
    const double tilt = 0.2618;  // 15°

    // 搜索中心 = 云台朝向 + π（本项目 yaw=外法向）。
    // 锚定连续云台朝向，替代原 rough_yaw（会翻面的 PnP 法向），翻面解被 ±70° 排除。
    const double center = gimbal.yaw + CV_PI;
    double best_yaw = center, best_err = 1e18;
    const double step = CV_PI / 180.0;              // 1°
    const double range = 70.0 * CV_PI / 180.0;      // ±70°（避免切换瞬间裁剪）
    for (double yaw = center - range; yaw <= center + range; yaw += step) {
        const cv::Mat R_a2w = armorToWorldRotation(yaw, tilt);
        std::vector<cv::Point3d> corners_cam;
        corners_cam.reserve(4);
        for (const auto& p : object_points) {
            const cv::Mat p_local = (cv::Mat_<double>(3, 1) << p.x, p.y, p.z);
            const cv::Mat p_c = R_w2c * (R_a2w * p_local + t_world) + t_w2c;
            corners_cam.emplace_back(p_c.at<double>(0), p_c.at<double>(1), p_c.at<double>(2));
        }
        std::vector<cv::Point2d> projected;
        cv::projectPoints(corners_cam, cv::Mat::zeros(3, 1, CV_64F),
                          cv::Mat::zeros(3, 1, CV_64F),
                          camera.camera_matrix, camera.distortion_coefficients, projected);
        double err = 0.0;
        for (int i = 0; i < 4; ++i) err += cv::norm(projected[i] - image_points[i]);
        if (err < best_err) { best_err = err; best_yaw = yaw; }
    }
    // 归一化到 (-π, π]（方位角标准表示，恢复负值）。跨 ±π 边界是物理连续（差 2π），
    // 不是翻面（差 π）；EKF 的 wrapPi 会正确消掉这个 2π 差。
    while (best_yaw > CV_PI) best_yaw -= 2.0 * CV_PI;
    while (best_yaw <= -CV_PI) best_yaw += 2.0 * CV_PI;
    return best_yaw;
}

}  // namespace

bool get_pic(cv::Mat& pic) {
    if (!ensureCameraStarted()) {
        pic.release();
        return false;
    }

    // 持久缓冲：避免每帧分配 + 拷贝。下一帧 grab 会覆盖它，
    // 所以调用方必须在下次 get_pic 之前用完本帧（main 主循环满足此约定）。
    static rmv::camera::Frame frame;
    if (!cameraInstance().grab(frame) ||
        frame.width == 0 || frame.height == 0 || frame.data.empty()) {
        pic.release();
        return false;
    }

    // frame.data 已是 cv::Mat（64 字节对齐），零拷贝共享，无需 clone（AVX2 安全）。
    pic = frame.data;
    cv::flip(pic, pic, -1);  // 机上相机倒装，画面旋转 180°
    return true;
}

std::vector<ArmorDetection> armor_detect(const cv::Mat& pic, TeamColor enemy_color) {
    // 单例:进程内只加载一次模型
    static rmv::detect::Detector detector("detect/models/armor_detector.onnx");
    static rmv::detect::ArmorClassifier classifier("detect/models/armor_classifier.onnx");
    if (pic.empty() || !detector.ok()) {
        return {};
    }

    const auto dets = detector.detect(pic);

    // 敌方颜色 -> 检测类别:我方红打蓝(保留 blue=2),我方蓝打红(保留 red=1)
    const int target_class = (enemy_color == TeamColor::Blue) ? 2 : 1;

    std::vector<ArmorDetection> results;
    for (const auto& d : dets) {
        if (d.class_id != target_class) {
            continue;
        }
        // 灯条提取 + 配对 + 大小判定(灰度阈值 + 最小二乘 + 几何约束)
        const auto armor = rmv::detect::extractArmor(pic, d.box, d.class_id);
        if (!armor.valid) continue;

        ArmorDetection ad;
        ad.corners = armor.corners;
        ad.size = (armor.size == 1) ? ArmorSize::Large : ArmorSize::Small;
        ad.target_id = rmv::detect::ArmorClassifier::toTargetId(
            classifier.classify(pic, armor.corners, armor.size));
        ad.confidence = d.confidence;

        // 本次任务只打底盘：只保留底盘类 2/3/4/6/7，排除 1/8(大装甲板)。
        const bool is_chassis = (ad.target_id == 2 || ad.target_id == 3 ||
                                 ad.target_id == 4 || ad.target_id == 6 ||
                                 ad.target_id == 7);
        if (!is_chassis) continue;

        results.push_back(ad);
    }
    return results;
}

std::vector<ArmorPose> armor_solve(const std::vector<ArmorDetection>& detections,
                                  const CameraParameters& camera,
                                  const GimbalState& gimbal) {
    std::vector<ArmorPose> results;
    g_camera_cache = camera;  // 供 ekf_predict 取用（见文件顶部注释）
    if (!camera.calibrated) return results;

    for (const auto& det : detections) {
        // 按装甲板尺寸建 3D 点(毫米 -> 米),顺序与 corners 一致:[左上, 右上, 右下, 左下]
        const double width = (det.size == ArmorSize::Large) ? 0.230 : 0.135;
        const double hw = width * 0.5, hh = 0.057 * 0.5;
        const std::vector<cv::Point3d> object_points = {
            {-hw, -hh, 0.0}, {hw, -hh, 0.0}, {hw, hh, 0.0}, {-hw, hh, 0.0}};
        // 角点统一转 double(Point2d)，与物点(Point3d)/内参(CV_64F)保持一致，
        // 否则 solvePnP/projectPoints 会因 float/double 混用触发类型断言崩溃。
        std::vector<cv::Point2d> image_points;
        image_points.reserve(4);
        for (const auto& c : det.corners) image_points.emplace_back(c.x, c.y);

        // PnP 解算（IPPE：平面位姿专用，对装甲板共面目标更稳）
        cv::Mat rvec, tvec;
        if (!cv::solvePnP(object_points, image_points, camera.camera_matrix,
                          camera.distortion_coefficients, rvec, tvec, false,
                          cv::SOLVEPNP_IPPE)) {
            continue;  // PnP 失败
        }

        // 深度非法拒绝(相机 z 轴朝前,z <= 0 在相机后方)
        if (tvec.at<double>(2) <= 0.0) continue;

        // 重投影误差(平均像素距离),误差异常拒绝
        std::vector<cv::Point2d> projected;
        cv::projectPoints(object_points, rvec, tvec, camera.camera_matrix,
                          camera.distortion_coefficients, projected);
        double reproj = 0.0;
        for (int i = 0; i < 4; ++i) reproj += cv::norm(projected[i] - image_points[i]);
        reproj /= 4.0;
        if (reproj > 10.0) continue;

        // 相机坐标 -> 云台坐标(固定外参)
        const cv::Mat p_gimbal_mat = camera.rotation_camera_to_gimbal * tvec +
                                     camera.translation_camera_to_gimbal;
        const cv::Vec3d p_cam(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
        const cv::Vec3d p_gimbal(p_gimbal_mat.at<double>(0), p_gimbal_mat.at<double>(1),
                                 p_gimbal_mat.at<double>(2));

        // 云台坐标 -> 世界坐标（用云台当前 yaw/pitch/roll 旋转）
        const cv::Vec3d p_world = gimbal_to_world(p_gimbal, gimbal);

        // 板法向 yaw（EKF 第 4 维观测）：重投影搜索，搜索中心锚定「云台朝向+π」（连续），
        // 不再取会翻面的 PnP 法向（IPPE 对平面目标法向 ± 歧义）。
        const double normal_yaw =
            optimizeArmorYaw(object_points, image_points, camera, gimbal, p_world);

        ArmorPose pose;
        pose.detection = det;
        pose.position_camera_m = p_cam;
        pose.position_world_m = p_world;   // 世界系坐标（前/左/上）
        pose.armor_normal_yaw = normal_yaw;
        pose.armor_tilt = 0.2618;  // 板倾角固定 75°（板面与水平面夹角），法向相对水平面上仰 15°；写死，不随 PnP 法向噪声变化
        pose.reprojection_error = reproj;
        results.push_back(pose);
    }
    return results;
}

PredictionResult ekf_predict(const std::vector<ArmorPose>& observations,
                             const GimbalState& gimbal, double timestamp_seconds) {
    // 单例：整个进程只维护一个旋转车体估计器和一个 Planner，状态跨帧保持。
    static rmv::tracker::EKFEstimator estimator(rmv::tracker::loadEKFConfigOrDefault());
    static rmv::tracker::Planner planner;

    estimator.update(observations, timestamp_seconds, gimbal, g_camera_cache);
    PredictionResult result = planner.plan(estimator.state(), estimator.trackingState(), gimbal,
                                           timestamp_seconds, estimator.targetId());
    result.target_obs_index = estimator.associatedObservationIndex(result.best_slot);
    return result;
}
