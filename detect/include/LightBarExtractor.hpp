#pragma once

#include <opencv2/core.hpp>

#include <array>

namespace rmv::detect {

struct Light {
    cv::Point2f top, bottom, center;
    float length = 0.f;      // 灯条长度(上下端点距离)
    float width = 0.f;       // 灯条宽度(面积/长度)
    float tilt_angle = 0.f;  // 倾斜角(度)
    int color = 0;           // 1=红, 2=蓝
};

// 一块装甲板的结果
struct ArmorResult {
    bool valid = false;
    std::array<cv::Point2f, 4> corners{};  // 左上、右上、右下、左下
    int size = 0;                          // 0=小装甲板, 1=大装甲板
    cv::Point2f center{};
    Light left_light, right_light;         // 左右灯条(数字透视变换用)
};

// 检测装甲板:灰度阈值找灯条 -> 颜色判断 -> 配对 -> 输出角点/大小/灯条。
// box 为 YOLO 检测的 bbox(原图坐标),detect_color: 1=红, 2=蓝。
ArmorResult extractArmor(const cv::Mat& bgr, const cv::Rect2f& box, int detect_color);

}  // namespace rmv::detect
