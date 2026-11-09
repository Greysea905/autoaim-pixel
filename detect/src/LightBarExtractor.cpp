#include "LightBarExtractor.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <limits>
#include <vector>

namespace rmv::detect {

namespace {

// 灯条筛选参数(与 Python test_match.py 一致)
// 分色灰度阈值：灰度 = 0.299R+0.587G+0.114B，蓝色权重(0.114)远低于红色(0.299)，
// 蓝色灯条在灰度图上更暗，需更低阈值才提取得全。实机实测：红 40、蓝 26。
constexpr float kGrayThresRed = 40.f;   // 红色灯条灰度阈值
constexpr float kGrayThresBlue = 26.f;  // 蓝色灯条灰度阈值
constexpr float kAreaThres = 50.f;      // 面积阈值
constexpr float kRatioThres = 1.8f;     // 宽高比(细长,放宽以容忍运动模糊把灯条拖宽)
constexpr float kMaxTilt = 65.f;        // 最大倾斜角(从竖直算,放宽以容忍镜头旋转)
constexpr float kParallelThres = 20.f;  // 两灯条平行差(放宽)
constexpr float kLenRatioThres = 0.75f;  // 长度比下限(短/长)

// 灰度梯度端点精修：在二值化端点邻域、沿灯条轴线取一条灰度剖面，
// 找过渡最陡（|梯度|最大）的位置作为亚像素端点。二值化端点只用来圈定
// 搜索邻域，灰度梯度用来精确定位，从而去掉「阈值切在斜坡上」带来的抖动。
// 不引入可调参数：搜索半窗 5px、垂直平均 3px 均为固定常量。
cv::Point2f refineEndpoint(const cv::Mat& gray, cv::Point2f endpoint, cv::Point2f dir_out) {
    constexpr int kHalfWin = 5;                  // 沿轴向搜索半窗（±5px）
    const float len = cv::norm(dir_out);
    if (len < 1e-3f) return endpoint;            // 退化保护（端点与中心重合）
    dir_out /= len;                              // 轴向单位方向（指向灯条外侧）
    const cv::Point2f n(-dir_out.y, dir_out.x);  // 垂直方向

    const int N = 2 * kHalfWin + 1;              // 采样点数 = 11
    std::vector<float> g(N, 0.f);
    for (int i = 0; i < N; ++i) {
        const cv::Point2f p = endpoint + dir_out * static_cast<float>(i - kHalfWin);
        float sum = 0.f; int cnt = 0;
        for (int k = -1; k <= 1; ++k) {          // 垂直 3px 平均，压横向噪声
            const cv::Point2f q = p + n * static_cast<float>(k);
            const int x = cvRound(q.x), y = cvRound(q.y);
            if (x >= 0 && x < gray.cols && y >= 0 && y < gray.rows) {
                sum += gray.at<uchar>(y, x); ++cnt;
            }
        }
        g[i] = cnt ? sum / cnt : 0.f;
    }

    // 中心差分梯度，找 |梯度| 最大处（背景↔灯条过渡最陡处）
    std::vector<float> grad(N, 0.f);
    int best = 0; float best_mag = -1.f;
    for (int i = 1; i < N - 1; ++i) {
        grad[i] = g[i + 1] - g[i - 1];
        const float mag = std::fabs(grad[i]);
        if (mag > best_mag) { best_mag = mag; best = i; }
    }

    // 亚像素：对梯度峰值做抛物线拟合，偏移量限制在 ±1px 内
    float sub = 0.f;
    if (best > 0 && best < N - 1) {
        const float denom = grad[best - 1] - 2.f * grad[best] + grad[best + 1];
        if (std::fabs(denom) > 1e-4f) {
            sub = 0.5f * (grad[best - 1] - grad[best + 1]) / denom;
            sub = std::max(-1.f, std::min(1.f, sub));
        }
    }
    return endpoint + dir_out * (static_cast<float>(best - kHalfWin) + sub);
}

// 找灯条:灰度二值化 + 形态学 + 形状筛选 + 竖向检查 + 边界颜色分类
std::vector<Light> findLights(const cv::Mat& roi, int detect_color) {
    const float thres = (detect_color == 1) ? kGrayThresRed : kGrayThresBlue;  // 按颜色选阈值
    cv::Mat gray;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    cv::Mat mask;
    cv::threshold(gray, mask, thres, 255, cv::THRESH_BINARY);
    const cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<Light> lights;
    for (const auto& c : contours) {
        if (c.size() < 5) continue;
        const double area = cv::contourArea(c);
        if (area < kAreaThres) continue;

        const cv::RotatedRect rr = cv::minAreaRect(c);
        const float w = rr.size.width, h = rr.size.height;
        if (std::min(w, h) < 1.f) continue;
        const float ratio = std::max(w, h) / std::min(w, h);
        if (ratio < kRatioThres) continue;

        // 中轴线:fitLine(最小二乘)拟合灯条轴线,把轮廓点投影到轴线上取最大/最小投影
        // 作为真实端点(比 minAreaRect 短边更抗噪声、更贴真实端点)
        cv::Vec4f line;
        cv::fitLine(c, line, cv::DIST_L2, 0, 0.01, 0.01);
        const cv::Point2f dir(line[0], line[1]);     // 归一化方向向量
        const cv::Point2f origin(line[2], line[3]);  // 直线上一点
        float min_proj = std::numeric_limits<float>::max();
        float max_proj = -std::numeric_limits<float>::max();
        for (const auto& p : c) {
            const float proj = (p.x - origin.x) * dir.x + (p.y - origin.y) * dir.y;
            min_proj = std::min(min_proj, proj);
            max_proj = std::max(max_proj, proj);
        }
        const cv::Point2f a = origin + dir * min_proj;
        const cv::Point2f b = origin + dir * max_proj;
        cv::Point2f top, bottom;
        if (a.y <= b.y) { top = a; bottom = b; }
        else            { top = b; bottom = a; }

        // 灰度梯度端点精修：去掉二值化阈值在斜坡上的抖动
        const cv::Point2f mid = (top + bottom) * 0.5f;
        top    = refineEndpoint(gray, top,    top - mid);    // top-mid 指向灯条上外侧
        bottom = refineEndpoint(gray, bottom, bottom - mid); // bottom-mid 指向灯条下外侧

        Light light;
        light.top = top;
        light.bottom = bottom;
        light.center = (top + bottom) * 0.5f;
        light.length = cv::norm(bottom - top);
        light.width = static_cast<float>(area) / std::max(light.length, 1e-3f);
        // 倾斜角从竖直算:竖直=0°,水平=90°
        light.tilt_angle = std::abs(90.f - std::abs(std::atan2(bottom.y - top.y, bottom.x - top.x)) *
                                                    180.f / CV_PI);
        if (light.tilt_angle > kMaxTilt) continue;  // 竖向检查

        // 颜色分类:只用轮廓边界像素(灯条中心过曝成白,填充会稀释颜色)
        int sum_r = 0, sum_b = 0;
        for (const auto& p : c) {
            const cv::Vec3b px = roi.at<cv::Vec3b>(p.y, p.x);
            sum_r += px[2];
            sum_b += px[0];
        }
        light.color = (sum_r > sum_b * 1.1f) ? 1 : (sum_b > sum_r * 1.1f ? 2 : 0);

        lights.push_back(light);
    }
    return lights;
}

}  // namespace

ArmorResult extractArmor(const cv::Mat& bgr, const cv::Rect2f& box, int detect_color) {
    ArmorResult result;

    // 外扩 bbox(小外延,包含灯条边缘)
    const float pad_x = box.width * 0.1f;
    const float pad_y = box.height * 0.1f;
    const int x0 = std::max(0, static_cast<int>(box.x - pad_x));
    const int y0 = std::max(0, static_cast<int>(box.y - pad_y));
    const int x1 = std::min(bgr.cols, static_cast<int>(box.x + box.width + pad_x));
    const int y1 = std::min(bgr.rows, static_cast<int>(box.y + box.height + pad_y));
    if (x1 <= x0 || y1 <= y0) return result;
    const cv::Mat roi = bgr(cv::Rect(x0, y0, x1 - x0, y1 - y0));

    std::vector<Light> lights = findLights(roi, detect_color);
    std::vector<Light> target;
    for (const auto& l : lights) {
        if (l.color == detect_color) target.push_back(l);
    }
    if (target.size() < 2) return result;

    // 按面积降序(优先大灯条)
    std::sort(target.begin(), target.end(),
              [](const Light& a, const Light& b) { return a.width * a.length > b.width * b.length; });

    // 任意配对:遍历所有灯条对,存在一对"平行 + 长度比"满足即匹配成功
    for (size_t i = 0; i < target.size(); ++i) {
        for (size_t j = i + 1; j < target.size(); ++j) {
            Light left = target[i], right = target[j];
            if (left.center.x > right.center.x) std::swap(left, right);

            const float par = std::abs(left.tilt_angle - right.tilt_angle);
            const float lr = std::min(left.length, right.length) / std::max(left.length, right.length);
            if (par < kParallelThres && lr > kLenRatioThres) {
                const float avg = (left.length + right.length) * 0.5f;
                const float cd = cv::norm(left.center - right.center) / std::max(avg, 1e-3f);
                result.valid = true;
                const cv::Point2f offset(static_cast<float>(x0), static_cast<float>(y0));
                result.corners = {left.top + offset, right.top + offset,
                                  right.bottom + offset, left.bottom + offset};
                result.size = (cd >= 3.0f) ? 1 : 0;  // 中心距离 >= 3 倍灯条长 → 大装甲板
                result.center = (left.center + right.center) * 0.5f + offset;
                left.top += offset; left.bottom += offset; left.center += offset;
                right.top += offset; right.bottom += offset; right.center += offset;
                result.left_light = left;
                result.right_light = right;
                return result;
            }
        }
    }
    return result;
}

}  // namespace rmv::detect
