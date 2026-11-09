// 灯条提取调试工具：实时相机 + YOLO 检测 + 灯条提取中间过程可视化。
// 用途：手动调灯条提取参数（灰度阈值/面积/宽高比/倾斜角/平行差/长度比）。
// 显示（一个窗口左右拼接）：
//   左 = 原图 + 可视化（绿线=fitLine直线, 红点=上端点, 蓝点=下端点, 白框=匹配框, 黄框=YOLO检测框）
//   右 = 形态学闭运算后的二值化 mask
// 用法：./build/lightbar_debug [red|blue]   默认 red（打红目标）
// 参数用窗口 trackbar 实时调；q 或 Esc 退出。需在项目根目录运行（检测模型是相对路径）。

#include "Detector.hpp"
#include "LightBarExtractor.hpp"
#include "StudentTasks.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using rmv::detect::Light;

namespace {

// ---- 可调参数（trackbar 用 int，使用时转 float）----
int g_red_gray_thres = 40;    // 红色灯条灰度阈值
int g_blue_gray_thres = 26;   // 蓝色灯条灰度阈值
int g_area_thres = 50;        // 面积阈值
int g_ratio_thres10 = 18;     // 宽高比阈值 ×0.1（18 → 1.8）
int g_max_tilt = 65;          // 最大倾斜角（从竖直算，度）
int g_parallel_thres = 20;    // 两灯条平行差阈值（度）
int g_len_ratio100 = 75;      // 长度比下限 ×0.01（75 → 0.75）
int g_red_gain10 = 0;        // 红色 R 通道增益 ×0.1（10 → 1.0 倍，不提亮；调大提亮）
int g_blue_gain10 = 0;       // 蓝色 B 通道增益 ×0.1

// 灰度梯度端点精修（与 LightBarExtractor.cpp 保持一致）
cv::Point2f refineEndpoint(const cv::Mat& gray, cv::Point2f endpoint, cv::Point2f dir_out) {
    constexpr int kHalfWin = 5;
    const float len = cv::norm(dir_out);
    if (len < 1e-3f) return endpoint;
    dir_out /= len;
    const cv::Point2f n(-dir_out.y, dir_out.x);

    const int N = 2 * kHalfWin + 1;
    std::vector<float> g(N, 0.f);
    for (int i = 0; i < N; ++i) {
        const cv::Point2f p = endpoint + dir_out * static_cast<float>(i - kHalfWin);
        float sum = 0.f; int cnt = 0;
        for (int k = -1; k <= 1; ++k) {
            const cv::Point2f q = p + n * static_cast<float>(k);
            const int x = cvRound(q.x), y = cvRound(q.y);
            if (x >= 0 && x < gray.cols && y >= 0 && y < gray.rows) {
                sum += gray.at<uchar>(y, x); ++cnt;
            }
        }
        g[i] = cnt ? sum / cnt : 0.f;
    }

    std::vector<float> grad(N, 0.f);
    int best = 0; float best_mag = -1.f;
    for (int i = 1; i < N - 1; ++i) {
        grad[i] = g[i + 1] - g[i - 1];
        const float mag = std::fabs(grad[i]);
        if (mag > best_mag) { best_mag = mag; best = i; }
    }

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

// ---- 灯条提取（复制 LightBarExtractor::findLights，加可视化）----
std::vector<Light> findLightsDebug(const cv::Mat& roi, int detect_color, cv::Mat& viz, cv::Mat& mask_out, cv::Mat& gray_out) {
    const int thres = (detect_color == 1) ? g_red_gray_thres : g_blue_gray_thres;  // 按颜色选阈值

    // 分色软件增益：红→R 通道×系数，蓝→B 通道×系数（缓解远处暗）
    cv::Mat enhanced = roi.clone();
    const int ch = (detect_color == 1) ? 2 : 0;  // BGR 里 R=2, B=0
    const float gain = (detect_color == 1 ? g_red_gain10 : g_blue_gain10) / 10.0f;
    if (std::fabs(gain - 1.0f) > 1e-3f) {
        for (int y = 0; y < roi.rows; ++y) {
            for (int x = 0; x < roi.cols; ++x) {
                const int v = cvRound(roi.at<cv::Vec3b>(y, x)[ch] * gain);
                enhanced.at<cv::Vec3b>(y, x)[ch] = static_cast<uchar>(std::min(v, 255));
            }
        }
    }

    cv::Mat gray;
    cv::cvtColor(enhanced, gray, cv::COLOR_BGR2GRAY);
    gray_out = gray.clone();
    cv::Mat mask;
    cv::threshold(gray, mask, static_cast<double>(thres), 255, cv::THRESH_BINARY);
    const cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k);
    mask_out = mask.clone();

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<Light> lights;
    for (const auto& c : contours) {
        if (c.size() < 5) continue;
        const double area = cv::contourArea(c);
        if (area < g_area_thres) continue;

        const cv::RotatedRect rr = cv::minAreaRect(c);
        const float w = rr.size.width, h = rr.size.height;
        if (std::min(w, h) < 1.f) continue;
        const float ratio = std::max(w, h) / std::min(w, h);
        if (ratio < g_ratio_thres10 * 0.1f) continue;

        // 中轴线 fitLine（最小二乘）
        cv::Vec4f line;
        cv::fitLine(c, line, cv::DIST_L2, 0, 0.01, 0.01);
        const cv::Point2f dir(line[0], line[1]);
        const cv::Point2f origin(line[2], line[3]);
        float min_proj = FLT_MAX, max_proj = -FLT_MAX;
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

        // 灰度梯度端点精修：精修前（灰点）用二值化投影，精修后用灰度梯度亚像素端点
        const cv::Point2f old_top = top, old_bottom = bottom;
        const cv::Point2f mid = (top + bottom) * 0.5f;
        top    = refineEndpoint(gray, top,    top - mid);
        bottom = refineEndpoint(gray, bottom, bottom - mid);

        Light light;
        light.top = top;
        light.bottom = bottom;
        light.center = (top + bottom) * 0.5f;
        light.length = cv::norm(bottom - top);
        light.width = static_cast<float>(area) / std::max(light.length, 1e-3f);
        light.tilt_angle = std::abs(90.f - std::abs(std::atan2(bottom.y - top.y, bottom.x - top.x)) *
                                                    180.f / CV_PI);
        if (light.tilt_angle > g_max_tilt) continue;

        // 颜色分类（复制 findLights：轮廓边界像素，灯条中心过曝成白）
        int sum_r = 0, sum_b = 0;
        for (const auto& p : c) {
            const cv::Vec3b px = roi.at<cv::Vec3b>(p.y, p.x);
            sum_r += px[2];
            sum_b += px[0];
        }
        light.color = (sum_r > sum_b * 1.1f) ? 1 : (sum_b > sum_r * 1.1f ? 2 : 0);

        // 可视化：fitLine 直线（绿）+ 精修前端点（灰）+ 精修后上端点（红）/下端点（蓝）
        cv::line(viz, top, bottom, cv::Scalar(0, 255, 0), 1);
        cv::circle(viz, old_top, 2, cv::Scalar(128, 128, 128), -1);
        cv::circle(viz, old_bottom, 2, cv::Scalar(128, 128, 128), -1);
        cv::circle(viz, top, 3, cv::Scalar(0, 0, 255), -1);
        cv::circle(viz, bottom, 3, cv::Scalar(255, 0, 0), -1);

        lights.push_back(light);
    }
    return lights;
}

// 灰度剖面可视化：沿灯条轴线采样灰度，画「剖面 + 梯度」曲线。
// 用途：判断端点波动根因——剖面像「台阶」=二值化锯齿（灰度精修收益大）；
//       像「斜坡」=灰度过渡平缓+噪声（灰度精修收益有限，得调曝光/多帧平滑）。
cv::Mat drawProfile(const cv::Mat& gray_roi, const cv::Point2f& top, const cv::Point2f& bottom) {
    cv::Point2f d = bottom - top;
    const float len = cv::norm(d);
    if (len < 1.f) return cv::Mat();
    d /= len;                                  // 轴线单位方向
    const cv::Point2f n(-d.y, d.x);            // 垂直方向

    const int N = 64;
    std::vector<float> g(N), grad(N, 0.f);
    for (int i = 0; i < N; ++i) {
        const cv::Point2f p = top + d * (len * i / (N - 1.0f));
        float sum = 0.f;
        int cnt = 0;
        for (int k = -1; k <= 1; ++k) {        // 垂直窄条 3px，平均横向噪声
            const cv::Point2f q = p + n * static_cast<float>(k);
            const int x = static_cast<int>(std::round(q.x));
            const int y = static_cast<int>(std::round(q.y));
            if (x >= 0 && x < gray_roi.cols && y >= 0 && y < gray_roi.rows) {
                sum += gray_roi.at<uchar>(y, x);
                ++cnt;
            }
        }
        g[i] = cnt ? sum / cnt : 0.f;
        if (i > 0) grad[i] = g[i] - g[i - 1];
    }

    const int W = 340, H = 220;
    cv::Mat curve(H, W, CV_8UC3, cv::Scalar(30, 30, 30));
    // 灰度剖面（橙）
    for (int i = 1; i < N; ++i) {
        cv::line(curve,
                 cv::Point((i - 1) * W / N, H - static_cast<int>(g[i - 1] * H / 255)),
                 cv::Point(i * W / N,     H - static_cast<int>(g[i] * H / 255)),
                 cv::Scalar(0, 165, 255), 1);
    }
    // 梯度（绿，以中线为零）
    float gmax = 0.f;
    for (int i = 1; i < N; ++i) gmax = std::max(gmax, std::fabs(grad[i]));
    if (gmax < 1.f) gmax = 1.f;
    for (int i = 1; i < N; ++i) {
        cv::line(curve,
                 cv::Point((i - 1) * W / N, H / 2 - static_cast<int>(grad[i - 1] * (H / 2 - 5) / gmax)),
                 cv::Point(i * W / N,     H / 2 - static_cast<int>(grad[i] * (H / 2 - 5) / gmax)),
                 cv::Scalar(0, 255, 0), 1);
    }
    cv::line(curve, cv::Point(0, H / 2), cv::Point(W, H / 2), cv::Scalar(80, 80, 80), 1);  // 中线
    return curve;
}

}  // namespace

int main(int argc, char** argv) {
    const int target_class = (argc >= 2 && std::string(argv[1]) == "blue") ? 2 : 1;

    rmv::detect::Detector detector("detect/models/armor_detector.onnx");
    if (!detector.ok()) { std::fprintf(stderr, "检测器加载失败\n"); return 1; }

    const std::string win = "lightbar_debug (左=原图可视化, 右=二值化mask, q/Esc=退出)";
    cv::namedWindow(win, cv::WINDOW_NORMAL);
    cv::createTrackbar("red_gray",       win, &g_red_gray_thres,   255);
    cv::createTrackbar("blue_gray",      win, &g_blue_gray_thres,  255);
    cv::createTrackbar("red_gain*0.1",   win, &g_red_gain10,     50);
    cv::createTrackbar("blue_gain*0.1",  win, &g_blue_gain10,    50);

    cv::Mat pic;
    while (true) {
        pic.release();
        if (!get_pic(pic) || pic.empty()) { cv::waitKey(1); continue; }

        cv::Mat viz = pic.clone();
        cv::Mat mask_canvas = cv::Mat::zeros(pic.rows, pic.cols, CV_8UC1);

        const auto dets = detector.detect(pic);
        for (const auto& d : dets) {
            if (d.class_id != target_class) continue;

            // YOLO 检测框（黄）
            cv::rectangle(viz, d.box, cv::Scalar(0, 255, 255), 1);

            // 外扩 bbox（同 extractArmor）
            const float pad_x = d.box.width * 0.1f, pad_y = d.box.height * 0.1f;
            const int x0 = std::max(0, static_cast<int>(d.box.x - pad_x));
            const int y0 = std::max(0, static_cast<int>(d.box.y - pad_y));
            const int x1 = std::min(pic.cols, static_cast<int>(d.box.x + d.box.width + pad_x));
            const int y1 = std::min(pic.rows, static_cast<int>(d.box.y + d.box.height + pad_y));
            if (x1 <= x0 || y1 <= y0) continue;

            const cv::Rect roi_rect(x0, y0, x1 - x0, y1 - y0);
            const cv::Mat roi = pic(roi_rect);
            cv::Mat roi_viz = viz(roi_rect);
            cv::Mat roi_mask, roi_gray;
            std::vector<Light> lights = findLightsDebug(roi, target_class, roi_viz, roi_mask, roi_gray);
            roi_mask.copyTo(mask_canvas(roi_rect));

            // 配对 + 匹配框（复制 extractArmor 配对逻辑，不筛颜色，成功即 break）
            std::sort(lights.begin(), lights.end(),
                      [](const Light& a, const Light& b) { return a.width * a.length > b.width * b.length; });
            for (size_t i = 0; i < lights.size(); ++i) {
                for (size_t j = i + 1; j < lights.size(); ++j) {
                    Light left = lights[i], right = lights[j];
                    if (left.center.x > right.center.x) std::swap(left, right);
                    const float par = std::abs(left.tilt_angle - right.tilt_angle);
                    const float lr = std::min(left.length, right.length) / std::max(left.length, right.length);
                    if (par < g_parallel_thres && lr > g_len_ratio100 * 0.01f) {
                        const float ox = static_cast<float>(x0), oy = static_cast<float>(y0);
                        std::vector<cv::Point> corners{
                            cv::Point(static_cast<int>(left.top.x + ox), static_cast<int>(left.top.y + oy)),
                            cv::Point(static_cast<int>(right.top.x + ox), static_cast<int>(right.top.y + oy)),
                            cv::Point(static_cast<int>(right.bottom.x + ox), static_cast<int>(right.bottom.y + oy)),
                            cv::Point(static_cast<int>(left.bottom.x + ox), static_cast<int>(left.bottom.y + oy)),
                        };
                        cv::polylines(viz, corners, true, cv::Scalar(255, 255, 255), 2);  // 白=匹配框

                        // 灰度剖面可视化：左/右灯条各画一条剖面+梯度曲线（用增强后的 gray）
                        cv::Mat pL = drawProfile(roi_gray, left.top, left.bottom);
                        cv::Mat pR = drawProfile(roi_gray, right.top, right.bottom);
                        cv::Mat profiles;
                        cv::hconcat(pL, pR, profiles);
                        cv::imshow("gray profile (橙=剖面, 绿=梯度; 左灯条|右灯条)", profiles);

                        break;
                    }
                }
            }
        }

        // 拼接：左 viz + 右 mask（转 BGR），缩 0.5 显示
        cv::Mat mask_bgr;
        cv::cvtColor(mask_canvas, mask_bgr, cv::COLOR_GRAY2BGR);
        cv::Mat side;
        cv::hconcat(viz, mask_bgr, side);
        cv::resize(side, side, cv::Size(), 0.5, 0.5);
        cv::imshow(win, side);

        const int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) break;
    }
    return 0;
}
