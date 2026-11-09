#include "ArmorClassifier.hpp"

#include <opencv2/imgproc.hpp>

#include <vector>

namespace rmv::detect {

namespace {

constexpr int kInputSize = 64;   // 模型输入
constexpr int kNumClasses = 7;   // 1,2,3,4,outpost,guard,base

const char* kLabels[kNumClasses] = {"1", "2", "3", "4",
                                    "outpost", "guard", "base"};

}  // namespace

ArmorClassifier::ArmorClassifier(const std::string& model_path) {
    try {
        auto model = core_.read_model(model_path);
        compiled_ = core_.compile_model(model, "CPU");
        infer_ = compiled_.create_infer_request();
        ok_ = true;
    } catch (const std::exception&) {
        ok_ = false;
    }
}

int ArmorClassifier::classify(const cv::Mat& bgr, const std::array<cv::Point2f, 4>& corners,
                              int size) {
    if (bgr.empty() || !ok_) return -1;

    // 透视变换(与 build_dataset.py extract_number 一致):
    // 把装甲板 4 角点 warp 成 w×28 的标准化矩形,数字在竖直中心带。
    const int w = (size == 1) ? 54 : 32;  // 大装甲板 54px,小 32px
    constexpr int H = 28, L = 12;
    const int ty = (H - L) / 2 - 1;  // 7
    const int by = ty + L;           // 19

    const std::vector<cv::Point2f> src{corners[3], corners[0], corners[1], corners[2]};
    const std::vector<cv::Point2f> dst{cv::Point2f(0, by), cv::Point2f(0, ty),
                                       cv::Point2f(w - 1, ty), cv::Point2f(w - 1, by)};
    const cv::Mat M = cv::getPerspectiveTransform(src, dst);
    cv::Mat warped;
    cv::warpPerspective(bgr, warped, M, cv::Size(w, H));
    const cv::Mat roi = warped(cv::Rect(3, 2, w - 6, H - 4));  // 裁掉左右灯条 + 上下各2行余量

    cv::Mat gray;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    // 灰度直接输入(重训用灰度,与训练数据同分布,不再二值化)
    cv::Mat rgb;
    cv::cvtColor(gray, rgb, cv::COLOR_GRAY2BGR);  // 灰度 -> 3 通道
    cv::resize(rgb, rgb, cv::Size(kInputSize, kInputSize));
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    // HWC -> CHW,直接填输入张量
    auto input_tensor = infer_.get_input_tensor();
    float* input = input_tensor.data<float>();
    for (int y = 0; y < kInputSize; ++y) {
        const float* row = rgb.ptr<float>(y);
        for (int x = 0; x < kInputSize; ++x) {
            for (int c = 0; c < 3; ++c) {
                input[c * kInputSize * kInputSize + y * kInputSize + x] = row[x * 3 + c];
            }
        }
    }

    infer_.infer();

    auto output = infer_.get_output_tensor();
    const float* out = output.data<float>();  // (1, 7)
    int best = 0;
    for (int c = 1; c < kNumClasses; ++c) {
        if (out[c] > out[best]) best = c;
    }
    return best;
}

const char* ArmorClassifier::label(int cls) {
    if (cls < 0 || cls >= kNumClasses) return "?";
    return kLabels[cls];
}

int ArmorClassifier::toTargetId(int cls) {
    if (cls < 0 || cls >= kNumClasses) return 0;  // 越界 -> 未知/无目标
    if (cls < 4) return cls + 1;  // 数字 1-4
    return cls + 2;               // outpost=6, guard=7, base=8
}

}  // namespace rmv::detect
