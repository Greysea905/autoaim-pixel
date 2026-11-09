#pragma once

#include <openvino/openvino.hpp>

#include <opencv2/core.hpp>

#include <array>
#include <string>

namespace rmv::detect {

// 数字分类器(模型B):透视变换裁装甲板中心数字区 -> 64x64 推理 -> 类别。
// 类别共 7 类:0-3 = 数字 1-4,4-6 = outpost/guard/base。
class ArmorClassifier {
public:
    explicit ArmorClassifier(const std::string& model_path);

    bool ok() const { return ok_; }

    // 返回类别索引 0..6,失败返回 -1。
    // corners 为装甲板 4 角点(左上、右上、右下、左下);size:0=小,1=大。
    int classify(const cv::Mat& bgr, const std::array<cv::Point2f, 4>& corners, int size);

    // 类别索引 -> 标签文本
    static const char* label(int cls);

    // 类别索引 -> target_id(1,2,3,4,6,7,8;越界/未知 -> 0)
    static int toTargetId(int cls);

private:
    ov::Core core_;
    ov::CompiledModel compiled_;
    ov::InferRequest infer_;
    bool ok_ = false;
};

}  // namespace rmv::detect
