#pragma once

#include <openvino/openvino.hpp>

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace rmv::detect {

// 单个检测结果
struct Detection {
    int class_id = -1;    // 0=dead, 1=red, 2=blue
    float confidence = 0.f;
    cv::Rect2f box;       // 原图坐标 (x, y, w, h)
};

// YOLO 检测器(OpenVINO 后端)
class Detector {
public:
    Detector(const std::string& model_path, int imgsz = 416,
             float conf_thres = 0.25f, float iou_thres = 0.45f);

    bool ok() const { return ok_; }

    std::vector<Detection> detect(const cv::Mat& bgr);

private:
    ov::Core core_;
    ov::CompiledModel compiled_;
    ov::InferRequest infer_;
    bool ok_ = false;
    int imgsz_ = 416;
    float conf_thres_ = 0.25f;
    float iou_thres_ = 0.45f;
};

}  // namespace rmv::detect
