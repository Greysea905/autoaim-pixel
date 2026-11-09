#include "Detector.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

namespace rmv::detect {

namespace {

// 简单 NMS:按 IoU 去重,返回保留的下标
std::vector<int> nms(const std::vector<cv::Rect2d>& boxes, const std::vector<float>& scores,
                     float iou_thres) {
    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return scores[a] > scores[b]; });

    std::vector<bool> suppressed(boxes.size(), false);
    std::vector<int> keep;
    for (const int i : order) {
        if (suppressed[i]) continue;
        keep.push_back(i);
        for (const int j : order) {
            if (suppressed[j]) continue;
            const auto& a = boxes[i];
            const auto& b = boxes[j];
            const double ix = std::max(0.0, std::min(a.x + a.width, b.x + b.width) - std::max(a.x, b.x));
            const double iy = std::max(0.0, std::min(a.y + a.height, b.y + b.height) - std::max(a.y, b.y));
            const double inter = ix * iy;
            if (inter <= 0) continue;
            const double uni = a.area() + b.area() - inter;
            if (inter / uni > iou_thres) suppressed[j] = true;
        }
    }
    return keep;
}

}  // namespace

Detector::Detector(const std::string& model_path, int imgsz, float conf_thres, float iou_thres)
    : imgsz_(imgsz), conf_thres_(conf_thres), iou_thres_(iou_thres) {
    try {
        auto model = core_.read_model(model_path);
        compiled_ = core_.compile_model(model, "CPU");
        infer_ = compiled_.create_infer_request();
        // 从模型输入张量自动读取实际输入尺寸(高=宽)，避免与硬编码 imgsz 不一致；
        // 以后换不同输入尺寸的权重无需再改代码。
        const ov::Shape shape = compiled_.input().get_shape();
        if (shape.size() == 4 && shape[2] > 0) {
            imgsz_ = static_cast<int>(shape[2]);
        }
        std::cout << "[detector] 模型输入尺寸 " << imgsz_ << "x" << imgsz_ << "\n";
        ok_ = true;
    } catch (const std::exception&) {
        ok_ = false;
    }
}

std::vector<Detection> Detector::detect(const cv::Mat& bgr) {
    std::vector<Detection> results;
    if (bgr.empty() || !ok_) return results;

    // 预处理:letterbox(等比缩放 + 灰边填充,与 ultralytics 训练/推理一致)
    const float scale = std::min(static_cast<float>(imgsz_) / bgr.cols,
                                 static_cast<float>(imgsz_) / bgr.rows);
    const int nw = static_cast<int>(std::round(bgr.cols * scale));
    const int nh = static_cast<int>(std::round(bgr.rows * scale));
    cv::Mat rgb;
    cv::resize(bgr, rgb, cv::Size(nw, nh));
    const int dw = (imgsz_ - nw) / 2;
    const int dh = (imgsz_ - nh) / 2;
    cv::copyMakeBorder(rgb, rgb, dh, imgsz_ - nh - dh, dw, imgsz_ - nw - dw,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    // HWC -> CHW,直接填输入张量
    auto input_tensor = infer_.get_input_tensor();
    float* input = input_tensor.data<float>();
    for (int y = 0; y < imgsz_; ++y) {
        const float* row = rgb.ptr<float>(y);
        for (int x = 0; x < imgsz_; ++x) {
            for (int c = 0; c < 3; ++c) {
                input[c * imgsz_ * imgsz_ + y * imgsz_ + x] = row[x * 3 + c];
            }
        }
    }

    // 推理
    infer_.infer();

    // 解码:输出 (1, 4+nc, num_preds)
    auto output = infer_.get_output_tensor();
    const float* out = output.data<float>();
    const ov::Shape& oshape = output.get_shape();
    const int num_preds = static_cast<int>(oshape[2]);
    const int nc = static_cast<int>(oshape[1]) - 4;

    std::vector<cv::Rect2d> boxes;
    std::vector<float> confs;
    std::vector<int> class_ids;

    for (int i = 0; i < num_preds; ++i) {
        int best_cls = -1;
        float best_score = -1.f;
        for (int c = 0; c < nc; ++c) {
            const float s = out[(4 + c) * num_preds + i];
            if (s > best_score) { best_score = s; best_cls = c; }
        }
        if (best_score < conf_thres_) continue;

        const float cx = out[0 * num_preds + i];
        const float cy = out[1 * num_preds + i];
        const float w = out[2 * num_preds + i];
        const float h = out[3 * num_preds + i];
        // 映射回原图:先减 padding,再除以 scale
        const float x0 = (cx - w * 0.5f - dw) / scale;
        const float y0 = (cy - h * 0.5f - dh) / scale;
        const float bw = w / scale, bh = h / scale;
        boxes.emplace_back(x0, y0, bw, bh);
        confs.push_back(best_score);
        class_ids.push_back(best_cls);
    }

    for (const int idx : nms(boxes, confs, iou_thres_)) {
        Detection d;
        d.class_id = class_ids[idx];
        d.confidence = confs[idx];
        d.box = boxes[idx];
        results.push_back(d);
    }
    return results;
}

}  // namespace rmv::detect
