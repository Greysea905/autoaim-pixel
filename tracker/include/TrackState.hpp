#pragma once

#include <Eigen/Dense>

#include <array>
#include <cmath>

namespace rmv::tracker {

constexpr double kPi = 3.14159265358979323846;

// 旋转车体的 10 维状态，在「世界坐标系(前/左/上，IMU 建立)」中表示。
//
// 字段：
//   cx, vx, cy, vy   车体旋转中心在世界系的位置 / 速度（m, m/s）
//   z0, z1           两对装甲板高度（m）：z0=板0/板2（对侧），z1=板1/板3（对侧）
//   r0, r1           两对半径（m）：r0=板0/板2（对侧），r1=板1/板3（对侧）
//   yaw, w           车体朝向角（rad）与角速度（rad/s）
//
// yaw 约定（采用「+r」）：
//   yaw = 板 0 相对中心的方向角 = atan2(T0.y - C.y, T0.x - C.x)
//   相对世界 +x 轴，逆时针为正，范围 (-π, π]。
//   因此 板 i 位置 = 中心 + r·(cos(yaw+off_i), sin(yaw+off_i))，off_i = i·π/2。
struct TrackState {
    double cx{0.0}, vx{0.0}, cy{0.0}, vy{0.0};
    double z0{0.0}, z1{0.0};   // z0=板0/2高度、z1=板1/3高度
    double r0{0.0}, r1{0.0};
    double yaw{0.0}, w{0.0};

    // 第 i 块装甲板的高度 / 半径（i=0..3）：高度偶数槽(0,2)取 z0、奇数槽(1,3)取 z1，
    // 半径偶数槽(0,2)用 r0、奇数槽(1,3)用 r1。
    double height(int i) const { return (i % 2 == 0) ? z0 : z1; }
    double radius(int i) const { return (i % 2 == 0) ? r0 : r1; }

    // 第 i 块装甲板的位置（i=0..3，世界系前/左/上，单位 m）。
    Eigen::Vector3d armorPosition(int i) const {
        const double off = static_cast<double>(i) * kPi * 0.5;
        const double a = yaw + off;
        return Eigen::Vector3d(cx + radius(i) * std::cos(a),
                               cy + radius(i) * std::sin(a),
                               height(i));
    }
};

}  // namespace rmv::tracker
