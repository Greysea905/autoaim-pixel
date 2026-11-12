#pragma once

#include "AutoAimTypes.hpp"
#include "TrackState.hpp"

#include <Eigen/Dense>

#include <array>
#include <cstdint>
#include <vector>

namespace rmv::tracker {

// 像素直滤 EKF（混合观测）可调参数。过程侧参数与 EKFConfig 完全一致；
// 观测侧 = 8 维角点像素 + 1 维板法向 yaw（来自 optimizeArmorYaw）。
struct PixelEKFConfig {
    // ---- 过程噪声 Q（同 EKFConfig）----
    double q_pos{3000.0};
    double q_yaw{2400.0};
    double q_z{0.03};
    double q_r{1e-8};   // 半径=物理常数：q_r 极小，靠初始 P0 收敛后自然锁定
    double q_mag{1e-6}; // 倾斜角 mag 过程噪声（准静态，极小）
    double q_azi{1e-6}; // 倾斜方向 azi 过程噪声（准静态，极小）
    double q_aw{0.0};   // 角加速度 aw 过程噪声（0=稳态 w 恒定，>0=机动 w 可 ramp）

    // ---- 观测噪声 R ----
    double sigma_pixel_u{3.0};   // u 方向像素噪声 (px)
    double sigma_pixel_v{3.0};   // v 方向像素噪声 (px)
    double sigma_yaw_rad{0.4};   // 板法向 yaw 观测噪声 (rad)

    // ---- 深度保护（透视除法分母下限，防 NaN/Inf）----
    double min_depth{0.1};       // 相机系深度 Z 的 clamp 下限 (m)

    // ---- 板尺寸（已知物理常数，单位 m）----
    // 本任务只打底盘：底盘为小板 135×57mm。大板 230×57mm 待后续支持。
    double plate_half_width{0.0675};
    double plate_half_height{0.0285};
    double plate_tilt_deg{15.0};  // 板固定安装角（后仰，法向上仰），单位度

    // ---- 关联 / 状态机 ----
    double mahalanobis_gate2{16.92};   // χ²(9) 95% 上分位（9 维观测）
    int converge_frames{3};
    double reset_lost_s{0.20};

    // ---- 初始化 ----
    double r0_init{0.30};
    double r1_init{0.30};
    double mag_init{0.0};   // 倾斜角初值（0 = 无倾斜起步）
    double azi_init{0.0};   // 倾斜方向初值（mag=0 时无意义，任意）

    // ---- 发散检测物理界 ----
    double radius_min{0.10};
    double radius_max{0.50};
    double w_max{200.0};
    double mag_max{0.6};    // 倾斜角物理界 (rad, ~34°)

    // ---- 一致性检验（NIS）----
    double nis_threshold{16.92};       // χ²(9) 95% 上分位
    int nis_fail_frames{3};
    double nis_stable_threshold{9.0};  // 最小 NIS 低于此值→Q 缩小（平滑）；高于→Q 全量
    double q_stable_scale{0.10};
};

// 像素直滤 EKF（混合观测）：
//   观测 = 8 维角点像素 + 1 维板法向 yaw（optimizeArmorYaw）。
//   位置/距离由角点像素驱动（诚实的高斯像素噪声），朝向由 yaw 直通观测钉住（正对期间不再漂移）。
// 状态 12 维 [cx,vx,cy,vy,z0,z1,r0,r1,yaw,w,mag,azi]（世界系）。
//   mag=底盘倾斜角（与 +z 夹角）、azi=倾斜方向（旋转轴在 xoy 投影朝向），准静态。
class PixelEKF {
public:
    enum StateIndex : int {
        CX = 0, VX, CY, VY, Z0, Z1, R0, R1, YAW, W, AW, MAG, AZI, kDim
    };

    explicit PixelEKF(const PixelEKFConfig& cfg = PixelEKFConfig{});

    // 每帧调用一次。observations 是 armor_solve 输出（含 corners + armor_normal_yaw + position_world）。
    // 只用 corners + armor_normal_yaw 作观测，position 仅用于初始化。
    void update(const std::vector<ArmorPose>& observations, double timestamp,
                const GimbalState& gimbal, const CameraParameters& camera);

    TrackState state() const;
    TrackingState trackingState() const;
    int stableStage() const;
    std::uint8_t targetId() const;
    void reset();

    // 观测函数 h：状态 x 下板 slot 的预测观测（9 维：[u0,v0,u1,v1,u2,v2,u3,v3, yaw]）。
    Eigen::VectorXd h(const Eigen::VectorXd& x, int slot) const;

    int associatedObservationIndex(int slot) const { return slot_obs_idx_[slot]; }

    // 底盘倾斜状态（观测模型用，供可视化/调试）
    double mag() const { return x_(MAG); }
    double azi() const { return x_(AZI); }

    // IMM 用：读写状态/协方差、读上一帧似然
    const Eigen::VectorXd& x() const { return x_; }
    void setX(const Eigen::VectorXd& x) { x_ = x; }
    const Eigen::MatrixXd& P() const { return P_; }
    void setP(const Eigen::MatrixXd& P) { P_ = P; }
    double lastLikelihood() const { return last_likelihood_; }

private:
    PixelEKFConfig cfg_;
    bool initialized_{false};
    Eigen::VectorXd x_;   // 后验状态(12)
    Eigen::MatrixXd P_;   // 后验协方差(12x12)
    TrackingState state_{TrackingState::NoTarget};
    double last_timestamp_{0.0};
    double last_obs_timestamp_{0.0};
    int converged_frames_{0};
    int stable_stage_{0};
    std::uint8_t last_target_id_{0};

    int slot_obs_idx_[4]{-1, -1, -1, -1};
    int nis_fail_frames_{0};
    double last_nis_{1e18};
    double last_likelihood_{1.0};  // 上一帧模型似然（IMM 概率更新用）

    // 跨 reset 保留的物理常数（同一辆车不变）：半径 r0/r1、高度差 dz=z0-z1。
    double saved_r0_{0.28}, saved_r1_{0.28}, saved_dz_{0.0};
    bool has_saved_{false};

    // 当前帧世界→相机变换（h/数值雅可比用，每帧 update 开头缓存）。
    Eigen::Matrix3d R_w2c_{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d t_w2c_{Eigen::Vector3d::Zero()};
    double fx_{0.0}, fy_{0.0}, cx_{0.0}, cy_{0.0};

    void predict(double dt);
    double mahalanobis2(const Eigen::VectorXd& z, int slot) const;
    void correct(const Eigen::VectorXd& obs, int slot);
    void initialize(const std::vector<ArmorPose>& observations, double ts);
    bool diverged() const;
    Eigen::Vector2d worldCornerToPixel(const Eigen::Vector3d& pw) const;
    std::array<Eigen::Vector3d, 4> plateCornersWorld(const Eigen::VectorXd& x, int slot) const;
    Eigen::MatrixXd pixelR() const;
    Eigen::MatrixXd numericalJacobian(const Eigen::VectorXd& x, int slot) const;
    static double wrapPi(double a);
};

// 从 YAML 加载 PixelEKF 参数（path 缺省 config/pixel_ekf.yaml）。
PixelEKFConfig loadPixelEKFConfigOrDefault(const std::string& path = "config/pixel_ekf.yaml");

}  // namespace rmv::tracker
