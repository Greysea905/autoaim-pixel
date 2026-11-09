#pragma once

#include "AutoAimTypes.hpp"
#include "TrackState.hpp"

#include <Eigen/Dense>

#include <vector>

namespace rmv::tracker {

// EKF 可调参数（供仿真调参/优化用）。默认值 = 上机手调出的当前值。
struct EKFConfig {
    // 过程噪声
    double q_pos{3000.0};
    double q_yaw{2400.0};
    double q_z{0.03};
    double q_r{1e-8};   // 半径=物理常数：q_r 极小，靠初始 P0 收敛后自然锁定
    // 观测噪声：前 3 维是相机系球坐标（随相机光轴锚定，天然对角），第 4 维是世界系法向 yaw。
    double sigma_cam_yaw{0.10};         // 板中心相机方位角 θ_cam (rad)：较准，不随距离
    double sigma_cam_pitch{0.10};       // 板中心相机俯仰角 φ_cam (rad)，不随距离
    // 距离线性缩放：d≤d_ref 用 base，d>d_ref 用 base+slope·(d-d_ref)
    double sigma_cam_distance{0.50};      // 距离噪声 base (m)，1m 基准
    double sigma_dist_slope{1.0};         // 距离噪声斜率（每米）
    double sigma_world_normal_yaw{0.4};    // 法向噪声 base (rad)，1m 基准
    double sigma_norm_slope{1.5};          // 法向噪声斜率（每米 rad）
    double sigma_scale_dref{1.0};          // 缩放起始距离 (m)
    // 关联 / 状态机
    double mahalanobis_gate2{9.49};
    int converge_frames{3};
    double reset_lost_s{0.20};
    // 初始化
    double r0_init{0.30};
    double r1_init{0.30};
    // 发散检测（物理合理范围，越界 → reset；不再是每帧 clamp）
    double radius_min{0.10};
    double radius_max{0.50};
    double w_max{200.0};
    // 一致性检验（NIS：新息平方和检验，连续超阈值则判定异常并重置）
    double nis_threshold{9.49};  // χ²(4) 95% 上分位
    int nis_fail_frames{3};
    double nis_stable_threshold{4.0};  // 上一帧最小 NIS 低于此值→模型拟合好→Q 缩小（平滑压波动）；高于→Q 全量（加速收敛）
    double q_stable_scale{0.10};
};

// EKF 估计器：观测 → 旋转车体状态（世界系）。
// 只负责状态估计（预测/更新/关联/初始化/状态机），不负责弹道与瞄准（Planner 的活）。
//
// 状态 10 维 [cx,vx,cy,vy,z0,z1,r0,r1,yaw,w]（世界系）：旋转中心 + 平移速度 + 绕中心匀速转动。
// z0=板0/2高度、z1=板1/3高度（对侧板等高）。
// 观测 4 维（混合系）：[θ_cam, φ_cam, d, yaw_world]
//   - 前 3 维 = 板中心在「相机系」的球坐标（方位角/俯仰角/距离）；噪声 R 随相机光轴锚定，无需每帧旋转。
//   - 第 4 维 = 板法向在「世界系」的方位角（= yaw + off，直接观测车体朝向）。
// 过程线性（匀速平移 + 匀速旋转），观测非线性（h：世界→相机→球坐标）。
// +r 约定见 TrackState::armorPosition。
class EKFEstimator {
public:
    // 状态向量索引（公开，供测试/调试构造状态用）。
    enum StateIndex : int {
        CX = 0, VX, CY, VY, Z0, Z1, R0, R1, YAW, W, kDim
    };

    explicit EKFEstimator(const EKFConfig& cfg = EKFConfig{});

    // 每帧调用一次。observations 是 PnP 输出（position_camera_m=相机系, armor_normal_yaw=世界系）。
    // gimbal/camera 用于把世界系状态投影回相机系，与观测对齐。
    void update(const std::vector<ArmorPose>& observations, double timestamp,
                const GimbalState& gimbal, const CameraParameters& camera);

    // 当前后验状态（世界系）。
    TrackState state() const;

    // 当前跟踪状态：NoTarget / Stable / Unstable。
    TrackingState trackingState() const;
    // 稳定期计数：每次从非稳定重新收敛到稳定就 +1（0 = 还没稳定过）。
    int stableStage() const;

    // 目标的板编号（来自分类结果，直接取最新非零值；0 = 未知）。
    std::uint8_t targetId() const;

    // 显式重置（丢目标超时内部也会调用）。
    void reset();

    // 观测函数 h：给定状态 x，板 slot 的预测观测（4 维：[θ_cam, φ_cam, d, yaw_world]）。
    Eigen::Vector4d h(const Eigen::VectorXd& x, int slot) const;

    // 上一帧关联结果：slot(0-3) 对应观测在 observations 里的下标（-1=无）。
    // 供可视化把 Planner 选中的「目标板槽位」映射回「当前帧的哪个观测」。
    int associatedObservationIndex(int slot) const { return slot_obs_idx_[slot]; }

private:
    EKFConfig cfg_;
    bool initialized_{false};
    Eigen::VectorXd x_;   // 后验状态(10)
    Eigen::MatrixXd P_;   // 后验协方差(10x10)
    TrackingState state_{TrackingState::NoTarget};
    double last_timestamp_{0.0};
    double last_obs_timestamp_{0.0};
    int converged_frames_{0};
    int stable_stage_{0};

    // 目标车编号（来自分类结果，0 = 未知）。
    std::uint8_t last_target_id_{0};

    // 上一帧关联结果：slot → observations 下标（-1=无观测）。
    int slot_obs_idx_[4]{-1, -1, -1, -1};

    // 连续新息超阈值帧数（NIS 发散检测）。
    int nis_fail_frames_{0};

    // 上一帧最小 NIS（驱动 predict 的自适应 Q：低→平滑，高→加速收敛）。
    double last_nis_{1e18};

    // 跨 reset 保留的物理常数（同一辆车不变）：半径 r0/r1、高度差 dz=z0-z1。
    double saved_r0_{0.28}, saved_r1_{0.28}, saved_dz_{0.0};
    bool has_saved_{false};

    // 当前帧世界→相机变换（h/数值雅可比用，每帧在 update 开头算一次缓存）。
    Eigen::Matrix3d R_w2c_{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d t_w2c_{Eigen::Vector3d::Zero()};

    void predict(double dt);
    double mahalanobis2(const Eigen::Vector4d& z, int slot) const;
    void correct(const Eigen::Vector4d& obs, int slot);
    void initialize(const std::vector<ArmorPose>& obs, double ts);
    // 发散检测：状态（半径/角速度）是否落在物理合理范围。
    bool diverged() const;

    // 世界系板位置 + 法向 yaw → 观测向量（走同一套 世界→相机→球坐标）。
    Eigen::Vector4d worldToObs(const Eigen::Vector3d& pw, double world_yaw) const;
    // 观测雅可比：中心差分（角度维用 wrapPi 差分，避免 ±π 处伪导数）。
    Eigen::MatrixXd numericalJacobian(const Eigen::VectorXd& x, int slot) const;

    static double wrapPi(double a);
};

// 从 YAML 加载 EKF 参数（path 缺省 config/ekf.yaml）。
// 文件缺失/解析失败时退回默认参数并打印警告，保证离线工具/实机都能跑。
EKFConfig loadEKFConfigOrDefault(const std::string& path = "config/ekf.yaml");

}  // namespace rmv::tracker
