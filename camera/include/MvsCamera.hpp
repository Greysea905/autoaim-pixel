#ifndef RMV_CAMERA_MVS_CAMERA_HPP_
#define RMV_CAMERA_MVS_CAMERA_HPP_

#include <string>
#include <cstdint>
#include <vector>
#include <atomic>
#include <mutex>

#include "MvCameraControl.h"

#include <opencv2/core.hpp>

namespace rmv::camera
{

// 一帧图像
struct Frame
{
  uint32_t width = 0;        // 图像宽（像素）
  uint32_t height = 0;       // 图像高（像素）
  uint64_t pixel_type = 0;   // SDK 像素格式枚举值（MvGvspPixelType）
  uint32_t frame_num = 0;    // 帧号
  cv::Mat data;  // 图像数据（cv::Mat 64 字节对齐，供上层零拷贝 + AVX2 安全）
};


// 封装海康 MVS SDK 的一台相机
// 负责设备枚举、选择、打开/关闭，以及采集（取流）；参数设置后续扩展
// 内部持有 SDK 句柄，采用 RAII：析构时自动释放资源，避免句柄泄漏
class MvsCamera
{
public:
  MvsCamera();
  ~MvsCamera();

  // 禁止拷贝/赋值：句柄代表与相机的独占连接
  MvsCamera(const MvsCamera &) = delete;
  MvsCamera & operator=(const MvsCamera &) = delete;

  // 枚举设备，把结果缓存到内部，返回找到的设备数量
  int enumerate();

  // 按序列号查找并打开相机（网口、USB 相机）
  bool openBySerial(const std::string & serial);

  // 按 IP 查找并打开相机（仅网口相机）
  bool openByIp(const std::string & ip);

  // 打开第一台在线设备（未固定序列号/IP 时的兜底）
  bool openFirst();

  // 当前是否处于打开状态
  bool isOpen() const { return is_open_.load(); }

  // 关闭并销毁句柄（析构时也会自动调用）
  void close();

  // 最近一次失败的原因描述，含 SDK 错误码，供上层打印日志
  std::string lastError() const { return last_error_; }

  // 开始取流
  bool startGrabbing();

  // 取一帧图，成功返回 true 并把数据写入 frame
  bool grab(Frame & frame);

  // 停止取流
  bool stopGrabbing();

  // 参数读写
  // 曝光时间，单位微秒(us)；设置前会自动关闭自动曝光
  bool setExposureTime(float time_us);
  bool getExposureTime(float & time_us);
  // 增益，单位 dB；设置前会自动关闭自动增益
  bool setGain(float gain_db);
  bool getGain(float & gain_db);
  // 帧率，单位 fps；会自动使能帧率控制
  bool setFrameRate(float fps);
  bool getFrameRate(float & fps);
  // 关闭帧率控制，让相机按「曝光+读出」自动跑满帧率（清除之前被 setFrameRate 锁住的低帧率）
  bool disableFrameRate();
  // 像素格式（MvGvspPixelType 枚举值）
  bool setPixelFormat(uint64_t pixel_type);
  bool getPixelFormat(uint64_t & pixel_type);

  // 查询参数支持的范围（连接后调用）
  bool getExposureRange(float & min, float & max);
  bool getGainRange(float & min, float & max);
  bool getFrameRateRange(float & min, float & max);



private:
  /// 从枚举结果中挑出序列号匹配的设备，找不到返回 nullptr
  MV_CC_DEVICE_INFO * findDeviceBySerial(const std::string & serial);
  /// 从枚举结果中挑出 IP 匹配的设备（网口），找不到返回 nullptr
  MV_CC_DEVICE_INFO * findDeviceByIp(const std::string & ip);

  /// 列出当前在线设备的"序列号(型号)"，用于报错提示
  std::string listOnlineDevices();

  /// 用指定设备信息创建 handle 并打开设备，成功返回 true
  bool openDevice(const MV_CC_DEVICE_INFO * dev);

private:
  void * handle_;                              // SDK handle：与相机连接的唯一标识
  std::atomic<bool> is_open_{false};           // 是否已打开（原子，多线程安全读）
  MV_CC_DEVICE_INFO_LIST st_dev_list_;         // 枚举结果缓存
  std::string last_error_;                     // 最近一次错误描述
  std::mutex handle_mutex_;                    // 保护 handle 的并发访问
};

}  // namespace rmv::camera

#endif  // RMV_CAMERA_MVS_CAMERA_HPP_
