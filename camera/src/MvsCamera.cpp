// MvsCamera 类的实现：直接调用海康 MVS SDK
// 上层只通过本类的公开方法与相机交互，不接触任何 SDK 类型
#include "MvsCamera.hpp"

#include <cstdio>
#include <cstring>

namespace rmv::camera
{

namespace
{
// 把 SDK 错误码转成文字表述，未知码只显示十六进制
std::string errToStr(int code)
{
  char buf[16];
  std::snprintf(buf, sizeof(buf), "0x%x", code);
  std::string hex = buf;
  switch (code) {
#ifdef MV_E_ACCESS_DENIED
    case MV_E_ACCESS_DENIED: return "设备被占用或无访问权限(" + hex + ")";
#endif
#ifdef MV_E_TIMEOUT
    case MV_E_TIMEOUT:       return "超时(" + hex + ")";
#endif
#ifdef MV_E_GC_TIMEOUT
    case MV_E_GC_TIMEOUT:    return "超时(" + hex + ")";
#endif
#ifdef MV_E_NODATA
    case MV_E_NODATA:        return "超时未收到数据(" + hex + ")";
#endif
#ifdef MV_E_HANDLE
    case MV_E_HANDLE:        return "无效句柄(" + hex + ")";
#endif
    default:                 return hex;
  }
}
}  


// 构造 / 析构
MvsCamera::MvsCamera()
: handle_(nullptr),
  is_open_(false)
{
  std::memset(&st_dev_list_, 0, sizeof(st_dev_list_));
  MV_CC_Initialize();
}

MvsCamera::~MvsCamera()
{
  // 只关设备、销毁句柄。不调用 MV_CC_Finalize()：
  // 实测海康 USB3 传输层(libMvUsb3vTL.so)在 MV_CC_Finalize() 阶段会与内部后台
  // 线程竞态，退出时稳定段错误(在它前面 sleep 等待也无法规避)。进程退出时由
  // 操作系统统一回收 SDK 资源，跳过 Finalize 即可，不影响运行期行为。
  close();
}


/* public */
// 枚举设备
int MvsCamera::enumerate()
{
  // 枚举网口 + USB 相机，结果写入缓存 st_dev_list_
  std::memset(&st_dev_list_, 0, sizeof(st_dev_list_));
  int nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &st_dev_list_);
  if (nRet != MV_OK) {
    last_error_ = "MV_CC_EnumDevices 失败: " + errToStr(nRet);
    return -1;
  }
  last_error_.clear();
  return static_cast<int>(st_dev_list_.nDeviceNum);
}

// 按序列号打开相机
bool MvsCamera::openBySerial(const std::string & serial)
{
  // 若已打开先关掉，保证重复调用安全
  if (is_open_) {
    close();
  }

  int nDev = enumerate();
  if (nDev < 0) {
    return false;  
  }
  if (nDev == 0) {
    last_error_ = "未发现任何相机";
    return false;
  }

  MV_CC_DEVICE_INFO * dev = findDeviceBySerial(serial);
  if (dev == nullptr) {
    last_error_ = "未找到序列号为 '" + serial + "' 的相机。当前在线: " + listOnlineDevices();
    return false;
  }
  return openDevice(dev);
}

// 按 IP 打开相机
bool MvsCamera::openByIp(const std::string & ip)
{
  if (is_open_) {
    close();
  }

  // 先校验 IP 格式，并给出错误反馈
  unsigned int a = 0, b = 0, c = 0, d = 0;
  if (std::sscanf(ip.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) != 4) {
    last_error_ = "IP 格式错误: '" + ip + "'（应为 x.x.x.x）";
    return false;
  }

  int nDev = enumerate();
  if (nDev < 0) {
    return false;
  }
  if (nDev == 0) {
    last_error_ = "未发现任何相机";
    return false;
  }

  MV_CC_DEVICE_INFO * dev = findDeviceByIp(ip);
  if (dev == nullptr) {
    last_error_ = "未找到 IP 为 '" + ip + "' 的网口相机。当前在线: " + listOnlineDevices();
    return false;
  }
  return openDevice(dev);
}

// 打开第一台在线设备
bool MvsCamera::openFirst()
{
  if (is_open_) {
    close();
  }

  int nDev = enumerate();
  if (nDev < 0) {
    return false;
  }
  if (nDev == 0) {
    last_error_ = "未发现任何相机";
    return false;
  }
  return openDevice(st_dev_list_.pDeviceInfo[0]);
}

// 关闭相机
void MvsCamera::close()
{
  std::lock_guard<std::mutex> lock(handle_mutex_);
  if (handle_ != nullptr) {
    if (is_open_.load()) {
      // 关连接前必须先停取流，否则 SDK 在退出清理时会崩溃。
      // 若从未开始取流，StopGrabbing 会返回错误码，忽略即可。
      MV_CC_StopGrabbing(handle_);
      MV_CC_CloseDevice(handle_);  // 关连接
      is_open_.store(false);
    }
    MV_CC_DestroyHandle(handle_);  // 退 handle
    handle_ = nullptr;
  }
}


// 取流
// 开始取流
bool MvsCamera::startGrabbing()
{
  std::lock_guard<std::mutex> lock(handle_mutex_);
  int nRet = MV_CC_StartGrabbing(handle_);
  if (nRet != MV_OK) {
    last_error_ = "MV_CC_StartGrabbing 失败: " + errToStr(nRet);
    return false;
  }
  last_error_.clear();
  return true;
}

// 取一帧图像
bool MvsCamera::grab(Frame & frame)
{
  std::lock_guard<std::mutex> lock(handle_mutex_);
  MV_FRAME_OUT stFrame;
  std::memset(&stFrame, 0, sizeof(stFrame));

  // 阻塞取一帧，超时 1000ms
  int nRet = MV_CC_GetImageBuffer(handle_, &stFrame, 1000);
  if (nRet != MV_OK) {
    last_error_ = "MV_CC_GetImageBuffer 失败: " + errToStr(nRet);
    return false;
  }

  frame.width = stFrame.stFrameInfo.nExtendWidth;
  frame.height = stFrame.stFrameInfo.nExtendHeight;
  frame.frame_num = stFrame.stFrameInfo.nFrameNum;
  frame.pixel_type = static_cast<uint64_t>(stFrame.stFrameInfo.enPixelType);

  // 只要不是 BGR8（3 通道），就统一用 SDK 转成 BGR8。相机默认可能是 YUV422/Bayer 等，
  // 直接用原数据会跟上层 CV_8UC3 的 3 通道假设不匹配导致越界崩。
  if (frame.pixel_type != static_cast<uint64_t>(PixelType_Gvsp_BGR8_Packed)) {
    // 直接写进 frame.data（cv::Mat，64 字节对齐），省掉中间的 std::vector 再拷贝。
    frame.data = cv::Mat(static_cast<int>(frame.height), static_cast<int>(frame.width), CV_8UC3);
    MV_CC_PIXEL_CONVERT_PARAM_EX param;
    std::memset(&param, 0, sizeof(param));
    param.nWidth = frame.width;
    param.nHeight = frame.height;
    param.enSrcPixelType = stFrame.stFrameInfo.enPixelType;  // 真实源格式（YUV422/Bayer/Mono…）
    param.pSrcData = stFrame.pBufAddr;
    param.nSrcDataLen = stFrame.stFrameInfo.nFrameLen;
    param.enDstPixelType = PixelType_Gvsp_BGR8_Packed;
    param.pDstBuffer = frame.data.data;
    param.nDstBufferSize = static_cast<unsigned int>(frame.width * frame.height * 3);

    nRet = MV_CC_ConvertPixelTypeEx(handle_, &param);
    if (nRet != MV_OK) {
      MV_CC_FreeImageBuffer(handle_, &stFrame);
      last_error_ = "像素转 BGR 失败: " + errToStr(nRet);
      return false;
    }
    frame.pixel_type = static_cast<uint64_t>(PixelType_Gvsp_BGR8_Packed);
  } else {
    frame.data = cv::Mat(static_cast<int>(frame.height), static_cast<int>(frame.width), CV_8UC3);
    std::memcpy(frame.data.data, stFrame.pBufAddr, stFrame.stFrameInfo.nFrameLen);
  }

  MV_CC_FreeImageBuffer(handle_, &stFrame);
  last_error_.clear();
  return true;
}

// 停止取流
bool MvsCamera::stopGrabbing()
{
  std::lock_guard<std::mutex> lock(handle_mutex_);
  int nRet = MV_CC_StopGrabbing(handle_);
  if (nRet != MV_OK) {
    last_error_ = "MV_CC_StopGrabbing 失败: " + errToStr(nRet);
    return false;
  }
  last_error_.clear();
  return true;
}


// 参数读写
// 设置曝光时间
bool MvsCamera::setExposureTime(float time_us)
{
  std::lock_guard<std::mutex> lock(handle_mutex_);
  // 手动曝光前先关闭自动曝光，否则自动模式会覆盖手动值
  int nRet = MV_CC_SetEnumValue(handle_, "ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF);
  if (nRet != MV_OK) {
    last_error_ = "关闭自动曝光失败: " + errToStr(nRet);
    return false;
  }
  nRet = MV_CC_SetFloatValue(handle_, "ExposureTime", time_us);
  if (nRet != MV_OK) {
    last_error_ = "设置曝光失败: " + errToStr(nRet);
    return false;
  }
  last_error_.clear();
  return true;
}

// 获取曝光时间
bool MvsCamera::getExposureTime(float & time_us)
{
  std::lock_guard<std::mutex> lock(handle_mutex_);
  MVCC_FLOATVALUE stVal;
  std::memset(&stVal, 0, sizeof(stVal));
  int nRet = MV_CC_GetFloatValue(handle_, "ExposureTime", &stVal);
  if (nRet != MV_OK) {
    last_error_ = "读取曝光失败: " + errToStr(nRet);
    return false;
  }
  time_us = stVal.fCurValue;
  last_error_.clear();
  return true;
}

// 设置增益值
bool MvsCamera::setGain(float gain_db)
{
  std::lock_guard<std::mutex> lock(handle_mutex_);
  // 手动增益前先关闭自动增益
  int nRet = MV_CC_SetEnumValue(handle_, "GainAuto", MV_GAIN_MODE_OFF);
  if (nRet != MV_OK) {
    last_error_ = "关闭自动增益失败: " + errToStr(nRet);
    return false;
  }
  nRet = MV_CC_SetFloatValue(handle_, "Gain", gain_db);
  if (nRet != MV_OK) {
    last_error_ = "设置增益失败: " + errToStr(nRet);
    return false;
  }
  last_error_.clear();
  return true;
}

// 获取增益值
bool MvsCamera::getGain(float & gain_db)
{
  std::lock_guard<std::mutex> lock(handle_mutex_);
  MVCC_FLOATVALUE stVal;
  std::memset(&stVal, 0, sizeof(stVal));
  int nRet = MV_CC_GetFloatValue(handle_, "Gain", &stVal);
  if (nRet != MV_OK) {
    last_error_ = "读取增益失败: " + errToStr(nRet);
    return false;
  }
  gain_db = stVal.fCurValue;
  last_error_.clear();
  return true;
}

// 设置帧率
bool MvsCamera::setFrameRate(float fps)
{
  std::lock_guard<std::mutex> lock(handle_mutex_);
  // 先使能帧率控制，再设帧率
  int nRet = MV_CC_SetBoolValue(handle_, "AcquisitionFrameRateEnable", true);
  if (nRet != MV_OK) {
    last_error_ = "使能帧率控制失败: " + errToStr(nRet);
    return false;
  }
  nRet = MV_CC_SetFloatValue(handle_, "AcquisitionFrameRate", fps);
  if (nRet != MV_OK) {
    last_error_ = "设置帧率失败: " + errToStr(nRet);
    return false;
  }
  last_error_.clear();
  return true;
}

// 获取帧率
bool MvsCamera::getFrameRate(float & fps)
{
  std::lock_guard<std::mutex> lock(handle_mutex_);
  MVCC_FLOATVALUE stVal;
  std::memset(&stVal, 0, sizeof(stVal));
  int nRet = MV_CC_GetFloatValue(handle_, "AcquisitionFrameRate", &stVal);
  if (nRet != MV_OK) {
    last_error_ = "读取帧率失败: " + errToStr(nRet);
    return false;
  }
  fps = stVal.fCurValue;
  last_error_.clear();
  return true;
}

// 关闭帧率控制
bool MvsCamera::disableFrameRate()
{
  std::lock_guard<std::mutex> lock(handle_mutex_);
  int nRet = MV_CC_SetBoolValue(handle_, "AcquisitionFrameRateEnable", false);
  if (nRet != MV_OK) {
    last_error_ = "关闭帧率控制失败: " + errToStr(nRet);
    return false;
  }
  last_error_.clear();
  return true;
}

// 设置像素格式
bool MvsCamera::setPixelFormat(uint64_t pixel_type)
{
  std::lock_guard<std::mutex> lock(handle_mutex_);
  int nRet = MV_CC_SetEnumValue(handle_, "PixelFormat", static_cast<unsigned int>(pixel_type));
  if (nRet != MV_OK) {
    last_error_ = "设置像素格式失败: " + errToStr(nRet);
    return false;
  }
  last_error_.clear();
  return true;
}

// 获取像素格式
bool MvsCamera::getPixelFormat(uint64_t & pixel_type)
{
  std::lock_guard<std::mutex> lock(handle_mutex_);
  MVCC_ENUMVALUE stVal;
  std::memset(&stVal, 0, sizeof(stVal));
  int nRet = MV_CC_GetEnumValue(handle_, "PixelFormat", &stVal);
  if (nRet != MV_OK) {
    last_error_ = "读取像素格式失败: " + errToStr(nRet);
    return false;
  }
  pixel_type = static_cast<uint64_t>(stVal.nCurValue);
  last_error_.clear();
  return true;
}


// 查询曝光时间范围
bool MvsCamera::getExposureRange(float & min, float & max)
{
  std::lock_guard<std::mutex> lock(handle_mutex_);
  MVCC_FLOATVALUE stVal;
  std::memset(&stVal, 0, sizeof(stVal));
  int nRet = MV_CC_GetFloatValue(handle_, "ExposureTime", &stVal);
  if (nRet != MV_OK) {
    last_error_ = "查询曝光范围失败: " + errToStr(nRet);
    return false;
  }
  min = stVal.fMin;
  max = stVal.fMax;
  last_error_.clear();
  return true;
}

// 查询增益范围
bool MvsCamera::getGainRange(float & min, float & max)
{
  std::lock_guard<std::mutex> lock(handle_mutex_);
  MVCC_FLOATVALUE stVal;
  std::memset(&stVal, 0, sizeof(stVal));
  int nRet = MV_CC_GetFloatValue(handle_, "Gain", &stVal);
  if (nRet != MV_OK) {
    last_error_ = "查询增益范围失败: " + errToStr(nRet);
    return false;
  }
  min = stVal.fMin;
  max = stVal.fMax;
  last_error_.clear();
  return true;
}

// 查询帧率范围
bool MvsCamera::getFrameRateRange(float & min, float & max)
{
  std::lock_guard<std::mutex> lock(handle_mutex_);
  MVCC_FLOATVALUE stVal;
  std::memset(&stVal, 0, sizeof(stVal));
  int nRet = MV_CC_GetFloatValue(handle_, "AcquisitionFrameRate", &stVal);
  if (nRet != MV_OK) {
    last_error_ = "查询帧率范围失败: " + errToStr(nRet);
    return false;
  }
  min = stVal.fMin;
  max = stVal.fMax;
  last_error_.clear();
  return true;
}

/* private */
MV_CC_DEVICE_INFO * MvsCamera::findDeviceBySerial(const std::string & serial)
{
  // 遍历枚举缓存，按传输层类型读取对应的序列号字段
  for (unsigned int i = 0; i < st_dev_list_.nDeviceNum; ++i) {
    MV_CC_DEVICE_INFO * pDev = st_dev_list_.pDeviceInfo[i];
    const char * serial_str = nullptr;
    if (pDev->nTLayerType == MV_USB_DEVICE) {
      serial_str = reinterpret_cast<const char *>(pDev->SpecialInfo.stUsb3VInfo.chSerialNumber);
    } else if (pDev->nTLayerType == MV_GIGE_DEVICE) {
      serial_str = reinterpret_cast<const char *>(pDev->SpecialInfo.stGigEInfo.chSerialNumber);
    }
    if (serial_str != nullptr && serial == serial_str) {
      return pDev;
    }
  }
  return nullptr;
}

MV_CC_DEVICE_INFO * MvsCamera::findDeviceByIp(const std::string & ip)
{
  // 把 "x.x.x.x" 转成 SDK 使用的大端 uint32 表示
  unsigned int a = 0, b = 0, c = 0, d = 0;
  if (std::sscanf(ip.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) != 4) {
    return nullptr;
  }
  unsigned int target = (a << 24) | (b << 16) | (c << 8) | d;

  // 仅网口相机 IP 匹配
  for (unsigned int i = 0; i < st_dev_list_.nDeviceNum; ++i) {
    MV_CC_DEVICE_INFO * pDev = st_dev_list_.pDeviceInfo[i];
    if (pDev->nTLayerType == MV_GIGE_DEVICE &&
        pDev->SpecialInfo.stGigEInfo.nCurrentIp == target) {
      return pDev;
    }
  }
  return nullptr;
}

// 列出当前在线设备的"序列号(型号)"，用于报错提示
std::string MvsCamera::listOnlineDevices()
{
  std::string result;
  for (unsigned int i = 0; i < st_dev_list_.nDeviceNum; ++i) {
    MV_CC_DEVICE_INFO * pDev = st_dev_list_.pDeviceInfo[i];
    const char * model = nullptr;
    const char * serial_str = nullptr;
    if (pDev->nTLayerType == MV_USB_DEVICE) {
      model = reinterpret_cast<const char *>(pDev->SpecialInfo.stUsb3VInfo.chModelName);
      serial_str = reinterpret_cast<const char *>(pDev->SpecialInfo.stUsb3VInfo.chSerialNumber);
    } else if (pDev->nTLayerType == MV_GIGE_DEVICE) {
      model = reinterpret_cast<const char *>(pDev->SpecialInfo.stGigEInfo.chModelName);
      serial_str = reinterpret_cast<const char *>(pDev->SpecialInfo.stGigEInfo.chSerialNumber);
    }
    if (i > 0) result += ", ";
    if (serial_str != nullptr) result += serial_str;
    if (model != nullptr) result += std::string("(") + model + ")";
  }
  return result.empty() ? "无" : result;
}

bool MvsCamera::openDevice(const MV_CC_DEVICE_INFO * dev)
{
  std::lock_guard<std::mutex> lock(handle_mutex_);
  // 1) 把 handel 绑定到指定设备
  int nRet = MV_CC_CreateHandle(&handle_, dev);
  if (nRet != MV_OK) {
    last_error_ = "MV_CC_CreateHandle 失败: " + errToStr(nRet);
    return false;
  }

  // 2) 打开设备连接
  nRet = MV_CC_OpenDevice(handle_);
  if (nRet != MV_OK) {
    last_error_ = "MV_CC_OpenDevice 失败: " + errToStr(nRet);
    // 打开失败要退回 handle，避免 handle 泄漏
    MV_CC_DestroyHandle(handle_);
    handle_ = nullptr;
    return false;
  }

  is_open_.store(true);
  last_error_.clear();
  return true;
}

}  // namespace rmv::camera
