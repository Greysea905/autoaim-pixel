#!/usr/bin/env python3
"""虚拟云台：向串口持续发送 64 字节云台帧，模拟电控/云台给视觉发的数据。

用途：没有真电控时，让 rmv_training 的读线程能收到合法帧，从而跑通整条主循环
（取流 -> 识别 -> PnP -> UI）。它只替代「电控发帧」，不替代相机。

用法（推荐：用 socat 建一对虚拟串口，无需任何硬件）：
  1) 起一对虚拟串口：
       socat -d -d pty,raw,echo=0 pty,raw,echo=0
     会打印两个设备名，例如 /dev/pts/3 和 /dev/pts/4
  2) 本脚本写其中一个，main 读另一个：
       python3 virtual_gimbal.py /dev/pts/3          # 发帧方
       ./build/rmv_training /dev/pts/4 115200 config/camera.yaml 8080   # 收帧方

也可以直接接一个 USB 转串口设备，把它的设备名（如 /dev/ttyUSB0）传给本脚本。

依赖：pyserial（pip install pyserial）
"""

import math
import struct
import sys
import time

FRAME_HEAD = 0x71
FRAME_TAIL = 0x4C

# 字段顺序严格对齐 Communication.hpp 里的 MessData_AutoAim（#pragma pack(1)，无填充）。
# 总长 64 字节：
#   B(1) + fff(3) + BBB(3) + fffffff(7) + B(1) + ffff(4) + H(1) + B(1) = 64
FRAME_FMT = "<BfffBBBfffffffBffffHB"

# 电控 -> 视觉 的字段：status 表示己方颜色（0=红方自瞄打蓝，5=蓝方自瞄打红）。
DEFAULT_STATUS = 0  # 默认红方（视觉打蓝方装甲板）


def build_frame(yaw=0.0, pitch=0.0, roll=0.0, status=DEFAULT_STATUS, bias=0.0):
    """打包一帧 64 字节。视觉关心的字段只有 head/tail/status/yaw/pitch/roll/bias。"""
    return struct.pack(
        FRAME_FMT,
        FRAME_HEAD,                       # head
        yaw, pitch, roll,                 # 云台当前 yaw/pitch/roll (rad)
        status, 0, 0,                     # status / is_far / armor_flag
        0.0, bias, 0.0, 0.0, 0.0, 0.0, 0.0,  # latency/bias/distance/pitch_offset/coo_x/coo_y/wheel_w
        0,                                # fire_allowance
        0.0, 0.0, 0.0, 0.0,               # target_yaw/target_pitch/yaw_vel/pitch_vel
        0,                                # crc（EKF 状态，这里视觉只看电控发来的，填 0 即可）
        FRAME_TAIL,                       # tail
    )


def main():
    if len(sys.argv) < 2:
        print("用法: python3 virtual_gimbal.py <串口设备> [波特率] [status]")
        print("  status: 0=红方打蓝(默认), 5=蓝方打红")
        print("示例: python3 virtual_gimbal.py /dev/pts/3 115200 5   # 打红方")
        return 2

    port = sys.argv[1]
    baud = int(sys.argv[2]) if len(sys.argv) > 2 else 115200
    status = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_STATUS

    try:
        import serial
    except ImportError:
        print("缺少 pyserial，请先安装：pip install pyserial")
        return 1

    ser = serial.Serial(port, baud, timeout=0)  # timeout=0：读非阻塞，用于清空回包

    rate = 165.0          # 发送频率 Hz
    period = 1.0 / rate
    t = 0.0
    print(f"[virtual_gimbal] 向 {port} 以 {rate:.0f}Hz 发送 64 字节帧 (status={status})，Ctrl+C 停止")

    try:
        while True:
            # 让云台角度缓慢摆动，方便在 UI 上看到 yaw/pitch 随帧变化
            yaw = 0.0 # 0.3 * math.sin(2 * math.pi * 0.5 * t)
            pitch = 0.0 # -0.1 * math.cos(2 * math.pi * 0.3 * t)
            frame = build_frame(yaw=yaw, pitch=pitch, roll=0.0, status=status, bias=0.0)
            ser.write(frame)
            ser.read(4096)  # 读并丢弃 main 写回的数据，防止缓冲塞满导致 main 写失败
            t += period
            time.sleep(period)
    except KeyboardInterrupt:
        print("\n[virtual_gimbal] 已停止")
    finally:
        ser.close()


if __name__ == "__main__":
    sys.exit(main())
