#!/usr/bin/env python3
"""棋盘格内参标定:读图 -> findChessboardCorners -> calibrateCamera -> 输出内参/畸变/误差。

用法:
    python3 calib/calibrate.py                 # 读 calib_images/*.jpg
    python3 calib/calibrate.py --dir 图片目录   # 指定目录

依赖: opencv-python, numpy
"""
import argparse
import glob
import sys

import cv2
import numpy as np

# ===== 你的棋盘格参数 =====
PATTERN = (11, 8)     # 内角点(列,行):实测 11列 x 8行(板实际是 12×9 方块)
SQUARE_SIZE = 0.025   # 格子边长(m),25mm。内参结果与这个无关,保留作记录
# ==========================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="calib_images", help="棋盘格图片目录")
    args = ap.parse_args()

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    files = sorted(f for f in glob.glob(f"{args.dir}/*") if f.lower().endswith(exts))
    print(f"找到 {len(files)} 张图")

    # 世界坐标点:棋盘格上每个内角点的 3D 坐标(棋盘平面 z=0)
    objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_SIZE

    objpoints, imgpoints = [], []
    img_shape = None

    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]  # (width, height)

        ok, corners = cv2.findChessboardCorners(gray, PATTERN, None)
        if not ok:
            print(f"  ✗ {f}: 未检测到棋盘格(检查角点是否清晰/完整入画)")
            continue

        # 亚像素精化:把角点定位到亚像素精度
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        objpoints.append(objp)
        imgpoints.append(corners2)
        print(f"  ✓ {f}")

    if len(objpoints) < 10:
        print(f"\n只有 {len(objpoints)} 张有效,至少 10 张(建议 15+)。多拍几张重来。")
        sys.exit(1)

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    print("\n========== 标定结果 ==========")
    print(f"有效图片数: {len(objpoints)} / {len(files)}")
    print(f"图像分辨率: {img_shape[0]} x {img_shape[1]}")
    print(f"重投影误差 RMS: {rms:.4f} px   (目标 < 0.5,可接受 < 1.0)")
    print("\n内参矩阵 K:")
    print(f"  fx = {K[0, 0]:.4f}")
    print(f"  fy = {K[1, 1]:.4f}")
    print(f"  cx = {K[0, 2]:.4f}")
    print(f"  cy = {K[1, 2]:.4f}")
    print("\n畸变系数 [k1, k2, p1, p2, k3]:")
    print("  " + np.array2string(dist.ravel(), precision=8))

    print("\n以上数值填进 config/camera.yaml 的 camera_matrix / distortion_coefficients。")


if __name__ == "__main__":
    main()
