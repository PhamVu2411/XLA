# -*- coding: utf-8 -*-
# =============================================================================
# FILE: run_panorama.py  -  CHẠY GHÉP ẢNH TỪ DÒNG LỆNH (không cần giao diện)
# -----------------------------------------------------------------------------
# Cách dùng (đứng ở thư mục gốc repo):
#   python python/run_panorama.py images/test1.png images/test2.png images/test3.png -o ketqua.png
# Ảnh phải được liệt kê theo thứ tự từ TRÁI sang PHẢI.
# OpenCV chỉ được dùng để ĐỌC / GHI file ảnh (được phép theo yêu cầu môn học).
# =============================================================================
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panorama_stitching import Panorama  # noqa: E402


def read_rgb(path):
    """Đọc ảnh (hỗ trợ đường dẫn có dấu tiếng Việt) và đổi BGR -> RGB."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    return img[..., ::-1]


def save_rgb(path, img_rgb):
    """Ghi ảnh RGB (float [0,1] hoặc uint8) ra file, hỗ trợ đường dẫn tiếng Việt."""
    if img_rgb.dtype != np.uint8:
        img_rgb = (np.clip(img_rgb, 0, 1) * 255).round().astype(np.uint8)
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, np.ascontiguousarray(img_rgb[..., ::-1]))
    if not ok:
        raise IOError(f"Không ghi được ảnh: {path}")
    buf.tofile(path)


def main():
    ap = argparse.ArgumentParser(description="Ghép ảnh panorama bằng SIFT + RANSAC (code from scratch)")
    ap.add_argument("images", nargs="+", help="các ảnh theo thứ tự trái -> phải")
    ap.add_argument("-o", "--output", default="panorama_result.png", help="file ảnh kết quả")
    ap.add_argument("--max-width", type=int, default=800, help="thu nhỏ ảnh rộng hơn giá trị này (0 = không thu nhỏ)")
    args = ap.parse_args()

    images = [read_rgb(p) for p in args.images]
    stitcher = Panorama(max_width=args.max_width or None, verbose=True)
    t = time.time()
    result = stitcher.get_panorama(images)
    print(f"Xong sau {time.time() - t:.1f} giây. Kích thước kết quả: {result.shape[1]}x{result.shape[0]}")
    save_rgb(args.output, result)
    print("Đã lưu:", args.output)


if __name__ == "__main__":
    main()
