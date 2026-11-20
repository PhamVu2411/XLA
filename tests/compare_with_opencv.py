# -*- coding: utf-8 -*-
# =============================================================================
# FILE: tests/compare_with_opencv.py  -  SO SÁNH CODE TỰ VIẾT VỚI OPENCV + TEST CASES
# -----------------------------------------------------------------------------
# OpenCV CHỈ được dùng trong file này để làm "đáp án" đối chiếu, không dùng trong thuật toán.
# Chạy (đứng ở thư mục gốc repo):
#     python tests/compare_with_opencv.py
# Kết quả: bảng so sánh in ra màn hình + ảnh trong thư mục results/
# =============================================================================
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import image_ops as ops                          # noqa: E402
from panorama_stitching import Panorama          # noqa: E402
from run_panorama import read_rgb, save_rgb      # noqa: E402

IMG = lambda name: os.path.join(ROOT, "images", name)
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)


def timed(f, *a, **k):
    t = time.perf_counter()
    r = f(*a, **k)
    return r, (time.perf_counter() - t) * 1000


def row(name, ours, ref, t1, t2, valid=None):
    diff = np.abs(np.asarray(ours, float) - np.asarray(ref, float))
    if valid is not None:
        diff = diff[valid]
    print(f"{name:<22}{diff.max():>14.2e}{diff.mean():>14.2e}{t1:>12.1f}{t2:>12.1f}")


# -----------------------------------------------------------------------------
# PHẦN 1: So sánh từng phép xử lý ảnh với hàm tương ứng của OpenCV
# -----------------------------------------------------------------------------
def test_basic_ops():
    print("\n=== PHẦN 1: SO SÁNH TỪNG HÀM VỚI OPENCV (ảnh img3.jpg, thang [0,1]) ===")
    print(f"{'Phép toán':<22}{'Sai số max':>14}{'Sai số TB':>14}{'Tự viết ms':>12}{'OpenCV ms':>12}")
    rgb = read_rgb(IMG("img3.jpg"))
    g = ops.rgb2gray(rgb)

    s = 1.6
    a, t1 = timed(ops.gaussian_blur, g, s)
    b, t2 = timed(cv2.GaussianBlur, g, (0, 0), s)
    row("Gaussian blur", a, b, t1, t2)

    f = 1 / np.sqrt(2)
    a, t1 = timed(ops.resize_bilinear, g, f)
    b, t2 = timed(cv2.resize, g, (0, 0), fx=f, fy=f, interpolation=cv2.INTER_LINEAR)
    row("Resize (song tuyến)", a, b, t1, t2)

    (gx, gy), t1 = timed(ops.sobel, g)
    bx, t2 = timed(cv2.Sobel, g, cv2.CV_64F, 1, 0, ksize=3)
    row("Sobel x", gx, bx, t1, t2)
    by = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    row("Sobel y", gy, by, t1, t2)

    H = np.array([[1.02, 0.05, 30], [-0.03, 0.98, 10], [1e-5, 2e-5, 1]])
    a, t1 = timed(ops.warp_perspective, g, H, (900, 700))
    b, t2 = timed(cv2.warpPerspective, g, H, (900, 700))
    inner = cv2.erode((b > 0).astype(np.uint8), np.ones((5, 5))) > 0   # bỏ mép vùng ảnh
    row("Warp perspective", a, b, t1, t2, inner)

    # Homography + RANSAC trên dữ liệu giả lập: 200 cặp điểm, 30% là cặp sai
    rng = np.random.default_rng(1)
    src = rng.uniform(0, 600, (200, 2))
    dst = ops.apply_homography(H, src) + rng.normal(0, 1, (200, 2))
    dst[:60] = rng.uniform(0, 600, (60, 2))
    (Ho, m1), t1 = timed(ops.ransac_homography, src, dst, 5.0)
    (Hc, m2), t2 = timed(cv2.findHomography, src, dst, cv2.RANSAC, 5.0)
    grid = np.array([[x, y] for x in range(0, 601, 100) for y in range(0, 601, 100)], float)
    err = np.linalg.norm(ops.apply_homography(Ho, grid) - ops.apply_homography(Hc, grid), axis=1)
    print(f"{'RANSAC homography':<22}{err.max():>14.2e}{err.mean():>14.2e}{t1:>12.1f}{t2:>12.1f}"
          f"   (sai số = độ lệch điểm chiếu, pixel; inlier: tự viết {m1.sum()}, OpenCV {int(m2.sum())})")


# -----------------------------------------------------------------------------
# PHẦN 2: Test cases ghép ảnh (so với cv2.Stitcher nếu có)
# -----------------------------------------------------------------------------
def make_split_case():
    """Test giả lập: cắt ảnh img.jpg thành 3 ảnh chồng lấn, ảnh giữa xoay 5 độ."""
    img = read_rgb(IMG("img.jpg"))
    a = img[:, :600]
    b = (ops.rotate(img[:, 380:980] / 255.0, 5) * 255).astype(np.uint8)
    c = img[:, 760:1366]
    return [a, b, c]


CASES = [
    ("8 ảnh test1-8", "case1_test1-8", lambda: [read_rgb(IMG(f"test{i}.png")) for i in range(1, 9)]),
    ("2 ảnh mosque", "case2_mosque", lambda: [read_rgb(IMG("mosque1.jpg")), read_rgb(IMG("mosque2.jpg"))]),
    ("3 ảnh cắt từ img.jpg", "case3_img_split", make_split_case),
]


def test_panorama():
    print("\n=== PHẦN 2: TEST CASES GHÉP PANORAMA ===")
    for name, fname, loader in CASES:
        images = loader()
        p = Panorama()
        try:
            res, t = timed(p.get_panorama, images)
            info = ", ".join(f"{m['inliers']}/{m['matches']}" for m in p.match_info)
            fn = os.path.join(OUT, fname + ".png")
            save_rgb(fn, res)
            print(f"[OK]  {name:<24} {t / 1000:6.1f}s  kích thước {res.shape[1]}x{res.shape[0]}"
                  f"  inlier/cặp khớp: {info}")
        except Exception as e:
            print(f"[LỖI] {name:<24} {e}")
            continue
        # Đối chiếu với bộ ghép có sẵn của OpenCV (chỉ để so sánh trực quan)
        try:
            st = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
            status, ref = st.stitch([np.ascontiguousarray(im[..., ::-1]) for im in images])
            if status == 0:
                save_rgb(fn.replace(".png", "_opencv.png"), ref[..., ::-1])
                print(f"      OpenCV Stitcher: {ref.shape[1]}x{ref.shape[0]} (đã lưu *_opencv.png)")
            else:
                print(f"      OpenCV Stitcher không ghép được (mã lỗi {status})")
        except Exception as e:
            print("      Không chạy được cv2.Stitcher:", e)


if __name__ == "__main__":
    test_basic_ops()
    test_panorama()
    print("\nẢnh kết quả nằm trong:", OUT)
