# -*- coding: utf-8 -*-
# =============================================================================
# FILE: sift_extractor.py  -  CÀI ĐẶT THUẬT TOÁN SIFT TỪ ĐẦU (CODE FROM SCRATCH)
# -----------------------------------------------------------------------------
# Nhiệm vụ: từ 1 ảnh xám, tìm các ĐIỂM ĐẶC TRƯNG (keypoint) và tạo cho mỗi điểm một
# DESCRIPTOR (vector 128 số, giống "dấu vân tay") để so khớp với ảnh khác.
#
# Quy trình (hàm extract_sift_data gọi lần lượt):
#   B1. downsample_img()     : tạo nhiều kích thước ảnh (các octave)
#   B2. create_octave()      : làm mờ Gaussian với nhiều sigma -> không gian tỉ lệ
#   B3. get_doG()            : hiệu 2 ảnh mờ liên tiếp (Difference of Gaussians)
#   B4. get_keypoints()      : tìm điểm cực trị trong lân cận 3x3x3 của DoG
#   B5. filter_keypoints()   : loại điểm nằm trên cạnh (dùng ma trận Hessian)
#   B6. get_dominant_angles(): gán hướng chính cho điểm -> bất biến với phép xoay
#   B7. get_descriptor()     : tạo vector 128 chiều quanh điểm
#
# Mọi phép xử lý ảnh (resize, Gaussian, Sobel, xoay) đều dùng hàm TỰ VIẾT trong
# image_ops.py - KHÔNG gọi OpenCV / scikit-image.
# =============================================================================
import numpy as np
import pandas as pd

try:
    from general_functions import get_hist_fast
    import image_ops as ops
except ImportError:
    from python.general_functions import get_hist_fast
    from python import image_ops as ops


class SIFT:
    def __init__(self, sigma=1.2, k=np.sqrt(1.8), num_downsamples=3, doG_thresh=0.003, r=10):
        """
        Tham số:
          sigma           : độ mờ Gaussian ban đầu
          k               : hệ số nhân giữa 2 mức mờ liên tiếp (sigma, k*sigma, k^2*sigma, ...)
          num_downsamples : số lần thu nhỏ ảnh (số octave thêm vào). Panorama dùng 0 cho nhanh
          doG_thresh      : ngưỡng loại điểm có độ tương phản thấp (|DoG| quá nhỏ = nhiễu)
          r               : ngưỡng tỉ lệ Hessian để loại điểm nằm trên cạnh (Lowe đề xuất r=10)
        """
        self.doG_thresh = doG_thresh
        self.r = r
        # 6 mức sigma -> 6 ảnh mờ -> 5 ảnh DoG trong mỗi octave
        self.sigmas = [sigma * k ** i for i in range(6)]
        # Hệ số thu nhỏ của các octave: 1, √2, 2, 2√2, ...
        self.downsampling_lvls = [1.0]
        for _ in range(num_downsamples):
            self.downsampling_lvls.append(np.sqrt(2) * self.downsampling_lvls[-1])

    # ---------------------------------------------------------------- B1
    def downsample_img(self, img):
        """
        [B1] Tạo các phiên bản ảnh với kích thước khác nhau (mỗi phiên bản = 1 octave),
        giúp SIFT tìm được điểm đặc trưng dù vật to hay nhỏ (bất biến tỉ lệ).
        Thu nhỏ bằng nội suy song tuyến tự viết (ops.resize_bilinear).
        """
        return [img if lvl == 1.0 else ops.resize_bilinear(img, 1.0 / lvl)
                for lvl in self.downsampling_lvls]

    # ---------------------------------------------------------------- B2
    def create_octave(self, img):
        """
        [B2] Làm mờ ảnh bằng Gaussian với 6 sigma khác nhau (ops.gaussian_blur tự viết).
        Sigma càng lớn ảnh càng mờ -> chỉ còn cấu trúc lớn. Tập ảnh này là "không gian tỉ lệ".
        """
        return [ops.gaussian_blur(img, s) for s in self.sigmas]

    # ---------------------------------------------------------------- B3
    def get_doG(self, imgs_list):
        """
        [B3] Difference of Gaussians: DoG_i = ảnh_mờ_(i) - ảnh_mờ_(i-1).
        DoG xấp xỉ toán tử Laplacian of Gaussian, làm nổi bật các "đốm" và góc.
        6 ảnh mờ -> 5 ảnh DoG.
        """
        return [imgs_list[i] - imgs_list[i - 1] for i in range(1, len(imgs_list))]

    # ---------------------------------------------------------------- B4
    @staticmethod
    def _max3(a):
        """Giá trị lớn nhất trong cửa sổ 3x3 quanh mỗi pixel (vector hóa)."""
        p = np.pad(a, 1, mode='edge')
        h, w = a.shape
        return np.max([p[i:i + h, j:j + w] for i in range(3) for j in range(3)], axis=0)

    @staticmethod
    def _min3(a):
        """Giá trị nhỏ nhất trong cửa sổ 3x3 quanh mỗi pixel (vector hóa)."""
        p = np.pad(a, 1, mode='edge')
        h, w = a.shape
        return np.min([p[i:i + h, j:j + w] for i in range(3) for j in range(3)], axis=0)

    @staticmethod
    def get_keypoints(doG_list, thresh=0.003):
        """
        [B4] Tìm keypoint ứng viên trên ảnh DoG ở giữa của bộ 3 (doG_list[0], [1], [2]).
        Pixel (x, y) của doG_list[1] là keypoint nếu:
          - |giá trị| > thresh (đủ tương phản, không phải nhiễu), và
          - nó là CỰC ĐẠI (hoặc CỰC TIỂU) so với 26 điểm lân cận:
              8 điểm quanh nó trong cùng ảnh + 9 điểm ở DoG trên + 9 điểm ở DoG dưới.
        Thay vì 2 vòng for qua từng pixel, ta tính max/min 3x3 cho cả ảnh cùng lúc.
        Trả về danh sách tọa độ (x, y).
        """
        d0, d1, d2 = doG_list[0], doG_list[1], doG_list[2]
        max_nb = np.maximum(SIFT._max3(d0), SIFT._max3(d2))   # max của DoG dưới và trên
        min_nb = np.minimum(SIFT._min3(d0), SIFT._min3(d2))   # min của DoG dưới và trên
        is_max = (d1 > max_nb) & (d1 == SIFT._max3(d1))       # lớn nhất trong 27 điểm
        is_min = (d1 < min_nb) & (d1 == SIFT._min3(d1))       # nhỏ nhất trong 27 điểm
        cond = (np.abs(d1) > thresh) & (is_max | is_min)
        cond[0, :] = cond[-1, :] = cond[:, 0] = cond[:, -1] = False   # bỏ viền ảnh
        ys, xs = np.nonzero(cond)
        return list(zip(xs.tolist(), ys.tolist()))

    # ---------------------------------------------------------------- B5
    def filter_keypoints(self, img, keypoints, dxx, dyy, dxy, r=10, patch_size=5):
        """
        [B5] Lọc bỏ keypoint nằm trên CẠNH, chỉ giữ điểm giống GÓC.
        Dựng ma trận 2x2 từ đạo hàm bậc 2 (dxx, dyy, dxy) trong vùng 5x5 quanh điểm
        (cách làm của repo gốc: tổng bình phương trên vùng lân cận), rồi tính
            ratio = (Tr(H))^2 / Det(H)
        Điểm trên cạnh: cong mạnh theo 1 hướng, yếu theo hướng kia -> ratio lớn -> loại.
        Giữ điểm nếu ratio < (r+1)^2 / r  (r = 10 -> ngưỡng 12.1).
        """
        filtered_keypoints = []
        thr = ((r + 1) ** 2) / r
        for x, y in keypoints:
            y0, y1 = int(y - patch_size / 2), int(y + patch_size / 2)
            x0, x1 = int(x - patch_size / 2), int(x + patch_size / 2)
            if y0 < 0 or x0 < 0:
                continue
            dxx_ = dxx[y0:y1, x0:x1]
            dyy_ = dyy[y0:y1, x0:x1]
            dxy_ = dxy[y0:y1, x0:x1]
            if dxx_.any():
                H = np.array([[np.sum(dxx_ ** 2), np.sum(dxy_ ** 2)],
                              [np.sum(dxy_ ** 2), np.sum(dyy_ ** 2)]])
                ratio = (H[0, 0] + H[1, 1]) ** 2 / (np.linalg.det(H) + 1e-8)
                if x < img.shape[1] - 1 and y < img.shape[0] - 1 and ratio < thr:
                    filtered_keypoints.append((x, y))
        return filtered_keypoints

    # ---------------------------------------------------------------- B6
    def get_dominant_angles(self, filtered_keypoints, gradient_magnitude, gradient_orientation, patch_size=8):
        """
        [B6] Gán HƯỚNG CHÍNH cho mỗi keypoint -> SIFT bất biến khi ảnh bị xoay.
        Lấy vùng 8x8 quanh điểm, lập histogram 36 bin (mỗi bin 10°) của hướng gradient,
        trọng số = độ lớn gradient. Bin cao nhất = hướng chính.
        Các bin khác cao >= 80% bin lớn nhất được thêm làm HƯỚNG PHỤ (như bài báo của Lowe).
        """
        orientation_data = {}
        new_filtered_keypoints = []
        half = patch_size // 2
        for kp in filtered_keypoints:
            x, y = kp
            if y - half < 0 or x - half < 0:
                continue
            mag = gradient_magnitude[y - half:y + half, x - half:x + half]
            ori = gradient_orientation[y - half:y + half, x - half:x + half]
            if mag.shape != (patch_size, patch_size):
                continue
            hist, bins = np.histogram(ori.ravel(), bins=36, range=(0, 360), weights=mag.ravel())
            centers = (bins[:-1] + bins[1:]) / 2            # tâm của mỗi bin
            max_bin = int(np.argmax(hist))
            angles = [centers[max_bin]]                     # hướng chính
            peaks = np.where(hist >= 0.8 * hist[max_bin])[0]
            angles += [centers[b] for b in peaks if b != max_bin]   # hướng phụ
            orientation_data[kp] = angles
            new_filtered_keypoints.append(kp)
        return orientation_data, new_filtered_keypoints

    # ---------------------------------------------------------------- B7
    def get_descriptor(self, doG_i, kp, rot_angle, patch_size=16, subarray_size=4):
        """
        [B7] Tạo DESCRIPTOR 128 chiều cho một keypoint:
          1. Cắt vùng 16x16 quanh keypoint.
          2. Xoay vùng theo hướng chính (ops.rotate tự viết) -> mô tả theo hướng chuẩn.
          3. Chia vùng thành lưới 4x4 = 16 ô, mỗi ô 4x4 pixel.
          4. Mỗi ô: histogram 8 hướng gradient -> 16 x 8 = 128 số (get_hist_fast).
          5. Chuẩn hóa vector về độ dài 1 -> ít bị ảnh hưởng khi độ sáng thay đổi.
        Trả về [] nếu keypoint quá sát biên.
        """
        half = patch_size // 2
        x, y = kp
        if x - half < 0 or y - half < 0:
            return []
        patch = doG_i[y - half:y + half, x - half:x + half]
        if patch.shape != (patch_size, patch_size):
            return []
        patch = ops.rotate(patch, rot_angle)
        feature_vector = get_hist_fast(patch, cell=subarray_size, nbins=8)
        norm = np.linalg.norm(feature_vector)
        return feature_vector / norm if norm else feature_vector

    # ---------------------------------------------------------------- HÀM CHÍNH
    def extract_sift_data(self, img):
        """
        Chạy toàn bộ SIFT trên 1 ảnh xám (giá trị trong [0, 1]).
        Trả về bảng pandas DataFrame, mỗi dòng là 1 keypoint:
          Position    : tọa độ (x, y) trong ảnh của octave tương ứng
          Size        : tỉ lệ của octave chứa điểm (1 = ảnh gốc)
          Orientation : hướng chính (độ)
          Descriptor  : vector 128 chiều
        """
        img = np.asarray(img, dtype=np.float64)
        rows = []
        downsampled_imgs = self.downsample_img(img)                       # B1
        for s, im in enumerate(downsampled_imgs):
            octave = self.create_octave(im)                               # B2
            doG_list = self.get_doG(octave)                               # B3
            # Duyệt các DoG ở giữa (có DoG dưới i-1 và DoG trên i+1 để so sánh)
            for i in range(1, len(doG_list) - 1):
                doG_i = doG_list[i]
                keypoints = self.get_keypoints(doG_list[i - 1:i + 2], self.doG_thresh)   # B4
                # Đạo hàm bậc 1 và bậc 2 bằng Sobel tự viết
                dx, dy = ops.sobel(doG_i)
                dxx, _ = ops.sobel(dx)
                dxy, dyy = ops.sobel(dy)
                gradient_magnitude, gradient_orientation = ops.cart_to_polar(dx, dy)
                filtered = self.filter_keypoints(doG_i, keypoints, dxx, dyy, dxy, r=self.r)   # B5
                orientation_data, filtered = self.get_dominant_angles(
                    filtered, gradient_magnitude, gradient_orientation)                       # B6
                for kp in filtered:                                                          # B7
                    for angle in orientation_data[kp]:
                        desc = self.get_descriptor(doG_i, kp, angle)
                        if len(desc):
                            rows.append([kp, 1 / self.downsampling_lvls[s], angle, desc])
        return pd.DataFrame(rows, columns=["Position", "Size", "Orientation", "Descriptor"])
