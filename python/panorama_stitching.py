# -*- coding: utf-8 -*-
# =============================================================================
# FILE: panorama_stitching.py  -  GHÉP NHIỀU ẢNH THÀNH ẢNH TOÀN CẢNH (PANORAMA)
# -----------------------------------------------------------------------------
# Hàm chính: Panorama.get_panorama(danh_sách_ảnh_RGB_đã_sắp_thứ_tự_từ_trái_sang_phải)
#
# Quy trình:
#   0. (Tùy chọn) thu nhỏ ảnh quá lớn cho nhanh                   (resize_bilinear)
#   1. Chuyển các ảnh sang ảnh xám                                 (rgb2gray)
#   2. Trích xuất SIFT cho từng ảnh                                (get_sift_descriptors)
#   3. Với mỗi cặp ảnh liền kề (i-1, i):
#        - so khớp descriptor: láng giềng gần nhất + ratio test   (match_descriptors)
#        - ước lượng Homography 3x3 bằng RANSAC + DLT              (ransac_homography)
#   4. Chọn ẢNH GIỮA làm tham chiếu, ghép lan dần sang trái và phải, nhân dồn các
#      ma trận                                                     (stitch_multiple_images)
#      -> ý tưởng của bài báo "Improved Algorithm of Fast Panorama Stitching":
#         bắt đầu từ ảnh giữa để giảm méo hình tích lũy.
#   5. Cắt bỏ viền đen xung quanh                                  (find_bounding_box)
#
# Toàn bộ dùng hàm TỰ VIẾT trong image_ops.py (không dùng OpenCV/skimage/sklearn).
# =============================================================================
import numpy as np

try:
    from general_functions import normalize_img
    from sift_extractor import SIFT
    import image_ops as ops
except ImportError:
    from python.general_functions import normalize_img
    from python.sift_extractor import SIFT
    from python import image_ops as ops


class Panorama:
    def __init__(self, max_width=800, ratio=0.8, ransac_thresh=5.0, verbose=False):
        """
        max_width     : ảnh rộng hơn giá trị này sẽ được thu nhỏ trước khi ghép (None = giữ nguyên)
        ratio         : ngưỡng ratio test khi so khớp descriptor (None = tắt)
        ransac_thresh : sai số chiếu tối đa (pixel) để 1 cặp điểm được tính là inlier
        verbose       : in thông tin từng bước ra màn hình
        """
        self.max_width = max_width
        self.ratio = ratio
        self.ransac_thresh = ransac_thresh
        self.verbose = verbose
        self.match_info = []          # lưu số cặp khớp / inlier của từng cặp ảnh (để báo cáo)

    def _log(self, *args):
        if self.verbose:
            print(*args)

    # ---------------------------------------------------------------- BƯỚC 3
    def compute_homography_matrix(self, src_sift, dst_sift, threshold_distance=1.0):
        """
        Tính ma trận Homography H (3x3) biến đổi tọa độ ảnh nguồn (src) sang ảnh đích (dst):
            [x', y', w]^T = H · [x, y, 1]^T
          1. So khớp descriptor (ops.match_descriptors): với mỗi điểm của ảnh nguồn tìm
             điểm gần nhất ở ảnh đích; giữ cặp có khoảng cách < threshold_distance và
             qua được ratio test.
          2. RANSAC (ops.ransac_homography) chọn H tốt nhất và loại cặp khớp sai.
        Nếu ratio test để lại quá ít cặp (< 10) thì thử lại không dùng ratio test.
        """
        d_src = np.stack(src_sift["Descriptor"].to_list())
        d_dst = np.stack(dst_sift["Descriptor"].to_list())
        p_src = np.array(src_sift["Position"].to_list(), dtype=np.float64)
        p_dst = np.array(dst_sift["Position"].to_list(), dtype=np.float64)

        i1, i2, _ = ops.match_descriptors(d_src, d_dst, threshold_distance, self.ratio)
        if len(i1) < 10 and self.ratio is not None:
            i1, i2, _ = ops.match_descriptors(d_src, d_dst, threshold_distance, None)

        source_points, target_points = p_src[i1], p_dst[i2]
        H, inliers = ops.ransac_homography(source_points, target_points, thresh=self.ransac_thresh)
        self.match_info.append({"matches": int(len(i1)), "inliers": int(inliers.sum())})
        self._log(f"   so khớp: {len(i1)} cặp, RANSAC giữ {int(inliers.sum())} inlier")
        if H is None:
            raise RuntimeError("Không đủ điểm khớp giữa 2 ảnh liền kề - kiểm tra thứ tự ảnh "
                               "và phần chồng lấn.")
        return H

    # ---------------------------------------------------------------- GHÉP 2 ẢNH
    def stitch_two_images(self, img1, img2, x_offset=0, img1_left=False):
        """
        Chồng 2 ảnh (đã cùng hệ tọa độ) thành 1 ảnh:
          - img1 được dịch sang phải x_offset pixel (khi có ảnh mới nằm bên trái cần chừa chỗ).
          - Vùng chỉ có 1 ảnh: lấy pixel của ảnh đó.
          - Vùng chồng lấn (cả 2 ảnh có dữ liệu): lấy TRUNG BÌNH CỘNG (blending đơn giản),
            riêng dải mép 2 pixel của mỗi ảnh thì ưu tiên ảnh kia để không bị vệt tối.
        """
        img1 = normalize_img(img1)
        img2 = normalize_img(img2)
        h1, w1 = img1.shape[:2]

        # Dịch img1 sang phải x_offset pixel (phép tịnh tiến bằng cách chép mảng)
        img1_shifted = np.zeros((h1, w1 + x_offset) + img1.shape[2:])
        img1_shifted[:, x_offset:] = img1

        stitched_img = np.zeros(img2.shape)
        if img1_left:
            stitched_img[:, :w1 + x_offset] = img1_shifted
        else:
            stitched_img[:, :w1] = img1_shifted[:, :w1]

        # Mặt nạ "có dữ liệu" của từng ảnh (pixel khác đen ở ít nhất 1 kênh màu)
        v1 = stitched_img.reshape(stitched_img.shape[:2] + (-1,)).max(axis=2) > 0
        v2 = img2.reshape(img2.shape[:2] + (-1,)).max(axis=2) > 0
        # "Lõi" = vùng có dữ liệu đã co vào 2 pixel. Dải mép ngoài cùng của ảnh sau khi uốn
        # bị nội suy với nền đen nên tối hơn; nếu đem lấy trung bình sẽ tạo vệt kẻ tối.
        c1 = ops.binary_erode(v1, 5)
        c2 = ops.binary_erode(v2, 5)
        avg = (v1 & v2) & (c1 == c2)        # chồng lấn và 2 ảnh "ngang hàng" -> lấy trung bình
        take1 = v1 & (~v2 | (c1 & ~c2))     # chỉ ảnh 1 có, hoặc ảnh 2 chỉ có mép -> lấy ảnh 1
        take2 = v2 & (~v1 | (c2 & ~c1))     # chỉ ảnh 2 có, hoặc ảnh 1 chỉ có mép -> lấy ảnh 2
        if img2.ndim == 3:
            avg, take1, take2 = avg[..., None], take1[..., None], take2[..., None]
        result_image = np.where(take1, stitched_img, 0.0) + np.where(take2, img2, 0.0)
        result_image = np.where(avg, (stitched_img + img2) / 2, result_image)
        return result_image

    # ---------------------------------------------------------------- BƯỚC 5
    def find_bounding_box(self, img):
        """
        Tìm hình chữ nhật lớn nhất nằm trọn trong vùng có ảnh để cắt bỏ viền đen:
          1. Phân ngưỡng: pixel có độ sáng > 1/255 là "có ảnh".
          2. Phép đóng (closing) 15x15 để lấp lỗ đen nhỏ bên trong.
          3. Giữ vùng liên thông lớn nhất, lấy hình chữ nhật bao quanh nó.
          4. Thu nhỏ dần hình chữ nhật (mỗi phía 2 pixel) tới khi bên trong hết pixel đen.
        Trả về (x, y, w, h).
        """
        gray = ops.rgb2gray(img) if img.ndim > 2 else np.asarray(img, dtype=np.float64)
        binary_mask = gray * 255 > 1
        result_mask = ops.binary_closing(binary_mask, 15)
        result_mask = ops.largest_component(result_mask)
        ys, xs = np.nonzero(result_mask)
        if len(xs) == 0:
            return 0, 0, img.shape[1], img.shape[0]
        x, y = int(xs.min()), int(ys.min())
        w, h = int(xs.max()) - x + 1, int(ys.max()) - y + 1

        mask = result_mask[y:y + h, x:x + w]
        while np.any(~mask) and w > 4 and h > 4:
            prev_mask = mask.copy()
            x, y, w, h = x + 2, y + 2, w - 4, h - 4
            mask = result_mask[y:y + h, x:x + w]
            if np.array_equal(mask, prev_mask):
                break
        return x, y, w, h

    # ---------------------------------------------------------------- BƯỚC 4
    def stitch_backward(self, current_image, previous_image, accumulated_transform_matrix, offset=0, is_left=False):
        """
        Uốn current_image theo ma trận tích lũy (ops.warp_perspective tự viết) vào khung
        rộng = tổng chiều rộng 2 ảnh, rồi ghép với previous_image.
        """
        output_size = (previous_image.shape[1] + current_image.shape[1], previous_image.shape[0])
        transformed = ops.warp_perspective(current_image, accumulated_transform_matrix, output_size)
        return self.stitch_two_images(previous_image, transformed, offset, is_left)

    def stitch_multiple_images(self, img_splits, affine_matrices):
        """
        Ghép nhiều ảnh, BẮT ĐẦU TỪ ẢNH GIỮA (ảnh tham chiếu k).
          affine_matrices[i] = H biến tọa độ ảnh i sang ảnh i+1.
          - Ảnh bên trái (k-1, k-2, ...): nhân dồn H[k-1] · H[k-2] ... để đưa về ảnh k.
          - Ảnh bên phải (k+1, k+2, ...): nhân dồn inv(H[k]) · inv(H[k+1]) ...
          Mỗi vòng lặp ghép thêm 1 ảnh bên trái và 1 ảnh bên phải (lan dần ra 2 phía).
          Nếu ảnh bên trái bị đẩy sang x âm thì dịch toàn bộ sang phải offset_x pixel.
        """
        num_of_imgs = len(img_splits)
        k = max(num_of_imgs // 2 - 1, 0)                 # chỉ số ảnh giữa
        ref_img = ops.to_float(img_splits[k])
        a_m_prev = np.eye(3)
        a_m_next = np.eye(3)

        for i in range(num_of_imgs):
            if k - i - 1 < 0 and k + i + 1 >= num_of_imgs:
                break
            offset_x = 0
            if k - i - 1 >= 0:                           # ---- ghép ảnh bên TRÁI
                prev_img = ops.to_float(img_splits[k - i - 1])
                a_m_prev = a_m_prev @ affine_matrices[k - i - 1]
                if a_m_prev[0, 2] < 0:                   # tịnh tiến x âm -> dịch sang phải
                    offset_x = int(abs(a_m_prev[0, 2]))
                    a_m_prev[0, 2] += offset_x
                self._log(f"   ghép ảnh {k - i} (bên trái)")
                ref_img = self.stitch_backward(prev_img, ref_img, a_m_prev, offset_x, True)
            if k + i + 1 < num_of_imgs:                  # ---- ghép ảnh bên PHẢI
                next_img = ops.to_float(img_splits[k + i + 1])
                a_m_next = a_m_next @ np.linalg.inv(affine_matrices[k + i])
                a_m_next[0, 2] += offset_x
                self._log(f"   ghép ảnh {k + i + 2} (bên phải)")
                ref_img = self.stitch_backward(next_img, ref_img, a_m_next, 0)
        return ref_img

    # ---------------------------------------------------------------- BƯỚC 2
    def get_sift_descriptors(self, image_splits, num_downsamples=0):
        """Chạy SIFT cho từng ảnh xám (1 octave cho nhanh). Trả về list các DataFrame."""
        sift_extractor = SIFT(num_downsamples=num_downsamples)
        sift_data = []
        for n, image in enumerate(image_splits):
            sift_summary = sift_extractor.extract_sift_data(image)
            self._log(f"   SIFT ảnh {n + 1}: {len(sift_summary)} keypoint")
            sift_data.append(sift_summary)
        return sift_data

    # ---------------------------------------------------------------- BƯỚC 3
    def compute_affine_matrices(self, gray_images, sift_data):
        """Tính Homography giữa mọi cặp ảnh liền kề (1-2, 2-3, ...).
        (Tên "affine" giữ theo repo gốc, thực chất là homography 3x3.)"""
        self.match_info = []
        affine_matrices = []
        for i in range(1, len(gray_images)):
            self._log(f"   cặp ảnh {i} - {i + 1}:")
            affine_matrices.append(self.compute_homography_matrix(sift_data[i - 1], sift_data[i]))
        return np.array(affine_matrices)

    # ---------------------------------------------------------------- HÀM CHÍNH
    def _prepare(self, images):
        """Bước 0: đổi sang float [0, 1]; thu nhỏ nếu ảnh rộng hơn max_width."""
        out = []
        for img in images:
            img = ops.to_float(img)
            if self.max_width and img.shape[1] > self.max_width:
                img = ops.resize_bilinear(img, self.max_width / img.shape[1])
            out.append(img)
        return out

    def get_panorama(self, images_list_rgb):
        """
        Nhận danh sách ảnh màu RGB ĐÃ SẮP XẾP từ trái sang phải -> trả về ảnh panorama
        (float trong [0, 1]). gui.py và run_panorama.py gọi hàm này.
        """
        if len(images_list_rgb) < 2:
            raise ValueError("Cần ít nhất 2 ảnh để ghép.")
        images = self._prepare(images_list_rgb)
        self._log("[1] Chuyển ảnh xám")
        gray_images = [ops.rgb2gray(img) for img in images]
        self._log("[2] Trích xuất SIFT")
        sift_data = self.get_sift_descriptors(gray_images)
        self._log("[3] So khớp + RANSAC")
        affine_matrices = self.compute_affine_matrices(gray_images, sift_data)
        self._log("[4] Ghép ảnh từ ảnh giữa")
        stitched_image = self.stitch_multiple_images(images, affine_matrices)
        self._log("[5] Cắt viền đen")
        x, y, w, h = self.find_bounding_box(stitched_image)
        return np.clip(stitched_image[y:y + h, x:x + w], 0, 1)
