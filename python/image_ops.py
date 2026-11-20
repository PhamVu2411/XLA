# -*- coding: utf-8 -*-
# =============================================================================
# FILE: image_ops.py  -  CÁC PHÉP XỬ LÝ ẢNH TỰ CÀI ĐẶT BẰNG NUMPY (CODE FROM SCRATCH)
# -----------------------------------------------------------------------------
# File này thay thế TOÀN BỘ các hàm có sẵn của OpenCV / scikit-image / scikit-learn
# mà bản gốc của repo đã dùng. Chỉ dùng NumPy (tính toán mảng, đại số tuyến tính).
#
#   Hàm tự viết                 Thay cho hàm có sẵn
#   -------------------------   ---------------------------------------------
#   rgb2gray                    skimage.color.rgb2gray
#   gaussian_blur               cv2.GaussianBlur
#   resize_bilinear             cv2.resize
#   sobel, cart_to_polar        cv2.Sobel, cv2.cartToPolar
#   rotate                      skimage.transform.rotate
#   warp_perspective            cv2.warpPerspective
#   compute_homography_dlt      (phần giải hệ của cv2.findHomography)
#   ransac_homography           cv2.findHomography(..., cv2.RANSAC)
#   match_descriptors           sklearn.neighbors.NearestNeighbors
#   binary_closing              cv2.threshold + cv2.morphologyEx
#   largest_component           cv2.findContours + cv2.boundingRect
#
# Quy ước tọa độ: điểm ảnh viết dạng (x, y) = (cột, hàng); mảng ảnh truy cập img[y, x].
# =============================================================================
import numpy as np


# -----------------------------------------------------------------------------
# 1. CHUYỂN ĐỔI KIỂU DỮ LIỆU VÀ ẢNH XÁM
# -----------------------------------------------------------------------------
def to_float(img):
    """Đổi ảnh sang float64. Ảnh uint8 (0..255) được chia 255 để về đoạn [0, 1]."""
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0
    return img.astype(np.float64)


def rgb2gray(img):
    """
    Chuyển ảnh màu RGB -> ảnh xám theo công thức độ sáng (chuẩn ITU-R BT.709,
    giống skimage):  Gray = 0.2125*R + 0.7154*G + 0.0721*B
    Mắt người nhạy nhất với màu xanh lá nên hệ số G lớn nhất.
    """
    img = to_float(img)
    if img.ndim == 2:
        return img
    return img[..., 0] * 0.2125 + img[..., 1] * 0.7154 + img[..., 2] * 0.0721


# -----------------------------------------------------------------------------
# 2. TÍCH CHẬP (CORRELATION) VÀ LÀM MỜ GAUSSIAN
# -----------------------------------------------------------------------------
def _pad_reflect(img, ph, pw):
    """Thêm viền bằng cách PHẢN XẠ ảnh qua mép (kiểu 'dcb|abcd|cba', giống
    BORDER_REFLECT_101 mặc định của OpenCV) để bộ lọc tính được ở sát biên."""
    pad = [(ph, ph), (pw, pw)] + [(0, 0)] * (img.ndim - 2)
    return np.pad(img, pad, mode='reflect')


def correlate2d(img, kernel):
    """
    Tương quan 2D (2D correlation) - phép "trượt kernel qua ảnh":
        out[y, x] = Σ_i Σ_j kernel[i, j] * img[y + i - kh/2, x + j - kw/2]
    Cài đặt kiểu vector hóa: thay vì lặp từng pixel, ta cộng dồn kh*kw ảnh đã
    dịch chuyển (mỗi ảnh nhân với 1 hệ số của kernel) -> chạy nhanh với NumPy.
    Hỗ trợ ảnh xám (H, W) và ảnh màu (H, W, C).
    """
    img = np.asarray(img, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    kh, kw = kernel.shape
    H, W = img.shape[:2]
    p = _pad_reflect(img, kh // 2, kw // 2)
    out = np.zeros_like(img)
    for i in range(kh):
        for j in range(kw):
            if kernel[i, j] != 0:
                out += kernel[i, j] * p[i:i + H, j:j + W]
    return out


def gaussian_kernel_1d(sigma):
    """
    Tạo kernel Gaussian 1 chiều:  g(t) = exp(-t^2 / (2*sigma^2)), rồi chuẩn hóa tổng = 1.
    Bán kính chọn giống OpenCV với ảnh số thực: ksize = round(8*sigma + 1) | 1.
    """
    ksize = int(round(sigma * 8 + 1)) | 1          # luôn là số lẻ
    r = ksize // 2
    t = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-(t ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def gaussian_blur(img, sigma):
    """
    Làm mờ Gaussian. Kernel Gaussian 2D TÁCH ĐƯỢC (separable): G(x, y) = g(x) * g(y)
    nên ta lọc theo hàng rồi theo cột bằng kernel 1D -> nhanh hơn nhiều so với 2D.
    """
    g = gaussian_kernel_1d(sigma)
    out = correlate2d(img, g[None, :])     # lọc theo chiều ngang
    out = correlate2d(out, g[:, None])     # lọc theo chiều dọc
    return out


# -----------------------------------------------------------------------------
# 3. ĐẠO HÀM ẢNH - BỘ LỌC SOBEL
# -----------------------------------------------------------------------------
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)


def sobel(img):
    """Trả về (gx, gy): đạo hàm theo x (ngang) và theo y (dọc) bằng kernel Sobel 3x3."""
    return correlate2d(img, SOBEL_X), correlate2d(img, SOBEL_Y)


def cart_to_polar(dx, dy):
    """Đổi gradient (dx, dy) -> (độ lớn, góc tính bằng độ trong [0, 360))."""
    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    angle = np.degrees(np.arctan2(dy, dx)) % 360.0
    return magnitude, angle


# -----------------------------------------------------------------------------
# 4. NỘI SUY SONG TUYẾN, THAY ĐỔI KÍCH THƯỚC, XOAY ẢNH
# -----------------------------------------------------------------------------
def bilinear_sample(img, xs, ys):
    """
    Lấy giá trị ảnh tại các tọa độ THỰC (xs, ys) bằng nội suy song tuyến:
    giá trị = trung bình có trọng số của 4 pixel nguyên xung quanh.
        f(x, y) ≈ (1-a)(1-b) f00 + a(1-b) f10 + (1-a)b f01 + ab f11
    với a, b là phần lẻ của x, y. Điểm nằm ngoài ảnh nhận giá trị 0.
    Trả về (giá_trị, mặt_nạ_hợp_lệ).
    """
    h, w = img.shape[:2]
    valid = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)
    xc = np.clip(xs, 0, w - 1)
    yc = np.clip(ys, 0, h - 1)
    x0 = np.floor(xc).astype(np.int64)
    y0 = np.floor(yc).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    a = xc - x0
    b = yc - y0
    if img.ndim == 3:
        a = a[..., None]
        b = b[..., None]
    out = (img[y0, x0] * (1 - a) * (1 - b) + img[y0, x1] * a * (1 - b)
           + img[y1, x0] * (1 - a) * b + img[y1, x1] * a * b)
    out = np.where(valid[..., None] if img.ndim == 3 else valid, out, 0.0)
    return out, valid


def resize_bilinear(img, fx, fy=None):
    """
    Thay đổi kích thước ảnh theo hệ số fx (ngang), fy (dọc) bằng nội suy song tuyến.
    Pixel đích (u, v) lấy giá trị tại điểm nguồn x = (u + 0.5)/fx - 0.5 (căn theo tâm
    pixel, giống cv2.resize với INTER_LINEAR).
    """
    fy = fx if fy is None else fy
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape[:2]
    nh, nw = max(1, int(round(h * fy))), max(1, int(round(w * fx)))
    xs = np.clip((np.arange(nw) + 0.5) / fx - 0.5, 0, w - 1)
    ys = np.clip((np.arange(nh) + 0.5) / fy - 0.5, 0, h - 1)
    X, Y = np.meshgrid(xs, ys)
    out, _ = bilinear_sample(img, X, Y)
    return out


def rotate(img, angle):
    """
    Xoay ảnh quanh tâm một góc `angle` độ (giữ nguyên kích thước, vùng trống = 0),
    cách làm giống skimage.transform.rotate.
    Dùng ÁNH XẠ NGƯỢC: với mỗi pixel đích p, tìm điểm nguồn
        q = R(angle) · (p - c) + c ,   R = [[cos, -sin], [sin, cos]],  c = tâm ảnh
    rồi nội suy song tuyến tại q. (Ánh xạ ngược giúp ảnh đích không bị "lỗ".)
    """
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape[:2]
    cx, cy = w / 2.0 - 0.5, h / 2.0 - 0.5
    t = np.deg2rad(angle)
    c, s = np.cos(t), np.sin(t)
    X, Y = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    xs = c * (X - cx) - s * (Y - cy) + cx
    ys = s * (X - cx) + c * (Y - cy) + cy
    out, _ = bilinear_sample(img, xs, ys)
    return out


# -----------------------------------------------------------------------------
# 5. HOMOGRAPHY: ÁP DỤNG, ƯỚC LƯỢNG (DLT) VÀ RANSAC
# -----------------------------------------------------------------------------
def apply_homography(H, pts):
    """
    Biến đổi các điểm (N, 2) bằng ma trận homography 3x3 (tọa độ thuần nhất):
        [x', y', w]^T = H · [x, y, 1]^T   ->   điểm mới = (x'/w, y'/w)
    """
    pts = np.asarray(pts, dtype=np.float64)
    ph = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    wz = ph[:, 2:3]
    wz = np.where(np.abs(wz) < 1e-12, 1e-12, wz)
    return ph[:, :2] / wz


def _normalize_points(pts):
    """
    Chuẩn hóa Hartley: dời tâm các điểm về gốc tọa độ và co giãn sao cho khoảng cách
    trung bình tới gốc = √2. Giúp hệ phương trình DLT ổn định số học hơn rất nhiều.
    Trả về (điểm_đã_chuẩn_hóa, ma_trận_T).
    """
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2) / (d + 1e-12)
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    return apply_homography(T, pts), T


def compute_homography_dlt(src, dst):
    """
    Ước lượng homography H (dst ≈ H·src) từ N >= 4 cặp điểm bằng DLT
    (Direct Linear Transform).
    Mỗi cặp (x, y) -> (u, v) cho 2 phương trình tuyến tính theo 9 phần tử h của H:
        [-x, -y, -1,  0,  0,  0, u*x, u*y, u] · h = 0
        [ 0,  0,  0, -x, -y, -1, v*x, v*y, v] · h = 0
    Xếp thành hệ A·h = 0 (A cỡ 2N x 9). Nghiệm (với ||h|| = 1) là vector kỳ dị
    phải ứng với giá trị kỳ dị NHỎ NHẤT của A  ->  hàng cuối của V^T trong SVD.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if len(src) < 4:
        return None
    s, T1 = _normalize_points(src)
    d, T2 = _normalize_points(dst)
    x, y = s[:, 0], s[:, 1]
    u, v = d[:, 0], d[:, 1]
    z, o = np.zeros_like(x), np.ones_like(x)
    A = np.empty((2 * len(x), 9))
    A[0::2] = np.stack([-x, -y, -o, z, z, z, u * x, u * y, u], axis=1)
    A[1::2] = np.stack([z, z, z, -x, -y, -o, v * x, v * y, v], axis=1)
    try:
        _, _, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(T2) @ Hn @ T1            # bỏ chuẩn hóa: H = T2^-1 · Hn · T1
    if abs(H[2, 2]) < 1e-12:
        return None
    return H / H[2, 2]


def _is_degenerate(pts):
    """4 điểm suy biến nếu có 3 điểm gần như thẳng hàng (diện tích tam giác ≈ 0)."""
    for i, j, k in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
        a, b, c = pts[i], pts[j], pts[k]
        area = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if area < 1e-3:
            return True
    return False


def ransac_homography(src, dst, thresh=5.0, max_iters=2000, confidence=0.995, seed=0):
    """
    RANSAC (RANdom SAmple Consensus) - ước lượng H bền vững khi có nhiều cặp khớp sai.
    Lặp lại:
      1. Chọn ngẫu nhiên 4 cặp điểm (số tối thiểu để giải H).
      2. Giải H bằng DLT.
      3. Chiếu mọi điểm src bằng H, cặp có sai số ||H·src - dst|| < thresh là INLIER.
      4. Giữ H có nhiều inlier nhất.
    Số vòng lặp được giảm dần theo công thức N = log(1 - p) / log(1 - w^4)
    (p = độ tin cậy, w = tỉ lệ inlier tốt nhất hiện tại).
    Cuối cùng giải lại H bằng DLT trên TOÀN BỘ inlier để chính xác hơn.
    Trả về (H, mặt_nạ_inlier).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = len(src)
    if n < 4:
        return None, np.zeros(n, dtype=bool)
    rng = np.random.default_rng(seed)       # seed cố định -> kết quả lặp lại được
    best_mask, best_count = None, 0
    needed, it = max_iters, 0
    while it < needed:
        it += 1
        idx = rng.choice(n, 4, replace=False)
        if _is_degenerate(src[idx]) or _is_degenerate(dst[idx]):
            continue
        H = compute_homography_dlt(src[idx], dst[idx])
        if H is None:
            continue
        err = np.linalg.norm(apply_homography(H, src) - dst, axis=1)
        mask = err < thresh
        count = int(mask.sum())
        if count > best_count:
            best_count, best_mask = count, mask
            w = count / n
            if w >= 1.0:
                break
            needed = min(max_iters, int(np.ceil(np.log(1 - confidence) / np.log(1 - w ** 4 + 1e-12))))
    if best_mask is None or best_count < 4:
        return None, np.zeros(n, dtype=bool)
    H = compute_homography_dlt(src[best_mask], dst[best_mask])
    if H is None:
        return None, np.zeros(n, dtype=bool)
    # Cập nhật lại tập inlier theo H cuối cùng
    err = np.linalg.norm(apply_homography(H, src) - dst, axis=1)
    return H, err < thresh


def warp_perspective(img, H, out_size, chunk_rows=256):
    """
    Uốn ảnh theo homography H vào khung đích kích thước out_size = (rộng, cao).
    Ánh xạ ngược: mỗi pixel đích p lấy giá trị từ điểm nguồn q = H^-1 · p (nội suy
    song tuyến). Pixel không có nguồn -> 0 (đen), tương đương BORDER_TRANSPARENT
    trên nền đen của bản gốc. Xử lý theo từng khối hàng để tiết kiệm bộ nhớ.
    """
    img = np.asarray(img, dtype=np.float64)
    W, Hh = int(out_size[0]), int(out_size[1])
    Hinv = np.linalg.inv(H)
    out = np.zeros((Hh, W) + img.shape[2:], dtype=np.float64)
    xs = np.arange(W, dtype=np.float64)
    for r0 in range(0, Hh, chunk_rows):
        r1 = min(Hh, r0 + chunk_rows)
        X, Y = np.meshgrid(xs, np.arange(r0, r1, dtype=np.float64))
        pts = np.stack([X.ravel(), Y.ravel()], axis=1)
        q = apply_homography(Hinv, pts)
        vals, _ = bilinear_sample(img, q[:, 0].reshape(X.shape), q[:, 1].reshape(X.shape))
        out[r0:r1] = vals
    return out


# -----------------------------------------------------------------------------
# 6. SO KHỚP DESCRIPTOR (LÁNG GIỀNG GẦN NHẤT)
# -----------------------------------------------------------------------------
def pairwise_distances(a, b):
    """Ma trận khoảng cách Euclid giữa mọi cặp vector:
       ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b   (tính 1 lần bằng phép nhân ma trận)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d2 = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2.0 * a @ b.T
    return np.sqrt(np.maximum(d2, 0.0))


def match_descriptors(desc1, desc2, max_distance=1.0, ratio=0.8):
    """
    So khớp: với mỗi descriptor của ảnh 1, tìm descriptor GẦN NHẤT của ảnh 2.
    Giữ cặp nếu:
      - khoảng cách < max_distance, và
      - (ratio test của Lowe) khoảng cách gần nhất < ratio * khoảng cách gần nhì
        -> loại các điểm "mập mờ" giống nhiều điểm cùng lúc. ratio=None để tắt.
    Trả về (chỉ_số_ở_ảnh_1, chỉ_số_ở_ảnh_2, khoảng_cách).
    """
    desc1 = np.asarray(desc1, dtype=np.float64)
    desc2 = np.asarray(desc2, dtype=np.float64)
    if len(desc1) == 0 or len(desc2) == 0:
        e = np.array([], dtype=np.int64)
        return e, e, np.array([])
    D = pairwise_distances(desc1, desc2)
    j1 = np.argmin(D, axis=1)
    rows = np.arange(len(desc1))
    d1 = D[rows, j1]
    keep = d1 < max_distance
    if ratio is not None and D.shape[1] >= 2:
        D[rows, j1] = np.inf
        d2 = D.min(axis=1)
        keep &= d1 < ratio * d2
    idx1 = rows[keep]
    return idx1, j1[keep], d1[keep]


# -----------------------------------------------------------------------------
# 7. HÌNH THÁI HỌC VÀ CẮT VIỀN ĐEN
# -----------------------------------------------------------------------------
def _sliding_extreme(mask, size, func, pad_value):
    """Lọc max/min với cửa sổ vuông size x size (tách thành 2 lần 1D: hàng rồi cột)."""
    r = size // 2
    out = mask
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        p = np.pad(out, pad, mode='constant', constant_values=pad_value)
        n = out.shape[axis]
        acc = None
        for k in range(size):
            sl = [slice(None), slice(None)]
            sl[axis] = slice(k, k + n)
            acc = p[tuple(sl)] if acc is None else func(acc, p[tuple(sl)])
        out = acc
    return out


def binary_dilate(mask, size):
    """Phép GIÃN (dilation): pixel = 1 nếu có ít nhất 1 pixel = 1 trong cửa sổ."""
    return _sliding_extreme(mask.astype(bool), size, np.logical_or, False)


def binary_erode(mask, size):
    """Phép CO (erosion): pixel = 1 chỉ khi mọi pixel trong cửa sổ đều = 1."""
    return _sliding_extreme(mask.astype(bool), size, np.logical_and, True)


def binary_closing(mask, size):
    """Phép ĐÓNG (closing) = giãn rồi co -> lấp các lỗ/khe đen nhỏ bên trong vùng trắng."""
    return binary_erode(binary_dilate(mask, size), size)


def largest_component(mask):
    """
    Giữ lại VÙNG LIÊN THÔNG lớn nhất (4-lân cận) của mặt nạ nhị phân, thay cho
    cv2.findContours + chọn contour lớn nhất. Cài đặt bằng gán nhãn lan truyền
    (mỗi pixel lấy nhãn nhỏ nhất của lân cận, lặp tới khi không đổi), chạy trên
    ảnh thu nhỏ để nhanh, sau đó phóng lại kích thước gốc.
    """
    h, w = mask.shape
    step = max(1, int(np.ceil(max(h, w) / 400)))       # thu nhỏ để lặp nhanh
    small = mask[::step, ::step]
    big = 10 ** 9
    labels = np.where(small, np.arange(small.size).reshape(small.shape), big)
    for _ in range(4 * sum(small.shape)):
        p = np.pad(labels, 1, mode='constant', constant_values=big)
        nb = np.minimum.reduce([p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
        nb = np.where(small, nb, big)
        if np.array_equal(nb, labels):
            break
        labels = nb
    vals, counts = np.unique(labels[small], return_counts=True)
    if len(vals) == 0:
        return mask
    keep_small = labels == vals[np.argmax(counts)]
    keep = np.repeat(np.repeat(keep_small, step, axis=0), step, axis=1)[:h, :w]
    return mask & keep
