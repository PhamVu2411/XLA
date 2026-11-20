# -*- coding: utf-8 -*-
# =============================================================================
# FILE: general_functions.py  -  CÁC HÀM TIỆN ÍCH DÙNG CHUNG
# -----------------------------------------------------------------------------
#   - normalize_img()              : chuẩn hóa giá trị pixel về đoạn [0, 1]
#   - show_images()                : hiển thị nhiều ảnh cạnh nhau (dùng trong notebook)
#   - sobel_numba()                : TỰ CÀI ĐẶT Sobel bằng vòng lặp (bản dễ đọc, để giải thích)
#   - compute_weighted_histogram() : TỰ CÀI ĐẶT histogram có trọng số (bản vòng lặp)
#   - get_hist()                   : histogram 8 hướng cho 1 ô 4x4 (bản vòng lặp)
#   - get_hist_fast()              : CÙNG KẾT QUẢ với get_hist nhưng vector hóa bằng NumPy,
#                                    xử lý 16 ô của 1 descriptor cùng lúc (SIFT dùng hàm này)
#   - concatenate_images()         : ghép ngang các ảnh để xem trước trên giao diện
#
# numba (@jit) chỉ để TĂNG TỐC vòng lặp, không phải thư viện xử lý ảnh. Nếu máy
# không cài numba, code vẫn chạy bình thường (chỉ chậm hơn ở các hàm vòng lặp).
# =============================================================================
import numpy as np

try:
    from numba import jit
except ImportError:                      # không có numba -> dùng decorator "rỗng"
    def jit(*args, **kwargs):
        def wrap(f):
            return f
        return wrap


def normalize_img(img):
    """
    Chuẩn hóa ảnh về [0, 1] theo công thức min-max:  (img - min) / (max - min).
    Mục đích: đưa mọi ảnh về cùng thang giá trị để các ngưỡng trong SIFT dùng chung được.
    """
    img = np.asarray(img, dtype=np.float64)
    img_min = np.min(img)
    img_max = np.max(img)
    if img_max - img_min < 1e-12:        # ảnh phẳng (1 màu) -> tránh chia cho 0
        return np.zeros_like(img)
    return (img - img_min) / (img_max - img_min)


def show_images(imgs, imgs_labels=None, figsize=(20, 10), show_axis=True):
    """Hiển thị danh sách ảnh theo hàng ngang bằng matplotlib (chỉ để minh họa)."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(imgs), figsize=figsize)
    axes = np.atleast_1d(axes)
    for i in range(len(imgs)):
        axes[i].imshow(imgs[i], cmap="gray")
        if imgs_labels:
            axes[i].set_title(imgs_labels[i])
        if not show_axis:
            axes[i].set_axis_off()
    plt.tight_layout()
    plt.show()


@jit(nopython=True)
def sobel_numba(img):
    """
    TỰ CÀI ĐẶT bộ lọc Sobel bằng vòng lặp (bản dễ hiểu để thuyết trình):
        Kx = [[-1,0,1],[-2,0,2],[-1,0,1]] -> gx ;  Ky = [[-1,-2,-1],[0,0,0],[1,2,1]] -> gy
        độ lớn = sqrt(gx^2 + gy^2) ;  hướng = arctan2(gy, gx) (độ, trong [-180, 180])
    Viền 1 pixel ngoài cùng không tính (gradient = 0).
    """
    rows, cols = img.shape
    grad_x = np.zeros_like(img)
    grad_y = np.zeros_like(img)
    kernel_x = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    kernel_y = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            grad_x[i, j] = np.sum(img[i-1:i+2, j-1:j+2] * kernel_x)
            grad_y[i, j] = np.sum(img[i-1:i+2, j-1:j+2] * kernel_y)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    orientation_deg = np.degrees(np.arctan2(grad_y, grad_x))
    return magnitude, orientation_deg


@jit(nopython=True)
def compute_weighted_histogram(window, weights, num_bins):
    """
    TỰ CÀI ĐẶT histogram có trọng số: chia [-180°, 180°) thành num_bins khoảng, mỗi pixel
    cộng "độ lớn gradient" của nó vào bin chứa "hướng gradient" của nó.
    """
    hist = np.zeros(num_bins)
    bin_width = 360.0 / num_bins
    for i in range(window.shape[0]):
        for j in range(window.shape[1]):
            b = int(np.floor((window[i, j] + 180.0) / bin_width))
            if 0 <= b < num_bins:
                hist[b] += weights[i, j]
    return hist


@jit(nopython=True)
def get_hist(array, nbins=8):
    """Histogram 8 hướng gradient của 1 ô nhỏ (bản vòng lặp). Nhân 256 vì ảnh ở thang [0, 1]."""
    gradient_magnitude, gradient_orientation = sobel_numba(array * 256)
    return compute_weighted_histogram(gradient_orientation, gradient_magnitude, nbins)


def get_hist_fast(patch, cell=4, nbins=8):
    """
    BẢN VECTOR HÓA của get_hist cho cả vùng 16x16 cùng lúc -> vector 128 chiều.
    Kết quả GIỐNG HỆT việc gọi get_hist cho từng ô 4x4 rồi nối lại:
      1. Cắt vùng thành lưới (16/cell) x (16/cell) ô, mỗi ô cell x cell pixel.
      2. Sobel trong từng ô (chỉ các pixel bên trong ô, giống sobel_numba).
      3. Mỗi ô lập histogram 8 bin (trọng số = độ lớn gradient) bằng np.bincount.
    """
    p = np.asarray(patch, dtype=np.float64) * 256
    n = p.shape[0] // cell
    cells = p.reshape(n, cell, n, cell).transpose(0, 2, 1, 3).reshape(n * n, cell, cell)
    # Sobel cho các pixel bên trong mỗi ô (bỏ viền của ô), tính cho mọi ô cùng lúc
    c = cells
    gx = ((c[:, :-2, 2:] - c[:, :-2, :-2]) + 2 * (c[:, 1:-1, 2:] - c[:, 1:-1, :-2])
          + (c[:, 2:, 2:] - c[:, 2:, :-2]))
    gy = ((c[:, 2:, :-2] - c[:, :-2, :-2]) + 2 * (c[:, 2:, 1:-1] - c[:, :-2, 1:-1])
          + (c[:, 2:, 2:] - c[:, :-2, 2:]))
    mag = np.sqrt(gx ** 2 + gy ** 2).reshape(n * n, -1)
    ang = np.degrees(np.arctan2(gy, gx)).reshape(n * n, -1)
    b = np.floor((ang + 180.0) / (360.0 / nbins)).astype(np.int64)
    ok = (b >= 0) & (b < nbins)
    cell_id = np.repeat(np.arange(n * n), mag.shape[1]).reshape(mag.shape)
    flat = (cell_id * nbins + b)[ok]
    hist = np.bincount(flat, weights=mag[ok], minlength=n * n * nbins)
    return hist


def concatenate_images(image_list, space_width=20):
    """
    Ghép ngang nhiều ảnh RGB (uint8) thành 1 ảnh, có dải xám giữa các ảnh. Ảnh thấp hơn
    được đệm thêm phía dưới để cùng chiều cao. Chỉ dùng để xem trước trong gui.py.
    """
    max_h = max(img.shape[0] for img in image_list)
    parts = []
    for img in image_list:
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        pad = np.full((max_h - img.shape[0], img.shape[1], 3), 41, dtype=np.uint8)
        space = np.full((max_h, space_width, 3), 41, dtype=np.uint8)
        parts += [space, np.concatenate([img.astype(np.uint8), pad], axis=0)]
    return np.concatenate(parts, axis=1)
