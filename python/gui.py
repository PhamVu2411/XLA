# -*- coding: utf-8 -*-
# =============================================================================
# FILE: gui.py  -  GIAO DIỆN NGƯỜI DÙNG "StitchMaster" (thư viện CustomTkinter)
# -----------------------------------------------------------------------------
# Cách chạy (đứng ở thư mục nào cũng được):
#     python python/gui.py
#
# Các chức năng:
#   - Bấm vào khung ảnh  : chọn nhiều ảnh (theo thứ tự từ trái sang phải)
#   - Swap               : nhập 2 số thứ tự (bắt đầu từ 1) để đổi chỗ 2 ảnh
#   - Start              : chạy ghép ảnh trong 1 luồng (thread) riêng + thanh tiến trình
#   - Save               : lưu kết quả vào thư mục results/ (panorama_img1.png, ...)
#
# OpenCV chỉ dùng để ĐỌC / GHI file ảnh. Thuật toán nằm trong panorama_stitching.py.
# Trong chương trình, ảnh luôn được giữ ở thứ tự kênh RGB.
# =============================================================================
import os
import sys
import threading

import customtkinter as ctk
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from general_functions import concatenate_images  # noqa: E402
from panorama_stitching import Panorama            # noqa: E402
from run_panorama import read_rgb, save_rgb        # noqa: E402

# Đường dẫn tính theo vị trí file -> chạy được từ bất kỳ thư mục nào, trên mọi hệ điều hành
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

upload_img = read_rgb(os.path.join(IMAGES_DIR, "upload.jpeg"))   # ảnh "bấm để tải ảnh lên"
upload_img_size = (600, 360)
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme(os.path.join(BASE_DIR, "python", "dark-blue.json"))

# Biến dùng chung giữa luồng giao diện và luồng ghép ảnh
is_done_stitching = False
stitch_error = None
stitcher = Panorama()
final_image = np.zeros((2, 2, 3), dtype=np.uint8)


def stitch(images):
    """Chạy trong luồng riêng: gọi thuật toán, đổi kết quả [0,1] -> uint8 [0,255]."""
    global is_done_stitching, final_image, stitch_error
    try:
        stitch_error = None
        final_image = (stitcher.get_panorama(images) * 255).round().astype("uint8")
    except Exception as e:                     # ví dụ: 2 ảnh không có phần chung
        stitch_error = str(e)
    is_done_stitching = True


class App(ctk.CTk):
    """Cửa sổ chính của ứng dụng."""

    def __init__(self):
        super().__init__()
        self.geometry("720x620")
        self.title("StitchMaster")
        self.resizable(False, False)
        try:
            self.iconbitmap(default=os.path.join(IMAGES_DIR, "icon1.ico"))   # icon chỉ có trên Windows
        except Exception:
            pass

        self.grid_columnconfigure((0, 1, 2), weight=1)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # Khung lớn hiển thị ảnh (ảnh đã tải lên hoặc ảnh kết quả); bấm vào để tải ảnh
        self.images_frame = ctk.CTkFrame(self, corner_radius=20)
        self.images_frame.grid(row=0, column=0, columnspan=3, sticky='nsew', padx=20, pady=20)
        self.images_frame_label = ctk.CTkLabel(self.images_frame, text="", corner_radius=20)
        self.images_frame_label.pack(padx=5, pady=35)
        self.add_image_to_frame(upload_img, "images_frame_label", upload_img_size)
        self.images_frame.bind("<Button-1>", self.upload_images)
        self.images_frame_label.bind("<Button-1>", self.upload_images)

        # Thanh tiến trình
        self.progress_bar = ctk.CTkProgressBar(self, height=1, corner_radius=10, border_width=2,
                                               border_color='gray', determinate_speed=0.5)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=1, column=0, sticky='nsew', columnspan=3, padx=20, pady=5)

        # 2 ô nhập số thứ tự ảnh cần đổi chỗ + nút Swap
        self.swap_frame = ctk.CTkFrame(self, corner_radius=20, fg_color='black')
        self.swap_frame.grid(row=2, column=1)
        self.swap_entry1 = ctk.CTkEntry(self.swap_frame, width=90)
        self.swap_entry2 = ctk.CTkEntry(self.swap_frame, width=90)
        self.swap_entry1.grid(row=0, column=0, pady=10)
        self.swap_entry2.grid(row=0, column=1, pady=10)

        # Các nút bấm
        self.start_button = ctk.CTkButton(self, text="Start", height=40, width=190, corner_radius=8)
        self.save_button = ctk.CTkButton(self, text="Save", height=40, width=190, corner_radius=8,
                                         state='disabled', fg_color="gray")
        self.swap_button = ctk.CTkButton(self.swap_frame, text="Swap", height=40, width=190, corner_radius=8)
        self.swap_button.bind("<Button-1>", self.swap_images)
        self.start_button.bind("<Button-1>", self.start_stitching)
        self.save_button.bind("<Button-1>", self.save_image)
        self.start_button.grid(row=2, column=0, pady=0)
        self.save_button.grid(row=2, column=2, pady=0)
        self.swap_button.grid(row=1, columnspan=2)

        self.grid_rowconfigure(0, weight=10)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, minsize=50, weight=2)
        self.images_frame.pack_propagate(False)

        self.images_list = []     # danh sách ảnh (RGB) người dùng đã tải lên
        self.i = 1                # số thứ tự dùng khi đặt tên file lưu

    def upload_images(self, e):
        """[SỰ KIỆN] Mở hộp thoại chọn nhiều ảnh, đọc ảnh (RGB) và hiển thị xem trước."""
        global is_done_stitching
        images_path = list(ctk.filedialog.askopenfilenames(
            filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp"), ("Tất cả", "*.*")]))
        if images_path:
            self.images_list = [read_rgb(p) for p in images_path]
            is_done_stitching = False
            self.show_images()

    def show_images(self):
        """Ghép ngang các ảnh đã tải để xem trước trong khung."""
        self.add_image_to_frame(concatenate_images(self.images_list), "images_frame_label", upload_img_size)

    def add_image_to_frame(self, img, frame_label, size):
        """Chuyển ảnh NumPy (RGB) -> ảnh PIL -> CTkImage rồi đặt vào khung giao diện."""
        img = ctk.CTkImage(Image.fromarray(img), size=size)
        label = getattr(self, frame_label)
        label.configure(image=img)

    def swap_images(self, e):
        """[SỰ KIỆN] Đổi chỗ 2 ảnh trong danh sách (khi tải ảnh lên sai thứ tự)."""
        n = len(self.images_list)
        try:
            a, b = int(self.swap_entry1.get()), int(self.swap_entry2.get())
        except ValueError:
            return
        if n and 0 < a <= n and 0 < b <= n:
            self.images_list[a - 1], self.images_list[b - 1] = self.images_list[b - 1], self.images_list[a - 1]
            self.show_images()

    def update_progress(self):
        """Cho thanh tiến trình chạy tới khi luồng ghép ảnh báo xong."""
        while not is_done_stitching:
            self.progress_bar.step()
            self.progress_bar.update_idletasks()
            self.update()
            self.after(100)
        self.progress_bar.set(1)

    def start_stitching(self, e):
        """[SỰ KIỆN] Nút Start: chạy stitch() ở luồng riêng để giao diện không bị treo."""
        global is_done_stitching
        if len(self.images_list) < 2:
            return
        is_done_stitching = False
        self.progress_bar.set(0)
        threading.Thread(target=stitch, args=(self.images_list,), daemon=True).start()
        self.update_progress()
        if stitch_error:
            ctk.CTkLabel(self, text="Lỗi: " + stitch_error, text_color="red").grid(row=3, column=0, columnspan=3)
            return
        self.add_image_to_frame(final_image, "images_frame_label", upload_img_size)
        self.save_button.configure(fg_color="#7e6aff", text_color="#4C067A", text="Save", state="normal")

    def save_image(self, e):
        """[SỰ KIỆN] Nút Save: lưu ảnh vào thư mục results/ rồi đặt lại giao diện."""
        os.makedirs(RESULTS_DIR, exist_ok=True)
        save_rgb(os.path.join(RESULTS_DIR, f"panorama_img{self.i}.png"), final_image)
        self.i += 1
        self.save_button.configure(fg_color="gray", text_color="gray28", text="Saved", state="disabled")
        self.add_image_to_frame(upload_img, "images_frame_label", upload_img_size)


if __name__ == "__main__":
    app = App()
    app.mainloop()
