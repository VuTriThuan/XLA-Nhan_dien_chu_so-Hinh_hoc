"""Thay đổi kích thước ảnh.

Hai cách: nội suy song tuyến tính (lấy 4 điểm lân cận, dùng khi phóng to) và trung bình theo diện tích
(lọc thông thấp trước khi lấy mẫu, dùng khi thu nhỏ). Song tuyến tính khi thu nhỏ mạnh bỏ sót nét mảnh nên
nét bị "rách" thành các chấm rời; trung bình theo diện tích giữ được năng lượng của nét.
"""
import numpy as np

from ._kiem_tra import LoiXuLyAnh, ma_tran, so_nguyen_duong

CHE_DO = ("dien_tich", "song_tuyen_tinh")


def noi_suy_song_tuyen_tinh(anh, cao, rong):
    """Nội suy song tuyến tính, ánh xạ tâm pixel và kẹp biên."""
    anh = ma_tran(anh)
    so_nguyen_duong(cao, "cao")
    so_nguyen_duong(rong, "rộng")
    if anh.dtype.kind not in "uif" or not np.isfinite(anh).all():
        raise LoiXuLyAnh("Nội suy cần ma trận số thực hữu hạn")
    h, w = anh.shape
    yy = np.clip((np.arange(cao) + .5) * h / cao - .5, 0, h - 1)
    xx = np.clip((np.arange(rong) + .5) * w / rong - .5, 0, w - 1)
    y0, x0 = np.floor(yy).astype(int), np.floor(xx).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    wy, wx = (yy - y0)[:, None], (xx - x0)[None, :]
    return ((1 - wy) * ((1 - wx) * anh[y0[:, None], x0] + wx * anh[y0[:, None], x1])
            + wy * ((1 - wx) * anh[y1[:, None], x0] + wx * anh[y1[:, None], x1]))


def _trong_so_dien_tich(n_vao, n_ra):
    """Ma trận (n_ra x n_vao): mỗi ô đầu ra là trung bình có trọng số phần chồng lấn với các pixel nguồn."""
    buoc = n_vao / n_ra
    bien = np.arange(n_ra + 1) * buoc
    trai, phai = bien[:-1, None], bien[1:, None]
    j = np.arange(n_vao)[None, :]
    chong = np.clip(np.minimum(phai, j + 1) - np.maximum(trai, j), 0, None)
    return chong / buoc


def thu_nho_trung_binh_dien_tich(anh, cao, rong):
    """Thu nhỏ ảnh 2D bằng trung bình theo diện tích (hỗ trợ tỉ lệ không nguyên)."""
    a = np.asarray(anh, dtype=np.float64)
    if a.ndim != 2 or cao < 1 or rong < 1 or cao > a.shape[0] or rong > a.shape[1]:
        raise LoiXuLyAnh("thu_nho_trung_binh_dien_tich chỉ dùng để thu nhỏ ảnh 2D")
    return _trong_so_dien_tich(a.shape[0], cao) @ a @ _trong_so_dien_tich(a.shape[1], rong).T


def thu_phong(anh, cao, rong, che_do="dien_tich"):
    """Thu nhỏ theo diện tích (mặc định) hoặc song tuyến tính; phóng to luôn dùng song tuyến tính."""
    if che_do not in CHE_DO:
        raise LoiXuLyAnh(f"chế độ thu phóng phải là một trong {CHE_DO}")
    h, w = ma_tran(anh).shape
    if che_do == "dien_tich" and cao <= h and rong <= w:
        return thu_nho_trung_binh_dien_tich(anh, cao, rong)
    return noi_suy_song_tuyen_tinh(anh, cao, rong)


def gioi_han_kich_thuoc(anh, canh_toi_da=512):
    """Thu nhỏ ảnh (H,W) hoặc (H,W,C) nếu cạnh dài vượt canh_toi_da (BFS thuần Python rất chậm trên ảnh lớn)."""
    a = np.asarray(anh)
    h, w = a.shape[:2]
    if max(h, w) <= canh_toi_da:
        return a
    ti_le = canh_toi_da / max(h, w)
    cao, rong = max(2, round(h * ti_le)), max(2, round(w * ti_le))
    if a.ndim == 2:
        kq = thu_nho_trung_binh_dien_tich(a, cao, rong)
    else:
        kq = np.stack([thu_nho_trung_binh_dien_tich(a[:, :, c], cao, rong) for c in range(a.shape[2])], axis=2)
    if a.dtype == np.uint8:
        return np.rint(kq).clip(0, 255).astype(np.uint8)
    return kq.astype(a.dtype if a.dtype.kind == "f" else np.float32)
