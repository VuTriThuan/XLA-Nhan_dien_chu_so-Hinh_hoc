"""Lọc nhiễu."""
import numpy as np

from ._kiem_tra import ma_tran, so_nguyen_duong


def loc_trung_vi(xam, kich_thuoc=3):
    """Lọc trung vị cửa sổ lẻ; lặp pixel rìa. Kích thước 1 giữ nguyên ảnh.

    Khử nhiễu muối tiêu rất tốt nhưng có thể xóa nét mảnh (nét 1 pixel biến mất với cửa sổ 3x3).
    """
    xam = ma_tran(xam)
    so_nguyen_duong(kich_thuoc, "Cửa sổ", le=True)
    nua = kich_thuoc // 2
    dem = np.pad(xam, nua, mode="edge")
    cua_so = np.lib.stride_tricks.sliding_window_view(dem, (kich_thuoc, kich_thuoc))
    return np.median(cua_so, axis=(-2, -1)).astype(np.uint8)
