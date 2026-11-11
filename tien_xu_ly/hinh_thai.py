"""Hình thái học nhị phân với phần tử cấu trúc vuông."""
import numpy as np

from ._kiem_tra import LoiXuLyAnh, ma_tran, so_nguyen_duong

PHEP = ("co", "gian", "mo", "dong")


def hinh_thai_nhi_phan(mat_na, phep, kich_thuoc=3):
    """Co / giãn / mở / đóng với phần tử cấu trúc vuông toàn 1, đệm nền 0.

    Co cần mọi vị trí của phần tử cấu trúc nằm trong đối tượng; giãn chỉ cần một vị trí chạm. Mở = co rồi giãn,
    đóng = giãn rồi co. Với mở/đóng, đệm r = k//2 pixel nền trước phép ghép và chỉ cắt về kích thước gốc sau
    cả hai bước (quy ước biên; không phải định nghĩa duy nhất đúng).
    """
    mat_na = ma_tran(mat_na).astype(bool)
    so_nguyen_duong(kich_thuoc, "SE", le=True)
    if phep in ("mo", "dong"):
        r = kich_thuoc // 2
        mo_rong = np.pad(mat_na, r, constant_values=False)
        dau, sau = ("co", "gian") if phep == "mo" else ("gian", "co")
        kq = hinh_thai_nhi_phan(hinh_thai_nhi_phan(mo_rong, dau, kich_thuoc), sau, kich_thuoc)
        return kq[r:-r, r:-r] if r else kq
    cua_so = np.lib.stride_tricks.sliding_window_view(
        np.pad(mat_na, kich_thuoc // 2, constant_values=False), (kich_thuoc, kich_thuoc))
    if phep == "co":
        return cua_so.all(axis=(-2, -1))
    if phep == "gian":
        return cua_so.any(axis=(-2, -1))
    raise LoiXuLyAnh(f"Phép hình thái phải là một trong {PHEP}")
