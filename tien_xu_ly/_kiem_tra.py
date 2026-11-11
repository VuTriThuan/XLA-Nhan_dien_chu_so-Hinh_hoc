"""Kiểm tra đối số dùng chung cho các bước xử lý ảnh (nội bộ)."""
import numpy as np


class LoiXuLyAnh(ValueError):
    """Gốc của mọi lỗi do ảnh/tham số không hợp lệ."""


class AnhDongNhat(LoiXuLyAnh):
    """Ảnh một màu: không có gì để tách."""


class KhongCoDoiTuong(LoiXuLyAnh):
    """Sau khi tách nền không còn pixel đối tượng nào."""


class NetQuaMong(LoiXuLyAnh):
    """Đối tượng biến mất khi thu nhỏ."""


def so_nguyen_duong(gia_tri, ten, le=False):
    sai = (isinstance(gia_tri, (bool, np.bool_)) or not isinstance(gia_tri, (int, np.integer))
           or gia_tri < 1 or (le and gia_tri % 2 == 0))
    if sai:
        raise LoiXuLyAnh(ten + " phải là số nguyên dương" + (" lẻ" if le else ""))


def ma_tran(anh):
    a = np.asarray(anh)
    if a.ndim != 2 or a.size == 0:
        raise LoiXuLyAnh("Cần ma trận 2 chiều không rỗng")
    return a
