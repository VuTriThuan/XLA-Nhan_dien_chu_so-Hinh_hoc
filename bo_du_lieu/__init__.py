"""Nạp và tạo dữ liệu cho hai bài toán."""
from loi.cau_hinh import lay_bai_toan
from .hinh_hoc import sinh_bo_du_lieu, tai_hinh_hoc, tai_meta_hinh_hoc, ve_hinh
from .mnist import chuan_bi_mnist, chia_phan_tang, tai_mnist


def tai_tap(bai_toan, tap):
    """(x float32 N x 1 x 28 x 28 trong [0,1], y int64) của một tập; hình học có thêm tập 'stress'."""
    ten = lay_bai_toan(bai_toan).ten
    return tai_mnist(tap) if ten == "chu_so" else tai_hinh_hoc(tap)


__all__ = ["sinh_bo_du_lieu", "tai_hinh_hoc", "tai_meta_hinh_hoc", "ve_hinh", "chuan_bi_mnist", "chia_phan_tang", "tai_mnist", "tai_tap"]
