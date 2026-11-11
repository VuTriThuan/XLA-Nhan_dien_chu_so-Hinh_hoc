"""Tăng cường dữ liệu bằng biến đổi affine tự cài (không dùng torchvision/OpenCV)."""
import math

import torch


def bien_doi_affine(x, goc, ti_le, dx, dy):
    """Ánh xạ ngược theo tâm pixel, nội suy song tuyến tính, ngoài canvas = 0. goc tính bằng radian."""
    n, _, cao, rong = x.shape
    thiet_bi = x.device
    yy, xx = torch.meshgrid(torch.arange(cao, device=thiet_bi), torch.arange(rong, device=thiet_bi), indexing="ij")
    xx = xx.float()[None] - (rong - 1) / 2 - dx[:, None, None]
    yy = yy.float()[None] - (cao - 1) / 2 - dy[:, None, None]
    c, s, z = torch.cos(goc)[:, None, None], torch.sin(goc)[:, None, None], ti_le[:, None, None]
    sx = (c * xx + s * yy) / z + (rong - 1) / 2
    sy = (-s * xx + c * yy) / z + (cao - 1) / 2
    x0, y0 = sx.floor().long(), sy.floor().long()
    wx, wy = sx - x0, sy - y0
    phang = x[:, 0].reshape(n, -1)

    def lay(ix, iy):
        hop_le = (ix >= 0) & (ix < rong) & (iy >= 0) & (iy < cao)
        chi_so = iy.clamp(0, cao - 1) * rong + ix.clamp(0, rong - 1)
        return phang.gather(1, chi_so.reshape(n, -1)).reshape(n, cao, rong) * hop_le

    kq = (1 - wy) * ((1 - wx) * lay(x0, y0) + wx * lay(x0 + 1, y0))
    kq += wy * ((1 - wx) * lay(x0, y0 + 1) + wx * lay(x0 + 1, y0 + 1))
    return kq[:, None].clamp(0, 1)


def tang_cuong(x, goc_toi_da, dich_toi_da, ti_le=(0.9, 1.1), do_sang=False, nhieu=0.0, generator=None):
    """Xoay (radian), dịch (pixel), co giãn ngẫu nhiên; tùy chọn đổi độ sáng 0,9-1,1 và cộng nhiễu Gauss.

    generator tách RNG tăng cường khỏi RNG của Dropout để kết quả tái lập được.
    """
    n = len(x)

    def ngau_nhien(*hinh):
        return torch.rand(*(hinh or (n,)), device=x.device, generator=generator)

    goc = (ngau_nhien() * 2 - 1) * goc_toi_da
    tl = ti_le[0] + ngau_nhien() * (ti_le[1] - ti_le[0])
    dx = (ngau_nhien() * 2 - 1) * dich_toi_da
    dy = (ngau_nhien() * 2 - 1) * dich_toi_da
    kq = bien_doi_affine(x, goc, tl, dx, dy)
    if do_sang:
        kq = kq * (0.9 + ngau_nhien(n, 1, 1, 1) * 0.2)
    if nhieu > 0:
        kq = kq + torch.randn(kq.shape, device=kq.device, generator=generator) * nhieu
    return kq.clamp(0, 1)


# Cấu hình tăng cường của từng bài toán (giữ đúng công thức đã dùng khi huấn luyện các mô hình hiện có)
def tang_cuong_chu_so(x, generator=None):
    return tang_cuong(x, math.radians(10), 2.0, generator=generator)


def tang_cuong_hinh_hoc(x, generator=None):
    return tang_cuong(x, 0.35, 1.5, do_sang=True, nhieu=0.012, generator=generator)
