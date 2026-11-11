"""Kiến trúc CNN dùng chung cho cả hai bài toán.

Conv(1->16, 3x3) -> [BatchNorm] -> ReLU -> MaxPool2 -> Conv(16->32, 3x3) -> [BatchNorm] -> ReLU -> MaxPool2
-> Flatten(800) -> [Dropout] -> Linear(800->64) -> ReLU -> Linear(64->so_lop).
Chữ số: 10 lớp, Dropout 0,25. Hình học: 3 lớp; cấu hình cải tiến thêm BatchNorm và Dropout 0,3.
"""
from torch import nn

from .cau_hinh import KICH_THUOC_CNN


class MangCNN(nn.Module):
    def __init__(self, so_lop, batch_norm=False, dropout=0.25):
        super().__init__()
        lop = []
        for vao, ra in ((1, 16), (16, 32)):
            lop.append(nn.Conv2d(vao, ra, 3))
            if batch_norm:
                lop.append(nn.BatchNorm2d(ra))
            lop += [nn.ReLU(), nn.MaxPool2d(2)]
        self.dac_trung = nn.Sequential(*lop)
        self.phan_loai = nn.Sequential(nn.Flatten(), nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                                       nn.Linear(32 * 5 * 5, 64), nn.ReLU(), nn.Linear(64, so_lop))

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, KICH_THUOC_CNN, KICH_THUOC_CNN):
            raise ValueError("Cần đầu vào N x 1 x 28 x 28")
        return self.phan_loai(self.dac_trung(x))


def tao_mang(kien_truc):
    """Dựng mạng từ dict kiến trúc lưu trong checkpoint: {so_lop, batch_norm, dropout}."""
    return MangCNN(kien_truc["so_lop"], kien_truc.get("batch_norm", False), kien_truc.get("dropout", 0.25))


def dem_tham_so(mang):
    return sum(p.numel() for p in mang.parameters() if p.requires_grad)
