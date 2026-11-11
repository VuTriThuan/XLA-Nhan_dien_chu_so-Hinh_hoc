"""Nhận dạng: nạp checkpoint, dự đoán từ đầu vào đã chuẩn hóa hoặc từ ảnh bất kỳ."""
import time
from pathlib import Path

import numpy as np
import torch

from .cau_hinh import BAI_TOAN, KICH_THUOC_CNN, PHIEN_BAN_DAU_VAO, lay_bai_toan
from .cnn import tao_mang
from tien_xu_ly import AnhDongNhat, xu_ly_anh


def duong_dan_mo_hinh(bai_toan, ten="tot_nhat"):
    return lay_bai_toan(bai_toan).tep_mo_hinh(ten)


def nap_checkpoint(duong_dan, bai_toan):
    """Nạp checkpoint (chỉ trọng số và kiểu dữ liệu cơ bản) và kiểm tra khớp bài toán, thứ tự lớp, đầu vào."""
    bt = lay_bai_toan(bai_toan)
    trang_thai = torch.load(duong_dan, map_location="cpu", weights_only=True)
    if trang_thai["bai_toan"] != bt.ten or trang_thai["lop"] != list(bt.lop):
        raise ValueError("Checkpoint không khớp bài toán hoặc thứ tự lớp")
    if trang_thai["phien_ban_tien_xu_ly"] != PHIEN_BAN_DAU_VAO[bt.ten]:
        raise ValueError("Checkpoint được huấn luyện với đầu vào khác phiên bản hiện tại")
    mang = tao_mang(trang_thai["kien_truc"])
    mang.load_state_dict(trang_thai["state_dict"], strict=True)
    mang.eval()
    return mang, trang_thai


class DuDoan:
    """Dự đoán cho MỘT bài toán. Đầu vào là ảnh 28x28 đã chuẩn hóa: float [0,1], nền 0, nét sáng."""

    def __init__(self, bai_toan, checkpoint=None, thiet_bi="cpu"):
        self.bai_toan = lay_bai_toan(bai_toan)
        self.thiet_bi = torch.device(thiet_bi)
        self.mang, self.trang_thai = nap_checkpoint(checkpoint or duong_dan_mo_hinh(bai_toan), bai_toan)
        self.mang.to(self.thiet_bi)
        self.bien_the = self.trang_thai["bien_the"]

    def xac_suat(self, x, co_lo=512):
        """Softmax cho một lô ảnh N x 1 x 28 x 28 (hoặc N x 28 x 28), trả về mảng NumPy N x so_lop."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 3:
            x = x[:, None]
        kq = []
        with torch.inference_mode():
            for dau in range(0, len(x), co_lo):
                kq.append(self.mang(torch.from_numpy(np.ascontiguousarray(x[dau:dau + co_lo])).to(self.thiet_bi))
                          .softmax(1).cpu().numpy())
        return np.concatenate(kq) if kq else np.zeros((0, self.bai_toan.so_lop), np.float32)

    def du_doan_dau_vao(self, anh):
        """Một ảnh (28,28), (1,28,28) hoặc (1,1,28,28), float trong [0,1]. Không tiền xử lý lại."""
        x = torch.as_tensor(np.asarray(anh).copy(), dtype=torch.float32)
        if tuple(x.shape) == (KICH_THUOC_CNN,) * 2:
            x = x[None, None]
        elif tuple(x.shape) == (1, KICH_THUOC_CNN, KICH_THUOC_CNN):
            x = x[None]
        if tuple(x.shape) != (1, 1, KICH_THUOC_CNN, KICH_THUOC_CNN) or not bool(torch.isfinite(x).all()):
            raise ValueError("Cần đúng một ảnh hữu hạn 1x1x28x28")
        if float(x.min()) < 0 or float(x.max()) > 1:
            raise ValueError("Đầu vào phải trong [0,1], không phải [0,255] hay đã chuẩn hóa mean/std")
        if float(x.max() - x.min()) < .01:
            raise AnhDongNhat("Ảnh đầu vào đã chuẩn hóa không có nét")
        p = self.xac_suat(x[0].numpy())[0]
        i = int(p.argmax())
        return {"lop": i, "ten_lop": self.bai_toan.lop[i], "nhan": self.bai_toan.nhan_vi[i],
                "do_tin_cay": float(p[i]), "xac_suat": {t: float(v) for t, v in zip(self.bai_toan.lop, p)},
                "bien_the": self.bien_the}


class BoNhanDang:
    """Nạp hai mô hình một lần; nhận ảnh bất kỳ, tiền xử lý rồi nhận dạng."""

    def __init__(self, thiet_bi="cpu", so_luong=2):
        torch.set_num_threads(so_luong)
        self.mo_hinh = {ten: DuDoan(ten, thiet_bi=thiet_bi) for ten in BAI_TOAN}

    def thong_tin(self):
        return {ten: {"bien_the": m.bien_the, "so_lop": m.bai_toan.so_lop} for ten, m in self.mo_hinh.items()}

    def nhan_dang(self, anh, bai_toan, **tuy_chon):
        """Trả về dict kết quả kèm `cac_buoc` (ảnh trung gian). Lỗi ảnh ném LoiXuLyAnh (là ValueError)."""
        bat_dau = time.perf_counter()
        xu_ly = xu_ly_anh(anh, bai_toan, **tuy_chon)
        kq = self.mo_hinh[lay_bai_toan(bai_toan).ten].du_doan_dau_vao(xu_ly.anh)
        kq.update(bai_toan=bai_toan, cuc_tinh=xu_ly.cuc_tinh, nguong=xu_ly.nguong, hop_bao=list(xu_ly.hop_bao),
                  ti_le_vien_toi=xu_ly.ti_le_vien_toi, canh_bao=list(xu_ly.canh_bao), cac_buoc=xu_ly.cac_buoc,
                  ms=round((time.perf_counter() - bat_dau) * 1000, 1))
        return kq
