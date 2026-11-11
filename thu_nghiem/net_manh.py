"""Chữ số nét mảnh: MNIST phóng to, làm mỏng nét bằng bộ xương Zhang-Suen rồi dày lại đến độ dày mong muốn.

Đo độ chính xác khi nét chỉ bằng vài phần trăm chiều cao chữ số (bút mảnh trên khung vẽ lớn), qua ba cấu hình thu nhỏ.
Ảnh do chương trình tạo từ MNIST, KHÔNG phải chữ viết tay thật: chỉ cho biết mức rủi ro.
"""
import numpy as np
from PIL import Image

from loi.du_doan import BoNhanDang
from bo_du_lieu import tai_mnist
from tien_xu_ly import hinh_thai_nhi_phan

from ._chung import CAU_HINH_TIEN_XU_LY, ghi_ket_qua, in_bang, nhan_dang_hoac_loi, thanh_xam_rgb

TI_LE_NET = (0.12, 0.08, 0.05, 0.03, 0.015)      # độ dày nét / chiều cao chữ số
SO_ANH = 150
HAT = 5


def lam_manh_zhang_suen(mat_na):
    """Làm mỏng ảnh nhị phân về bộ xương 1 pixel (Zhang-Suen), vector hóa bằng NumPy."""
    a = np.pad(mat_na.astype(np.uint8), 1)
    while True:
        thay_doi = False
        for buoc in (0, 1):
            p2, p3, p4, p5 = a[:-2, 1:-1], a[:-2, 2:], a[1:-1, 2:], a[2:, 2:]
            p6, p7, p8, p9 = a[2:, 1:-1], a[2:, :-2], a[1:-1, :-2], a[:-2, :-2]
            tam = a[1:-1, 1:-1]
            lang_gieng = [p2, p3, p4, p5, p6, p7, p8, p9]
            b = sum(n.astype(int) for n in lang_gieng)
            chuyen = sum(((lang_gieng[i] == 0) & (lang_gieng[(i + 1) % 8] == 1)).astype(int) for i in range(8))
            if buoc == 0:
                dk = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                dk = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            xoa = (tam == 1) & (b >= 2) & (b <= 6) & (chuyen == 1) & dk
            if xoa.any():
                thay_doi = True
                tam[xoa] = 0
        if not thay_doi:
            return a[1:-1, 1:-1].astype(bool)


def ve_chu_so_manh(anh28, ti_le_net, rng, xuong=None):
    """Đặt chữ số vào khung lớn, nét dày xấp xỉ ti_le_net x chiều cao chữ số. Trả về ảnh xám uint8 (H, W)."""
    canh = int(rng.integers(300, 480))
    d = int(canh * rng.uniform(0.5, 0.8))
    lon = np.array(Image.fromarray(anh28).resize((d, d), Image.BICUBIC)) > 127
    if xuong is None:
        xuong = lam_manh_zhang_suen(lon)
    chieu_cao = d * 20 / 28
    r = max(0, int(round(ti_le_net * chieu_cao / 2)))
    muc = xuong
    for _ in range(r):
        muc = hinh_thai_nhi_phan(muc, "gian", 3)           # giãn vuông r lần: nét dày xấp xỉ 2r+1 pixel
    khung = np.zeros((canh, canh), bool)
    y0, x0 = int(rng.integers(0, canh - d + 1)), int(rng.integers(0, canh - d + 1))
    khung[y0:y0 + d, x0:x0 + d] = muc
    return np.where(khung, 0, 255).astype(np.uint8)


def chay(so_anh=SO_ANH):
    bo = BoNhanDang()
    x, y = tai_mnist("test")
    rng = np.random.default_rng(HAT)
    chi_so = rng.choice(len(y), so_anh, replace=False)
    ket_qua, dong = {}, []
    for ti_le in TI_LE_NET:
        dung = {ten: 0 for ten in CAU_HINH_TIEN_XU_LY}
        rs = np.random.default_rng(321)
        for i in chi_so:
            anh = thanh_xam_rgb(ve_chu_so_manh(np.rint(x[i, 0] * 255).astype(np.uint8), ti_le, rs))
            for ten, cau_hinh in CAU_HINH_TIEN_XU_LY.items():
                dung[ten] += nhan_dang_hoac_loi(bo, anh, "chu_so", **cau_hinh) == int(y[i])
        ket_qua[str(ti_le)] = {ten: v / so_anh for ten, v in dung.items()}
        dong.append({"ti_le_net_tren_chieu_cao": ti_le, **{ten: round(v / so_anh, 4) for ten, v in dung.items()}})
    ghi_ket_qua("net_manh_chu_so", {"mo_ta": "MNIST test phóng to, làm mỏng nét (Zhang-Suen) rồi dày lại; ảnh tổng hợp, không phải chữ viết tay thật",
                                     "so_anh_moi_muc": so_anh, "hat": HAT, "do_chinh_xac": ket_qua}, dong)
    in_bang("Chữ số nét mảnh: độ chính xác theo độ dày nét", ["nét/chiều cao"] + list(CAU_HINH_TIEN_XU_LY),
            [[d["ti_le_net_tren_chieu_cao"]] + [f"{d[t]:.1%}" for t in CAU_HINH_TIEN_XU_LY] for d in dong])
    return ket_qua
