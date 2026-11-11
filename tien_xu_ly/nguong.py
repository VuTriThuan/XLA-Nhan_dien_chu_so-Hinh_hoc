"""Phân ngưỡng Otsu và xác định cực tính (nét tối trên nền sáng hay ngược lại)."""
import numpy as np

from ._kiem_tra import LoiXuLyAnh, ma_tran

CUC_TINH = ("tu_dong", "net_toi", "net_sang")


def nguong_otsu(xam):
    """Chọn ngưỡng tối đa hóa phương sai giữa hai lớp nền/nét, duyệt histogram 256 mức."""
    xam = ma_tran(xam)
    if xam.dtype != np.uint8:
        raise LoiXuLyAnh("Otsu cần ảnh uint8")
    hist = np.bincount(xam.ravel(), minlength=256).astype(float)
    trong_so = np.cumsum(hist)
    mo_men = np.cumsum(hist * np.arange(256))
    con_lai = xam.size - trong_so
    diem = np.full(256, -1.0)
    hop_le = (trong_so > 0) & (con_lai > 0)
    diem[hop_le] = ((mo_men[-1] * trong_so[hop_le] - mo_men[hop_le] * xam.size) ** 2
                    / (trong_so[hop_le] * con_lai[hop_le]))
    return int(np.argmax(diem)) if hop_le.any() else int(xam.flat[0])


def xac_dinh_cuc_tinh(xam, nguong, cuc_tinh="tu_dong"):
    """Trả về (net_toi, ti_le_vien_toi, canh_bao).

    Chế độ tự động đoán nền từ trung vị mức xám trên viền ảnh. Đây chỉ là quy tắc kinh nghiệm: cảnh báo khi
    tỉ lệ viền tối nằm trong [0,35; 0,65], không phải độ tin cậy của CNN.
    """
    if cuc_tinh not in CUC_TINH:
        raise LoiXuLyAnh(f"cực tính phải là một trong {CUC_TINH}")
    vien = np.concatenate((xam[0], xam[-1], xam[:, 0], xam[:, -1]))   # bốn góc được đếm hai lần
    ti_le = float(np.mean(vien <= nguong))
    canh_bao = None
    if cuc_tinh == "tu_dong" and 0.35 <= ti_le <= 0.65:
        canh_bao = (f"Cực tính tự động không chắc chắn: {ti_le:.0%} mẫu viền tối. Kiểm tra ảnh đầu ra; "
                    "chọn cực tính 'net_toi' hoặc 'net_sang' nếu sai. Đây là cảnh báo kinh nghiệm, không phải độ tin cậy CNN.")
    net_toi = cuc_tinh == "net_toi" or (cuc_tinh == "tu_dong" and np.median(vien) > nguong)
    return bool(net_toi), ti_le, canh_bao
