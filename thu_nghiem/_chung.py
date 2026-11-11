"""Tiện ích dùng chung cho các thí nghiệm (nội bộ)."""
import numpy as np

from loi.cau_hinh import KET_QUA, ghi_csv, ghi_json
from tien_xu_ly import LoiXuLyAnh

THU_MUC_RA = KET_QUA / "tich_hop"

# Ba cấu hình của khâu thu nhỏ/chuẩn hóa được so sánh: cách cũ -> chống alias -> chống alias + kéo giãn độ sáng (mặc định)
CAU_HINH_TIEN_XU_LY = {
    "cu_song_tuyen_tinh": dict(che_do_thu_nho="song_tuyen_tinh", keo_gian=False),
    "dien_tich": dict(che_do_thu_nho="dien_tich", keo_gian=False),
    "dien_tich_keo_gian": dict(che_do_thu_nho="dien_tich", keo_gian=True),
}


def thanh_xam_rgb(xam_uint8):
    return np.repeat(np.asarray(xam_uint8, dtype=np.uint8)[..., None], 3, axis=2)


def nhan_dang_hoac_loi(bo, anh, bai_toan, **tuy_chon):
    """Chỉ số lớp dự đoán hoặc -1 nếu tiền xử lý không tách được đối tượng (tính là sai)."""
    try:
        return bo.nhan_dang(anh, bai_toan, **tuy_chon)["lop"]
    except LoiXuLyAnh:
        return -1


def ghi_ket_qua(ten, du_lieu, dong=None):
    THU_MUC_RA.mkdir(parents=True, exist_ok=True)
    ghi_json(THU_MUC_RA / f"{ten}.json", du_lieu)
    if dong:
        ghi_csv(THU_MUC_RA / f"{ten}.csv", dong)


def in_bang(tieu_de, cot, dong):
    print(tieu_de)
    print("  " + " | ".join(f"{c:>20s}" for c in cot))
    for d in dong:
        print("  " + " | ".join(f"{str(v):>20s}" for v in d))
