"""Thu ảnh chữ số viết tay và hình vẽ tay THẬT để đánh giá (không bao giờ sinh ảnh tự động).

Nhãn do người thu chọn TRƯỚC khi lưu; công cụ không hiển thị dự đoán để tránh sửa ảnh/nhãn theo mô hình.
Ảnh lưu vào du_lieu/<bai_toan>/<viet_tay|ve_tay>/<lop>/ kèm tệp .json ghi nguồn; chỉ dùng để đánh giá,
không dùng để huấn luyện hay chỉnh mô hình đã chốt.
"""
import io
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from loi.cau_hinh import ghi_json, lay_bai_toan, sha256
from tien_xu_ly import doc_anh, xu_ly_anh

NGUON_HOP_LE = ("ve_tren_khung", "tai_len")
TOI_DA_BYTE = 8 * 1024 * 1024
TOI_DA_PIXEL = 16_000_000
_KHOA = threading.Lock()


def dem_mau(bai_toan, thu_muc=None):
    bt = lay_bai_toan(bai_toan)
    goc = Path(thu_muc) if thu_muc else bt.thu_muc_anh_that_day_du
    return {ten: len(list((goc / ten).glob("*.png"))) for ten in bt.lop}


def luu_mau(bai_toan, nhan, du_lieu_anh, nguoi_ve="", nguon="ve_tren_khung", thu_muc=None):
    """Kiểm tra và lưu một ảnh; trả về {'da_luu': đường dẫn, 'so_luong': đếm theo lớp}. Lỗi đầu vào ném ValueError."""
    from PIL import Image
    bt = lay_bai_toan(bai_toan)
    if nhan not in bt.lop:
        raise ValueError(f"Nhãn phải là một trong {list(bt.lop)}")
    if nguon not in NGUON_HOP_LE:
        raise ValueError("Nguồn ảnh không hợp lệ")
    if not 0 < len(du_lieu_anh) <= TOI_DA_BYTE:
        raise ValueError("Ảnh quá lớn (tối đa 8 MB) hoặc rỗng")
    try:
        with Image.open(io.BytesIO(du_lieu_anh)) as im:
            if im.width * im.height > TOI_DA_PIXEL:
                raise ValueError("Giới hạn 16 triệu pixel")
            im.load()
            anh = im.convert("RGBA")
    except OSError as loi:                       # gồm UnidentifiedImageError, tệp hỏng
        raise ValueError("Không đọc được ảnh (dùng PNG hoặc JPEG)") from loi
    xu_ly_anh(doc_anh(du_lieu_anh), bt.ten)           # từ chối ảnh trống hoặc không tách được đối tượng
    goc = Path(thu_muc) if thu_muc else bt.thu_muc_anh_that_day_du
    thu_muc_lop = goc / nhan
    thu_muc_lop.mkdir(parents=True, exist_ok=True)
    khoa = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    with _KHOA:
        tep = thu_muc_lop / f"{khoa}.png"
        anh.save(tep)
        ghi_json(tep.with_suffix(".json"), {"bai_toan": bt.ten, "lop": nhan, "nguoi_ve": str(nguoi_ve)[:40], "nguon": nguon,
                                            "sha256": sha256(tep), "tao_luc_utc": datetime.now(timezone.utc).isoformat(),
                                            "nhan_ghi_truoc_khi_du_doan": True,
                                            "su_dung": "chỉ để đánh giá; không huấn luyện hay chỉnh mô hình trên các ảnh này"})
    return {"da_luu": str(tep.relative_to(goc.parent)), "so_luong": dem_mau(bt.ten, goc)}
