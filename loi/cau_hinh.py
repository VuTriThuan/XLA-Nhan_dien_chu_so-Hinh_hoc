"""Cấu hình dùng chung: đường dẫn, hai bài toán nhận dạng và các tiện ích đọc/ghi.

Mọi đường dẫn tính từ vị trí của file này nên chạy được từ bất kỳ thư mục nào.
"""
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

GOC = Path(__file__).resolve().parents[1]
DU_LIEU = GOC / "du_lieu"
MO_HINH = GOC / "mo_hinh"
KET_QUA = GOC / "ket_qua"
ANH_MAU = DU_LIEU / "anh_mau"
BAO_CAO = GOC / "bao_cao"

KICH_THUOC_CNN = 28                       # cả hai CNN nhận ảnh 28x28, một kênh, [0,1], nền đen, nét sáng
PHIEN_BAN_TIEN_XU_LY = "thong-nhat-1"     # phiên bản tiền xử lý dùng để tạo đầu vào huấn luyện CNN hình học
# Định nghĩa đầu vào mà mỗi mô hình được huấn luyện; checkpoint phải khớp thì mới được nạp.
PHIEN_BAN_DAU_VAO = {"chu_so": "mnist-goc-chia-255", "hinh_hoc": PHIEN_BAN_TIEN_XU_LY}
BIEN_THE = ("co_so", "cai_tien")          # cấu hình cơ sở và cấu hình cải tiến (tăng cường dữ liệu, ...)


@dataclass(frozen=True)
class BaiToan:
    ten: str                    # chu_so | hinh_hoc
    lop: tuple                  # tên lớp theo thứ tự chỉ số của mạng
    nhan_vi: tuple              # nhãn hiển thị
    le: int                     # lề (px) khi căn đối tượng vào canvas 28x28
    du_lieu_goc: str            # thư mục trong du_lieu/ chứa bộ dữ liệu huấn luyện
    thu_muc_anh_that: str       # thư mục trong du_lieu/ chứa ảnh người thật viết/vẽ

    @property
    def so_lop(self):
        return len(self.lop)

    @property
    def noi_dung_toi_da(self):
        return KICH_THUOC_CNN - 2 * self.le

    @property
    def thu_muc_du_lieu(self):
        return DU_LIEU / self.du_lieu_goc

    @property
    def thu_muc_anh_that_day_du(self):
        return DU_LIEU / self.thu_muc_anh_that

    def tep_mo_hinh(self, ten, thu_muc=None):
        """mo_hinh/<bai_toan>_<ten>.pth, ten = co_so | cai_tien | tot_nhat (bản sao của cấu hình được chọn)."""
        return Path(thu_muc or MO_HINH) / f"{self.ten}_{ten}.pth"

    @property
    def thu_muc_ket_qua(self):
        return KET_QUA / self.ten


CHU_SO = BaiToan("chu_so", tuple(str(i) for i in range(10)), tuple(f"Chữ số {i}" for i in range(10)), 4, "mnist",
                 "chu_so_viet_tay")
HINH_HOC = BaiToan("hinh_hoc", ("tron", "tam_giac", "chu_nhat"), ("Tròn", "Tam giác", "Chữ nhật / vuông"), 3,
                   "hinh_hoc_tu_sinh", "hinh_ve_tay")
BAI_TOAN = {CHU_SO.ten: CHU_SO, HINH_HOC.ten: HINH_HOC}


def lay_bai_toan(ten):
    if ten not in BAI_TOAN:
        raise ValueError(f"bài toán phải là một trong {sorted(BAI_TOAN)}")
    return BAI_TOAN[ten]


def tep_lua_chon(thu_muc=None):
    """mo_hinh/lua_chon.json: lựa chọn bằng validation của cả hai bài toán, khóa theo tên bài toán."""
    return Path(thu_muc or MO_HINH) / "lua_chon.json"


def doc_lua_chon(bai_toan, thu_muc=None):
    return doc_json(tep_lua_chon(thu_muc))[lay_bai_toan(bai_toan).ten]


def ghi_lua_chon(bai_toan, gia_tri, thu_muc=None):
    tep = tep_lua_chon(thu_muc)
    tat_ca = doc_json(tep) if tep.exists() else {}
    tat_ca[lay_bai_toan(bai_toan).ten] = gia_tri
    ghi_json(tep, dict(sorted(tat_ca.items())))


# ---------------------------------------------------------------------------------- đọc / ghi
def ghi_json(duong_dan, gia_tri):
    duong_dan = Path(duong_dan)
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    duong_dan.write_text(json.dumps(gia_tri, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def doc_json(duong_dan):
    return json.loads(Path(duong_dan).read_text(encoding="utf-8"))


def ghi_csv(duong_dan, dong, cot=None):
    duong_dan = Path(duong_dan)
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    dong = list(dong)
    with duong_dan.open("w", encoding="utf-8-sig", newline="") as f:
        viet = csv.DictWriter(f, fieldnames=cot or list(dong[0]))
        viet.writeheader()
        viet.writerows(dong)


def sha256(duong_dan):
    h = hashlib.sha256()
    with Path(duong_dan).open("rb") as f:
        for khoi in iter(lambda: f.read(1024 * 1024), b""):
            h.update(khoi)
    return h.hexdigest()
