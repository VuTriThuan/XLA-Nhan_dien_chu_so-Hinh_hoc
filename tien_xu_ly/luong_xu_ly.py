"""Luồng tiền xử lý: ảnh bất kỳ chứa MỘT đối tượng -> ảnh 28x28 nét sáng nền đen cho CNN.

Các bước: (giới hạn kích thước) -> xám -> lọc trung vị -> Otsu -> cực tính -> (hình thái) -> thành phần liên thông
lớn nhất -> hộp bao -> thu nhỏ giữ tỉ lệ -> căn giữa canvas -> (kéo giãn độ sáng).
"""
from dataclasses import dataclass, field

import numpy as np

from loi.cau_hinh import KICH_THUOC_CNN, lay_bai_toan
from ._kiem_tra import AnhDongNhat, KhongCoDoiTuong, LoiXuLyAnh, NetQuaMong, so_nguyen_duong
from .anh import chuyen_xam
from .hinh_thai import PHEP, hinh_thai_nhi_phan
from .lien_thong import thanh_phan_lien_thong_lon_nhat
from .loc import loc_trung_vi
from .nguong import nguong_otsu, xac_dinh_cuc_tinh
from .thu_phong import CHE_DO, gioi_han_kich_thuoc, thu_phong

TEN_BUOC = {"xam": "Ảnh xám", "loc": "Lọc trung vị", "nguong": "Phân ngưỡng Otsu", "mat_na": "Vùng lớn nhất",
            "cat": "Cắt theo hộp bao", "thu_nho": "Thu nhỏ giữ tỉ lệ", "dau_vao_cnn": "Đầu vào CNN 28×28"}


@dataclass
class KetQuaXuLy:
    anh: np.ndarray                 # float32 (H, W) trong [0,1], nét sáng / nền đen: đầu vào của CNN
    cac_buoc: dict                  # tên bước -> ảnh uint8 để hiển thị
    hop_bao: tuple                  # (trái, trên, phải, dưới); phải/dưới không bao gồm
    nguong: int
    cuc_tinh: str = ""              # net_toi: nét tối trên ảnh đầu vào; net_sang: nét sáng
    ti_le_vien_toi: float = -1.0    # tỉ lệ mẫu viền có mức xám <= ngưỡng (không phải độ tin cậy)
    canh_bao: list = field(default_factory=list)

    def tensor(self, bo_cuc="NCHW"):
        """Tensor một ảnh (NumPy) cho Keras (NHWC) hoặc PyTorch (NCHW)."""
        if bo_cuc == "NHWC":
            return self.anh[None, :, :, None]
        if bo_cuc == "NCHW":
            return self.anh[None, None, :, :]
        raise LoiXuLyAnh("bố cục phải là NHWC hoặc NCHW")


def xu_ly_anh(anh, bai_toan="chu_so", cuc_tinh="tu_dong", kich_thuoc_loc=1, hinh_thai=None, chi_giu_lon_nhat=True,
              kich_thuoc_dau_ra=KICH_THUOC_CNN, le=None, che_do_thu_nho="dien_tich", keo_gian=True, canh_toi_da=512):
    """Tiền xử lý một ảnh. Mặc định không lọc và không dùng hình thái học để khỏi xóa nét mảnh.

    bai_toan        chu_so (nội dung tối đa 20x20 trong 28x28) hoặc hinh_hoc (22x22).
    cuc_tinh        tu_dong | net_toi | net_sang.
    kich_thuoc_loc  1 = tắt lọc trung vị; số lẻ lớn hơn 1 = cửa sổ lọc.
    hinh_thai       None | co | gian | mo | dong (SE 3x3). Bật thì đầu ra lấy từ mặt nạ nhị phân.
    che_do_thu_nho  dien_tich (chống alias, mặc định) | song_tuyen_tinh (cách cũ, để so sánh).
    keo_gian        chia cho giá trị lớn nhất để nét sáng nhất bằng 1 (như ảnh huấn luyện).
    canh_toi_da     thu nhỏ trước nếu cạnh dài vượt giá trị này; None để tắt.
    """
    bt = lay_bai_toan(bai_toan)
    if hinh_thai is not None and hinh_thai not in PHEP:
        raise LoiXuLyAnh(f"hình thái phải là None hoặc một trong {PHEP}")
    if che_do_thu_nho not in CHE_DO:
        raise LoiXuLyAnh(f"chế độ thu nhỏ phải là một trong {CHE_DO}")
    if not isinstance(chi_giu_lon_nhat, (bool, np.bool_)):
        raise LoiXuLyAnh("chi_giu_lon_nhat phải là bool")
    so_nguyen_duong(kich_thuoc_dau_ra, "kich_thuoc_dau_ra")
    ria = le if le is not None else round(kich_thuoc_dau_ra * bt.le / KICH_THUOC_CNN)
    if isinstance(ria, (bool, np.bool_)) or not isinstance(ria, (int, np.integer)) or ria < 0 \
            or kich_thuoc_dau_ra <= 2 * ria:
        raise LoiXuLyAnh("kich_thuoc_dau_ra phải lớn hơn 2*lề, lề >= 0")

    anh = np.asarray(anh)
    if canh_toi_da:
        anh = gioi_han_kich_thuoc(anh, canh_toi_da)
    xam = chuyen_xam(anh)
    loc = loc_trung_vi(xam, kich_thuoc_loc)
    if loc.max() == loc.min():
        raise AnhDongNhat("Ảnh một màu, không có đối tượng để tách")
    nguong = nguong_otsu(loc)
    net_toi, ti_le_vien, canh_bao = xac_dinh_cuc_tinh(loc, nguong, cuc_tinh)
    mat_na = loc <= nguong if net_toi else loc > nguong
    cac_buoc = {"xam": xam, "loc": loc, "nguong": mat_na.astype(np.uint8) * 255}
    if hinh_thai:
        mat_na = hinh_thai_nhi_phan(mat_na, hinh_thai)
    if chi_giu_lon_nhat:
        mat_na = thanh_phan_lien_thong_lon_nhat(mat_na)
    cac_buoc["mat_na"] = mat_na.astype(np.uint8) * 255
    ys, xs = np.nonzero(mat_na)
    if not len(xs):
        raise KhongCoDoiTuong("Không còn đối tượng: thử tắt lọc/hình thái hoặc đổi cực tính")
    hop_bao = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
    trai, tren, phai, duoi = hop_bao
    # Mức xám của nét lấy từ ảnh đã lọc (để nhiễu không quay lại); khi bật hình thái dùng mặt nạ nhị phân
    # để các pixel mới thêm vào thực sự xuất hiện ở đầu ra.
    tien_canh = (mat_na.astype(np.uint8) * 255 if hinh_thai else (255 - loc if net_toi else loc) * mat_na)
    cat = tien_canh[tren:duoi, trai:phai]
    ti_le = (kich_thuoc_dau_ra - 2 * ria) / max(cat.shape)
    cao, rong = max(1, round(cat.shape[0] * ti_le)), max(1, round(cat.shape[1] * ti_le))
    nho = thu_phong(cat.astype(float), cao, rong, che_do_thu_nho)
    canvas = np.zeros((kich_thuoc_dau_ra, kich_thuoc_dau_ra), dtype=np.float32)
    y, x = (kich_thuoc_dau_ra - cao) // 2, (kich_thuoc_dau_ra - rong) // 2
    canvas[y:y + cao, x:x + rong] = nho / 255
    if not canvas.any():
        raise NetQuaMong("Nét quá mỏng bị mất khi thu nhỏ: cần ảnh rõ hơn hoặc tăng kích thước đầu ra")
    cac_buoc.update(cat=cat, thu_nho=np.rint(nho).astype(np.uint8), dau_ra=np.rint(canvas * 255).astype(np.uint8))
    dau_vao = canvas
    if keo_gian and canvas.max() > 0:
        dau_vao = np.clip(canvas / canvas.max(), 0, 1).astype(np.float32)
    cac_buoc["dau_vao_cnn"] = np.rint(dau_vao * 255).astype(np.uint8)
    return KetQuaXuLy(dau_vao, cac_buoc, hop_bao, nguong, "net_toi" if net_toi else "net_sang", ti_le_vien,
                      [canh_bao] if canh_bao else [])
