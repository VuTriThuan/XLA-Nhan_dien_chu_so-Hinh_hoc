"""Hình vẽ kiểu Paint bằng Pillow, có rung tay, với nhiều độ dày nét; đo ba cấu hình thu nhỏ/chuẩn hóa.

Mã vẽ hoàn toàn độc lập với bộ sinh dữ liệu huấn luyện (khác bộ vẽ, kích thước 300-500 px) nhưng cùng các họ hình;
ảnh do chương trình tạo, KHÔNG phải hình người vẽ thật.
"""
import math

import numpy as np
from PIL import Image, ImageDraw

from loi.du_doan import BoNhanDang

from ._chung import CAU_HINH_TIEN_XU_LY, ghi_ket_qua, in_bang, nhan_dang_hoac_loi, thanh_xam_rgb

HO_HINH = ("tron", "tam_giac_deu", "tam_giac_vuong", "tam_giac_bat_ky", "chu_nhat_vua", "chu_nhat_dai")
LOP = {"tron": 0, "tam_giac_deu": 1, "tam_giac_vuong": 1, "tam_giac_bat_ky": 1, "chu_nhat_vua": 2, "chu_nhat_dai": 2}
TI_LE_NET = (0.012, 0.025, 0.05, 0.09)      # độ dày nét / kích thước hình
SO_ANH = 40
HAT = 424242


def xoay(diem, theta):
    c, s = math.cos(theta), math.sin(theta)
    return diem @ np.array([[c, -s], [s, c]]).T


def diem_hinh(ho, d, rng):
    """Các điểm của đường viền kín quanh gốc (0,0), kích thước ~d, và cờ 'đường cong mượt'."""
    if ho == "tron":
        t = np.linspace(0, 2 * math.pi, 90, endpoint=False)
        r = d / 2 * (1 + 0.04 * np.sin(rng.integers(2, 4) * t + rng.uniform(0, 6.28)))
        return np.column_stack((r * np.cos(t) * rng.uniform(0.8, 1.2), r * np.sin(t))), True
    if ho == "tam_giac_deu":
        g = np.arange(3) * 2 * math.pi / 3 - math.pi / 2
        return np.column_stack((np.cos(g), np.sin(g))) * d / 2, False
    if ho == "tam_giac_vuong":
        w, h = d * rng.uniform(.7, 1), d * rng.uniform(.6, 1)
        return np.array([[-w / 2, h / 2], [w / 2, h / 2], [-w / 2, -h / 2]]), False
    if ho == "tam_giac_bat_ky":
        while True:
            p = rng.uniform(-d / 2, d / 2, (3, 2))
            u, v = p[1] - p[0], p[2] - p[0]
            if abs(u[0] * v[1] - u[1] * v[0]) / 2 > 0.12 * d * d:      # diện tích; np.cross 2D bị bỏ từ NumPy 2
                return p, False
    ti_le = rng.uniform(.55, 1) if ho == "chu_nhat_vua" else rng.uniform(.25, .45)
    w = d / math.sqrt(1 + ti_le ** 2)
    h = w * ti_le
    return np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]), False


def ve_hinh_pil(ho, ti_le_net, rng, nghieng=30, rung=0.015):
    canh = int(rng.integers(300, 500))
    d = canh * rng.uniform(0.4, 0.8)
    diem, muot = diem_hinh(ho, d, rng)
    diem = xoay(diem, math.radians(rng.uniform(-nghieng, nghieng)))
    if not muot:
        diem = diem + rng.normal(0, rung * d, diem.shape)             # rung đỉnh
        day = []
        for i in range(len(diem)):
            a, b = diem[i], diem[(i + 1) % len(diem)]
            for t in np.linspace(0, 1, 20, endpoint=False):
                day.append(a + (b - a) * t + rng.normal(0, rung * d * .25, 2))
        diem = np.array(day)
    thap, cao = diem.min(0), diem.max(0)
    diem = diem + rng.uniform(-canh * .05, canh * .05, 2) + canh / 2 - (thap + cao) / 2
    anh = Image.new("L", (canh, canh), 255)
    vong = [tuple(p) for p in diem]
    ImageDraw.Draw(anh).line(vong + [vong[0]], fill=0, width=max(1, int(round(ti_le_net * d))), joint="curve")
    return np.array(anh)


def chay(so_anh=SO_ANH):
    bo = BoNhanDang()
    ket_qua, dong, bang = {}, [], []
    for ho in HO_HINH:
        hang = [ho]
        ket_qua[ho] = {}
        for si, ti_le in enumerate(TI_LE_NET):
            dung = {ten: 0 for ten in CAU_HINH_TIEN_XU_LY}
            rng = np.random.default_rng(HAT + 100 * si + HO_HINH.index(ho))
            for _ in range(so_anh):
                anh = thanh_xam_rgb(ve_hinh_pil(ho, ti_le, rng))
                for ten, cau_hinh in CAU_HINH_TIEN_XU_LY.items():
                    dung[ten] += nhan_dang_hoac_loi(bo, anh, "hinh_hoc", **cau_hinh) == LOP[ho]
            ket_qua[ho][str(ti_le)] = {t: v / so_anh for t, v in dung.items()}
            dong.append({"ho_hinh": ho, "ti_le_net": ti_le, **{t: round(v / so_anh, 4) for t, v in dung.items()}})
            hang.append("/".join(f"{dung[t] / so_anh:.0%}" for t in CAU_HINH_TIEN_XU_LY))
        bang.append(hang)
    tong = {t: float(np.mean([d[t] for d in dong])) for t in CAU_HINH_TIEN_XU_LY}
    ghi_ket_qua("ve_pil_hinh_hoc", {"mo_ta": "Hình vẽ bằng Pillow có rung tay; ảnh tổng hợp, không phải hình người vẽ thật",
                                    "so_anh_moi_o": so_anh, "hat": HAT, "cau_hinh": list(CAU_HINH_TIEN_XU_LY),
                                    "do_chinh_xac": ket_qua, "trung_binh": tong}, dong)
    in_bang("Hình vẽ PIL: cũ / diện tích / diện tích+kéo giãn", ["họ hình"] + [f"nét {t}" for t in TI_LE_NET], bang)
    print("  trung bình:", {t: f"{v:.1%}" for t, v in tong.items()})
    return ket_qua
