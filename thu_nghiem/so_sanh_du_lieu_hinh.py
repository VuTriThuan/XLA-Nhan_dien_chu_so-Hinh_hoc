"""So sánh dữ liệu huấn luyện hình học: đơn giản (tam giác đều, chữ nhật 0,55-1, tròn hoàn hảo) và đa dạng.

Cùng kiến trúc, công thức, hạt và tiền xử lý; chỉ khác bộ sinh dữ liệu và số ảnh. Hai mô hình phụ được huấn luyện trong thư
mục tạm (không đụng tới mo_hinh/ chính) rồi cả ba (cùng mô hình chính 13.500 ảnh) được đo trên các bộ giống nhau.
Mọi bộ đo đều do chương trình tạo, KHÔNG phải hình người vẽ thật.
"""
import tempfile
from pathlib import Path

import numpy as np

from loi.cau_hinh import HINH_HOC
from loi.du_doan import DuDoan, duong_dan_mo_hinh
from bo_du_lieu import sinh_bo_du_lieu, tai_hinh_hoc
from loi.huan_luyen import huan_luyen
from tien_xu_ly import LoiXuLyAnh, xu_ly_anh

from ._chung import ghi_ket_qua, in_bang, thanh_xam_rgb
from .ve_pil import HAT, HO_HINH, LOP, TI_LE_NET, ve_hinh_pil

SO_ANH_PIL = 40


def dung_tren_tap(du_doan, x, y):
    return float((du_doan.xac_suat(x).argmax(1) == y).mean())


def dung_ve_pil(du_doan):
    ket_qua = {}
    for ho in HO_HINH:
        dung = tong = 0
        for si, ti_le in enumerate(TI_LE_NET):
            rng = np.random.default_rng(HAT + 100 * si + HO_HINH.index(ho))
            for _ in range(SO_ANH_PIL):
                anh = thanh_xam_rgb(ve_hinh_pil(ho, ti_le, rng))
                try:
                    dung += du_doan.du_doan_dau_vao(xu_ly_anh(anh, "hinh_hoc").anh)["lop"] == LOP[ho]
                except LoiXuLyAnh:
                    pass
                tong += 1
        ket_qua[ho] = dung / tong
    return ket_qua


def chay():
    cau_hinh = {"don_gian_4500": dict(da_dang=False, ti_le=1), "da_dang_4500": dict(da_dang=True, ti_le=1)}
    mo_hinh = {}
    with tempfile.TemporaryDirectory() as tam:
        tam = Path(tam)
        for ten, kw in cau_hinh.items():
            thu_muc_dl = tam / ten / "du_lieu"
            print(f"Sinh dữ liệu {ten} ...", flush=True)
            sinh_bo_du_lieu(thu_muc_dl, anh_mau=None, **kw)
            huan_luyen("hinh_hoc", "cai_tien", so_luong=4, in_ra=False,
                       nguon_du_lieu=lambda bt, tap, d=thu_muc_dl: tai_hinh_hoc(tap, d),
                       thu_muc_mo_hinh=tam / ten / "mo_hinh", thu_muc_ket_qua=tam / ten / "ket_qua")
            mo_hinh[ten] = (DuDoan("hinh_hoc", HINH_HOC.tep_mo_hinh("cai_tien", tam / ten / "mo_hinh")), thu_muc_dl)
        mo_hinh["da_dang_13500 (mô hình chính)"] = (DuDoan("hinh_hoc", duong_dan_mo_hinh("hinh_hoc", "cai_tien")), None)
        x_don, y_don = tai_hinh_hoc("test", mo_hinh["don_gian_4500"][1])
        x_da, y_da = tai_hinh_hoc("test")
        x_st, y_st = tai_hinh_hoc("stress")
        ket_qua, bang = {}, []
        for ten, (du_doan, _) in mo_hinh.items():
            pil = dung_ve_pil(du_doan)
            ket_qua[ten] = {"test_du_lieu_don_gian": dung_tren_tap(du_doan, x_don, y_don),
                            "test_du_lieu_da_dang": dung_tren_tap(du_doan, x_da, y_da),
                            "stress_da_dang": dung_tren_tap(du_doan, x_st, y_st),
                            "ve_pil_theo_ho": pil, "ve_pil_trung_binh": float(np.mean(list(pil.values())))}
            bang.append([ten, f"{ket_qua[ten]['test_du_lieu_don_gian']:.1%}", f"{ket_qua[ten]['test_du_lieu_da_dang']:.1%}",
                         f"{ket_qua[ten]['stress_da_dang']:.1%}", f"{ket_qua[ten]['ve_pil_trung_binh']:.1%}"] +
                        [f"{pil[h]:.0%}" for h in HO_HINH])
    ghi_ket_qua("so_sanh_du_lieu_hinh", {
        "mo_ta": "Cùng kiến trúc/công thức/hạt/tiền xử lý; chỉ khác bộ sinh dữ liệu. Mô hình phụ huấn luyện trong thư mục tạm.",
        "so_anh_pil_moi_ho": SO_ANH_PIL * len(TI_LE_NET), "ho_hinh": list(HO_HINH), "ket_qua": ket_qua,
        "luu_y": "Mọi bộ đo là ảnh tổng hợp; bộ vẽ PIL cùng họ hình với dữ liệu đa dạng nên đo khả năng tổng quát hóa sang bộ vẽ khác, "
                 "không phải sang hình người vẽ thật hay hình lạ. Một hạt mỗi cấu hình."},
              [{"mo_hinh": t, **{k: v for k, v in r.items() if k != "ve_pil_theo_ho"},
                **{f"pil_{h}": v for h, v in r["ve_pil_theo_ho"].items()}} for t, r in ket_qua.items()])
    in_bang("So sánh dữ liệu huấn luyện hình học",
            ["mô hình", "test đơn giản", "test đa dạng", "stress đa dạng", "PIL (TB)"] + [h[:12] for h in HO_HINH], bang)
    return ket_qua
