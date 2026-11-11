"""Kiểm thử đánh giá: chỉ số, đánh giá ảnh thật (trùng, ảnh hỏng, trạng thái), chẩn đoán."""
import numpy as np
import pytest
from PIL import Image

from loi.cau_hinh import CHU_SO, HINH_HOC
from loi.danh_gia import (chi_so_phan_loai, chi_so_theo_nhom, danh_gia_anh_that, khoang_wilson,
                                tap_chan_doan_chu_so)
from bo_du_lieu import tai_mnist, ve_hinh


def test_chi_so_truong_hop_biet_truoc():
    m = chi_so_phan_loai([0, 0, 1, 1], [0, 1, 1, 1], HINH_HOC)
    assert m["so_dung"] == 3 and m["do_chinh_xac"] == .75 and m["so_sai"] == 1
    lop = {r["lop"]: r for r in m["theo_lop"]}
    assert lop["tron"]["recall"] == .5 and lop["tam_giac"]["precision"] == pytest.approx(2 / 3)
    assert lop["chu_nhat"]["so_mau"] == 0 and lop["chu_nhat"]["f1"] == 0              # lớp không có mẫu: không chia cho 0
    assert m["ma_tran_nham_lan"] == [[1, 1, 0], [0, 2, 0], [0, 0, 0]]


def test_anh_khong_hop_le_tinh_la_sai_va_khong_vao_ma_tran():
    m = chi_so_phan_loai([0, 1, 2], [0, -1, 2], HINH_HOC)
    assert m["do_chinh_xac"] == pytest.approx(2 / 3) and m["so_khong_hop_le"] == 1
    assert m["theo_lop"][1]["recall"] == 0 and sum(map(sum, m["ma_tran_nham_lan"])) == 2


def test_khoang_wilson_hop_ly():
    thap, cao = khoang_wilson(9915, 10000)
    assert thap < .9915 < cao and 0 <= thap and cao <= 1 and cao - thap < .005
    assert khoang_wilson(0, 0) == [0.0, 0.0]
    assert khoang_wilson(10, 10)[1] == pytest.approx(1.0)


def test_chi_so_cho_chu_so_dung_10_lop():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 10, 500)
    m = chi_so_phan_loai(y, y, CHU_SO)
    assert m["do_chinh_xac"] == 1 and m["f1_macro"] == 1 and len(m["theo_lop"]) == 10


def test_chi_so_theo_nhom_bo_qua_goc_cua_hinh_tron():
    y = np.array([0, 1, 1, 2])
    du_doan = np.array([0, 1, 0, 2])
    meta = [{"goc_do": 10.0, "do_day": 1.0}, {"goc_do": 10.0, "do_day": 2.0}, {"goc_do": 70.0, "do_day": 2.0},
            {"goc_do": 70.0, "do_day": 3.0}]
    nhom = {(r["dac_trung"], r["tu"]): r for r in chi_so_theo_nhom(y, du_doan, meta)}
    assert nhom[("goc_do", 0)]["so_anh"] == 1 and nhom[("goc_do", 0)]["do_chinh_xac"] == 1    # hình tròn không tính vào góc
    assert nhom[("goc_do", 60)]["so_anh"] == 2 and nhom[("goc_do", 60)]["do_chinh_xac"] == .5


class DuDoanGia:
    """Luôn đoán 'tron'; dùng để kiểm tra cách tính, không cần mô hình thật."""
    def du_doan_dau_vao(self, x):
        return {"lop": 0}


def test_anh_that_thieu_trung_va_anh_hong(tmp_path):
    goc, ra = tmp_path / "anh", tmp_path / "ra"
    for ten in HINH_HOC.lop:
        (goc / ten).mkdir(parents=True)
    ra.mkdir()
    a, _ = ve_hinh(0, np.random.default_rng(100))
    b, _ = ve_hinh(2, np.random.default_rng(101))
    Image.fromarray(a).save(goc / "tron" / "a.png")
    (goc / "tron" / "trung.png").write_bytes((goc / "tron" / "a.png").read_bytes())
    Image.fromarray(np.zeros((28, 28), np.uint8)).save(goc / "tam_giac" / "trang.png")
    Image.fromarray(b).save(goc / "chu_nhat" / "b.png")
    kq = danh_gia_anh_that(HINH_HOC, DuDoanGia(), ra, goc)
    assert kq["trang_thai"] == "partial" and kq["so_anh"] == 3 and kq["so_anh_trung_bi_bo_qua"] == 1
    assert kq["chi_so"]["do_chinh_xac"] == pytest.approx(1 / 3) and kq["chi_so"]["so_khong_hop_le"] == 1
    assert len(kq["anh_loi_dau_vao"]) == 1 and (ra / "external_metrics.json").exists()


def test_anh_that_trong_va_du_so_luong(tmp_path):
    kq = danh_gia_anh_that(HINH_HOC, DuDoanGia(), tmp_path, tmp_path / "khong_co")
    assert kq["trang_thai"] == "pending" and kq["so_anh"] == 0 and kq["chi_so"] is None
    goc = tmp_path / "du"
    rng = np.random.default_rng(0)
    for lop, ten in enumerate(HINH_HOC.lop):
        (goc / ten).mkdir(parents=True)
        for i in range(10):
            Image.fromarray(ve_hinh(lop, rng)[0]).save(goc / ten / f"{i}.png")
    kq = danh_gia_anh_that(HINH_HOC, DuDoanGia(), tmp_path, goc)
    assert kq["trang_thai"] == "ready" and kq["so_anh"] == 30


def test_tap_chan_doan_chu_so_co_dinh_va_phan_tang():
    x, y = tai_mnist("test")
    chi_so, dieu_kien = tap_chan_doan_chu_so(x, y)
    assert len(chi_so) == 1000 and np.array_equal(np.bincount(y[chi_so]), [100] * 10)
    assert set(dieu_kien) == {"goc", "dich_3px", "xoay_20_do", "nhieu_gauss_0.15"}
    chi_so2, dk2 = tap_chan_doan_chu_so(x, y)
    assert np.array_equal(chi_so, chi_so2) and np.array_equal(dieu_kien["dich_3px"], dk2["dich_3px"])
    assert np.array_equal(dieu_kien["goc"][:, 0], x[chi_so, 0])
