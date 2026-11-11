"""Kiểm thử nạp/sinh dữ liệu và thu ảnh thật."""
import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from loi.cau_hinh import CHU_SO, DU_LIEU, HINH_HOC, doc_json, sha256
from bo_du_lieu import chia_phan_tang, tai_hinh_hoc, tai_meta_hinh_hoc, tai_mnist, tai_tap, ve_hinh
from bo_du_lieu.hinh_hoc import GOC_TAM_GIAC_TOI_THIEU, dau_vao_cnn, dinh_tam_giac
from bo_du_lieu.mnist import SHA256_DUNG
from bo_du_lieu.thu_thap import dem_mau, luu_mau


# ------------------------------------------------------------------------------ MNIST
def test_chia_phan_tang_tai_lap_roi_nhau_va_can_bang():
    with np.load(CHU_SO.thu_muc_du_lieu / "mnist.npz", allow_pickle=False) as d:
        y = d["y_train"]
    a, b = chia_phan_tang(y)
    c, e = chia_phan_tang(y)
    assert np.array_equal(a, c) and np.array_equal(b, e)
    assert len(a) == 54000 and len(b) == 6000
    assert not set(a) & set(b) and set(a) | set(b) == set(range(60000))
    assert np.max(np.abs(np.bincount(y[b]) - np.bincount(y) * .1)) < 1


def test_mnist_dung_checksum_va_chia_tap_da_luu():
    thu_muc = CHU_SO.thu_muc_du_lieu
    assert sha256(thu_muc / "mnist.npz") == SHA256_DUNG
    with np.load(thu_muc / "splits.npz") as s:
        train, val = chia_phan_tang(np.load(thu_muc / "mnist.npz")["y_train"])
        assert np.array_equal(s["train"], train) and np.array_equal(s["validation"], val)


def test_tai_mnist_hinh_dang_va_khong_roi_ri():
    x, y = tai_mnist("validation")
    assert x.shape == (6000, 1, 28, 28) and x.dtype == np.float32 and 0 <= x.min() and x.max() <= 1 and y.dtype == np.int64
    assert tai_mnist("test")[0].shape == (10000, 1, 28, 28)
    with pytest.raises(ValueError):
        tai_mnist("stress")


# ------------------------------------------------------------------------------ bộ sinh hình
def test_sinh_hinh_tai_lap_va_trong_gioi_han():
    for lop in range(3):
        a, ma = ve_hinh(lop, np.random.default_rng(17))
        b, mb = ve_hinh(lop, np.random.default_rng(17))
        assert np.array_equal(a, b) and ma == mb
        assert a.shape == (28, 28) and a.dtype == np.uint8
        assert 7 <= ma["duong_kinh"] <= 21 and 8.4 <= ma["tam_x"] <= 19.6 and 8.4 <= ma["tam_y"] <= 19.6


def test_tam_giac_du_loai_va_goc_hop_le():
    rng = np.random.default_rng(5)
    cac_loai = set()
    for _ in range(400):
        loai, v = dinh_tam_giac(rng, 10.0)
        cac_loai.add(loai)
        assert np.allclose(np.hypot(v[:, 0], v[:, 1]), 10.0)                 # đỉnh nằm trên đường tròn ngoại tiếp
        canh = [np.linalg.norm(v[i] - v[(i + 1) % 3]) for i in range(3)]
        goc = []
        for i in range(3):                                                    # góc đối diện cạnh i (định lý cos)
            a, b, c = canh[i], canh[(i + 1) % 3], canh[(i + 2) % 3]
            goc.append(np.degrees(np.arccos(np.clip((b * b + c * c - a * a) / (2 * b * c), -1, 1))))
        assert min(goc) >= GOC_TAM_GIAC_TOI_THIEU - 1e-6 and abs(sum(goc) - 180) < 1e-6
        if loai == "deu":
            assert np.allclose(goc, 60)
        if loai == "vuong":
            assert abs(max(goc) - 90) < 1e-6
        if loai == "can":
            assert min(abs(goc[i] - goc[j]) for i in range(3) for j in range(i + 1, 3)) < 1e-6
    assert cac_loai == {"deu", "vuong", "can", "bat_ky"}


def test_cac_bien_the_khoang_gia_tri_va_tai_lap_stress():
    rng = np.random.default_rng(8)
    thay = {0: set(), 1: set(), 2: set()}
    for i in range(600):
        lop = i % 3
        tho, m = ve_hinh(lop, rng)
        thay[lop].add(m["bien_the"])
        if lop == 2:
            assert .29 <= m["ti_le_canh"] <= 1.0
        if lop == 0:
            assert .7 <= m["ti_le_canh"] <= 1.0
        assert 0 <= m["do_lech_dinh"] <= .035 + 1e-9
    assert {"tron", "elip", "tron_luon_song", "elip_luon_song"} <= thay[0]
    assert {"chu_nhat", "vuong"} <= thay[2] and len(thay[1]) == 4
    a, ma = ve_hinh(1, np.random.default_rng(3), stress=True)
    b, mb = ve_hinh(1, np.random.default_rng(3), stress=True)
    assert np.array_equal(a, b) and ma == mb


def test_goc_va_do_day_bat_buoc_duoc_ton_trong():
    for lop in (1, 2):
        _, m = ve_hinh(lop, np.random.default_rng(4), goc=123.0, do_day=2)
        assert m["goc_do"] == 123.0 and m["do_day"] == 2.0


def test_dau_vao_cnn_la_dau_ra_cua_tien_xu_ly_thong_nhat():
    from tien_xu_ly import xu_ly_anh
    tho, _ = ve_hinh(1, np.random.default_rng(11))
    np.testing.assert_array_equal(dau_vao_cnn(tho), np.rint(xu_ly_anh(tho, "hinh_hoc").anh * 255).astype(np.uint8))


@pytest.mark.skipif(not (HINH_HOC.thu_muc_du_lieu / "shapes_dataset.npz").exists(), reason="chưa sinh dữ liệu")
def test_bo_du_lieu_hinh_hoc_dem_roi_nhau_khong_trung():
    import hashlib
    ti_le = doc_json(HINH_HOC.thu_muc_du_lieu / "data_stats.json")["ti_le"]
    with np.load(HINH_HOC.thu_muc_du_lieu / "shapes_dataset.npz", allow_pickle=False) as d:
        chi_so = [set(d[t].tolist()) for t in ("train", "validation", "test")]
        assert [len(s) for s in chi_so] == [3150 * ti_le, 675 * ti_le, 675 * ti_le]
        assert not (chi_so[0] & chi_so[1] or chi_so[0] & chi_so[2] or chi_so[1] & chi_so[2])
        for tap, moi_lop in (("train", 1050), ("validation", 225), ("test", 225)):
            assert np.array_equal(np.bincount(d["labels"][d[tap]], minlength=3), [moi_lop * ti_le] * 3)
        assert len({hashlib.sha256(a.tobytes()).hexdigest() for a in d["images"]}) == 4500 * ti_le
        assert d["inputs"].shape == (4500 * ti_le, 28, 28)
        vao = [{hashlib.sha256(a.tobytes()).hexdigest() for a in d["inputs"][d[t]]} for t in ("train", "validation", "test")]
        assert not (vao[0] & vao[1] or vao[0] & vao[2] or vao[1] & vao[2])


@pytest.mark.skipif(not (HINH_HOC.thu_muc_du_lieu / "shapes_dataset.npz").exists(), reason="chưa sinh dữ liệu")
def test_tai_hinh_hoc_va_meta_cung_thu_tu():
    x, y = tai_tap("hinh_hoc", "test")
    meta = tai_meta_hinh_hoc("test")
    assert x.shape[1:] == (1, 28, 28) and len(meta) == len(y)
    assert all(m["lop"] == lop for m, lop in zip(meta[:50], y[:50]))
    xs, ys = tai_tap("hinh_hoc", "stress")
    assert len(ys) == 300 == len(tai_meta_hinh_hoc("stress"))


# ------------------------------------------------------------------------------ thu ảnh thật
def anh_png(ve):
    im = Image.new("RGB", (280, 280), "white")
    ve(ImageDraw.Draw(im))
    bo = io.BytesIO()
    im.save(bo, format="PNG")
    return bo.getvalue()


def test_luu_mau_ghi_anh_va_tep_nguon(tmp_path):
    du_lieu = anh_png(lambda d: d.ellipse([50, 50, 230, 230], outline="black", width=6))
    kq = luu_mau("hinh_hoc", "tron", du_lieu, "nguoi_01", "ve_tren_khung", tmp_path)
    assert kq["so_luong"] == {"tron": 1, "tam_giac": 0, "chu_nhat": 0}
    png = next((tmp_path / "tron").glob("*.png"))
    meta = doc_json(png.with_suffix(".json"))
    assert meta["lop"] == "tron" and meta["nguoi_ve"] == "nguoi_01" and meta["nhan_ghi_truoc_khi_du_doan"] is True
    assert meta["sha256"] == sha256(png)
    assert dem_mau("hinh_hoc", tmp_path)["tron"] == 1


def test_luu_mau_tu_choi_dau_vao_sai(tmp_path):
    tot = anh_png(lambda d: d.ellipse([50, 50, 230, 230], outline="black", width=6))
    for kw in (dict(bai_toan="hinh_hoc", nhan="sao", du_lieu_anh=tot), dict(bai_toan="chu_so", nhan="10", du_lieu_anh=tot),
               dict(bai_toan="hinh_hoc", nhan="tron", du_lieu_anh=tot, nguon="khac"),
               dict(bai_toan="hinh_hoc", nhan="tron", du_lieu_anh=b"khong phai anh"),
               dict(bai_toan="hinh_hoc", nhan="tron", du_lieu_anh=b""),
               dict(bai_toan="hinh_hoc", nhan="tron", du_lieu_anh=anh_png(lambda d: None)),          # ảnh trống
               dict(bai_toan="khong", nhan="tron", du_lieu_anh=tot)):
        with pytest.raises(ValueError):
            luu_mau(thu_muc=tmp_path, **kw)
    assert not list(tmp_path.rglob("*.png"))
