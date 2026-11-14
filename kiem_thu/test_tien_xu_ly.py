"""Kiểm thử tiền xử lý: đối chiếu độc lập bằng vòng lặp, đáp án dựng tay và các hồi quy đã biết.

CU = chế độ cũ (nội suy song tuyến tính, không kéo giãn) để khẳng định giá trị pixel chính xác của bản gốc.
"""
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from tien_xu_ly import (AnhDongNhat, KetQuaXuLy, KhongCoDoiTuong, LoiXuLyAnh, NetQuaMong, chuyen_xam,
                                  doc_anh, gioi_han_kich_thuoc, hinh_thai_nhi_phan, loc_trung_vi, nguong_otsu,
                                  noi_suy_song_tuyen_tinh, thanh_phan_lien_thong_lon_nhat,
                                  thu_nho_trung_binh_dien_tich, thu_phong, xu_ly_anh)

CU = dict(che_do_thu_nho="song_tuyen_tinh", keo_gian=False)
DU_LIEU_THU = Path(__file__).parent / "du_lieu_thu"


def mau_chu_L():
    a = np.full((80, 100), 255, dtype=np.uint8)
    a[15:65, 35:45] = 0
    a[15:25, 35:70] = 0
    return a


def khoi_vuong():
    a = np.full((25, 25), 255, np.uint8)
    a[6:19, 6:19] = 0
    return a


# ------------------------------------------------------------------------------ luồng xử lý
def test_dau_ra_tuong_duong_khi_dao_cuc_tinh_va_hop_bao():
    a = mau_chu_L()
    r = xu_ly_anh(a)
    np.testing.assert_allclose(r.anh, xu_ly_anh(255 - a).anh)
    assert r.hop_bao == (35, 15, 70, 65)
    assert r.tensor("NHWC").shape == (1, 28, 28, 1) and r.tensor("NCHW").shape == (1, 1, 28, 28)
    assert r.anh.dtype == np.float32 and 0 <= r.anh.min() <= r.anh.max() == 1.0
    assert (r.cuc_tinh, xu_ly_anh(255 - a).cuc_tinh) == ("net_toi", "net_sang")
    with pytest.raises(LoiXuLyAnh):
        r.tensor("sai")


def test_bat_bien_theo_dich_chuyen_va_nhieu_roi_rac():
    a = mau_chu_L()
    np.testing.assert_allclose(xu_ly_anh(a).anh, xu_ly_anh(np.roll(a, 5, axis=1)).anh)
    b = a.copy()
    b[2, 2] = 0
    np.testing.assert_allclose(xu_ly_anh(a).anh, xu_ly_anh(b).anh)


@pytest.mark.parametrize("gia_tri", [0, 255, 100])
def test_anh_dong_nhat_bi_tu_choi(gia_tri):
    with pytest.raises(AnhDongNhat):
        xu_ly_anh(np.full((20, 20), gia_tri, dtype=np.uint8))


def test_nen_trong_suot_va_float():
    a = mau_chu_L()
    rgba = np.zeros((80, 100, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255 - a
    np.testing.assert_allclose(xu_ly_anh(a).anh, xu_ly_anh(rgba).anh)
    np.testing.assert_allclose(xu_ly_anh(a).anh, xu_ly_anh(a.astype(float) / 255).anh)


def test_hinh_hoc_kich_thuoc_64_va_le():
    r = xu_ly_anh(mau_chu_L(), "hinh_hoc", kich_thuoc_dau_ra=64, le=6)
    assert r.anh.shape == (64, 64) and not r.anh[:6].any()


@pytest.mark.parametrize("bai_toan,noi_dung", [("chu_so", 20), ("hinh_hoc", 22)])
def test_hop_dong_dau_vao_cnn(bai_toan, noi_dung):
    r = xu_ly_anh(mau_chu_L(), bai_toan)
    ys, xs = np.nonzero(r.anh)
    assert r.anh.shape == (28, 28) and max(np.ptp(ys), np.ptp(xs)) + 1 <= noi_dung
    assert r.anh[:(28 - noi_dung) // 2].sum() == 0 and r.anh[:, :(28 - noi_dung) // 2].sum() == 0


def test_anh_nho_va_hop_dong_can_giua():
    a = np.array([[255, 255], [255, 0]], np.uint8)
    for bai_toan, n, le in (("chu_so", 28, 4), ("hinh_hoc", 64, 6)):
        r = xu_ly_anh(a, bai_toan, cuc_tinh="net_toi", kich_thuoc_dau_ra=n, le=le)
        mong_doi = np.zeros((n, n), np.float32)
        mong_doi[le:n - le, le:n - le] = 1
        np.testing.assert_allclose(r.anh, mong_doi)
    with pytest.raises(AnhDongNhat):
        xu_ly_anh(np.zeros((1, 1), np.uint8))


def test_ti_le_khung_hinh_hop_bao_va_tam_giua():
    a = np.full((30, 40), 255, np.uint8)
    a[3:13, 7:27] = 0
    r = xu_ly_anh(a)
    assert r.hop_bao == (7, 3, 27, 13) and r.cac_buoc["thu_nho"].shape == (10, 20)
    mong_doi = np.zeros((28, 28), np.float32)
    mong_doi[9:19, 4:24] = 1
    np.testing.assert_array_equal(r.anh, mong_doi)
    np.testing.assert_array_equal(r.tensor("NCHW")[0, 0], r.tensor("NHWC")[0, :, :, 0])


def test_cuc_tinh_mo_ho_co_canh_bao_va_ghi_de():
    a = np.full((60, 60), 255, np.uint8)
    a[:40] = 0
    r = xu_ly_anh(a)
    assert r.cuc_tinh == "net_sang" and r.hop_bao == (0, 40, 60, 60)      # cảnh báo không tự sửa lựa chọn sai
    assert len(r.canh_bao) == 1 and "Cực tính" in r.canh_bao[0]
    assert r.ti_le_vien_toi == pytest.approx(140 / 240)
    for khung, cuc_tinh, ky_vong in ((a, "net_toi", "net_toi"), (255 - a, "net_sang", "net_sang")):
        r2 = xu_ly_anh(khung, cuc_tinh=cuc_tinh)
        assert r2.canh_bao == [] and r2.hop_bao == (0, 0, 60, 40) and r2.cuc_tinh == ky_vong


def test_anh_sach_khong_canh_bao_va_khong_doi_pixel():
    a = mau_chu_L()
    for khung, cuc_tinh, ti_le in ((a, "net_toi", 0.0), (255 - a, "net_sang", 1.0)):
        r = xu_ly_anh(khung, **CU)
        assert r.canh_bao == [] and r.cuc_tinh == cuc_tinh and r.ti_le_vien_toi == ti_le
        np.testing.assert_array_equal(r.anh, xu_ly_anh(khung, cuc_tinh=cuc_tinh, **CU).anh)
        assert float(r.anh.sum()) == pytest.approx(120.0)


def test_cuc_tinh_ghi_de_khi_vat_chiem_vien():
    a = np.zeros((20, 20), np.uint8)
    a[5:15, 5:15] = 255
    assert xu_ly_anh(a).hop_bao == (5, 5, 15, 15)
    assert xu_ly_anh(a, cuc_tinh="net_toi").hop_bao == (0, 0, 20, 20)


def test_net_dut_va_net_mong():
    a = np.full((30, 30), 255, np.uint8)
    a[4:12, 14] = 0
    a[15:26, 14] = 0
    assert xu_ly_anh(a).hop_bao == (14, 15, 15, 26)                         # chỉ giữ đoạn lớn nhất
    assert xu_ly_anh(a, chi_giu_lon_nhat=False).hop_bao == (14, 4, 15, 26)
    with pytest.raises(AnhDongNhat):                                          # median 3x3 xóa nét 1 pixel
        xu_ly_anh(a, kich_thuoc_loc=3)


def test_loc_va_hinh_thai_sua_duoc_nhieu_den_dau_ra():
    a = khoi_vuong()
    a[12, 12] = 255
    for kw in (dict(kich_thuoc_loc=3), dict(hinh_thai="dong")):
        r = xu_ly_anh(a, **kw)
        assert r.cac_buoc["cat"][6, 6] == 255 and r.anh[14, 14] > .9


def test_hinh_thai_cho_dau_ra_nhi_phan():
    mo = np.full((60, 60), 255, dtype=np.uint8)
    mo[15:45, 20:40] = 200
    r = xu_ly_anh(mo, hinh_thai="dong")
    assert set(np.unique(r.cac_buoc["cat"]).tolist()) <= {0, 255}


def test_vi_du_chu_so_nhieu_khop_dau_ra_da_luu():
    dau_vao = np.array(Image.open(DU_LIEU_THU / "digit_noise" / "input.png"))
    da_luu = np.array(Image.open(DU_LIEU_THU / "digit_noise" / "07_output.png"))
    r = xu_ly_anh(dau_vao, kich_thuoc_loc=3, **CU)
    np.testing.assert_array_equal(r.cac_buoc["dau_ra"], da_luu)
    assert int(r.cac_buoc["dau_ra"][7, 18]) == 93 and int(r.cac_buoc["dau_ra"][9, 17]) == 224


def test_chan_canvas_rong_chi_o_che_do_cu():
    a = np.full((300, 300), 255, dtype=np.uint8)
    a[10, 10] = 0
    a[290, 290] = 0
    with pytest.raises(NetQuaMong):                       # song tuyến tính bỏ sót hai chấm khi thu nhỏ 14 lần
        xu_ly_anh(a, chi_giu_lon_nhat=False, **CU)
    assert xu_ly_anh(a, chi_giu_lon_nhat=False).anh.max() == 1.0   # trung bình diện tích giữ được chúng


def test_khong_con_doi_tuong():
    a = np.full((20, 20), 255, np.uint8)
    a[0:2, 0:2] = 0                                 # khối 2x2 bị phép co 3x3 xóa hết
    with pytest.raises(KhongCoDoiTuong):
        xu_ly_anh(a, hinh_thai="co", cuc_tinh="net_toi")


@pytest.mark.parametrize("a", [np.empty((0, 2)), np.zeros((3, 3, 2)), np.array([[np.nan]]), np.array([[np.inf]]),
                               np.array([[-1]]), np.array([[256]]), np.ones((2, 2), complex), np.array([["x"]])])
def test_anh_khong_hop_le(a):
    with pytest.raises(ValueError):
        xu_ly_anh(a)


@pytest.mark.parametrize("kw", [dict(kich_thuoc_loc=3.0), dict(kich_thuoc_loc=True), dict(kich_thuoc_loc=0),
                                dict(kich_thuoc_loc=2), dict(kich_thuoc_dau_ra=True), dict(kich_thuoc_dau_ra=0),
                                dict(kich_thuoc_dau_ra=28.0), dict(le=-1), dict(le=14), dict(le=False),
                                dict(hinh_thai=""), dict(hinh_thai="sai"), dict(chi_giu_lon_nhat="yes"),
                                dict(bai_toan="sai"), dict(cuc_tinh="sai"), dict(che_do_thu_nho="sai")])
def test_tham_so_khong_hop_le(kw):
    with pytest.raises(ValueError):
        xu_ly_anh(khoi_vuong(), **kw)


def test_canh_toi_da_thu_nho_anh_lon_truoc():
    im = Image.new("L", (2400, 1800), 255)
    ImageDraw.Draw(im).ellipse([500, 300, 1900, 1500], outline=0, width=40)
    r = xu_ly_anh(np.array(im))
    assert r.cac_buoc["xam"].shape[0] <= 512 and r.cac_buoc["xam"].shape[1] <= 512


# ------------------------------------------------------------------------------ từng bước
def test_chuyen_xam_mau_co_ban_va_alpha():
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], np.uint8)
    np.testing.assert_array_equal(chuyen_xam(rgb), [[76, 150, 29]])
    np.testing.assert_array_equal(chuyen_xam(np.array([[[0, 0, 0, 0], [0, 0, 0, 255], [0, 0, 0, 128]]], np.uint8)),
                                  [[255, 0, 127]])
    a = khoi_vuong()
    np.testing.assert_array_equal(chuyen_xam(a.astype(float)), a)


def test_doc_anh_nen_trong_suot_thanh_trang():
    bo = io.BytesIO()
    Image.fromarray(np.zeros((6, 6, 4), np.uint8)).save(bo, format="PNG")
    rgb = doc_anh(bo.getvalue())
    assert rgb.shape == (6, 6, 3) and (rgb == 255).all()


def test_loc_trung_vi_voi_tham_chieu_vong_lap():
    rng = np.random.default_rng(17)
    a = rng.integers(0, 256, (7, 9), dtype=np.uint8)
    for co in (1, 3, 5):
        r = co // 2
        mong_doi = np.zeros_like(a)
        for y in range(7):
            for x in range(9):
                v = sorted(int(a[min(6, max(0, y + dy)), min(8, max(0, x + dx))])
                           for dy in range(-r, r + 1) for dx in range(-r, r + 1))
                mong_doi[y, x] = v[len(v) // 2]
        np.testing.assert_array_equal(loc_trung_vi(a, co), mong_doi)
    nhieu = np.full((5, 5), 100, dtype=np.uint8)
    nhieu[2, 2] = 255
    np.testing.assert_array_equal(loc_trung_vi(nhieu), np.full((5, 5), 100))


def test_otsu_voi_tham_chieu_trung_binh_hai_lop():
    rng = np.random.default_rng(22)
    for _ in range(12):
        a = rng.integers(0, 256, (9, 13), dtype=np.uint8)
        diem = []
        for t in range(256):
            thap, cao = a[a <= t], a[a > t]
            diem.append(len(thap) * len(cao) * (thap.mean() - cao.mean()) ** 2 if len(thap) and len(cao) else -1)
        assert diem[nguong_otsu(a)] == pytest.approx(max(diem), abs=1e-5)
    assert nguong_otsu(np.array([[0, 0, 255, 255]], np.uint8)) == 0


def tham_chieu_hinh_thai(mat_na, phep, co):
    """Vòng lặp Boolean độc lập: mở rộng đủ miền cho cả phép ghép, cắt một lần ở cuối."""
    r = co // 2
    dem = 2 * r + 1
    nguon = np.pad(mat_na, dem)

    def nguyen_thuy(a, gian):
        kq = np.zeros_like(a)
        for y in range(r, a.shape[0] - r):
            for x in range(r, a.shape[1] - r):
                v = [a[y + dy, x + dx] for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
                kq[y, x] = any(v) if gian else all(v)
        return kq

    if phep == "gian":
        kq = nguyen_thuy(nguon, True)
    elif phep == "co":
        kq = nguyen_thuy(nguon, False)
    elif phep == "mo":
        kq = nguyen_thuy(nguyen_thuy(nguon, False), True)
    else:
        kq = nguyen_thuy(nguyen_thuy(nguon, True), False)
    return kq[dem:-dem, dem:-dem]


@pytest.mark.parametrize("co", [1, 3, 5])
@pytest.mark.parametrize("phep", ["co", "gian", "mo", "dong"])
def test_hinh_thai_voi_tham_chieu_vong_lap(phep, co):
    rng = np.random.default_rng(25)
    for a in (np.ones((5, 7), bool), np.zeros((5, 7), bool), rng.random((7, 9)) > .4):
        np.testing.assert_array_equal(hinh_thai_nhi_phan(a, phep, co), tham_chieu_hinh_thai(a, phep, co))


def tham_chieu_tung_buoc(mat_na, phep, co=3):
    """Quy ước biên khác: cắt về miền ảnh gốc sau từng bước rồi đệm 0 lại."""
    def nt(a, gian):
        r = co // 2
        cua = np.lib.stride_tricks.sliding_window_view(np.pad(a, r, constant_values=False), (co, co))
        return cua.any(axis=(-2, -1)) if gian else cua.all(axis=(-2, -1))
    return {"co": lambda m: nt(m, False), "gian": lambda m: nt(m, True),
            "mo": lambda m: nt(nt(m, False), True), "dong": lambda m: nt(nt(m, True), False)}[phep](mat_na)


def test_dong_khac_tham_chieu_tung_buoc_o_bien_nhung_giu_vat_cham_vien():
    rng = np.random.default_rng(3)
    for co in (3, 5, 7):
        r = co // 2
        for _ in range(6):
            m = rng.random((20, 22)) < 0.45
            np.testing.assert_array_equal(hinh_thai_nhi_phan(m, "dong", co)[r:-r, r:-r],
                                          tham_chieu_tung_buoc(m, "dong", co)[r:-r, r:-r])
            np.testing.assert_array_equal(hinh_thai_nhi_phan(m, "mo", co), tham_chieu_tung_buoc(m, "mo", co))
    mot = np.ones((5, 5), dtype=bool)
    assert int(hinh_thai_nhi_phan(mot, "dong").sum()) == 25 and int(tham_chieu_tung_buoc(mot, "dong").sum()) == 9


def test_hinh_thai_dap_an_biet_truoc():
    cham = np.zeros((7, 7), dtype=bool)
    cham[3, 3] = True
    assert hinh_thai_nhi_phan(cham, "gian").sum() == 9 and hinh_thai_nhi_phan(cham, "mo").sum() == 0
    khoi = np.zeros((7, 7), dtype=bool)
    khoi[1:6, 1:6] = True
    khoi[3, 3] = False
    assert hinh_thai_nhi_phan(khoi, "dong")[3, 3]
    with pytest.raises(LoiXuLyAnh):
        hinh_thai_nhi_phan(cham, "sai")


def test_thanh_phan_lien_thong():
    cheo = np.eye(5, dtype=bool)
    np.testing.assert_array_equal(thanh_phan_lien_thong_lon_nhat(cheo), cheo)          # 8 láng giềng
    a = np.zeros((8, 8), bool)
    a[1, 1:3] = 1
    a[5, 5:7] = 1
    mong_doi = np.zeros_like(a)
    mong_doi[1, 1:3] = 1
    np.testing.assert_array_equal(thanh_phan_lien_thong_lon_nhat(a), mong_doi)          # hòa: giữ vùng gặp trước
    assert not thanh_phan_lien_thong_lon_nhat(np.zeros((2, 2), bool)).any()


def test_noi_suy_song_tuyen_tinh_voi_tham_chieu_vong_lap():
    a = np.array([[0., 40., 100.], [150., 200., 255.]])
    for cao, rong in [(1, 1), (3, 7), (2, 3), (1, 5), (8, 1)]:
        mong_doi = np.zeros((cao, rong))
        for y in range(cao):
            for x in range(rong):
                sy = min(1, max(0, (y + .5) * 2 / cao - .5))
                sx = min(2, max(0, (x + .5) * 3 / rong - .5))
                y0, x0 = int(sy), int(sx)
                fy, fx = sy - y0, sx - x0
                mong_doi[y, x] = sum(a[min(y0 + dy, 1), min(x0 + dx, 2)] * (fy if dy else 1 - fy) * (fx if dx else 1 - fx)
                                     for dy in (0, 1) for dx in (0, 1))
        np.testing.assert_allclose(noi_suy_song_tuyen_tinh(a, cao, rong), mong_doi)
    np.testing.assert_allclose(noi_suy_song_tuyen_tinh(np.array([[0., 100.]]), 1, 3), [[0, 50, 100]])
    for cao, rong in [(0, 1), (1, 0), (1.5, 2), (True, 2)]:
        with pytest.raises(ValueError):
            noi_suy_song_tuyen_tinh(np.ones((2, 2)), cao, rong)


# ------------------------------------------------------------------------------ thu nhỏ theo diện tích
def test_dien_tich_giu_hang_so_va_trung_binh():
    a = np.random.default_rng(0).random((97, 131))
    kq = thu_nho_trung_binh_dien_tich(a, 22, 29)                                        # tỉ lệ không nguyên
    assert kq.shape == (22, 29) and kq.mean() == pytest.approx(a.mean(), abs=1e-10)
    assert np.allclose(thu_nho_trung_binh_dien_tich(np.full((50, 40), 0.3), 7, 9), 0.3)


def test_dien_tich_bang_trung_binh_khoi_khi_ti_le_nguyen():
    a = np.arange(12 * 20, dtype=float).reshape(12, 20)
    assert np.allclose(thu_nho_trung_binh_dien_tich(a, 3, 5), a.reshape(3, 4, 5, 4).mean(axis=(1, 3)))


def test_dien_tich_net_mot_pixel_khong_bien_mat():
    a = np.zeros((100, 100))
    a[37, :] = 1
    kq = thu_nho_trung_binh_dien_tich(a, 10, 10)
    assert kq.sum() == pytest.approx(1.0) and kq.max() == pytest.approx(0.1)


def test_thu_phong_chon_che_do_va_gioi_han():
    a = np.random.default_rng(1).random((40, 40))
    np.testing.assert_allclose(thu_phong(a, 10, 10), thu_nho_trung_binh_dien_tich(a, 10, 10))
    np.testing.assert_allclose(thu_phong(a, 10, 10, "song_tuyen_tinh"), noi_suy_song_tuyen_tinh(a, 10, 10))
    np.testing.assert_allclose(thu_phong(a, 80, 80), noi_suy_song_tuyen_tinh(a, 80, 80))   # phóng to: song tuyến tính
    with pytest.raises(LoiXuLyAnh):
        thu_phong(a, 10, 10, "sai")
    with pytest.raises(LoiXuLyAnh):
        thu_nho_trung_binh_dien_tich(np.ones((4, 4)), 8, 8)
    lon = np.full((1000, 800, 3), 200, np.uint8)
    nho = gioi_han_kich_thuoc(lon)
    assert max(nho.shape[:2]) == 512 and nho.shape[2] == 3 and (nho == 200).all()
    be = np.zeros((50, 60, 3), np.uint8)
    assert gioi_han_kich_thuoc(be) is be


@pytest.mark.parametrize("kich_thuoc, mong_doi", [((1, 513), (1, 512)), ((513, 1), (512, 1)),
                                                  ((1, 2000, 3), (1, 512, 3)), ((2, 513), (2, 512))])
def test_gioi_han_kich_thuoc_anh_mot_hang_hoac_mot_cot(kich_thuoc, mong_doi):
    a = np.full(kich_thuoc, 90, np.uint8)
    kq = gioi_han_kich_thuoc(a)
    assert kq.shape == mong_doi and kq.dtype == np.uint8 and (kq == 90).all()


def test_ket_qua_la_dataclass_day_du():
    r = xu_ly_anh(mau_chu_L())
    assert isinstance(r, KetQuaXuLy)
    assert list(r.cac_buoc) == ["xam", "loc", "nguong", "mat_na", "cat", "thu_nho", "dau_ra", "dau_vao_cnn"]
