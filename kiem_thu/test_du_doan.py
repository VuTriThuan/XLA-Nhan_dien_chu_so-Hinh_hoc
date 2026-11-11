"""Kiểm thử bộ nhận dạng: khớp kết quả đã lưu, đầu vào sai, checkpoint sai, hồi quy số liệu và các ca ảnh thật giả lập."""
import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from loi.cau_hinh import ANH_MAU, BAI_TOAN, CHU_SO, HINH_HOC, doc_json, doc_lua_chon, lay_bai_toan, sha256
from loi.du_doan import DuDoan, duong_dan_mo_hinh, nap_checkpoint
from bo_du_lieu import tai_tap
from tien_xu_ly import AnhDongNhat, doc_anh_tu_tep


def anh_xam(ve, kich_thuoc=400):
    im = Image.new("L", (kich_thuoc, kich_thuoc), 255)
    ve(ImageDraw.Draw(im))
    return np.repeat(np.array(im)[..., None], 3, axis=2)


def tron_manh(do_day=3):
    return anh_xam(lambda d: d.ellipse([80, 80, 320, 320], outline=0, width=do_day))


def so_bay_manh(do_day=4):
    return anh_xam(lambda d: d.line([(110, 100), (290, 100), (190, 320)], fill=0, width=do_day, joint="curve"))


@pytest.fixture(scope="module")
def du_doan_chu_so():
    return DuDoan("chu_so")


# ------------------------------------------------------------------------------ DuDoan
@pytest.mark.parametrize("hinh", [(28, 28), (1, 28, 28), (1, 1, 28, 28)])
def test_khop_ket_qua_da_luu(du_doan_chu_so, hinh):
    x, _ = tai_tap("chu_so", "test")
    kq = du_doan_chu_so.du_doan_dau_vao(x[0].reshape(hinh))
    da_luu = np.load(CHU_SO.thu_muc_ket_qua / du_doan_chu_so.bien_the / "test_predictions.npz")["xac_suat"][0]
    assert kq["lop"] == int(da_luu.argmax())
    assert np.allclose(list(kq["xac_suat"].values()), da_luu, atol=2e-6) and abs(sum(kq["xac_suat"].values()) - 1) < 2e-6


@pytest.mark.parametrize("sai", [np.zeros((2, 28, 28)), np.ones((28, 28)) * 255, np.ones((28, 28)) * np.nan,
                                 np.ones((28, 28)) * -.1, np.zeros((27, 28))])
def test_dau_vao_chuan_hoa_sai(du_doan_chu_so, sai):
    with pytest.raises(ValueError):
        du_doan_chu_so.du_doan_dau_vao(sai)


def test_dau_vao_khong_co_net_bi_tu_choi(du_doan_chu_so):
    with pytest.raises(AnhDongNhat):
        du_doan_chu_so.du_doan_dau_vao(np.full((28, 28), .5))


@pytest.mark.parametrize("bai_toan", list(BAI_TOAN))
def test_checkpoint_sai_bi_tu_choi(bai_toan, tmp_path):
    bt = lay_bai_toan(bai_toan)
    goc = torch.load(duong_dan_mo_hinh(bai_toan), map_location="cpu", weights_only=True)
    for khoa, gia_tri in (("lop", list(reversed(bt.lop))), ("bai_toan", "khac"), ("phien_ban_tien_xu_ly", "cu")):
        hong = {**goc, khoa: gia_tri}
        tep = tmp_path / f"{khoa}.pth"
        torch.save(hong, tep)
        with pytest.raises(ValueError):
            nap_checkpoint(tep, bai_toan)
    with pytest.raises(ValueError):                                        # nạp checkpoint của bài toán này cho bài toán kia
        nap_checkpoint(duong_dan_mo_hinh(bai_toan), "hinh_hoc" if bai_toan == "chu_so" else "chu_so")


# ------------------------------------------------------------------------------ hồi quy số liệu đã báo cáo
def test_hoi_quy_mnist_9915_tren_10000(du_doan_chu_so):
    x, y = tai_tap("chu_so", "test")
    dung = int((du_doan_chu_so.xac_suat(x).argmax(1) == y).sum())
    assert (du_doan_chu_so.bien_the, dung) == ("cai_tien", 9915)
    assert int((DuDoan("chu_so", duong_dan_mo_hinh("chu_so", "co_so")).xac_suat(x).argmax(1) == y).sum()) == 9916


@pytest.mark.parametrize("bai_toan", list(BAI_TOAN))
def test_ket_qua_da_luu_khop_checkpoint_va_lua_chon_bang_validation(bai_toan):
    bt = lay_bai_toan(bai_toan)
    lua_chon = doc_lua_chon(bt.ten)
    ung_vien = lua_chon["ung_vien"]
    assert lua_chon["bien_the_duoc_chon"] == max(ung_vien, key=lambda b: (ung_vien[b]["accuracy_validation"],
                                                                         -ung_vien[b]["loss_validation"]))
    assert lua_chon["khong_dung_test"] is True
    assert sha256(bt.tep_mo_hinh("tot_nhat")) == sha256(bt.tep_mo_hinh(lua_chon["bien_the_duoc_chon"]))
    x, y = tai_tap(bai_toan, "test")
    for b in ("co_so", "cai_tien"):
        du_doan = DuDoan(bai_toan, bt.tep_mo_hinh(b))
        xs = du_doan.xac_suat(x)
        luu = doc_json(bt.thu_muc_ket_qua / b / "test_metrics.json")
        assert luu["so_dung"] == int((xs.argmax(1) == y).sum()) and luu["so_anh"] == len(y)
        assert np.allclose(xs, np.load(bt.thu_muc_ket_qua / b / "test_predictions.npz")["xac_suat"], atol=1e-6)


# ------------------------------------------------------------------------------ BoNhanDang: cả đường xử lý
@pytest.mark.parametrize("so", range(10))
def test_anh_mau_chu_so(bo_nhan_dang, so):
    kq = bo_nhan_dang.nhan_dang(doc_anh_tu_tep(ANH_MAU / f"chu_so_{so}.png"), "chu_so")
    assert kq["lop"] == so


@pytest.mark.parametrize("lop", range(3))
def test_anh_mau_hinh_hoc(bo_nhan_dang, lop):
    kq = bo_nhan_dang.nhan_dang(doc_anh_tu_tep(ANH_MAU / f"hinh_{HINH_HOC.lop[lop]}.png"), "hinh_hoc")
    assert kq["lop"] == lop


def test_ket_qua_day_du_cac_truong(bo_nhan_dang):
    kq = bo_nhan_dang.nhan_dang(so_bay_manh(12), "chu_so")
    assert kq["ten_lop"] == "7" and kq["nhan"] == "Chữ số 7" and len(kq["xac_suat"]) == 10
    assert sum(kq["xac_suat"].values()) == pytest.approx(1, abs=1e-5)
    assert list(kq["cac_buoc"]) == ["xam", "loc", "nguong", "mat_na", "cat", "thu_nho", "dau_ra", "dau_vao_cnn"]
    assert kq["cuc_tinh"] == "net_toi" and kq["ms"] > 0 and kq["canh_bao"] == []
    assert bo_nhan_dang.nhan_dang(tron_manh(8), "hinh_hoc")["nhan"] == "Tròn"


def test_dao_cuc_tinh_cho_cung_lop(bo_nhan_dang):
    a = tron_manh(8)
    ket_qua = [bo_nhan_dang.nhan_dang(x, "hinh_hoc") for x in (a, 255 - a)]
    assert [k["lop"] for k in ket_qua] == [0, 0] and [k["cuc_tinh"] for k in ket_qua] == ["net_toi", "net_sang"]


def test_net_manh_can_thu_nho_theo_dien_tich(bo_nhan_dang):
    """Hồi quy cho lỗi đã phát hiện: nét mảnh bị rách khi thu nhỏ bằng song tuyến tính."""
    assert bo_nhan_dang.nhan_dang(tron_manh(3), "hinh_hoc")["lop"] == 0
    cu = bo_nhan_dang.nhan_dang(tron_manh(3), "hinh_hoc", che_do_thu_nho="song_tuyen_tinh", keo_gian=False)
    assert cu["lop"] != 0                                                      # cách xử lý cũ nhận sai hình tròn nét mảnh
    assert bo_nhan_dang.nhan_dang(so_bay_manh(4), "chu_so")["lop"] == 7


def test_chu_nhat_dai(bo_nhan_dang):
    anh = anh_xam(lambda d: d.rectangle([40, 150, 360, 250], outline=0, width=5))        # tỉ lệ cạnh 0,31
    assert bo_nhan_dang.nhan_dang(anh, "hinh_hoc")["lop"] == 2


def test_anh_chup_rat_lon(bo_nhan_dang):
    im = Image.new("L", (3000, 2400), 255)
    ImageDraw.Draw(im).ellipse([600, 400, 2400, 2000], outline=0, width=40)
    assert bo_nhan_dang.nhan_dang(np.repeat(np.array(im)[..., None], 3, axis=2), "hinh_hoc")["lop"] == 0


def test_anh_trong_va_bai_toan_sai(bo_nhan_dang):
    with pytest.raises(ValueError):
        bo_nhan_dang.nhan_dang(np.full((100, 100, 3), 255, np.uint8), "chu_so")
    with pytest.raises(ValueError):
        bo_nhan_dang.nhan_dang(tron_manh(8), "khong_co")
