"""Kiểm thử các hàm hỗ trợ của thu_nghiem/ (không chạy các thí nghiệm đầy đủ)."""
import numpy as np
import pytest

from tien_xu_ly import thanh_phan_lien_thong_lon_nhat
from thu_nghiem.net_manh import lam_manh_zhang_suen, ve_chu_so_manh
from thu_nghiem.ve_pil import HO_HINH, LOP, ve_hinh_pil


def test_zhang_suen_thanh_bo_xuong_mot_pixel_va_giu_lien_thong():
    thanh = np.zeros((40, 60), bool)
    thanh[15:25, 5:55] = True                                     # thanh ngang dày 10 px
    xuong = lam_manh_zhang_suen(thanh)
    assert 0 < xuong.sum() < thanh.sum() / 5
    assert np.array_equal(thanh_phan_lien_thong_lon_nhat(xuong), xuong)         # vẫn là một thành phần
    assert xuong.sum(axis=0).max() <= 2                                          # mỗi cột còn ~1 pixel
    chu_l = np.zeros((50, 50), bool)
    chu_l[5:45, 10:18] = True
    chu_l[37:45, 10:40] = True
    assert np.array_equal(thanh_phan_lien_thong_lon_nhat(lam_manh_zhang_suen(chu_l)), lam_manh_zhang_suen(chu_l))


def test_ve_chu_so_manh_do_day_theo_ti_le():
    anh28 = np.zeros((28, 28), np.uint8)
    anh28[4:24, 12:16] = 255                                      # chữ số "1" giả
    dam = ve_chu_so_manh(anh28, 0.12, np.random.default_rng(0))
    manh = ve_chu_so_manh(anh28, 0.015, np.random.default_rng(0))
    assert dam.dtype == np.uint8 and dam.shape == manh.shape
    assert (dam == 0).sum() > 2 * (manh == 0).sum() > 0             # nét dày có nhiều pixel mực hơn nhiều


@pytest.mark.parametrize("ho", HO_HINH)
def test_ve_hinh_pil_tai_lap_va_co_muc(ho):
    a = ve_hinh_pil(ho, 0.05, np.random.default_rng(1))
    b = ve_hinh_pil(ho, 0.05, np.random.default_rng(1))
    assert np.array_equal(a, b) and a.ndim == 2 and 300 <= a.shape[0] < 500 and (a == 0).any() and (a == 255).any()
    assert LOP[ho] in (0, 1, 2)


def test_ve_pil_khong_goi_np_cross_vector_2d(monkeypatch):
    """np.cross với vector 2D bị bỏ từ NumPy 2.0; tam giác bất kỳ phải vẽ được mà không cần nó."""
    def cam(*_a, **_k):
        raise AssertionError("không được gọi np.cross")
    monkeypatch.setattr(np, "cross", cam)
    for hat in range(20):
        assert (ve_hinh_pil("tam_giac_bat_ky", 0.05, np.random.default_rng(hat)) == 0).any()
