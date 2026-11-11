"""Kiểm thử kiến trúc CNN và tăng cường dữ liệu."""
import math

import numpy as np
import pytest
import torch

from loi.cnn import MangCNN, dem_tham_so
from loi.tang_cuong import bien_doi_affine, tang_cuong, tang_cuong_chu_so, tang_cuong_hinh_hoc


@pytest.mark.parametrize("so_lop,batch_norm,dropout,so_tham_so",
                         [(10, False, 0.25, 56714), (3, False, 0.0, 56259), (3, True, 0.3, 56355)])
def test_so_tham_so_va_hinh_dang_dau_ra(so_lop, batch_norm, dropout, so_tham_so):
    mang = MangCNN(so_lop, batch_norm, dropout).eval()
    assert mang(torch.zeros(2, 1, 28, 28)).shape == (2, so_lop)
    assert dem_tham_so(mang) == so_tham_so
    with pytest.raises(ValueError):
        mang(torch.zeros(2, 3, 28, 28))


def test_dropout_chi_tac_dung_khi_huan_luyen():
    mang = MangCNN(3, dropout=0.5)
    x = torch.rand(4, 1, 28, 28)
    mang.eval()
    np.testing.assert_array_equal(mang(x).detach().numpy(), mang(x).detach().numpy())


def test_affine_dong_nhat_va_khong_cuon_pixel():
    x = torch.arange(784, dtype=torch.float32).reshape(1, 1, 28, 28) / 783
    z, mot = torch.zeros(1), torch.ones(1)
    assert torch.allclose(bien_doi_affine(x, z, mot, z, z), x, atol=2e-6)
    dich = bien_doi_affine(x, z, mot, mot * 3, z)
    assert float(dich[0, 0, :, :3].abs().max()) == 0                      # phần bị dịch ra ngoài là nền 0, không cuộn
    assert torch.allclose(dich[0, 0, :, 3:], x[0, 0, :, :-3], atol=2e-6)


def test_tang_cuong_tach_rng_va_tai_lap():
    x = torch.zeros(2, 1, 28, 28)
    x[:, :, 5:21, 10:15] = 1
    torch.manual_seed(9)
    truoc = torch.random.get_rng_state().clone()
    a = tang_cuong_chu_so(x, torch.Generator().manual_seed(3))
    b = tang_cuong_chu_so(x, torch.Generator().manual_seed(3))
    assert torch.equal(a, b) and not torch.equal(a, x)
    assert torch.equal(torch.random.get_rng_state(), truoc)               # RNG toàn cục (Dropout) không bị động tới


def test_tang_cuong_hinh_hoc_giu_khoang_gia_tri():
    x = torch.zeros(8, 1, 28, 28)
    x[:, :, 8:20, 8:20] = 1
    y = tang_cuong_hinh_hoc(x)
    assert y.shape == x.shape and torch.isfinite(y).all() and 0 <= float(y.min()) <= float(y.max()) <= 1


def test_tang_cuong_tham_so_tuy_chinh():
    x = torch.zeros(4, 1, 28, 28)
    x[:, :, 10:18, 10:18] = 1
    khong_doi = tang_cuong(x, 0.0, 0.0, ti_le=(1.0, 1.0))
    assert torch.allclose(khong_doi, x, atol=1e-5)
    xoay = tang_cuong(x, math.radians(30), 0.0, ti_le=(1.0, 1.0), generator=torch.Generator().manual_seed(1))
    assert not torch.allclose(xoay, x, atol=1e-3)
