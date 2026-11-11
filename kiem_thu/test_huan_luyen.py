"""Kiểm thử huấn luyện trên dữ liệu nhỏ, ghi vào thư mục tạm (không đụng tới mo_hinh/ và ket_qua/ thật)."""
import numpy as np
import pytest
import torch

from loi import cau_hinh, huan_luyen as hl
from loi.cau_hinh import doc_json, lay_bai_toan, sha256
from loi.du_doan import DuDoan
from bo_du_lieu import tai_mnist, ve_hinh
from bo_du_lieu.hinh_hoc import dau_vao_cnn


@pytest.fixture(scope="module")
def du_lieu_nho():
    rng = np.random.default_rng(0)

    def hinh(n):
        y = np.repeat(np.arange(3), n // 3)
        x = np.stack([dau_vao_cnn(ve_hinh(int(c), rng)[0]) for c in y])[:, None].astype(np.float32) / 255
        return x, y.astype(np.int64)
    x, y = tai_mnist("train")
    x2, y2 = tai_mnist("validation")
    return {"hinh_hoc": {"train": hinh(150), "validation": hinh(60)},
            "chu_so": {"train": (x[:400], y[:400]), "validation": (x2[:150], y2[:150])}}


@pytest.fixture
def thu_muc_tam(tmp_path, monkeypatch, du_lieu_nho):
    """Chuyển đầu ra sang thư mục tạm và thay nguồn dữ liệu bằng tập nhỏ; đọc tập test là lỗi."""
    monkeypatch.setattr(cau_hinh, "MO_HINH", tmp_path / "mo_hinh")
    monkeypatch.setattr(cau_hinh, "KET_QUA", tmp_path / "ket_qua")
    da_doc = []

    def tai_gia(bai_toan, tap):
        da_doc.append(tap)
        if tap not in ("train", "validation"):
            raise AssertionError(f"Huấn luyện không được đọc tập {tap}")
        return du_lieu_nho[bai_toan][tap]
    monkeypatch.setattr(hl, "tai_tap", tai_gia)
    return tmp_path, da_doc


@pytest.mark.parametrize("bai_toan", ["hinh_hoc", "chu_so"])
def test_huan_luyen_hai_cau_hinh_va_chon_bang_validation(bai_toan, thu_muc_tam):
    tmp, da_doc = thu_muc_tam
    bt = lay_bai_toan(bai_toan)
    lua_chon = hl.huan_luyen_ca_hai(bai_toan, so_epoch=2, so_luong=2, in_ra=False)
    assert set(da_doc) == {"train", "validation"}                                    # không bao giờ đọc test
    for b in ("co_so", "cai_tien"):
        ket_qua = tmp / "ket_qua" / bt.ten / b
        lich_su = doc_json(ket_qua / "history.json")
        assert [r["epoch"] for r in lich_su] == [1, 2]
        cau_hinh_ = doc_json(ket_qua / "config.json")
        assert cau_hinh_["tang_cuong"] == (b == "cai_tien") and cau_hinh_["so_epoch_hoan_thanh"] == 2
        assert (tmp / "mo_hinh" / f"{bt.ten}_{b}.pth").exists() and (ket_qua / "history.csv").exists()
    chon = lua_chon["bien_the_duoc_chon"]
    ung_vien = lua_chon["ung_vien"]
    assert chon == max(ung_vien, key=lambda b: (ung_vien[b]["accuracy_validation"], -ung_vien[b]["loss_validation"]))
    assert sha256(tmp / "mo_hinh" / f"{bt.ten}_tot_nhat.pth") == sha256(tmp / "mo_hinh" / f"{bt.ten}_{chon}.pth")
    mo_hinh = DuDoan(bai_toan, tmp / "mo_hinh" / f"{bt.ten}_tot_nhat.pth")                 # checkpoint mới nạp và dự đoán được
    x = np.zeros((28, 28), np.float32)
    x[8:20, 8:20] = 1
    assert mo_hinh.du_doan_dau_vao(x)["ten_lop"] in bt.lop


def test_huan_luyen_tai_lap_cung_hat(thu_muc_tam):
    tmp, _ = thu_muc_tam
    a = hl.huan_luyen("hinh_hoc", "cai_tien", so_epoch=2, so_luong=2, in_ra=False)
    trong_so_a = torch.load(tmp / "mo_hinh" / "hinh_hoc_cai_tien.pth", weights_only=True)["state_dict"]
    lich_su_a = doc_json(tmp / "ket_qua" / "hinh_hoc" / "cai_tien" / "history.json")
    hl.huan_luyen("hinh_hoc", "cai_tien", so_epoch=2, so_luong=2, in_ra=False)
    trong_so_b = torch.load(tmp / "mo_hinh" / "hinh_hoc_cai_tien.pth", weights_only=True)["state_dict"]
    lich_su_b = doc_json(tmp / "ket_qua" / "hinh_hoc" / "cai_tien" / "history.json")
    assert [r["validation_accuracy"] for r in lich_su_a] == [r["validation_accuracy"] for r in lich_su_b]
    assert all(torch.equal(trong_so_a[k], trong_so_b[k]) for k in trong_so_a)
    assert a["so_tham_so"] == 56355


def test_chon_mo_hinh_phat_hien_checkpoint_bi_sua(thu_muc_tam):
    tmp, _ = thu_muc_tam
    for b in ("co_so", "cai_tien"):
        hl.huan_luyen("hinh_hoc", b, so_epoch=1, so_luong=2, in_ra=False)
    tep = tmp / "mo_hinh" / "hinh_hoc_co_so.pth"
    tep.write_bytes(tep.read_bytes() + b"x")
    with pytest.raises(ValueError, match="thay đổi"):
        hl.chon_mo_hinh("hinh_hoc")


def test_tham_so_khong_hop_le(thu_muc_tam):
    with pytest.raises(ValueError):
        hl.huan_luyen("hinh_hoc", "khong_co", so_epoch=1, in_ra=False)
    with pytest.raises(ValueError):
        hl.huan_luyen("khong_co", "co_so", so_epoch=1, in_ra=False)
