"""Huấn luyện CNN cho hai bài toán; chọn checkpoint tốt nhất và chọn cấu hình cuối chỉ bằng validation.

Hai cấu hình so sánh: `co_so` (không tăng cường dữ liệu) và `cai_tien` (có tăng cường; với hình học còn thêm
BatchNorm và Dropout). Tập test không bao giờ được đọc trong hàm này.
"""
import os
import platform
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .cau_hinh import BIEN_THE, PHIEN_BAN_DAU_VAO, doc_json, ghi_csv, ghi_json, ghi_lua_chon, lay_bai_toan, sha256
from bo_du_lieu import tai_tap
from .cnn import dem_tham_so, tao_mang
from .tang_cuong import tang_cuong_chu_so, tang_cuong_hinh_hoc

# Công thức huấn luyện của từng bài toán (giữ đúng công thức đã dùng cho các mô hình hiện có).
CONG_THUC = {
    "chu_so": dict(so_epoch=12, co_lo=128, toc_do_hoc=1e-3, hat=42, buoc_giam=(8, 0.3), lech_hat_lo=1,
                   hat_tang_cuong=1000, tang_cuong=tang_cuong_chu_so,
                   mo_ta_tang_cuong="xoay ±10°, dịch ±2 px, co giãn [0,9; 1,1]"),
    "hinh_hoc": dict(so_epoch=20, co_lo=64, toc_do_hoc=1e-3, hat=42, buoc_giam=None, lech_hat_lo=0,
                     hat_tang_cuong=None, tang_cuong=tang_cuong_hinh_hoc,
                     mo_ta_tang_cuong="xoay ±20°, dịch ±1,5 px, co giãn [0,9; 1,1], độ sáng ×[0,9; 1,1], nhiễu Gauss 0,012"),
}
KIEN_TRUC = {
    "chu_so": {"co_so": dict(batch_norm=False, dropout=0.25), "cai_tien": dict(batch_norm=False, dropout=0.25)},
    "hinh_hoc": {"co_so": dict(batch_norm=False, dropout=0.0), "cai_tien": dict(batch_norm=True, dropout=0.3)},
}


def dat_hat(hat):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(hat)
    np.random.seed(hat)
    torch.manual_seed(hat)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(hat)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def danh_gia_lo(mang, x, y, thiet_bi, co_lo=512):
    mang.eval()
    loss, dung = 0.0, 0
    with torch.inference_mode():
        for dau in range(0, len(y), co_lo):
            xb, yb = x[dau:dau + co_lo].to(thiet_bi), y[dau:dau + co_lo].to(thiet_bi)
            logits = mang(xb)
            loss += nn.functional.cross_entropy(logits, yb, reduction="sum").item()
            dung += int((logits.argmax(1) == yb).sum())
    return loss / len(y), dung / len(y)


def huan_luyen(bai_toan, bien_the, *, so_epoch=None, co_lo=None, toc_do_hoc=None, hat=None, thiet_bi="cpu", so_luong=6,
               in_ra=True, nguon_du_lieu=None, thu_muc_mo_hinh=None, thu_muc_ket_qua=None):
    """Huấn luyện một cấu hình; ghi checkpoint tốt nhất vào mo_hinh/<bai_toan>_<bien_the>.pth và lịch sử vào ket_qua.

    nguon_du_lieu (callable(bai_toan, tap)) và hai thư mục đầu ra cho phép chạy thí nghiệm mà không đụng tới dữ liệu và
    mô hình chính của dự án.
    """
    bt = lay_bai_toan(bai_toan)
    if bien_the not in BIEN_THE:
        raise ValueError(f"biến thể phải là một trong {BIEN_THE}")
    ct = {**CONG_THUC[bt.ten]}
    for khoa, gia_tri in (("so_epoch", so_epoch), ("co_lo", co_lo), ("toc_do_hoc", toc_do_hoc), ("hat", hat)):
        if gia_tri is not None:
            ct[khoa] = gia_tri
    kien_truc = {"so_lop": bt.so_lop, **KIEN_TRUC[bt.ten][bien_the]}
    co_tang_cuong = bien_the == "cai_tien"
    torch.set_num_threads(so_luong)
    dat_hat(ct["hat"])
    tai = nguon_du_lieu or tai_tap
    x_tr, y_tr = tai(bt.ten, "train")              # chỉ đọc train và validation, không đọc test
    x_va, y_va = tai(bt.ten, "validation")
    x_tr, y_tr, x_va, y_va = map(torch.from_numpy, (x_tr, y_tr, x_va, y_va))
    nap = DataLoader(TensorDataset(x_tr, y_tr), batch_size=ct["co_lo"], shuffle=True,
                     generator=torch.Generator().manual_seed(ct["hat"] + ct["lech_hat_lo"]), num_workers=0)
    mang = tao_mang(kien_truc).to(thiet_bi)
    toi_uu = torch.optim.Adam(mang.parameters(), lr=ct["toc_do_hoc"])
    lich = torch.optim.lr_scheduler.StepLR(toi_uu, *ct["buoc_giam"]) if ct["buoc_giam"] else None
    rng_tc = torch.Generator(device=thiet_bi).manual_seed(ct["hat"] + ct["hat_tang_cuong"]) \
        if ct["hat_tang_cuong"] is not None else None
    ket_qua = Path(thu_muc_ket_qua or bt.thu_muc_ket_qua) / bien_the
    tep_mo_hinh = bt.tep_mo_hinh(bien_the, thu_muc_mo_hinh)
    tep_mo_hinh.parent.mkdir(parents=True, exist_ok=True)
    ket_qua.mkdir(parents=True, exist_ok=True)
    cau_hinh = {"bai_toan": bt.ten, "bien_the": bien_the, "hat": ct["hat"], "so_epoch_yeu_cau": ct["so_epoch"],
                "co_lo": ct["co_lo"], "toc_do_hoc": ct["toc_do_hoc"], "bo_toi_uu": "Adam",
                "lich_giam_toc_do": f"StepLR(step={ct['buoc_giam'][0]}, gamma={ct['buoc_giam'][1]})" if ct["buoc_giam"] else None,
                "ham_mat_mat": "CrossEntropyLoss trên logits", "kien_truc": kien_truc, "so_tham_so": dem_tham_so(mang),
                "tang_cuong": co_tang_cuong, "mo_ta_tang_cuong": ct["mo_ta_tang_cuong"] if co_tang_cuong else None,
                "so_train": len(y_tr), "so_validation": len(y_va), "thiet_bi": thiet_bi, "so_luong": so_luong,
                "phien_ban": {"torch": str(torch.__version__), "numpy": np.__version__, "python": platform.python_version(),
                              "he_dieu_hanh": platform.platform()},
                "tieu_chi_chon": "accuracy validation cao nhất, hòa thì loss validation thấp nhất; không đọc test",
                "bat_dau_utc": datetime.now(timezone.utc).isoformat()}
    ghi_json(ket_qua / "config.json", cau_hinh)
    tot_nhat, epoch_tot, lich_su = (-1.0, -float("inf")), 0, []
    bat_dau = time.perf_counter()
    for epoch in range(1, ct["so_epoch"] + 1):
        mang.train()
        loss_tong, dung, tong = 0.0, 0, 0
        t0 = time.perf_counter()
        toc_do = float(toi_uu.param_groups[0]["lr"])
        for xb, yb in nap:
            xb, yb = xb.to(thiet_bi), yb.to(thiet_bi)
            if co_tang_cuong:
                xb = ct["tang_cuong"](xb, rng_tc) if rng_tc is not None else ct["tang_cuong"](xb)
            toi_uu.zero_grad(set_to_none=True)
            logits = mang(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            loss.backward()
            toi_uu.step()
            loss_tong += float(loss.detach()) * len(yb)
            dung += int((logits.argmax(1) == yb).sum())
            tong += len(yb)
        loss_va, acc_va = danh_gia_lo(mang, x_va, y_va, thiet_bi)
        dong = {"epoch": epoch, "train_loss": loss_tong / tong, "train_accuracy": dung / tong, "validation_loss": loss_va,
                "validation_accuracy": acc_va, "toc_do_hoc": toc_do, "giay": time.perf_counter() - t0}
        lich_su.append(dong)
        if (acc_va, -loss_va) > tot_nhat:
            tot_nhat, epoch_tot = (acc_va, -loss_va), epoch
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in mang.state_dict().items()}, "bai_toan": bt.ten,
                        "bien_the": bien_the, "lop": list(bt.lop), "kien_truc": kien_truc,
                        "phien_ban_tien_xu_ly": PHIEN_BAN_DAU_VAO[bt.ten],
                        "cau_hinh": {k: v for k, v in cau_hinh.items() if k not in ("phien_ban",)},
                        "epoch_tot_nhat": epoch, "do_chinh_xac_validation": acc_va, "loss_validation": loss_va},
                       tep_mo_hinh)
        ghi_json(ket_qua / "history.json", lich_su)
        if lich is not None:
            lich.step()
        if in_ra:
            print(f"{bt.ten}/{bien_the} {epoch:02d}/{ct['so_epoch']}: train={dung / tong:.5f} val={acc_va:.5f} "
                  f"loss={loss_va:.5f} ({dong['giay']:.1f}s)", flush=True)
    ghi_csv(ket_qua / "history.csv", lich_su)
    cau_hinh.update(so_epoch_hoan_thanh=len(lich_su), epoch_tot_nhat=epoch_tot, accuracy_validation_tot_nhat=tot_nhat[0],
                    loss_validation_tot_nhat=-tot_nhat[1], giay_huan_luyen=time.perf_counter() - bat_dau,
                    sha256_checkpoint=sha256(tep_mo_hinh))
    ghi_json(ket_qua / "config.json", cau_hinh)
    return cau_hinh


def chon_mo_hinh(bai_toan):
    """Chọn cấu hình cuối chỉ bằng validation (accuracy rồi loss), ghi mo_hinh/lua_chon.json và sao chép thành <bai_toan>_tot_nhat.pth."""
    bt = lay_bai_toan(bai_toan)
    cau_hinh = {b: doc_json(bt.thu_muc_ket_qua / b / "config.json") for b in BIEN_THE}
    for b, c in cau_hinh.items():
        if c["sha256_checkpoint"] != sha256(bt.tep_mo_hinh(b)):
            raise ValueError(f"Checkpoint {b} đã thay đổi so với lúc huấn luyện")
    for khoa in ("hat", "co_lo", "toc_do_hoc", "so_epoch_yeu_cau", "so_train", "so_validation"):
        if cau_hinh["co_so"][khoa] != cau_hinh["cai_tien"][khoa]:
            raise ValueError(f"Hai cấu hình không so sánh được vì khác {khoa}")
    chon = max(cau_hinh, key=lambda b: (cau_hinh[b]["accuracy_validation_tot_nhat"], -cau_hinh[b]["loss_validation_tot_nhat"]))
    shutil.copyfile(bt.tep_mo_hinh(chon), bt.tep_mo_hinh("tot_nhat"))
    lua_chon = {"bien_the_duoc_chon": chon, "tieu_chi": "accuracy validation, rồi loss validation; không đọc test",
                "khong_dung_test": True,
                "ung_vien": {b: {"epoch": c["epoch_tot_nhat"], "accuracy_validation": c["accuracy_validation_tot_nhat"],
                                 "loss_validation": c["loss_validation_tot_nhat"], "sha256_checkpoint": c["sha256_checkpoint"]}
                             for b, c in cau_hinh.items()},
                "sha256_tot_nhat": sha256(bt.tep_mo_hinh("tot_nhat")),
                "chot_luc_utc": datetime.now(timezone.utc).isoformat()}
    ghi_lua_chon(bt.ten, lua_chon)
    return lua_chon


def huan_luyen_ca_hai(bai_toan, **tuy_chon):
    """Huấn luyện `co_so` và `cai_tien` rồi chọn cấu hình cuối."""
    for bien_the in BIEN_THE:
        huan_luyen(bai_toan, bien_the, **tuy_chon)
    return chon_mo_hinh(bai_toan)
