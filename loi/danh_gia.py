"""Đánh giá sau khi đã chốt cấu hình bằng validation: chỉ số, ma trận nhầm lẫn, ảnh lỗi, độ bền, ảnh người thật.

Mọi đầu ra ghi vào ket_qua/<bai_toan>/. Các bộ "chẩn đoán" và "stress" lấy từ dữ liệu tổng hợp/test nên KHÔNG phải ảnh
người viết/vẽ; chỉ đánh giá `anh_that` mới nói được về ảnh thật.
"""
import math
import time
from pathlib import Path

import numpy as np
import torch

from .cau_hinh import BIEN_THE, doc_json, doc_lua_chon, ghi_csv, ghi_json, lay_bai_toan, sha256
from .du_doan import DuDoan
from bo_du_lieu import tai_meta_hinh_hoc, tai_tap
from .tang_cuong import bien_doi_affine
from tien_xu_ly import LoiXuLyAnh, doc_anh_tu_tep, xu_ly_anh

DUOI_ANH = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
YEU_CAU_ANH_THAT = {"chu_so": (50, 5), "hinh_hoc": (30, 10)}     # (tổng tối thiểu, tối thiểu mỗi lớp)


# ------------------------------------------------------------------------------------ chỉ số

def ghi_ma_tran_nham_lan_csv(duong_dan, ma_tran):
    """Ghi ma trận nhầm lẫn ra CSV UTF-8 (np.savetxt nhận đường dẫn sẽ dùng mã hóa mặc định, cp1252 trên Windows)."""
    with Path(duong_dan).open("w", encoding="utf-8", newline="") as f:
        np.savetxt(f, np.asarray(ma_tran), delimiter=",", fmt="%d",
                   header="hàng = lớp thật; cột = lớp dự đoán", comments="# ")

def khoang_wilson(dung, n, z=1.959963984540054):
    if not n:
        return [0.0, 0.0]
    p = dung / n
    tam = (p + z * z / (2 * n)) / (1 + z * z / n)
    nua = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, tam - nua), min(1.0, tam + nua)]


def chi_so_phan_loai(y, du_doan, bt):
    """Chỉ số cho một lô. du_doan = -1 là ảnh không xử lý được: tính là sai nhưng không vào ma trận nhầm lẫn."""
    y, du_doan, k = np.asarray(y), np.asarray(du_doan), bt.so_lop
    hop_le = du_doan >= 0
    cm = np.bincount(y[hop_le].astype(int) * k + du_doan[hop_le].astype(int), minlength=k * k).reshape(k, k)
    ho_tro = np.bincount(y.astype(int), minlength=k)
    tp = cm.diagonal().astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(k), where=cm.sum(0) > 0)
    recall = np.divide(tp, ho_tro, out=np.zeros(k), where=ho_tro > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(k), where=(precision + recall) > 0)
    n, dung = len(y), int((y == du_doan).sum())
    return {"so_anh": n, "so_dung": dung, "so_sai": n - dung, "so_khong_hop_le": int((~hop_le).sum()),
            "do_chinh_xac": dung / n if n else None, "khoang_wilson_95": khoang_wilson(dung, n),
            "precision_macro": float(precision.mean()), "recall_macro": float(recall.mean()), "f1_macro": float(f1.mean()),
            "ma_tran_nham_lan": cm.tolist(),
            "theo_lop": [{"lop": bt.lop[i], "so_mau": int(ho_tro[i]), "precision": float(precision[i]),
                          "recall": float(recall[i]), "f1": float(f1[i])} for i in range(k)]}


# ------------------------------------------------------------------------------------ biểu đồ
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def ve_ma_tran_nham_lan(cm, bt, tieu_de, thu_muc):
    plt = _plt()
    cm = np.asarray(cm)
    k = bt.so_lop
    co = max(5.6, .7 * k)
    hinh, t = plt.subplots(figsize=(co + 1, co))
    anh = t.imshow(cm, cmap="Blues", vmin=0)
    hinh.colorbar(anh, ax=t, shrink=.8)
    for i in range(k):
        for j in range(k):
            t.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8 if k > 3 else 14,
                   color="white" if cm[i, j] > cm.max() * .5 else "black")
    t.set(xticks=range(k), yticks=range(k), xticklabels=bt.lop, yticklabels=bt.lop, xlabel="Lớp dự đoán",
          ylabel="Lớp thật", title=tieu_de)
    hinh.tight_layout()
    hinh.savefig(thu_muc / "confusion_matrix.png", dpi=180)
    plt.close(hinh)
    ti_le = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    hinh, t = plt.subplots(figsize=(co + 1, co))
    anh = t.imshow(ti_le, vmin=0, vmax=1, cmap="Blues")
    hinh.colorbar(anh, ax=t, shrink=.8, label="Tỉ lệ theo hàng (recall)")
    for i in range(k):
        for j in range(k):
            if cm[i, j]:
                t.text(j, i, f"{ti_le[i, j] * 100:.1f}", ha="center", va="center", fontsize=7 if k > 3 else 12,
                       color="white" if ti_le[i, j] > .5 else "black")
    t.set(xticks=range(k), yticks=range(k), xticklabels=bt.lop, yticklabels=bt.lop, xlabel="Lớp dự đoán",
          ylabel="Lớp thật", title=tieu_de + " (chuẩn hóa theo hàng, %)")
    hinh.tight_layout()
    hinh.savefig(thu_muc / "confusion_matrix_normalized.png", dpi=180)
    plt.close(hinh)


def ve_duong_cong_hoc(thu_muc, tieu_de):
    plt = _plt()
    lich_su = doc_json(thu_muc / "history.json")
    epoch_tot = doc_json(thu_muc / "config.json").get("epoch_tot_nhat")
    hinh, truc = plt.subplots(1, 2, figsize=(10, 3.6))
    for t, khoa, ten in zip(truc, ("loss", "accuracy"), ("Cross-entropy", "Độ chính xác")):
        for tien_to, nhan in (("train", "Huấn luyện (có tăng cường/dropout)"), ("validation", "Validation (chế độ eval)")):
            t.plot([r["epoch"] for r in lich_su], [r[f"{tien_to}_{khoa}"] for r in lich_su], marker=".", label=nhan)
        if epoch_tot:
            t.axvline(epoch_tot, ls=":", c="gray", label="Epoch được chọn")
        t.set(xlabel="Epoch", ylabel=ten)
        t.grid(alpha=.2)
        t.legend(fontsize=7)
    hinh.suptitle(f"{tieu_de}: lịch sử huấn luyện")
    hinh.tight_layout()
    hinh.savefig(thu_muc / "learning_curves.png", dpi=180)
    plt.close(hinh)


def ghi_anh_loi(x, y, xs, bt, thu_muc, tieu_de, meta=None, toi_da=24):
    """Danh sách mọi lỗi (json, csv), ảnh các lỗi softmax cao nhất và các cặp nhầm lẫn có hướng."""
    plt = _plt()
    du_doan = xs.argmax(1)
    sai = np.flatnonzero(du_doan != y)
    thu_tu = sai[np.argsort(-xs[sai, du_doan[sai]], kind="stable")]
    ban_ghi = []
    for i in thu_tu:
        b = {"chi_so": int(i), "lop_that": bt.lop[y[i]], "lop_du_doan": bt.lop[du_doan[i]],
             "do_tin_cay": float(xs[i, du_doan[i]]), "diem_lop_that": float(xs[i, y[i]])}
        if meta is not None:
            b["bien_the"] = meta[int(i)].get("bien_the")
        ban_ghi.append(b)
    ghi_json(thu_muc / "failure_cases.json", {"tong_so_loi": len(sai), "cac_loi": ban_ghi})
    if ban_ghi:
        ghi_csv(thu_muc / "failure_cases.csv", ban_ghi, cot=[c for c in ban_ghi[0]])
    anh_png = thu_muc / "failure_cases.png"
    if len(thu_tu):
        hien = thu_tu[:toi_da]
        cot = 6
        hang = math.ceil(len(hien) / cot)
        hinh, truc = plt.subplots(hang, cot, figsize=(10, 1.65 * hang), squeeze=False)
        for t in truc.flat:
            t.axis("off")
        for t, i in zip(truc.flat, hien):
            t.imshow(x[i, 0], cmap="gray", vmin=0, vmax=1)
            t.set_title(f"#{i} {bt.lop[y[i]]}→{bt.lop[du_doan[i]]}\nsoftmax {xs[i, du_doan[i]]:.3f}", fontsize=7)
        hinh.suptitle(f"{tieu_de}: {len(hien)}/{len(sai)} lỗi thật, softmax cao nhất trước", fontsize=11)
        hinh.tight_layout(rect=(0, 0, 1, .95))
        hinh.savefig(anh_png, dpi=180)
        plt.close(hinh)
    else:
        anh_png.unlink(missing_ok=True)
    cm = np.bincount(y * bt.so_lop + du_doan, minlength=bt.so_lop ** 2).reshape(bt.so_lop, bt.so_lop)
    ti_le = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    cap = [{"lop_that": bt.lop[i], "lop_du_doan": bt.lop[j], "so_luong": int(cm[i, j]),
            "ti_le_cua_lop_that": float(ti_le[i, j])}
           for i in range(bt.so_lop) for j in range(bt.so_lop) if i != j and cm[i, j] > 0]
    cap.sort(key=lambda r: (-r["so_luong"], r["lop_that"], r["lop_du_doan"]))
    ghi_json(thu_muc / "confused_pairs.json", cap)
    if cap:
        ghi_csv(thu_muc / "confused_pairs.csv", cap)


# ------------------------------------------------------------------------------------ đánh giá ảnh người thật
def danh_gia_anh_that(bt, du_doan, thu_muc_ra, thu_muc_anh=None):
    """Đánh giá trên ảnh người thật viết/vẽ, đi qua ĐÚNG đường xử lý của ứng dụng (tiền xử lý thống nhất).

    Ảnh trùng byte bị bỏ qua và được đếm; ảnh không xử lý được tính là sai (đếm riêng). Trạng thái: pending (chưa có ảnh),
    partial (chưa đủ số lượng), ready (đủ số lượng; không xác nhận nguồn gốc hay chất lượng).
    """
    goc = Path(thu_muc_anh) if thu_muc_anh else bt.thu_muc_anh_that_day_du
    duong, nhan, da_thay, trung = [], [], set(), 0
    for lop, ten in enumerate(bt.lop):
        thu_muc = goc / ten
        for p in sorted(thu_muc.iterdir()) if thu_muc.exists() else []:
            if p.suffix.lower() not in DUOI_ANH:
                continue
            ma = sha256(p)
            if ma in da_thay:
                trung += 1
                continue
            da_thay.add(ma)
            duong.append(p)
            nhan.append(lop)
    dem = {ten: nhan.count(i) for i, ten in enumerate(bt.lop)}
    tong_toi_thieu, moi_lop_toi_thieu = YEU_CAU_ANH_THAT[bt.ten]
    trang_thai = ("ready" if len(duong) >= tong_toi_thieu and min(dem.values()) >= moi_lop_toi_thieu
                  else "partial" if duong else "pending")
    kq = {"trang_thai": trang_thai, "thu_muc": str(goc), "so_anh": len(duong), "so_luong_moi_lop": dem,
          "yeu_cau": f"ít nhất {tong_toi_thieu} ảnh khác nhau, ít nhất {moi_lop_toi_thieu} ảnh mỗi lớp",
          "so_anh_trung_bi_bo_qua": trung,
          "nguon_goc": "tệp do người dùng cung cấp; nguồn gốc 'người thật' do người thu khai báo, không suy ra bằng mô hình",
          "luu_y": "chỉ dùng để đánh giá sau khi chốt mô hình; không dùng để huấn luyện hay chỉnh mô hình"}
    if not duong:
        kq["chi_so"] = None
        ghi_json(Path(thu_muc_ra) / "external_metrics.json", kq)
        return kq
    du_doan_lop, loi, mau = [], [], []
    for p, lop in zip(duong, nhan):
        try:
            x = xu_ly_anh(doc_anh_tu_tep(p), bt.ten).anh
            du_doan_lop.append(du_doan.du_doan_dau_vao(x)["lop"])
        except (ValueError, OSError) as e:
            du_doan_lop.append(-1)
            loi.append({"tep": str(p.name), "loi": str(e)})
        mau.append({"tep": p.name, "sha256": sha256(p), "nhan": bt.lop[lop],
                    "du_doan": bt.lop[du_doan_lop[-1]] if du_doan_lop[-1] >= 0 else "khong_xu_ly_duoc"})
    kq["chi_so"] = chi_so_phan_loai(np.array(nhan), np.array(du_doan_lop), bt)
    kq["anh_loi_dau_vao"] = loi
    kq["mau"] = mau
    ghi_json(Path(thu_muc_ra) / "external_metrics.json", kq)
    return kq


# ------------------------------------------------------------------------------------ bài toán chữ số: chẩn đoán độ bền
def tap_chan_doan_chu_so(x, y):
    """Tập con cố định 100 ảnh/lớp của test MNIST với 4 điều kiện; KHÔNG độc lập với test, không dùng để chọn mô hình."""
    rng = np.random.default_rng(20261110)
    chi_so = np.concatenate([rng.choice(np.flatnonzero(y == i), 100, replace=False) for i in range(10)])
    goc_ = torch.from_numpy(x[chi_so])
    n = len(chi_so)
    gen = torch.Generator().manual_seed(20261110)

    def ngau_nhien():
        return torch.rand(n, generator=gen)
    dich = bien_doi_affine(goc_, torch.zeros(n), torch.ones(n), (ngau_nhien() * 2 - 1) * 3, (ngau_nhien() * 2 - 1) * 3)
    xoay = bien_doi_affine(goc_, (ngau_nhien() * 2 - 1) * math.radians(20), torch.ones(n), torch.zeros(n), torch.zeros(n))
    nhieu = (goc_ + torch.randn(goc_.shape, generator=gen) * .15).clamp(0, 1)
    return chi_so, {"goc": goc_.numpy(), "dich_3px": dich.numpy(), "xoay_20_do": xoay.numpy(), "nhieu_gauss_0.15": nhieu.numpy()}


def chan_doan_chu_so(du_doan, x, y, thu_muc_ra):
    """Độ chính xác trên 4 điều kiện + so sánh đường vào MNIST gốc với đường qua tiền xử lý thống nhất."""
    bt = du_doan.bai_toan
    chi_so, dieu_kien = tap_chan_doan_chu_so(x, y)
    yc = y[chi_so]
    ket_qua, dong = {}, []
    for ten, anh in dieu_kien.items():
        m = chi_so_phan_loai(yc, du_doan.xac_suat(anh).argmax(1), bt)
        ket_qua[ten] = m
        dong.append({"dieu_kien": ten, "so_anh": len(chi_so), "do_chinh_xac": m["do_chinh_xac"], "f1_macro": m["f1_macro"]})
    ghi_json(Path(thu_muc_ra) / "diagnostic_metrics.json", ket_qua)
    qua_tien_xu_ly, loi = [], 0
    for a in x[chi_so, 0]:
        try:
            qua_tien_xu_ly.append(xu_ly_anh(np.rint(a * 255).astype(np.uint8), bt.ten).anh)
        except LoiXuLyAnh:
            qua_tien_xu_ly.append(np.zeros_like(a))
            loi += 1
    p_tx = du_doan.xac_suat(np.stack(qua_tien_xu_ly)).argmax(1)
    so_sanh = {"ghi_chu": "Cùng 1.000 chỉ số test, không phải mẫu ngoài mới",
               "mnist_goc": chi_so_phan_loai(yc, du_doan.xac_suat(x[chi_so]).argmax(1), bt),
               "qua_tien_xu_ly_thong_nhat": chi_so_phan_loai(yc, p_tx, bt), "so_anh_loi_tien_xu_ly": loi,
               "chinh_sach": "Kết quả chính trên MNIST KHÔNG cắt/căn lại ảnh chính thức; đường tiền xử lý chỉ để so sánh"}
    ghi_json(Path(thu_muc_ra) / "preprocessing_comparison.json", so_sanh)
    return chi_so, dong


# ------------------------------------------------------------------------------------ bài toán hình học: stress, nhóm, quét góc-độ dày
def chi_so_theo_nhom(y, du_doan, meta):
    kq = []
    for dac_trung, cac_khoang in (("goc_do", [(i, i + 60) for i in range(0, 360, 60)]), ("do_day", [(.5, 1.5), (1.5, 2.5), (2.5, 3.5)])):
        gia_tri = np.array([float(m[dac_trung]) for m in meta])
        for thap, cao in cac_khoang:
            mat_na = (gia_tri >= thap) & (gia_tri < cao)
            if dac_trung == "goc_do":
                mat_na &= y != 0                  # hướng của hình tròn không xác định: không đưa vào phân tích góc
            kq.append({"dac_trung": dac_trung, "tu": thap, "den_khong_gom": cao, "so_anh": int(mat_na.sum()),
                       "do_chinh_xac": float((y[mat_na] == du_doan[mat_na]).mean()) if mat_na.any() else None})
    return kq


def quet_goc_do_day(thu_muc_ra=None):
    """Quét ghép cặp 40 đối tượng gốc x 12 góc x 3 độ dày (tam giác và chữ nhật, bộ stress) cho cả hai cấu hình."""
    from bo_du_lieu import ve_hinh
    from bo_du_lieu.hinh_hoc import dau_vao_cnn
    plt = _plt()
    bt = lay_bai_toan("hinh_hoc")
    ra = Path(thu_muc_ra) if thu_muc_ra else bt.thu_muc_ket_qua
    goc_do, do_day = list(range(0, 360, 30)), [1, 2, 3]
    dau_vao, nhan, meta = [], [], []
    for d in do_day:
        for g in goc_do:
            for lop in (1, 2):
                for co_so in range(20):
                    hat = 777000 + lop * 1000 + co_so
                    tho, _ = ve_hinh(lop, np.random.default_rng(hat), stress=True, goc=g, do_day=d)
                    dau_vao.append(dau_vao_cnn(tho))
                    nhan.append(lop)
                    meta.append({"goc": g, "do_day": d})
    x, y = np.array(dau_vao)[:, None].astype(np.float32) / 255, np.array(nhan)
    hang, hinh_ve = [], plt.subplots(1, 2, figsize=(12, 3.8))
    hinh, truc = hinh_ve
    for t, bien_the in zip(truc, BIEN_THE):
        du_doan = DuDoan("hinh_hoc", bt.tep_mo_hinh(bien_the)).xac_suat(x).argmax(1)
        ma_tran = np.zeros((3, 12))
        for ti, d in enumerate(do_day):
            for ai, g in enumerate(goc_do):
                mat_na = np.array([m["goc"] == g and m["do_day"] == d for m in meta])
                ma_tran[ti, ai] = (du_doan[mat_na] == y[mat_na]).mean() * 100
                hang.append({"bien_the": bien_the, "goc_do": g, "do_day": d, "so_anh": int(mat_na.sum()),
                             "so_dung": int((du_doan[mat_na] == y[mat_na]).sum()), "do_chinh_xac": float(ma_tran[ti, ai] / 100)})
        t.imshow(ma_tran, cmap="YlGnBu", vmin=50, vmax=100, aspect="auto")
        for i in range(3):
            for j in range(12):
                t.text(j, i, f"{ma_tran[i, j]:.0f}", ha="center", va="center", fontsize=8,
                       color="white" if ma_tran[i, j] > 80 else "black")
        t.set(xticks=range(12), xticklabels=goc_do, yticks=range(3), yticklabels=do_day, xlabel="Góc xoay (độ)",
              ylabel="Độ dày nét (px)", title=f"{bien_the}: độ chính xác (%)")
    hinh.suptitle("Quét ghép cặp: 40 đối tượng khớp nhau mỗi ô (dữ liệu tổng hợp, không phải ảnh người vẽ)")
    hinh.tight_layout()
    hinh.savefig(ra / "rotation_thickness_sweep.png", dpi=180)
    plt.close(hinh)
    ghi_json(ra / "robustness_sweep.json", {"so_anh_moi_mo_hinh": len(y), "cac_o": hang,
                                            "muc_dich": "chẩn đoán sau khi chọn mô hình, không dùng để chọn hay chỉnh mô hình",
                                            "nguon": "ảnh tổng hợp, không phải ảnh người vẽ"})
    return hang


# ------------------------------------------------------------------------------------ điều phối
def danh_gia(bai_toan, thu_muc_anh_that=None, in_ra=True):
    """Đánh giá cả hai cấu hình trên tập test (và stress/chẩn đoán/ảnh thật tùy bài toán); ghi toàn bộ vào ket_qua."""
    bt = lay_bai_toan(bai_toan)
    lua_chon = doc_lua_chon(bt.ten)
    if sha256(bt.tep_mo_hinh("tot_nhat")) != lua_chon["sha256_tot_nhat"]:
        raise ValueError("Checkpoint cuối đã thay đổi so với lúc chọn bằng validation")
    x, y = tai_tap(bt.ten, "test")
    meta_test = tai_meta_hinh_hoc("test") if bt.ten == "hinh_hoc" else None
    xac_suat_tat_ca, ket_qua = {}, {}
    for bien_the in BIEN_THE:
        tep = bt.tep_mo_hinh(bien_the)
        if sha256(tep) != lua_chon["ung_vien"][bien_the]["sha256_checkpoint"]:
            raise ValueError(f"Checkpoint {bien_the} đã thay đổi so với lúc chọn bằng validation")
        du_doan = DuDoan(bt.ten, tep)
        ra = bt.thu_muc_ket_qua / bien_the
        ra.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        xs = du_doan.xac_suat(x)
        giay = time.perf_counter() - t0
        pred = xs.argmax(1)
        m = chi_so_phan_loai(y, pred, bt)
        m.update(loss=float(-np.log(np.clip(xs[np.arange(len(y)), y], 1e-12, 1)).mean()), giay_suy_luan=giay,
                 sha256_checkpoint=sha256(tep), nguon="tập test, giữ nguyên")
        ghi_json(ra / "test_metrics.json", m)
        np.savez_compressed(ra / "test_predictions.npz", y_that=y, y_du_doan=pred, xac_suat=xs, chi_so=np.arange(len(y)))
        ghi_csv(ra / "per_class_metrics.csv", m["theo_lop"])
        ghi_ma_tran_nham_lan_csv(ra / "confusion_matrix.csv", m["ma_tran_nham_lan"])
        ve_ma_tran_nham_lan(m["ma_tran_nham_lan"], bt, f"{bien_the}: tập test (n={len(y):,})", ra)
        ve_duong_cong_hoc(ra, bien_the)
        ghi_anh_loi(x, y, xs, bt, ra, bien_the, meta_test)
        if bt.ten == "hinh_hoc":
            ghi_json(ra / "grouped_metrics.json", chi_so_theo_nhom(y, pred, meta_test))
            xt, yt = tai_tap(bt.ten, "stress")
            xs_s = du_doan.xac_suat(xt)
            ghi_json(ra / "stress_metrics.json", chi_so_phan_loai(yt, xs_s.argmax(1), bt))
            (ra / "stress").mkdir(exist_ok=True)
            ghi_anh_loi(xt, yt, xs_s, bt, ra / "stress", f"{bien_the} (stress)", tai_meta_hinh_hoc("stress"))
        else:
            chi_so_cd, dong_cd = chan_doan_chu_so(du_doan, x, y, ra)
            ket_qua.setdefault("chan_doan", []).extend({"bien_the": bien_the, **d} for d in dong_cd)
        danh_gia_anh_that(bt, du_doan, ra, thu_muc_anh_that)
        ket_qua[bien_the] = m
        xac_suat_tat_ca[bien_the] = xs
        if in_ra:
            print(f"{bt.ten}/{bien_the}: test {m['so_dung']}/{m['so_anh']} = {m['do_chinh_xac']:.4%}; "
                  f"F1 macro {m['f1_macro']:.4%}", flush=True)
    # So sánh ghép cặp trên cùng ảnh test (McNemar chính xác), không phải hai lần chạy độc lập.
    co_so = xac_suat_tat_ca["co_so"].argmax(1) == y
    cai_tien = xac_suat_tat_ca["cai_tien"].argmax(1) == y
    n01, n10 = int((~co_so & cai_tien).sum()), int((co_so & ~cai_tien).sum())
    n = n01 + n10
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(n01, n10) + 1)) / (2 ** n)) if n else 1.0
    so_sanh = {"bien_the_duoc_chon": lua_chon["bien_the_duoc_chon"], "nguon_chon": "chỉ validation",
               "chenh_lech_accuracy_cai_tien_tru_co_so": float(cai_tien.mean() - co_so.mean()),
               "co_so_sai_cai_tien_dung": n01, "co_so_dung_cai_tien_sai": n10, "ca_hai_sai": int((~co_so & ~cai_tien).sum()),
               "ca_hai_dung": int((co_so & cai_tien).sum()), "mcnemar_chinh_xac_hai_phia_p": p, "so_hat": 1,
               "han_che": "Một hạt, một bộ đánh giá. Kiểm định ghép cặp không chứng minh hiệu năng trên ảnh ngoài."}
    ghi_json(bt.thu_muc_ket_qua / "comparison.json", so_sanh)
    dong = []
    for b in BIEN_THE:
        c = doc_json(bt.thu_muc_ket_qua / b / "config.json")
        dong.append({"bien_the": b, "so_tham_so": c["so_tham_so"], "epoch_tot_nhat": c["epoch_tot_nhat"],
                     "accuracy_validation": c["accuracy_validation_tot_nhat"], "accuracy_test": ket_qua[b]["do_chinh_xac"],
                     "f1_macro_test": ket_qua[b]["f1_macro"], "giay_huan_luyen": c["giay_huan_luyen"]})
        if bt.ten == "hinh_hoc":
            dong[-1]["accuracy_stress"] = doc_json(bt.thu_muc_ket_qua / b / "stress_metrics.json")["do_chinh_xac"]
    ghi_csv(bt.thu_muc_ket_qua / "comparison.csv", dong)
    if "chan_doan" in ket_qua:
        ghi_csv(bt.thu_muc_ket_qua / "diagnostic_comparison.csv", ket_qua["chan_doan"])
    ve_so_sanh(bt, dong, ket_qua.get("chan_doan"))
    return ket_qua


def ve_so_sanh(bt, dong, chan_doan):
    plt = _plt()
    ten = ["Cơ sở", "Cải tiến"]
    nhan_cot = [("accuracy_test", "Test (%)")] + ([("accuracy_stress", "Stress tổng hợp (%)")] if bt.ten == "hinh_hoc" else [])
    hinh, t = plt.subplots(figsize=(7, 4))
    vi_tri = np.arange(2)
    do_rong = .8 / len(nhan_cot)
    for k, (khoa, nhan) in enumerate(nhan_cot):
        gia_tri = [r[khoa] * 100 for r in dong]
        cot = t.bar(vi_tri + (k - (len(nhan_cot) - 1) / 2) * do_rong, gia_tri, do_rong, label=nhan)
        t.bar_label(cot, fmt="%.2f", padding=3)
    t.set(xticks=vi_tri, xticklabels=ten, ylabel="Độ chính xác (%)", ylim=(0, 110), title=f"{bt.ten}: so sánh hai cấu hình")
    t.legend(loc="lower right")
    t.grid(axis="y", alpha=.2)
    t.set_axisbelow(True)
    hinh.tight_layout()
    hinh.savefig(bt.thu_muc_ket_qua / "model_comparison.png", dpi=180)
    plt.close(hinh)
    if chan_doan:
        ten_dk = list(dict.fromkeys(r["dieu_kien"] for r in chan_doan))
        hinh, t = plt.subplots(figsize=(9, 4))
        xx = np.arange(len(ten_dk))
        for j, b in enumerate(BIEN_THE):
            t.bar(xx + (j - .5) * .35, [r["do_chinh_xac"] * 100 for r in chan_doan if r["bien_the"] == b], .35, label=b)
        t.set(xticks=xx, xticklabels=ten_dk, ylim=(0, 100), ylabel="Độ chính xác (%)",
              title="Biến đổi chẩn đoán trên cùng 1.000 ảnh test MNIST")
        t.legend()
        t.grid(axis="y", alpha=.2)
        t.set_axisbelow(True)
        hinh.tight_layout()
        hinh.savefig(bt.thu_muc_ket_qua / "robustness_comparison.png", dpi=180)
        plt.close(hinh)
