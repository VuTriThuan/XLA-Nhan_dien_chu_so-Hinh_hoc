"""MNIST chính thức: 54.000 train / 6.000 validation / 10.000 test (test giữ nguyên, không dùng để chọn mô hình)."""
import hashlib
import urllib.request
from pathlib import Path

import numpy as np

from loi.cau_hinh import ANH_MAU, CHU_SO, ghi_csv, ghi_json, sha256

THU_MUC = CHU_SO.thu_muc_du_lieu
URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
SHA256_DUNG = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
HAT_CHIA = 20261109


def chia_phan_tang(y, so_validation=6000, hat=HAT_CHIA):
    """Chia phân tầng theo phần dư lớn nhất để đủ đúng so_validation mẫu; hòa thì theo thứ tự chữ số."""
    y = np.asarray(y)
    dem = np.bincount(y, minlength=10)
    tho = dem * so_validation / len(y)
    han_ngach = np.floor(tho).astype(int)
    du = so_validation - int(han_ngach.sum())
    han_ngach[np.argsort(-(tho - han_ngach), kind="stable")[:du]] += 1
    rng = np.random.default_rng(hat)
    train, val = [], []
    for so in range(10):
        idx = rng.permutation(np.flatnonzero(y == so))
        val.extend(idx[:han_ngach[so]])
        train.extend(idx[han_ngach[so]:])
    return rng.permutation(train).astype(np.int64), rng.permutation(val).astype(np.int64)


def chuan_bi_mnist(thu_muc=THU_MUC, anh_mau=ANH_MAU, in_ra=True):
    """Tải MNIST nếu thiếu, kiểm tra SHA-256, tạo/đối chiếu chỉ số chia tập, ghi thống kê và ảnh mẫu."""
    thu_muc = Path(thu_muc)
    thu_muc.mkdir(parents=True, exist_ok=True)
    tep = thu_muc / "mnist.npz"
    if not tep.exists():
        tam = tep.with_suffix(".download")
        with urllib.request.urlopen(URL, timeout=90) as r, tam.open("wb") as f:
            for khoi in iter(lambda: r.read(1024 * 1024), b""):
                f.write(khoi)
        if sha256(tam) != SHA256_DUNG:
            raise ValueError("MNIST tải về có SHA-256 không đúng; giữ .download để kiểm tra")
        tam.replace(tep)
    if sha256(tep) != SHA256_DUNG:
        raise ValueError("SHA-256 của MNIST không khớp")
    with np.load(tep, allow_pickle=False) as d:
        x, y, xt, yt = (d[k] for k in ("x_train", "y_train", "x_test", "y_test"))
    if x.shape != (60000, 28, 28) or xt.shape != (10000, 28, 28) or x.dtype != np.uint8 or xt.dtype != np.uint8:
        raise ValueError("Mảng MNIST chính thức không như mong đợi")
    if y.shape != (60000,) or yt.shape != (10000,) or set(np.unique(y)) != set(range(10)):
        raise ValueError("Nhãn MNIST không như mong đợi")
    train, val = chia_phan_tang(y)
    tep_chia = thu_muc / "splits.npz"
    if tep_chia.exists():
        with np.load(tep_chia, allow_pickle=False) as s:
            if not np.array_equal(s["train"], train) or not np.array_equal(s["validation"], val):
                raise ValueError("Chỉ số chia tập hiện có khác kết quả tính lại; hãy dùng thư mục thí nghiệm mới")
    else:
        np.savez_compressed(tep_chia, train=train, validation=val)
    ma = [hashlib.sha256(a.tobytes()).hexdigest() for a in x]
    ma_test = {hashlib.sha256(a.tobytes()).hexdigest() for a in xt}
    ma_train, ma_val = {ma[i] for i in train}, {ma[i] for i in val}
    trung = {"train_validation": len(ma_train & ma_val), "train_test": len(ma_train & ma_test),
             "validation_test": len(ma_val & ma_test)}
    dong = [{"chu_so": i, "train": int(np.sum(y[train] == i)), "validation": int(np.sum(y[val] == i)),
             "test": int(np.sum(yt == i))} for i in range(10)]
    ghi_csv(thu_muc / "class_counts.csv", dong)
    thong_ke = {"nguon": URL, "tai_lieu_nguon": "https://keras.io/api/datasets/mnist/", "sha256_bo_du_lieu": SHA256_DUNG,
                "sha256_chia_tap": sha256(tep_chia), "hat_chia": HAT_CHIA, "so_train": len(train),
                "so_validation": len(val), "so_test": len(yt), "kich_thuoc_anh": [28, 28],
                "chuan_hoa": "float32 / 255; không trừ mean/chia std; giữ căn chỉnh gốc của MNIST",
                "so_luong_moi_lop": dong, "so_anh_trung_pixel_giua_cac_tap": trung,
                "chinh_sach_test": "10.000 ảnh test chính thức giữ nguyên, không dùng để chọn checkpoint hay mô hình",
                "giay_phep": "CC BY-SA 3.0; Yann LeCun và Corinna Cortes"}
    ghi_json(thu_muc / "data_stats.json", thong_ke)
    ve_anh_xem_truoc(x, y, train, thu_muc / "dataset_preview.png")
    if anh_mau:
        from PIL import Image
        anh_mau = Path(anh_mau)
        anh_mau.mkdir(parents=True, exist_ok=True)
        nguon = []
        for so in range(10):
            i = int(np.flatnonzero(yt == so)[0])
            Image.fromarray(xt[i]).save(anh_mau / f"chu_so_{so}.png")
            nguon.append({"tep": f"chu_so_{so}.png", "nguon": "MNIST test chính thức (không phải ảnh ngoài)",
                          "chi_so_test": i, "chu_so": so})
        ghi_json(anh_mau / "nguon_chu_so.json", nguon)
    if in_ra:
        print(f"MNIST: {len(train)} train / {len(val)} validation / {len(yt)} test", flush=True)
    return thong_ke


def ve_anh_xem_truoc(x, y, chi_so_train, duong_dan):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hinh, truc = plt.subplots(10, 8, figsize=(8, 9.5))
    for so in range(10):
        chon = chi_so_train[y[chi_so_train] == so][:8]
        for t, i in zip(truc[so], chon):
            t.imshow(x[i], cmap="gray", vmin=0, vmax=255)
            t.set_xticks([])
            t.set_yticks([])
        truc[so, 0].set_ylabel(str(so), rotation=0, labelpad=10)
    hinh.suptitle("MNIST: 8 ảnh huấn luyện mỗi chữ số (không phải ảnh ngoài)")
    hinh.tight_layout(rect=(0, 0, 1, .97))
    hinh.savefig(duong_dan, dpi=150)
    plt.close(hinh)


def tai_mnist(tap, thu_muc=THU_MUC):
    """Trả về (x float32 N x 1 x 28 x 28 trong [0,1], y int64). tap: train | validation | test."""
    thu_muc = Path(thu_muc)
    if tap not in ("train", "validation", "test"):
        raise ValueError("tap phải là train, validation hoặc test")
    with np.load(thu_muc / "mnist.npz", allow_pickle=False) as d:
        if tap == "test":
            x, y = d["x_test"], d["y_test"]
        else:
            with np.load(thu_muc / "splits.npz", allow_pickle=False) as s:
                i = s[tap]
            x, y = d["x_train"][i], d["y_train"][i]
    return x[:, None].astype(np.float32) / 255, y.astype(np.int64)
