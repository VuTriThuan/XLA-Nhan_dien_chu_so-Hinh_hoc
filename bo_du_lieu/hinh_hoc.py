"""Bộ sinh ảnh hình học tổng hợp: tròn, tam giác, chữ nhật.

Rasterizer giải tích bằng NumPy với siêu lấy mẫu 4x4, không dùng hàm vẽ của OpenCV/Pillow. Mỗi ảnh gốc 28x28 được
đưa qua CHÍNH tiền xử lý của hệ thống (`xu_ly_anh`) để tạo đầu vào CNN, nên huấn luyện và nhận dạng dùng chung một
đường xử lý. Ba tập train/validation/test chia phân tầng 70/15/15 và tách rời nhau; bộ "stress" là tập khó riêng,
không dùng để huấn luyện hay chọn mô hình.
"""
import csv
import hashlib
import math
from pathlib import Path

import numpy as np

from loi.cau_hinh import ANH_MAU, HINH_HOC, doc_json, ghi_csv, ghi_json
from tien_xu_ly import LoiXuLyAnh, xu_ly_anh

THU_MUC = HINH_HOC.thu_muc_du_lieu
HAT_MAC_DINH = 20261108
TI_LE_MAC_DINH = 3                      # 4.500 x 3 = 13.500 ảnh
GOC_TAM_GIAC_TOI_THIEU = 18.0           # độ: mọi góc trong của tam giác >= giá trị này
MO_TA_DA_DANG = {
    "tam_giac": "đều 25%, vuông 20%, cân 20%, bất kỳ 35%; mọi góc trong >= 18 độ",
    "chu_nhat": "tỉ lệ cạnh U(0,3; 1), 25% hình vuông; cạnh ngắn >= độ dày nét + 2 px để lõi không bị lấp đặc",
    "tron": "50% elip (tỉ lệ trục U(0,7; 1)), 50% lượn sóng bán kính tới 5%, còn lại tròn hoàn hảo",
    "rung_dinh": "tam giác/chữ nhật: 50% có nhiễu Gauss lên đỉnh, sigma tới 2,5% kích thước (bộ stress: luôn 3,5%)",
}


def khoang_cach_den_doan(x, y, a, b):
    vx, vy = b - a
    t = np.clip(((x - a[0]) * vx + (y - a[1]) * vy) / (vx * vx + vy * vy), 0, 1)
    return np.sqrt((x - (a[0] + t * vx)) ** 2 + (y - (a[1] + t * vy)) ** 2)


def dinh_tam_giac(rng, ban_kinh):
    """Tam giác nội tiếp đường tròn bán kính `ban_kinh` (đường kính ngoại tiếp = kích thước hình).

    Cung giữa hai đỉnh liên tiếp bằng 2 x góc đối diện, nên chọn cung chính là chọn góc trong.
    """
    loai = str(rng.choice(["deu", "vuong", "can", "bat_ky"], p=[.25, .20, .20, .35]))
    thap = math.radians(2 * GOC_TAM_GIAC_TOI_THIEU)           # cung nhỏ nhất ứng với góc trong 18 độ
    if loai == "deu":
        cung = [2 * math.pi / 3] * 3
    elif loai == "vuong":                                       # cạnh huyền là đường kính: một cung bằng pi
        khac = float(rng.uniform(thap, math.pi - thap))
        cung = [math.pi, khac, math.pi - khac]
    elif loai == "can":
        a = float(rng.uniform(thap, math.pi - thap / 2))
        cung = [a, a, 2 * math.pi - 2 * a]
    else:
        cung = list(thap + (2 * math.pi - 3 * thap) * rng.dirichlet([1.0, 1.0, 1.0]))
    cung = [cung[i] for i in rng.permutation(3)]
    pha = np.concatenate(([0.0], np.cumsum(cung[:-1])))
    return loai, np.column_stack((np.cos(pha), np.sin(pha))) * ban_kinh


def ve_hinh(lop, rng, *, stress=False, goc=None, do_day=None, da_dang=True):
    """Vẽ một hình (lop: 0 tròn, 1 tam giác, 2 chữ nhật). Trả về (ảnh uint8 28x28, siêu dữ liệu).

    da_dang=False tái hiện kiểu dữ liệu đơn giản của bản đầu (chỉ tam giác đều, chữ nhật tỉ lệ 0,55-1, hình tròn
    hoàn hảo, không rung đỉnh) để thí nghiệm so sánh; mặc định dùng bộ sinh đa dạng.
    """
    goc_ep, do_day_ep = goc, do_day                # giá trị bắt buộc của người gọi (None = ngẫu nhiên), dùng khi vẽ lại
    kich_thuoc = float(rng.uniform(7, 21))        # đường kính ngoại tiếp, KHÔNG phải bán kính 21 trên canvas 28
    do_day = float(do_day_ep if do_day_ep is not None else rng.choice([1, 2, 3]))
    goc = float(goc_ep if goc_ep is not None else rng.uniform(0, 360))
    theta = math.radians(goc)
    ban_kinh = kich_thuoc / 2
    ti_le_canh, bien_the, lech = 1.0, HINH_HOC.lop[lop], 0.0
    dinh = None    # đa giác (3-4 đỉnh) hoặc đường cong kín (đường tròn méo); None = đường tròn hoàn hảo
    if lop == 0:
        ti_le_truc = (float(rng.uniform(.7, 1.0)) if (stress or rng.random() < .5) else 1.0) if da_dang else 1.0
        luon_song = (float(rng.uniform(.01, .05)) if (stress or rng.random() < .5) else 0.0) if da_dang else 0.0
        if ti_le_truc < 1.0 or luon_song > 0.0:
            t = np.linspace(0, 2 * math.pi, 96, endpoint=False)
            r = ban_kinh * (1 + luon_song * np.sin(int(rng.integers(2, 5)) * t + float(rng.uniform(0, 2 * math.pi))))
            dinh = np.column_stack((r * np.cos(t), ti_le_truc * r * np.sin(t)))
            ti_le_canh = ti_le_truc
            bien_the = ("elip_luon_song" if luon_song > 0 else "elip") if ti_le_truc < 1.0 else "tron_luon_song"
    elif lop == 1 and not da_dang:
        pha = np.arange(3) * 2 * math.pi / 3 - math.pi / 2
        dinh = np.column_stack((np.cos(pha), np.sin(pha))) * ban_kinh
        bien_the = "tam_giac_deu"
    elif lop == 1:
        loai, dinh = dinh_tam_giac(rng, ban_kinh)
        bien_the = "tam_giac_" + loai
    else:
        ti_le_canh = float(rng.uniform(.3, 1)) if da_dang else float(rng.uniform(.55, 1))
        if rng.random() < .25:
            ti_le_canh = 1.0
        rong = kich_thuoc / math.sqrt(1 + ti_le_canh ** 2)
        cao = rong * ti_le_canh
        if da_dang and cao < do_day + 2.0:                      # giữ phần lõi rỗng, không biến thành một vệt đặc
            cao = min(do_day + 2.0, kich_thuoc * .7)
            rong = math.sqrt(max(kich_thuoc ** 2 - cao ** 2, 1.0))
            ti_le_canh = cao / rong
        dinh = np.array([[-rong / 2, -cao / 2], [rong / 2, -cao / 2], [rong / 2, cao / 2], [-rong / 2, cao / 2]])
        bien_the = "vuong" if ti_le_canh >= .999 else "chu_nhat"
    if lop != 0 and (da_dang or stress):
        lech = kich_thuoc * .035 if stress else (kich_thuoc * float(rng.uniform(0, .025)) if rng.random() < .5 else 0.0)
        if lech > 0:
            dinh = dinh + rng.normal(0, lech, dinh.shape)
    if dinh is not None:
        quay = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        dinh = dinh @ quay.T
        thap, cao_nhat = dinh.min(axis=0), dinh.max(axis=0)
    else:
        thap, cao_nhat = np.full(2, -ban_kinh), np.full(2, ban_kinh)
    le = do_day / 2 + .6
    # Miền tâm phụ thuộc kích thước đối tượng nên đường viền luôn nằm trọn trong ảnh.
    duoi = np.maximum(14 - 5.6, le - thap)
    tren = np.minimum(14 + 5.6, 28 - le - cao_nhat)
    if np.any(duoi > tren):
        return ve_hinh(lop, rng, stress=stress, goc=goc_ep, do_day=do_day_ep, da_dang=da_dang)
    tam = rng.uniform(duoi, tren)
    ss = 4
    yy, xx = np.meshgrid((np.arange(28 * ss) + .5) / ss, (np.arange(28 * ss) + .5) / ss, indexing="ij")
    if dinh is None:
        kc = np.abs(np.sqrt((xx - tam[0]) ** 2 + (yy - tam[1]) ** 2) - ban_kinh)
    else:
        dinh = dinh + tam
        kc = np.minimum.reduce([khoang_cach_den_doan(xx, yy, dinh[j], dinh[(j + 1) % len(dinh)])
                                for j in range(len(dinh))])
    cao_do = (kc <= do_day / 2).astype(np.float32)
    anh = cao_do.reshape(28, ss, 28, ss).mean(axis=(1, 3))
    sigma = float(rng.uniform(0, .05 if not stress else .10))
    muoi_tieu = float(rng.uniform(0, .05 if not stress else .08)) if rng.random() < .35 else 0.0
    if stress:
        # Phép thử độ bền tổng hợp, KHÔNG phải chữ viết/hình vẽ tay của người.
        anh *= rng.uniform(.65, 1.0)
        if rng.random() < .4:
            cot = int(rng.integers(6, 22))
            anh[:, cot:cot + 1] *= .15
    anh = np.clip(anh + rng.normal(0, sigma, anh.shape), 0, 1)
    ngau_nhien = rng.random(anh.shape)
    anh[ngau_nhien < muoi_tieu / 2] = 0
    anh[ngau_nhien > 1 - muoi_tieu / 2] = 1
    tho = np.rint(anh * 255).astype(np.uint8)
    meta = {"lop": lop, "ten_lop": HINH_HOC.lop[lop], "bien_the": bien_the, "duong_kinh": kich_thuoc,
            "tam_x": float(tam[0]), "tam_y": float(tam[1]), "do_day": do_day, "goc_do": goc,
            "ti_le_canh": ti_le_canh, "do_lech_dinh": lech / kich_thuoc, "sigma_gauss": sigma,
            "xac_suat_muoi_tieu": muoi_tieu, "nguon": "tong_hop_kho" if stress else "tong_hop"}
    return tho, meta


def dau_vao_cnn(tho):
    """Ảnh gốc 28x28 -> đầu vào CNN uint8, bằng đúng tiền xử lý của hệ thống."""
    return np.rint(xu_ly_anh(tho, "hinh_hoc").anh * 255).astype(np.uint8)


def sinh_bo_du_lieu(thu_muc=THU_MUC, hat=HAT_MAC_DINH, ti_le=TI_LE_MAC_DINH, anh_mau=ANH_MAU, da_dang=True):
    """Sinh tập train/validation/test (4.500 x ti_le ảnh) và bộ stress 300 ảnh; ghi .npz, metadata, thống kê."""
    thu_muc = Path(thu_muc)
    thu_muc.mkdir(parents=True, exist_ok=True)
    anh_tho, dau_vao, nhan, ban_ghi, chi_so = [], [], [], [], {}
    da_thay = set()
    for so_tap, (tap, moi_lop) in enumerate((("train", 1050 * ti_le), ("validation", 225 * ti_le), ("test", 225 * ti_le))):
        rng = np.random.default_rng(np.random.SeedSequence([hat, so_tap]))
        muc_tieu = np.repeat(np.arange(3), moi_lop)
        rng.shuffle(muc_tieu)
        dau = len(anh_tho)
        for lop in muc_tieu:
            while True:
                tho, meta = ve_hinh(int(lop), rng, da_dang=da_dang)
                ma = hashlib.sha256(tho.tobytes()).hexdigest()
                if ma in da_thay:
                    continue
                try:
                    xu_ly = dau_vao_cnn(tho)
                except LoiXuLyAnh:
                    continue
                break
            da_thay.add(ma)
            meta.update(chi_so=len(anh_tho), tap=tap, sha256=ma)
            anh_tho.append(tho)
            dau_vao.append(xu_ly)
            nhan.append(lop)
            ban_ghi.append(meta)
        chi_so[tap] = np.arange(dau, len(anh_tho), dtype=np.int64)
        print(f"Đã sinh {tap}: {len(muc_tieu)}", flush=True)
    anh_tho, dau_vao, nhan = np.stack(anh_tho), np.stack(dau_vao), np.array(nhan, dtype=np.int64)
    np.savez_compressed(thu_muc / "shapes_dataset.npz", images=anh_tho, inputs=dau_vao, labels=nhan, **chi_so)
    ghi_csv(thu_muc / "metadata.csv", ban_ghi)
    thong_ke = {"hat": hat, "ti_le": ti_le, "cac_lop": list(HINH_HOC.lop), "kich_thuoc_anh": [28, 28], "tong": len(nhan),
                "so_anh_khac_nhau": len(da_thay),
                "so_luong_moi_tap": {t: {HINH_HOC.lop[c]: int((nhan[i] == c).sum()) for c in range(3)}
                                     for t, i in chi_so.items()},
                "cac_tap_roi_nhau": True, "dinh_nghia_kich_thuoc": "đường kính ngoại tiếp trong [7, 21] pixel",
                "dich_chuyen": "tâm trong 14 ± 5,6 pixel, bị ràng buộc để hình nằm trọn trong ảnh",
                "nhieu": "Gauss sigma U(0, 0,05), muối tiêu p U(0, 0,05) trên 35% ảnh",
                "sieu_lay_mau": 4, "da_dang": MO_TA_DA_DANG if da_dang else "KHÔNG: dữ liệu đơn giản (tam giác đều, chữ nhật 0,55-1, tròn hoàn hảo)",
                "tien_xu_ly_dau_vao": "tien_xu_ly.xu_ly_anh(bai_toan='hinh_hoc'), mặc định",
                "sha256_bo_du_lieu": hashlib.sha256((thu_muc / "shapes_dataset.npz").read_bytes()).hexdigest(),
                "sha256_ma_sinh": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    # Bộ stress độc lập: chỉ để đo độ bền, không dùng để huấn luyện hay chọn mô hình.
    rng = np.random.default_rng(np.random.SeedSequence([hat, 999]))
    tho_s, vao_s, nhan_s, meta_s = [], [], [], []
    for lop in np.repeat(np.arange(3), 100):
        while True:
            tho, meta = ve_hinh(int(lop), rng, stress=True)
            try:
                xu_ly = dau_vao_cnn(tho)
                break
            except LoiXuLyAnh:
                continue
        tho_s.append(tho)
        vao_s.append(xu_ly)
        nhan_s.append(lop)
        meta_s.append(meta)
    np.savez_compressed(thu_muc / "synthetic_stress.npz", images=np.stack(tho_s), inputs=np.stack(vao_s),
                        labels=np.array(nhan_s, dtype=np.int64))
    ghi_json(thu_muc / "stress_metadata.json", meta_s)
    thong_ke["so_anh_stress"] = len(nhan_s)
    thong_ke["so_anh_nguoi_that"] = 0
    ghi_json(thu_muc / "data_stats.json", thong_ke)
    ve_anh_xem_truoc(anh_tho, nhan, thu_muc / "dataset_preview.png")
    if anh_mau:
        luu_anh_mau(anh_tho, nhan, chi_so["test"], Path(anh_mau))
    return thong_ke


def luu_anh_mau(anh_tho, nhan, chi_so_test, thu_muc):
    """Lưu mỗi lớp một ảnh 28x28 đầu tiên của tập test để thử nhanh; không phải ảnh người vẽ."""
    from PIL import Image
    thu_muc.mkdir(parents=True, exist_ok=True)
    nguon = []
    for lop, ten in enumerate(HINH_HOC.lop):
        i = int(chi_so_test[np.flatnonzero(nhan[chi_so_test] == lop)[0]])
        Image.fromarray(anh_tho[i]).save(thu_muc / f"hinh_{ten}.png")
        nguon.append({"tep": f"hinh_{ten}.png", "nguon": "tập test tổng hợp (không phải ảnh người vẽ)", "chi_so": i,
                      "lop": ten})
    ghi_json(thu_muc / "nguon_hinh_hoc.json", nguon)


def ve_anh_xem_truoc(anh_tho, nhan, duong_dan):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hinh, truc = plt.subplots(3, 8, figsize=(10, 4))
    for c in range(3):
        for j, i in enumerate(np.flatnonzero(nhan == c)[:8]):
            truc[c, j].imshow(anh_tho[i], cmap="gray", vmin=0, vmax=255)
            truc[c, j].axis("off")
            if j == 0:
                truc[c, j].set_title(HINH_HOC.nhan_vi[c], fontsize=10)
    hinh.suptitle("Ảnh hình học tổng hợp (gốc 28 × 28)")
    hinh.tight_layout()
    hinh.savefig(duong_dan, dpi=180)
    plt.close(hinh)


def tai_hinh_hoc(tap, thu_muc=THU_MUC):
    """Trả về (x float32 N x 1 x 28 x 28 trong [0,1], y int64). tap: train | validation | test | stress."""
    if tap == "stress":
        with np.load(Path(thu_muc) / "synthetic_stress.npz", allow_pickle=False) as d:
            return d["inputs"][:, None].astype(np.float32) / 255, d["labels"].astype(np.int64)
    if tap not in ("train", "validation", "test"):
        raise ValueError("tap phải là train, validation, test hoặc stress")
    with np.load(Path(thu_muc) / "shapes_dataset.npz", allow_pickle=False) as d:
        i = d[tap]
        return d["inputs"][i, None].astype(np.float32) / 255, d["labels"][i].astype(np.int64)


def tai_meta_hinh_hoc(tap, thu_muc=THU_MUC):
    """Siêu dữ liệu từng ảnh của một tập, cùng thứ tự với `tai_hinh_hoc(tap)`; các trường số được đổi sang float."""
    thu_muc = Path(thu_muc)
    if tap == "stress":
        return doc_json(thu_muc / "stress_metadata.json")
    with (thu_muc / "metadata.csv").open(encoding="utf-8-sig", newline="") as f:
        dong = [r for r in csv.DictReader(f) if r["tap"] == tap]
    chu = {"ten_lop", "bien_the", "nguon", "tap", "sha256"}
    return [{k: (v if k in chu else float(v)) for k, v in r.items()} for r in dong]
