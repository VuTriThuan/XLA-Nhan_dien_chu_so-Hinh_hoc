"""Vẽ biểu đồ từ kết quả thí nghiệm trong ket_qua/tich_hop/ (đọc JSON, không chạy lại thí nghiệm)."""
import numpy as np

from loi.cau_hinh import doc_json

from ._chung import THU_MUC_RA

NHAN_CAU_HINH = {"cu_song_tuyen_tinh": "Song tuyến tính (cách cũ)", "dien_tich": "Trung bình diện tích",
                 "dien_tich_keo_gian": "Diện tích + kéo giãn độ sáng"}
TEN_HO = {"tron": "Tròn", "tam_giac_deu": "Tam giác\nđều", "tam_giac_vuong": "Tam giác\nvuông", "tam_giac_bat_ky": "Tam giác\nbất kỳ",
          "chu_nhat_vua": "Chữ nhật\n(0,55–1)", "chu_nhat_dai": "Chữ nhật dài\n(0,25–0,45)"}
MAU = {"cu_song_tuyen_tinh": "#c0504d", "dien_tich": "#4f81bd", "dien_tich_keo_gian": "#2e8b57"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def ve_net_manh_chu_so():
    tep = THU_MUC_RA / "net_manh_chu_so.json"
    if not tep.exists():
        return None
    plt = _plt()
    d = doc_json(tep)["do_chinh_xac"]
    ti_le = list(d)
    hinh, t = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(ti_le))
    for k, ten in enumerate(NHAN_CAU_HINH):
        t.bar(x + (k - 1) * .27, [d[r][ten] * 100 for r in ti_le], .27, label=NHAN_CAU_HINH[ten], color=MAU[ten])
    t.set(xticks=x, xticklabels=[f"{float(r) * 100:g}%" for r in ti_le], ylim=(0, 105), ylabel="Độ chính xác (%)",
          xlabel="Độ dày nét / chiều cao chữ số", title="Chữ số nét mảnh (MNIST phóng to, làm mỏng nét)")
    t.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8, frameon=False)
    t.grid(axis="y", alpha=.2)
    t.set_axisbelow(True)
    hinh.tight_layout()
    hinh.savefig(THU_MUC_RA / "net_manh_chu_so.png", dpi=180)
    plt.close(hinh)
    return THU_MUC_RA / "net_manh_chu_so.png"


def ve_ve_pil():
    tep = THU_MUC_RA / "ve_pil_hinh_hoc.json"
    if not tep.exists():
        return None
    plt = _plt()
    d = doc_json(tep)["do_chinh_xac"]
    ho = list(d)
    hinh, t = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(ho))
    for k, ten in enumerate(NHAN_CAU_HINH):
        trung_binh = [np.mean([d[h][r][ten] for r in d[h]]) * 100 for h in ho]
        t.bar(x + (k - 1) * .27, trung_binh, .27, label=NHAN_CAU_HINH[ten], color=MAU[ten])
    t.set(xticks=x, xticklabels=[TEN_HO.get(h, h) for h in ho], ylim=(0, 105), ylabel="Độ chính xác (%)",
          title="Hình vẽ Pillow có rung tay (trung bình 4 độ dày nét)")
    t.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8, frameon=False)
    t.grid(axis="y", alpha=.2)
    t.set_axisbelow(True)
    hinh.tight_layout()
    hinh.savefig(THU_MUC_RA / "ve_pil_hinh_hoc.png", dpi=180)
    plt.close(hinh)
    return THU_MUC_RA / "ve_pil_hinh_hoc.png"


def ve_so_sanh_du_lieu_hinh():
    tep = THU_MUC_RA / "so_sanh_du_lieu_hinh.json"
    if not tep.exists():
        return None
    plt = _plt()
    kq = doc_json(tep)["ket_qua"]
    ten = list(kq)
    cac_bo = [("test_du_lieu_don_gian", "Test dữ liệu đơn giản"), ("test_du_lieu_da_dang", "Test dữ liệu đa dạng"),
              ("stress_da_dang", "Stress đa dạng"), ("ve_pil_trung_binh", "Hình vẽ Pillow (TB)")]
    hinh, t = plt.subplots(figsize=(9, 4))
    x = np.arange(len(cac_bo))
    do_rong = .8 / len(ten)
    for k, m in enumerate(ten):
        gia_tri = [kq[m][khoa] * 100 for khoa, _ in cac_bo]
        cot = t.bar(x + (k - (len(ten) - 1) / 2) * do_rong, gia_tri, do_rong, label=m)
        t.bar_label(cot, fmt="%.1f", fontsize=7, padding=2)
    t.set(xticks=x, xticklabels=[n for _, n in cac_bo], ylim=(40, 108), ylabel="Độ chính xác (%)",
          title="Dữ liệu huấn luyện hình học: đơn giản và đa dạng")
    t.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8, frameon=False)
    t.grid(axis="y", alpha=.2)
    t.set_axisbelow(True)
    hinh.tight_layout()
    hinh.savefig(THU_MUC_RA / "so_sanh_du_lieu_hinh.png", dpi=180)
    plt.close(hinh)
    return THU_MUC_RA / "so_sanh_du_lieu_hinh.png"


def ve_tat_ca():
    return [p for p in (ve_net_manh_chu_so(), ve_ve_pil(), ve_so_sanh_du_lieu_hinh()) if p]
