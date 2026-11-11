"""Biên dịch báo cáo LaTeX: chép hình mới nhất từ du_lieu/ và ket_qua/ vào hinh/ rồi gọi tectonic hoặc xelatex.

    python3 bao_cao/latex/bien_dich.py                    đồng bộ hình + biên dịch bao_cao.tex
    python3 bao_cao/latex/bien_dich.py --chi-dong-bo-hinh

Không có tectonic/xelatex: tải thư mục bao_cao/latex lên Overleaf, chọn XeLaTeX, file chính bao_cao.tex.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

THU_MUC = Path(__file__).resolve().parent
GOC = THU_MUC.parents[1]

# tên trong hinh/ -> nguồn trong dự án; đồng bộ trước khi biên dịch để hình luôn khớp kết quả hiện tại
HINH_BAO_CAO = {
    "chu_so_xem_truoc.png": "du_lieu/mnist/dataset_preview.png",
    "chu_so_ma_tran.png": "ket_qua/chu_so/cai_tien/confusion_matrix.png",
    "chu_so_loi.png": "ket_qua/chu_so/cai_tien/failure_cases.png",
    "hinh_hoc_xem_truoc.png": "du_lieu/hinh_hoc_tu_sinh/dataset_preview.png",
    "hinh_hoc_ma_tran.png": "ket_qua/hinh_hoc/cai_tien/confusion_matrix.png",
    "hinh_hoc_quet.png": "ket_qua/hinh_hoc/rotation_thickness_sweep.png",
    "net_manh_chu_so.png": "ket_qua/tich_hop/net_manh_chu_so.png",
    "ve_pil_hinh_hoc.png": "ket_qua/tich_hop/ve_pil_hinh_hoc.png",
    "so_sanh_du_lieu_hinh.png": "ket_qua/tich_hop/so_sanh_du_lieu_hinh.png",
}


def dong_bo_hinh():
    dich = THU_MUC / "hinh"
    dich.mkdir(exist_ok=True)
    thieu = [nguon for nguon in HINH_BAO_CAO.values() if not (GOC / nguon).exists()]
    for ten, nguon in HINH_BAO_CAO.items():
        if (GOC / nguon).exists():
            shutil.copyfile(GOC / nguon, dich / ten)
    if thieu:
        print("Thiếu hình nguồn (chạy lại bước tạo chúng):", *thieu, sep="\n  ", file=sys.stderr)
    return not thieu


def bien_dich():
    xelatex, tectonic, bibtex = shutil.which("xelatex"), shutil.which("tectonic"), shutil.which("bibtex")
    if tectonic:
        return subprocess.call([tectonic, "-X", "compile", "bao_cao.tex"], cwd=THU_MUC)
    if xelatex:
        buoc = [[xelatex, "-interaction=nonstopmode", "-halt-on-error", "bao_cao.tex"]]
        if bibtex:
            buoc += [[bibtex, "bao_cao"], buoc[0]]
        for lenh in buoc + [buoc[0]]:
            ma = subprocess.call(lenh, cwd=THU_MUC)
            if ma:
                return ma
        return 0
    print("Chưa có xelatex hoặc tectonic. Có thể tải thư mục bao_cao/latex lên Overleaf, chọn XeLaTeX, file chính "
          "bao_cao.tex.", file=sys.stderr)
    return 2


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chi-dong-bo-hinh", action="store_true", help="chỉ chép hình vào hinh/, không biên dịch")
    args = p.parse_args(argv)
    dong_bo_hinh()
    return 0 if args.chi_dong_bo_hinh else bien_dich()


if __name__ == "__main__":
    sys.exit(main())
