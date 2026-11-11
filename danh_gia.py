"""Đánh giá mô hình đã chọn và chạy các thí nghiệm đo độ bền (chạy được từ bất kỳ thư mục nào).

    python3 danh_gia.py --bai-toan hinh_hoc        test, stress/chẩn đoán, ảnh thật -> ket_qua/<bai_toan>/
    python3 danh_gia.py --thu-nghiem               cả ba thí nghiệm -> ket_qua/tich_hop/
    python3 danh_gia.py --thu-nghiem net_manh      một thí nghiệm
"""
import sys

sys.dont_write_bytecode = True
for _luong in (sys.stdout, sys.stderr):       # Windows in bằng cp1252 khi chuyển hướng đầu ra, làm hỏng tiếng Việt
    if hasattr(_luong, "reconfigure"):
        _luong.reconfigure(encoding="utf-8")

import argparse
import importlib
from pathlib import Path

THU_NGHIEM = ["net_manh", "ve_pil", "so_sanh_du_lieu_hinh"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    viec = p.add_mutually_exclusive_group(required=True)
    viec.add_argument("--bai-toan", choices=["chu_so", "hinh_hoc", "ca_hai"])
    viec.add_argument("--thu-nghiem", nargs="?", const="tat_ca", choices=["tat_ca", *THU_NGHIEM])
    p.add_argument("--thu-muc-anh-that", type=Path, help="thư mục ảnh thật ở nơi khác (có thư mục con theo lớp)")
    p.add_argument("--so-luong", type=int, default=6, help="số luồng CPU")
    args = p.parse_args(argv)

    import torch
    torch.set_num_threads(args.so_luong)
    if args.thu_nghiem:
        for ten in THU_NGHIEM if args.thu_nghiem == "tat_ca" else [args.thu_nghiem]:
            print(f"\n=== thu_nghiem/{ten} ===", flush=True)
            importlib.import_module(f"thu_nghiem.{ten}").chay()
        from loi.cau_hinh import GOC
        from thu_nghiem.bieu_do import ve_tat_ca
        for duong_dan in ve_tat_ca():
            print("Đã vẽ", duong_dan.relative_to(GOC))
        return 0

    from loi.danh_gia import danh_gia, quet_goc_do_day
    for bai_toan in ("chu_so", "hinh_hoc") if args.bai_toan == "ca_hai" else (args.bai_toan,):
        danh_gia(bai_toan, args.thu_muc_anh_that)
        if bai_toan == "hinh_hoc":
            quet_goc_do_day()
    return 0


if __name__ == "__main__":
    sys.exit(main())
