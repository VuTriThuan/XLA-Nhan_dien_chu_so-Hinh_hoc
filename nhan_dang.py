"""Nhận diện một ảnh và in kết quả dạng JSON (chạy được từ bất kỳ thư mục nào).

    python3 nhan_dang.py anh.png --bai-toan chu_so
    python3 nhan_dang.py anh.png --bai-toan hinh_hoc --loc 3 --luu-buoc cac_buoc/
"""
import sys

sys.dont_write_bytecode = True
for _luong in (sys.stdout, sys.stderr):       # Windows in bằng cp1252 khi chuyển hướng đầu ra, làm hỏng tiếng Việt
    if hasattr(_luong, "reconfigure"):
        _luong.reconfigure(encoding="utf-8")

import argparse
import json
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("anh", type=Path)
    p.add_argument("--bai-toan", choices=["chu_so", "hinh_hoc"], required=True)
    p.add_argument("--cuc-tinh", choices=["tu_dong", "net_toi", "net_sang"], default="tu_dong")
    p.add_argument("--loc", type=int, choices=[1, 3, 5], default=1, help="cửa sổ lọc trung vị (1 = tắt)")
    p.add_argument("--luu-buoc", type=Path, help="thư mục lưu ảnh từng bước tiền xử lý")
    args = p.parse_args(argv)

    from loi.du_doan import BoNhanDang
    from tien_xu_ly import LoiXuLyAnh, doc_anh_tu_tep
    try:
        ket_qua = BoNhanDang().nhan_dang(doc_anh_tu_tep(args.anh), args.bai_toan, cuc_tinh=args.cuc_tinh,
                                         kich_thuoc_loc=args.loc)
    except (OSError, LoiXuLyAnh) as loi:
        print("Lỗi:", loi, file=sys.stderr)
        return 2
    cac_buoc = ket_qua.pop("cac_buoc")
    if args.luu_buoc:
        from PIL import Image
        args.luu_buoc.mkdir(parents=True, exist_ok=True)
        for ten, anh in cac_buoc.items():
            Image.fromarray(anh).save(args.luu_buoc / f"{ten}.png")
    print(json.dumps(ket_qua, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
