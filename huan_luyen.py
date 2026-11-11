"""Huấn luyện CNN (chạy được từ bất kỳ thư mục nào).

    python3 huan_luyen.py --sinh-du-lieu [--bai-toan ca_hai] [--ti-le 3]    chuẩn bị MNIST và sinh bộ dữ liệu hình học
    python3 huan_luyen.py --bai-toan hinh_hoc                              huấn luyện co_so + cai_tien rồi chọn bằng validation
    python3 huan_luyen.py --bai-toan chu_so --bien-the cai_tien --so-epoch 5

Kết quả: mo_hinh/<bai_toan>_<bien_the>.pth, mo_hinh/<bai_toan>_tot_nhat.pth, mo_hinh/lua_chon.json, ket_qua/<bai_toan>/.
"""
import sys

sys.dont_write_bytecode = True
for _luong in (sys.stdout, sys.stderr):       # Windows in bằng cp1252 khi chuyển hướng đầu ra, làm hỏng tiếng Việt
    if hasattr(_luong, "reconfigure"):
        _luong.reconfigure(encoding="utf-8")

import argparse


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bai-toan", choices=["chu_so", "hinh_hoc", "ca_hai"], default="ca_hai")
    p.add_argument("--sinh-du-lieu", action="store_true", help="chỉ chuẩn bị/sinh dữ liệu, không huấn luyện")
    p.add_argument("--ti-le", type=int, default=3, help="(sinh dữ liệu) nhân số ảnh hình học: 1 = 4.500 ảnh, 3 = 13.500")
    p.add_argument("--bien-the", choices=["co_so", "cai_tien", "ca_hai"], default="ca_hai")
    p.add_argument("--so-epoch", type=int)
    p.add_argument("--co-lo", type=int)
    p.add_argument("--toc-do-hoc", type=float)
    p.add_argument("--hat", type=int)
    p.add_argument("--so-luong", type=int, default=6, help="số luồng CPU")
    args = p.parse_args(argv)
    cac_bai_toan = ("chu_so", "hinh_hoc") if args.bai_toan == "ca_hai" else (args.bai_toan,)

    if args.sinh_du_lieu:
        from bo_du_lieu import chuan_bi_mnist, sinh_bo_du_lieu
        if "chu_so" in cac_bai_toan:
            chuan_bi_mnist()
        if "hinh_hoc" in cac_bai_toan:
            sinh_bo_du_lieu(ti_le=args.ti_le, **({"hat": args.hat} if args.hat is not None else {}))
        return 0

    from loi.huan_luyen import chon_mo_hinh, huan_luyen
    bien_the = ("co_so", "cai_tien") if args.bien_the == "ca_hai" else (args.bien_the,)
    for bai_toan in cac_bai_toan:
        for b in bien_the:
            huan_luyen(bai_toan, b, so_epoch=args.so_epoch, co_lo=args.co_lo, toc_do_hoc=args.toc_do_hoc, hat=args.hat,
                       so_luong=args.so_luong)
        if args.bien_the == "ca_hai":      # chỉ khi có đủ hai cấu hình mới khóa lựa chọn cuối bằng validation
            print(f"{bai_toan}: chọn bằng validation:", chon_mo_hinh(bai_toan)["bien_the_duoc_chon"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
