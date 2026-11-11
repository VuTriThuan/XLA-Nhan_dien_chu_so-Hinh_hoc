"""Mở ứng dụng web: nhận diện ảnh và thu ảnh viết/vẽ tay thật (chạy được từ bất kỳ thư mục nào).

    python3 ung_dung.py                     http://127.0.0.1:8780
    python3 ung_dung.py --thu-anh           mở thẳng trang thu ảnh thật (/thu-anh)
    python3 ung_dung.py --port 9000 --mo-trinh-duyet
"""
import sys

sys.dont_write_bytecode = True
for _luong in (sys.stdout, sys.stderr):       # Windows in bằng cp1252 khi chuyển hướng đầu ra, làm hỏng tiếng Việt
    if hasattr(_luong, "reconfigure"):
        _luong.reconfigure(encoding="utf-8")

import argparse
import webbrowser


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8780)
    p.add_argument("--thu-anh", action="store_true", help="mở trang thu ảnh thật")
    p.add_argument("--mo-trinh-duyet", action="store_true", help="tự mở trình duyệt")
    args = p.parse_args(argv)

    from giao_dien.may_chu import chay
    trang = "/thu-anh" if args.thu_anh else ""
    if args.thu_anh:
        print(f"Trang thu ảnh thật: http://127.0.0.1:{args.port}{trang}", flush=True)
    if args.mo_trinh_duyet:
        webbrowser.open(f"http://127.0.0.1:{args.port}{trang}")
    chay(args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
