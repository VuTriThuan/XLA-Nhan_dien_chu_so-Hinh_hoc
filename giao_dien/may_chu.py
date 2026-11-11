"""Máy chủ web: nhận dạng ảnh vẽ/tải lên và thu ảnh thật để đánh giá. Chỉ lắng nghe 127.0.0.1."""
import base64
import binascii
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

from loi.cau_hinh import BAI_TOAN, lay_bai_toan
from loi.du_doan import BoNhanDang
from bo_du_lieu.thu_thap import dem_mau, luu_mau
from tien_xu_ly import (TEN_BUOC, AnhDongNhat, KhongCoDoiTuong, LoiXuLyAnh, NetQuaMong, doc_anh)
from tien_xu_ly.nguong import CUC_TINH

GIAO_DIEN = Path(__file__).resolve().parent
TOI_DA_BODY = 25 * 1024 * 1024
DATA_URL = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,")
BUOC_HIEN_THI = ("xam", "loc", "nguong", "mat_na", "cat", "thu_nho", "dau_vao_cnn")
THONG_BAO_LOI = (
    (AnhDongNhat, "Ảnh trống hoặc chỉ một màu. Hãy vẽ hoặc chọn ảnh có nét rõ trên nền đồng đều."),
    (KhongCoDoiTuong, "Không tách được nét nào. Thử vẽ đậm hơn hoặc đổi cực tính ở mục Tùy chọn."),
    (NetQuaMong, "Nét quá mảnh nên mất khi thu nhỏ. Hãy vẽ đậm hơn hoặc dùng ảnh rõ hơn."),
)


def png_data_url(a):
    from PIL import Image
    import io
    bo = io.BytesIO()
    Image.fromarray(np.asarray(a, dtype=np.uint8)).save(bo, format="PNG")
    return "data:image/png;base64," + base64.b64encode(bo.getvalue()).decode()


def tach_data_url(chuoi):
    if not isinstance(chuoi, str) or not DATA_URL.match(chuoi):
        raise ValueError("ảnh phải là data URL")
    try:
        return base64.b64decode(DATA_URL.sub("", chuoi, count=1), validate=True)
    except binascii.Error as e:
        raise ValueError("base64 không hợp lệ") from e


def dinh_dang_ket_qua(kq):
    """dict từ BoNhanDang.nhan_dang -> JSON trả về giao diện (ảnh trung gian thành PNG data URL)."""
    bt = lay_bai_toan(kq["bai_toan"])
    ten_thanh = [f"Số {t}" if bt.ten == "chu_so" else bt.nhan_vi[i] for i, t in enumerate(bt.lop)]
    return {"bai_toan": bt.ten, "nhan": kq["nhan"], "lop": kq["lop"], "do_tin_cay": kq["do_tin_cay"],
            "xac_suat": [{"ten": ten_thanh[i], "p": kq["xac_suat"][t]} for i, t in enumerate(bt.lop)],
            "bien_the": kq["bien_the"], "cuc_tinh": kq["cuc_tinh"], "nguong": kq["nguong"], "hop_bao": kq["hop_bao"],
            "canh_bao": kq["canh_bao"], "ms": kq["ms"],
            "cac_buoc": [{"ten": TEN_BUOC[k], "png": png_data_url(kq["cac_buoc"][k])} for k in BUOC_HIEN_THI]}


class XuLyYeuCau(BaseHTTPRequestHandler):
    bo_nhan_dang = None
    thu_muc_anh_that = None          # chỉ dùng trong kiểm thử để không ghi vào du_lieu/
    server_version = "NhanDang/1.0"
    khoa = threading.Lock()

    def _gui(self, ma, noi_dung, loai="application/json; charset=utf-8"):
        than = noi_dung if isinstance(noi_dung, bytes) else json.dumps(noi_dung, ensure_ascii=False).encode("utf-8")
        self.send_response(ma)
        self.send_header("Content-Type", loai)
        self.send_header("Content-Length", str(len(than)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(than)

    def _doc_json(self):
        try:
            dai = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ValueError("Content-Length không hợp lệ")
        if not 0 < dai <= TOI_DA_BODY:
            raise ValueError("Ảnh quá lớn (tối đa 25 MB) hoặc rỗng")
        return json.loads(self.rfile.read(dai))

    def do_HEAD(self):
        self.send_response(200 if urlsplit(self.path).path in ("/", "/index.html", "/thu-anh") else 404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        url = urlsplit(self.path)
        if url.path in ("/", "/index.html"):
            self._gui(200, (GIAO_DIEN / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/thu-anh":
            self._gui(200, (GIAO_DIEN / "thu_thap.html").read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/thong-tin":
            self._gui(200, self.bo_nhan_dang.thong_tin())
        elif url.path == "/api/thu-anh/dem":
            ten = parse_qs(url.query).get("bai_toan", [""])[0]
            if ten not in BAI_TOAN:
                return self._gui(400, {"loi": "bai_toan không hợp lệ"})
            self._gui(200, {"bai_toan": ten, "so_luong": dem_mau(ten, self.thu_muc_anh_that and self.thu_muc_anh_that / ten)})
        else:
            self._gui(404, {"loi": "Không tìm thấy"})

    def do_POST(self):
        duong = urlsplit(self.path).path
        if duong == "/api/nhan-dang":
            return self._nhan_dang()
        if duong == "/api/thu-anh":
            return self._thu_anh()
        self._gui(404, {"loi": "Không tìm thấy"})

    def _nhan_dang(self):
        try:
            yc = self._doc_json()
            bai_toan, cuc_tinh = yc["bai_toan"], yc.get("cuc_tinh", "tu_dong")
            loc, giu = int(yc.get("kich_thuoc_loc", 1)), bool(yc.get("chi_giu_lon_nhat", True))
            if bai_toan not in BAI_TOAN or cuc_tinh not in CUC_TINH or loc not in (1, 3, 5):
                raise ValueError
            du_lieu = tach_data_url(yc["anh"])
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            return self._gui(400, {"loi": "Yêu cầu không hợp lệ (cần anh dạng data URL, bai_toan chu_so/hinh_hoc)"})
        try:
            rgb = doc_anh(du_lieu)
        except Exception:
            return self._gui(422, {"loi": "Không đọc được ảnh. Hãy dùng PNG hoặc JPEG."})
        try:
            with self.khoa:
                kq = self.bo_nhan_dang.nhan_dang(rgb, bai_toan, cuc_tinh=cuc_tinh, kich_thuoc_loc=loc, chi_giu_lon_nhat=giu)
        except LoiXuLyAnh as loi:
            thong_bao = next((m for kieu, m in THONG_BAO_LOI if isinstance(loi, kieu)), str(loi))
            return self._gui(422, {"loi": thong_bao})
        self._gui(200, dinh_dang_ket_qua(kq))

    def _thu_anh(self):
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{self.headers.get('Host')}":
            return self._gui(403, {"loi": "Yêu cầu từ nguồn khác bị từ chối"})
        try:
            yc = self._doc_json()
            bai_toan = yc["bai_toan"]
            if bai_toan not in BAI_TOAN:
                raise ValueError("bai_toan không hợp lệ")
            thu_muc = self.thu_muc_anh_that / bai_toan if self.thu_muc_anh_that else None
            kq = luu_mau(bai_toan, yc["nhan"], tach_data_url(yc["anh"]), yc.get("nguoi_ve", ""),
                         yc.get("nguon", "ve_tren_khung"), thu_muc)
        except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as loi:
            return self._gui(400, {"loi": str(loi) or "Yêu cầu không hợp lệ"})
        self._gui(200, kq)

    def log_message(self, dinh_dang, *tham_so):
        sys.stderr.write("[ung_dung] " + (dinh_dang % tham_so) + "\n")


def tao_may_chu(cong=8780, bo_nhan_dang=None, thu_muc_anh_that=None):
    XuLyYeuCau.bo_nhan_dang = bo_nhan_dang or BoNhanDang()
    XuLyYeuCau.thu_muc_anh_that = Path(thu_muc_anh_that) if thu_muc_anh_that else None
    return ThreadingHTTPServer(("127.0.0.1", cong), XuLyYeuCau)


def chay(cong=8780):
    may_chu = tao_may_chu(cong)
    print(f"Ứng dụng sẵn sàng: http://127.0.0.1:{may_chu.server_address[1]}  (Ctrl+C để dừng)", flush=True)
    try:
        may_chu.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        may_chu.server_close()
