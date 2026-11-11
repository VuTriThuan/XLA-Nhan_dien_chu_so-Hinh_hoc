"""Kiểm thử ứng dụng web qua HTTP thật (cổng tạm): nhận dạng, lỗi đầu vào, thu ảnh thật."""
import base64
import io
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
from PIL import Image, ImageDraw

from giao_dien.may_chu import dinh_dang_ket_qua, png_data_url, tach_data_url, tao_may_chu


def anh_png(ve, kich_thuoc=400, che_do="RGB"):
    im = Image.new(che_do, (kich_thuoc, kich_thuoc), "white")
    ve(ImageDraw.Draw(im))
    bo = io.BytesIO()
    im.save(bo, format="PNG")
    return "data:image/png;base64," + base64.b64encode(bo.getvalue()).decode()


def so_bay(do_day=12):
    return anh_png(lambda d: d.line([(110, 100), (290, 100), (190, 320)], fill="black", width=do_day, joint="curve"))


def hinh_tron(do_day=10):
    return anh_png(lambda d: d.ellipse([80, 80, 320, 320], outline="black", width=do_day))


@pytest.fixture(scope="module")
def may_chu(bo_nhan_dang, tmp_path_factory):
    thu_muc = tmp_path_factory.mktemp("anh_that")
    ms = tao_may_chu(0, bo_nhan_dang, thu_muc)
    luong = threading.Thread(target=ms.serve_forever, daemon=True)
    luong.start()
    yield f"http://127.0.0.1:{ms.server_address[1]}", thu_muc
    ms.shutdown()
    ms.server_close()


def goi(url, duong, noi_dung=None, tieu_de=None, tho=None, phuong_thuc=None):
    du_lieu = tho if tho is not None else (json.dumps(noi_dung).encode() if noi_dung is not None else None)
    yc = urllib.request.Request(url + duong, data=du_lieu, headers={"Content-Type": "application/json", **(tieu_de or {})},
                                method=phuong_thuc)
    try:
        with urllib.request.urlopen(yc, timeout=60) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_trang_chu_trang_thu_anh_thong_tin_va_404(may_chu):
    url, _ = may_chu
    for duong, chu in (("/", "Nhận diện chữ số"), ("/thu-anh", "Thu ảnh thật")):
        with urllib.request.urlopen(url + duong, timeout=10) as r:
            assert r.status == 200 and chu in r.read().decode("utf-8")
    ma, tt = goi(url, "/api/thong-tin")
    assert ma == 200 and set(tt) == {"chu_so", "hinh_hoc"} and tt["chu_so"]["so_lop"] == 10
    assert goi(url, "/khong-co")[0] == 404
    assert urllib.request.urlopen(urllib.request.Request(url + "/", method="HEAD"), timeout=10).status == 200


def test_nhan_dang_chu_so_va_hinh(may_chu):
    url, _ = may_chu
    ma, kq = goi(url, "/api/nhan-dang", {"anh": so_bay(), "bai_toan": "chu_so"})
    assert ma == 200 and kq["lop"] == 7 and kq["nhan"] == "Chữ số 7" and len(kq["cac_buoc"]) == 7
    assert kq["xac_suat"][0]["ten"] == "Số 0" and kq["cac_buoc"][-1]["ten"] == "Đầu vào CNN 28×28"
    assert all(b["png"].startswith("data:image/png;base64,") for b in kq["cac_buoc"])
    ma, kq = goi(url, "/api/nhan-dang", {"anh": hinh_tron(), "bai_toan": "hinh_hoc"})
    assert ma == 200 and kq["lop"] == 0 and kq["xac_suat"][0]["ten"] == "Tròn"


def test_nhan_dang_voi_tuy_chon(may_chu):
    url, _ = may_chu
    ma, kq = goi(url, "/api/nhan-dang", {"anh": hinh_tron(), "bai_toan": "hinh_hoc", "cuc_tinh": "net_toi",
                                         "kich_thuoc_loc": 3, "chi_giu_lon_nhat": True})
    assert (ma, kq["lop"], kq["cuc_tinh"]) == (200, 0, "net_toi")


@pytest.mark.parametrize("noi_dung", [
    {"anh": "{}", "bai_toan": "sai"}, {"anh": "{}", "bai_toan": "chu_so", "cuc_tinh": "x"},
    {"anh": "{}", "bai_toan": "chu_so", "kich_thuoc_loc": 4}, {"bai_toan": "chu_so"},
    {"anh": "http://xau/x.png", "bai_toan": "chu_so"}, {"anh": "data:image/png;base64,@@@", "bai_toan": "chu_so"}])
def test_yeu_cau_khong_hop_le(may_chu, noi_dung):
    url, _ = may_chu
    if noi_dung.get("anh") == "{}":
        noi_dung = {**noi_dung, "anh": so_bay()}
    assert goi(url, "/api/nhan-dang", noi_dung)[0] == 400


def test_json_hong_va_anh_khong_doc_duoc_hoac_trong(may_chu):
    url, _ = may_chu
    assert goi(url, "/api/nhan-dang", tho=b"{khong phai json")[0] == 400
    hong = "data:image/png;base64," + base64.b64encode(b"khong phai anh").decode()
    assert goi(url, "/api/nhan-dang", {"anh": hong, "bai_toan": "chu_so"})[0] == 422
    ma, kq = goi(url, "/api/nhan-dang", {"anh": anh_png(lambda d: None, 80), "bai_toan": "chu_so"})
    assert ma == 422 and "trống" in kq["loi"]


def test_thu_anh_dem_luu_va_tu_choi(may_chu):
    url, thu_muc = may_chu
    ma, kq = goi(url, "/api/thu-anh/dem?bai_toan=hinh_hoc")
    assert ma == 200 and kq["so_luong"] == {"tron": 0, "tam_giac": 0, "chu_nhat": 0}
    ma, kq = goi(url, "/api/thu-anh", {"bai_toan": "hinh_hoc", "nhan": "tron", "anh": hinh_tron(),
                                       "nguoi_ve": "nguoi_01", "nguon": "ve_tren_khung"})
    assert ma == 200 and kq["so_luong"]["tron"] == 1
    assert len(list((thu_muc / "hinh_hoc" / "tron").glob("*.png"))) == 1                # ghi vào thư mục tạm của test
    assert goi(url, "/api/thu-anh/dem?bai_toan=sai")[0] == 400
    for sai in ({"bai_toan": "hinh_hoc", "nhan": "sao", "anh": hinh_tron()}, {"bai_toan": "sai", "nhan": "tron", "anh": hinh_tron()},
                {"bai_toan": "hinh_hoc", "nhan": "tron", "anh": anh_png(lambda d: None, 80)}, {"nhan": "tron"}):
        assert goi(url, "/api/thu-anh", sai)[0] == 400
    ma, kq = goi(url, "/api/thu-anh", {"bai_toan": "hinh_hoc", "nhan": "tron", "anh": hinh_tron()},
                 tieu_de={"Origin": "http://trang-la.example"})
    assert ma == 403


def test_ham_phu_tro():
    a = np.zeros((4, 4), np.uint8)
    assert tach_data_url(png_data_url(a))[:4] == b"\x89PNG"
    with pytest.raises(ValueError):
        tach_data_url("http://x")
    with pytest.raises(ValueError):
        tach_data_url("data:image/png;base64,@@")
    kq = dinh_dang_ket_qua({"bai_toan": "hinh_hoc", "nhan": "Tròn", "lop": 0, "do_tin_cay": .9,
                            "xac_suat": {"tron": .9, "tam_giac": .05, "chu_nhat": .05}, "bien_the": "cai_tien",
                            "cuc_tinh": "net_toi", "nguong": 100, "hop_bao": [0, 0, 1, 1], "canh_bao": [], "ms": 1.0,
                            "cac_buoc": {k: a for k in ("xam", "loc", "nguong", "mat_na", "cat", "thu_nho", "dau_vao_cnn")}})
    assert [x["ten"] for x in kq["xac_suat"]] == ["Tròn", "Tam giác", "Chữ nhật / vuông"]
