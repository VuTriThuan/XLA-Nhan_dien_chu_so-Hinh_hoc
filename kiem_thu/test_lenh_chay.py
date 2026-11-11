"""Kiểm thử các file chạy ở thư mục gốc: chạy từ thư mục khác, mã thoát và đầu ra."""
import json
import subprocess
import sys

from loi.cau_hinh import ANH_MAU, GOC


def chay(tep, *doi_so, cwd):
    return subprocess.run([sys.executable, "-B", "-X", "utf8", str(GOC / tep), *map(str, doi_so)], cwd=cwd, text=True,
                          encoding="utf-8", capture_output=True, timeout=120)


def test_nhan_dang_tu_thu_muc_khac_in_json(tmp_path):
    qt = chay("nhan_dang.py", ANH_MAU / "chu_so_7.png", "--bai-toan", "chu_so", cwd=tmp_path)
    assert qt.returncode == 0, qt.stderr
    kq = json.loads(qt.stdout)
    assert kq["ten_lop"] == "7" and "cac_buoc" not in kq and kq["bai_toan"] == "chu_so"


def test_luu_buoc_ghi_anh_tung_buoc(tmp_path):
    qt = chay("nhan_dang.py", ANH_MAU / "hinh_tam_giac.png", "--bai-toan", "hinh_hoc", "--luu-buoc", tmp_path / "buoc",
              cwd=tmp_path)
    assert qt.returncode == 0, qt.stderr
    assert sorted(p.stem for p in (tmp_path / "buoc").glob("*.png")) == sorted(
        ["xam", "loc", "nguong", "mat_na", "cat", "thu_nho", "dau_ra", "dau_vao_cnn"])


def test_loi_gon_khi_tep_khong_ton_tai_hoac_anh_trong(tmp_path):
    qt = chay("nhan_dang.py", tmp_path / "khong_co.png", "--bai-toan", "chu_so", cwd=tmp_path)
    assert qt.returncode == 2 and "Lỗi" in qt.stderr
    from PIL import Image
    Image.new("RGB", (50, 50), "white").save(tmp_path / "trang.png")
    qt = chay("nhan_dang.py", tmp_path / "trang.png", "--bai-toan", "chu_so", cwd=tmp_path)
    assert qt.returncode == 2 and "Lỗi" in qt.stderr


def test_tro_giup_cua_moi_file_chay(tmp_path):
    for tep, tu_khoa in (("huan_luyen.py", "--sinh-du-lieu"), ("nhan_dang.py", "--bai-toan"),
                         ("danh_gia.py", "--thu-nghiem"), ("ung_dung.py", "--thu-anh"),
                         ("bao_cao/latex/bien_dich.py", "--chi-dong-bo-hinh")):
        qt = chay(tep, "--help", cwd=tmp_path)
        assert qt.returncode == 0 and tu_khoa in qt.stdout, (tep, qt.stderr)


def test_doi_so_sai_bao_ma_2(tmp_path):
    assert chay("huan_luyen.py", "--bai-toan", "sai", cwd=tmp_path).returncode == 2
    assert chay("danh_gia.py", cwd=tmp_path).returncode == 2                       # phải chọn --bai-toan hoặc --thu-nghiem
    assert chay("danh_gia.py", "--bai-toan", "chu_so", "--thu-nghiem", cwd=tmp_path).returncode == 2
    assert chay("nhan_dang.py", cwd=tmp_path).returncode == 2
