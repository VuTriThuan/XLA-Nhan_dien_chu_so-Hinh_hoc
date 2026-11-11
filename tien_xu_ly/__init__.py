"""Xử lý ảnh tự cài bằng NumPy; mỗi mô-đun một nhóm kỹ thuật."""
from ._kiem_tra import AnhDongNhat, KhongCoDoiTuong, LoiXuLyAnh, NetQuaMong
from .anh import chuyen_xam, doc_anh, doc_anh_tu_tep
from .hinh_thai import hinh_thai_nhi_phan
from .lien_thong import thanh_phan_lien_thong_lon_nhat
from .loc import loc_trung_vi
from .luong_xu_ly import TEN_BUOC, KetQuaXuLy, xu_ly_anh
from .nguong import nguong_otsu, xac_dinh_cuc_tinh
from .thu_phong import (gioi_han_kich_thuoc, noi_suy_song_tuyen_tinh, thu_nho_trung_binh_dien_tich, thu_phong)

__all__ = ["AnhDongNhat", "KhongCoDoiTuong", "LoiXuLyAnh", "NetQuaMong", "chuyen_xam", "doc_anh", "doc_anh_tu_tep",
           "hinh_thai_nhi_phan", "thanh_phan_lien_thong_lon_nhat", "loc_trung_vi", "TEN_BUOC", "KetQuaXuLy",
           "xu_ly_anh", "nguong_otsu", "xac_dinh_cuc_tinh", "gioi_han_kich_thuoc", "noi_suy_song_tuyen_tinh",
           "thu_nho_trung_binh_dien_tich", "thu_phong"]
