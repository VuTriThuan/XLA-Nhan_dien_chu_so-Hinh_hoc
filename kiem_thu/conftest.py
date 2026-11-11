"""Cấu hình pytest dùng chung: đưa thư mục gốc dự án vào sys.path và không tạo __pycache__ trong dự án."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
GOC = Path(__file__).resolve().parents[1]
if str(GOC) not in sys.path:
    sys.path.insert(0, str(GOC))


import pytest  # noqa: E402


@pytest.fixture(scope="session")
def bo_nhan_dang():
    """Nạp hai mô hình một lần cho cả phiên kiểm thử."""
    from loi.du_doan import BoNhanDang
    return BoNhanDang()
