"""Thành phần liên thông."""
from collections import deque

import numpy as np

from ._kiem_tra import ma_tran


def thanh_phan_lien_thong_lon_nhat(mat_na):
    """BFS 8 láng giềng, giữ thành phần nhiều pixel nhất (hòa thì giữ thành phần gặp trước khi quét từ trên xuống).

    8-liên thông để nét chéo vẫn là cùng một thành phần. Nhược điểm: nét bị đứt thành nhiều đoạn chỉ còn đoạn lớn nhất.
    """
    mat_na = ma_tran(mat_na).astype(bool)
    cao, rong = mat_na.shape
    da_tham = np.zeros_like(mat_na, dtype=bool)
    tot_nhat = []
    for y, x in zip(*np.nonzero(mat_na)):
        if da_tham[y, x]:
            continue
        hang_doi = deque([(y, x)])
        da_tham[y, x] = True
        thanh_phan = []
        while hang_doi:
            cy, cx = hang_doi.popleft()
            thanh_phan.append((cy, cx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < cao and 0 <= nx < rong and mat_na[ny, nx] and not da_tham[ny, nx]:
                        da_tham[ny, nx] = True
                        hang_doi.append((ny, nx))
        if len(thanh_phan) > len(tot_nhat):
            tot_nhat = thanh_phan
    kq = np.zeros_like(mat_na, dtype=bool)
    if tot_nhat:
        yy, xx = zip(*tot_nhat)
        kq[yy, xx] = True
    return kq
