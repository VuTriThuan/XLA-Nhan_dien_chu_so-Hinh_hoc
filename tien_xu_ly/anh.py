"""Đọc ảnh và chuyển sang ảnh xám."""
import io

import numpy as np

from ._kiem_tra import LoiXuLyAnh


def chuyen_xam(anh):
    """Nhận uint8, hoặc float trong [0,1]/[0,255]; RGBA được ghép lên nền trắng.

    Y = 0,299 R + 0,587 G + 0,114 B. Ảnh float tự suy ra thang đo từ giá trị lớn nhất (<= 1 là [0,1]).
    """
    a = np.asarray(anh)
    if a.size == 0 or a.dtype.kind not in "uif":
        raise LoiXuLyAnh("Ảnh rỗng hoặc không phải dữ liệu số")
    if not np.isfinite(a).all() or a.min() < 0 or a.max() > 255:
        raise LoiXuLyAnh("Pixel phải hữu hạn và trong [0,255]")
    la_float = np.issubdtype(a.dtype, np.floating)
    a = a.astype(np.float32)
    if la_float and a.max() <= 1:
        a = a * 255
    if a.ndim == 3 and a.shape[2] in (3, 4):
        if a.shape[2] == 4:
            alpha = a[:, :, 3:4] / 255
            a = a[:, :, :3] * alpha + 255 * (1 - alpha)
        a = .299 * a[:, :, 0] + .587 * a[:, :, 1] + .114 * a[:, :, 2]
    elif a.ndim != 2:
        raise LoiXuLyAnh("Cần ảnh HxW, RGB hoặc RGBA; thứ tự kênh RGB, không phải BGR")
    return np.rint(a).clip(0, 255).astype(np.uint8)


def doc_anh(du_lieu):
    """Bytes ảnh (PNG/JPEG/...) -> RGB uint8.

    Pillow chỉ giải mã. Xoay theo EXIF và ghép nền trắng cho ảnh trong suốt làm bằng NumPy.
    """
    from PIL import Image
    with Image.open(io.BytesIO(du_lieu)) as im:
        huong = im.getexif().get(274, 1)
        a = np.array(im.convert("RGBA"))
    phep = {2: lambda x: x[:, ::-1], 3: lambda x: np.rot90(x, 2), 4: lambda x: x[::-1],
            5: lambda x: np.swapaxes(x, 0, 1), 6: lambda x: np.rot90(x, -1),
            7: lambda x: np.swapaxes(x, 0, 1)[::-1, ::-1], 8: lambda x: np.rot90(x)}
    a = phep.get(huong, lambda x: x)(a)
    alpha = a[:, :, 3:4].astype(np.float32) / 255
    rgb = a[:, :, :3] * alpha + 255 * (1 - alpha)
    return np.rint(rgb).astype(np.uint8)


def doc_anh_tu_tep(duong_dan):
    from pathlib import Path
    return doc_anh(Path(duong_dan).read_bytes())
