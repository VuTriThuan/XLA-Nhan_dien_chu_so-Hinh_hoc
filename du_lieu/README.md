# Dữ liệu

| Thư mục | Nội dung |
|---|---|
| `mnist/` | MNIST chính thức (`mnist.npz`), chỉ số chia train/validation (`splits.npz`), thống kê, ảnh xem trước |
| `chu_so_viet_tay/` | Ảnh chữ số viết tay **thật** của nhóm, để đánh giá — **58 ảnh của một người viết** (đủ ngưỡng số lượng 50) |
| `hinh_hoc_tu_sinh/` | 13.500 ảnh hình học tự sinh (`shapes_dataset.npz`), bộ stress 300 ảnh, `metadata.csv`, thống kê |
| `hinh_ve_tay/` | Ảnh hình vẽ tay **thật** của nhóm, để đánh giá — **43 ảnh của một người vẽ** (đủ ngưỡng số lượng 30) |
| `anh_mau/` | 10 ảnh chữ số (từ MNIST test) và 3 ảnh hình (từ test tự sinh) để thử nhanh; `nguon_*.json` ghi nguồn |

## Nguồn và điều kiện sử dụng

**MNIST**: Yann LeCun và Corinna Cortes, có gốc từ NIST. Tệp `mnist.npz` tải từ
`https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz` (SHA-256 `731c5ac6…b673d1`, được kiểm tra khi chuẩn bị dữ liệu); mô tả và giấy phép theo
`https://keras.io/api/datasets/mnist/`: Creative Commons Attribution-Share Alike 3.0. Hãy ghi nguồn khi chia sẻ ảnh mẫu, biểu đồ có ảnh MNIST hoặc dữ liệu phái sinh.
`splits.npz` là chỉ số chia lại 60.000 ảnh nguồn thành 54.000 train / 6.000 validation; 10.000 ảnh test giữ nguyên.

**Dữ liệu hình học** do chính dự án sinh bằng `python3 huan_luyen.py --sinh-du-lieu` (hạt cố định, mã ở `bo_du_lieu/hinh_hoc.py`); không phải ảnh người vẽ.
Ảnh trong `anh_mau/` **không** phải ảnh thu từ người dùng.
