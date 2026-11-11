# Thiết kế hệ thống

## 1. Luồng nhận dạng

```
ảnh (vẽ / tải lên / tệp)
  └─ tiền xử lý  tien_xu_ly/
       giới hạn kích thước (cạnh dài <= 512) → xám → lọc trung vị (tùy chọn) → Otsu → cực tính
       → [hình thái] → thành phần liên thông lớn nhất → hộp bao → thu nhỏ giữ tỉ lệ (trung bình diện tích)
       → căn giữa canvas 28×28 → kéo giãn độ sáng
  └─ chọn bài toán (người dùng chọn)
       ├─ chu_so   → CNN 10 lớp
       └─ hinh_hoc → CNN 3 lớp (tròn, tam giác, chữ nhật/vuông)
  └─ softmax → lớp dự đoán + điểm từng lớp (chưa hiệu chuẩn)
```

## 2. Quy ước đầu vào của CNN

| | Chữ số | Hình học |
|---|---|---|
| Ảnh | 28×28, một kênh, `float32` trong [0,1], **nền 0, nét sáng** | giống nhau |
| Nội dung tối đa | 20×20 (lề 4 px) | 22×22 (lề 3 px) |
| Căn giữa | theo tâm hộp bao | theo tâm hộp bao |
| Chuẩn hóa | chia 255, **không** trừ mean/chia std; nét sáng nhất = 1 | giống nhau |

CNN chữ số được huấn luyện trực tiếp trên MNIST (đã được căn theo trọng tâm, nét sáng nhất = 1). CNN hình học được huấn luyện trên đầu vào do
**chính** `xu_ly_anh` tạo ra từ ảnh gốc 28×28, nên huấn luyện và nhận dạng dùng chung một đường xử lý.

## 3. Cấu trúc mã nguồn

| Thư mục / file | Trách nhiệm |
|---|---|
| `huan_luyen.py`, `nhan_dang.py`, `danh_gia.py`, `ung_dung.py` | Bốn file chạy ở thư mục gốc, mỗi file một việc; chỉ đọc đối số rồi gọi mã bên dưới |
| `loi/cau_hinh.py` | Đường dẫn, định nghĩa hai bài toán (lớp, nhãn, lề), đọc/ghi JSON/CSV, SHA-256 |
| `tien_xu_ly/` | Xử lý ảnh tự cài bằng NumPy; mỗi mô-đun một nhóm kỹ thuật (`anh`, `loc`, `nguong`, `hinh_thai`, `lien_thong`, `thu_phong`, `luong_xu_ly`) |
| `loi/cnn.py`, `loi/tang_cuong.py` | `MangCNN` (kiến trúc dùng chung) và tăng cường dữ liệu affine tự cài |
| `bo_du_lieu/` | Nạp MNIST, bộ sinh hình tổng hợp, thu ảnh thật |
| `loi/huan_luyen.py` | Vòng huấn luyện dùng chung, chọn checkpoint tốt nhất và cấu hình cuối chỉ bằng validation |
| `loi/danh_gia.py` | Chỉ số, ma trận nhầm lẫn, ảnh lỗi, McNemar, stress/chẩn đoán, quét góc–độ dày, ảnh thật |
| `loi/du_doan.py` | `DuDoan` (một bài toán, đầu vào đã chuẩn hóa) và `BoNhanDang` (ảnh bất kỳ → kết quả) |
| `giao_dien/` | Máy chủ web (thư viện chuẩn) và hai trang HTML |

Quy tắc phụ thuộc một chiều: `loi/cau_hinh` ← `tien_xu_ly`, `loi/cnn`, `loi/tang_cuong` ← `bo_du_lieu` ← `loi/huan_luyen`, `loi/du_doan` ← `loi/danh_gia`, `giao_dien`.
Tiền xử lý không phụ thuộc PyTorch; Pillow chỉ dùng để giải mã/ghi tệp ảnh.

## 4. Kiến trúc CNN

`Conv(1→16, 3×3) → [BN] → ReLU → MaxPool2 → Conv(16→32, 3×3) → [BN] → ReLU → MaxPool2 → Flatten(800) → [Dropout] → Linear(800→64) → ReLU → Linear(64→số lớp)`.

| Mô hình | Số lớp | BatchNorm | Dropout | Tham số |
|---|---|---|---|---|
| Chữ số (cả hai cấu hình) | 10 | không | 0,25 | 56.714 |
| Hình học `co_so` | 3 | không | không | 56.259 |
| Hình học `cai_tien` | 3 | có | 0,3 | 56.355 |

Cấu hình `cai_tien` luôn có tăng cường dữ liệu khi huấn luyện; cấu hình cuối được chọn bằng accuracy validation rồi loss validation, **không đọc tập test**.

## 5. Định dạng checkpoint (`mo_hinh/<bai_toan>_<cấu hình>.pth`)

`state_dict`, `bai_toan`, `bien_the`, `lop` (thứ tự lớp), `kien_truc` (`so_lop`, `batch_norm`, `dropout`), `phien_ban_tien_xu_ly`, `cau_hinh`,
`epoch_tot_nhat`, `do_chinh_xac_validation`, `loss_validation`. Chỉ chứa kiểu dữ liệu cơ bản nên nạp được bằng `torch.load(weights_only=True)`.
`DuDoan` từ chối checkpoint sai bài toán, sai thứ tự lớp hoặc sai định nghĩa đầu vào (`phien_ban_tien_xu_ly`).
`mo_hinh/lua_chon.json` ghi (cho từng bài toán) ứng viên, tiêu chí chọn và SHA-256; `danh_gia` từ chối chạy nếu checkpoint đã đổi sau khi chọn.

Checkpoint chữ số là trọng số huấn luyện gốc (9915/10.000 trên MNIST test), chỉ đổi tên khóa và siêu dữ liệu sang định dạng trên. Checkpoint hình học
được huấn luyện lại với tiền xử lý thống nhất.

## 6. Quyết định thiết kế đáng chú ý

- **Thu nhỏ bằng trung bình theo diện tích** thay cho nội suy song tuyến tính: song tuyến tính chỉ lấy 4 điểm lân cận nên nét mảnh trên ảnh lớn bị "rách" thành chấm
  rời khi thu nhỏ mạnh (ảnh 400 px về 22 px), CNN nhận sai. Số liệu: `ket_qua/tich_hop/` và `thuc_nghiem.md`.
- **Kéo giãn độ sáng** (chia cho giá trị lớn nhất) vì nét mảnh sau khi lấy trung bình bị mờ, trong khi ảnh huấn luyện có nét sáng nhất ≈ 1.
- **Thu nhỏ trước khi xử lý ảnh lớn** (cạnh dài ≤ 512): thành phần liên thông viết bằng BFS thuần Python nên rất chậm trên ảnh nhiều megapixel.
- **8-liên thông**: nét chéo vẫn là một thành phần. Giữ thành phần lớn nhất loại được chấm nhiễu nhưng cũng bỏ mất đoạn nét bị đứt (có tùy chọn tắt).
- **Không bật mặc định lọc trung vị và hình thái học**: chúng có thể xóa nét mảnh hoặc lấp lỗ vốn phân biệt các chữ số.
- **Cực tính tự động** đoán nền từ viền ảnh và cảnh báo khi tỉ lệ viền tối nằm trong [0,35; 0,65]; đây là quy tắc kinh nghiệm, không phải độ tin cậy.
- **Dữ liệu hình học đa dạng** (tam giác bất kỳ, chữ nhật tỉ lệ 0,3–1, hình tròn méo, rung đỉnh): xem `thuc_nghiem.md`.
- **Thu ảnh thật tách khỏi huấn luyện**: nhãn ghi trước khi dự đoán, giao diện thu ảnh không hiển thị dự đoán, ảnh chỉ dùng để đánh giá.

## 7. Giới hạn

Ảnh viết/vẽ tay thật gồm 101 ảnh của một người viết (đủ số lượng tối thiểu nhưng chưa có người viết khác); trên đó chữ số đạt 50/58, thấp hơn nhiều so với MNIST; chọn Chữ số/Hình học là thao tác tay; không có lớp "không phải"; softmax chưa hiệu chuẩn;
Otsu toàn cục kém khi nền loang sáng nặng; chỉ xử lý một đối tượng mỗi ảnh; một hạt cho mỗi cấu hình huấn luyện.
