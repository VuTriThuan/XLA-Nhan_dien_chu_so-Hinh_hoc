# Mô hình đã huấn luyện

Mỗi bài toán (`chu_so_…`, `hinh_hoc_…`) có `co_so.pth` (không tăng cường dữ liệu), `cai_tien.pth` (tăng cường dữ liệu; hình học thêm BatchNorm và Dropout), `tot_nhat.pth` (bản sao của cấu hình được chọn bằng validation);
`lua_chon.json` ghi ứng viên, tiêu chí, SHA-256 cho cả hai bài toán. Chỉ chứa trọng số và kiểu dữ liệu cơ bản, nạp bằng `torch.load(weights_only=True)`. Định dạng: `bao_cao/thiet_ke.md`.

- `chu_so_*.pth`: trọng số huấn luyện gốc trên MNIST (9915/10.000 trên test), chỉ đổi tên khóa và siêu dữ liệu sang định dạng hiện tại.
- `hinh_hoc_*.pth`: huấn luyện trên 13.500 ảnh tự sinh với tiền xử lý thống nhất.
