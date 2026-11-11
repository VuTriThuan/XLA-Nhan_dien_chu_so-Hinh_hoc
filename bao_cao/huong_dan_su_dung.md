# Hướng dẫn sử dụng

Mọi lệnh chạy từ thư mục gốc của dự án (hoặc gọi `python3 /đường/dẫn/BTL/<file>.py ...` từ nơi khác).

## 1. Cài đặt

Cần Python 3.10 trở lên.

```bash
python3 -m pip install -r requirements.txt
```

Chỉ dùng CPU, không cần GPU. Dữ liệu (MNIST, bộ hình tổng hợp) và trọng số đã huấn luyện có sẵn trong thư mục dự án nên
không phải huấn luyện lại để dùng.

## 2. Ứng dụng web

```bash
python3 ung_dung.py            # mở http://127.0.0.1:8780   (đổi cổng: --port 9000)
```

- Chọn **Chữ số 0–9** hoặc **Hình học** trước khi vẽ. Hai mạng không có lớp "không phải", nên ảnh sai loại vẫn bị gán một lớp.
- Vẽ bằng chuột hoặc ngón tay, hoặc tải / kéo thả / dán ảnh (PNG, JPEG). Ảnh chỉ nên chứa **một** đối tượng chính trên nền khá đồng đều.
- Kết quả hiện lớp dự đoán, điểm softmax của mọi lớp và từng bước tiền xử lý. Điểm softmax **không phải độ chính xác** và chưa hiệu chuẩn.
- Mục *Tùy chọn tiền xử lý*: cực tính (tự động hay chỉ định nét tối/sáng), lọc trung vị (khử nhiễu muối tiêu, có thể xóa nét mảnh),
  chỉ giữ vùng liên thông lớn nhất (tắt nếu nét bị đứt).
- Máy chủ chỉ lắng nghe `127.0.0.1`, không gửi ảnh ra ngoài.

## 3. Nhận dạng một ảnh từ dòng lệnh

```bash
python3 nhan_dang.py anh.png --bai-toan chu_so
python3 nhan_dang.py hinh.png --bai-toan hinh_hoc --cuc-tinh net_toi --luu-buoc ket_qua_buoc/
```

In kết quả dạng JSON; `--luu-buoc` lưu ảnh từng bước tiền xử lý. Mã thoát 2 và thông báo ngắn nếu tệp không đọc được hoặc ảnh trống.

## 4. Thu ảnh viết tay / vẽ tay thật để đánh giá

Hiện đã có **101 ảnh thật của một người viết** (58 chữ số trong `du_lieu/chu_so_viet_tay/`, 43 hình trong `du_lieu/hinh_ve_tay/`), đủ số lượng tối thiểu nên trạng thái đánh giá là `ready` (chỉ xét số lượng, không xét số người viết); xem kết quả sơ bộ ở `tai_lieu/thuc_nghiem.md` (mục 5).
Ảnh cũng có thể lấy từ ảnh chụp một tờ giấy nhiều ký hiệu: cắt từng ký hiệu thành một ảnh riêng (một chữ số hoặc một hình mỗi ảnh, chừa lề quanh nét) và đặt vào thư mục lớp tương ứng; ghi nhãn trước khi chạy đánh giá.

```bash
python3 ung_dung.py --thu-anh             # mở trang thu ảnh: http://127.0.0.1:8780/thu-anh
```

Chọn **nhãn thật trước**, đặt mã người viết/vẽ (tự đặt, ví dụ `nguoi_01`), rồi vẽ trên khung hoặc tải ảnh. Trang này **không hiển thị dự đoán**
để tránh sửa ảnh theo mô hình. Cần tối thiểu: chữ số 50 ảnh (≥ 5 ảnh mỗi chữ số), hình học 30 ảnh (≥ 10 ảnh mỗi hình), nên có nhiều người.
Cũng có thể tự đặt ảnh vào `du_lieu/chu_so_viet_tay/<0-9>/` và `du_lieu/hinh_ve_tay/<tron|tam_giac|chu_nhat>/`.

Sau khi thu đủ:

```bash
python3 danh_gia.py --bai-toan chu_so
python3 danh_gia.py --bai-toan hinh_hoc
```

Kết quả ở `ket_qua/<bai_toan>/<bien_the>/external_metrics.json`. Trạng thái `pending` (chưa có ảnh), `partial` (chưa đủ số lượng) hoặc `ready`;
ảnh trùng byte bị bỏ qua, ảnh không xử lý được tính là sai và được đếm riêng. **Không** dùng các ảnh này để huấn luyện hay chỉnh mô hình rồi
gọi chính chúng là kiểm thử độc lập.

## 5. Huấn luyện lại (tùy chọn)

```bash
python3 huan_luyen.py --sinh-du-lieu --bai-toan hinh_hoc     # sinh 13.500 ảnh (--ti-le 1 cho 4.500 ảnh); chữ số: --bai-toan chu_so
python3 huan_luyen.py --bai-toan hinh_hoc                   # huấn luyện hai cấu hình rồi chọn bằng validation
python3 danh_gia.py --bai-toan hinh_hoc
```

`huan_luyen.py` ghi đè `mo_hinh/<bai_toan>_*.pth` và `ket_qua/<bai_toan>/`; hãy sao lưu trước. Chỉ chạy đủ hai cấu hình (`--bien-the ca_hai`, mặc định) mới chốt
`<bai_toan>_tot_nhat.pth` và `mo_hinh/lua_chon.json` bằng validation. Huấn luyện chạy CPU, có tái lập (cùng hạt, cùng máy cho cùng kết quả).

## 6. Kiểm thử và thí nghiệm

```bash
python3 -m pytest                                  # toàn bộ test (khoảng 30 giây)
python3 -m pytest kiem_thu/test_tien_xu_ly.py -q   # một file, có thể thêm tùy chọn pytest
python3 danh_gia.py --thu-nghiem                   # thí nghiệm đo độ bền (vài chục phút); kết quả ở ket_qua/tich_hop/
```

## 7. Xử lý sự cố

| Hiện tượng | Cách xử lý |
|---|---|
| `No module named pytest` | `python3 -m pip install -r requirements.txt` |
| Báo "Ảnh trống hoặc chỉ một màu" | Vẽ/chọn ảnh có nét rõ trên nền đồng đều, một đối tượng |
| Hình tròn, chữ số bị nhận sai dù trông đúng | Xem ảnh "Đầu vào CNN 28×28" ở các bước; thử đổi cực tính hoặc tắt lọc trung vị |
| Nền loang sáng nặng, nhiều bóng đổ | Ngưỡng Otsu toàn cục kém trên nền không đều; chụp lại với ánh sáng đều |
| Cổng 8780 đã dùng | `python3 ung_dung.py --port 8781` |
| Muốn xem từng bước tiền xử lý | `--luu-buoc` của `nhan_dang.py`, hoặc xem mục "Các bước tiền xử lý" trong ứng dụng |
