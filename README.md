# Nhận diện chữ số và hình dạng đơn giản bằng mạng neural

Bài tập lớn môn Xử lý ảnh. Hệ thống nhận một ảnh chứa **một chữ số viết tay (0–9)** hoặc **một hình học đơn giản (tròn, tam giác, chữ nhật/vuông)**,
tiền xử lý bằng các thuật toán xử lý ảnh tự cài bằng NumPy, rồi nhận dạng bằng CNN (PyTorch). Có ứng dụng web để vẽ hoặc tải ảnh và xem từng bước xử lý.

## Chạy nhanh

Cần Python 3.10+; cài thư viện một lần:

```bash
python3 -m pip install -r requirements.txt
```

```bash
python3 ung_dung.py                                   # ứng dụng web: http://127.0.0.1:8780 (thêm --thu-anh: trang thu ảnh thật)
python3 nhan_dang.py anh.png --bai-toan chu_so        # nhận diện một ảnh (chu_so | hinh_hoc), in JSON
python3 huan_luyen.py --bai-toan hinh_hoc             # huấn luyện lại (--sinh-du-lieu: chuẩn bị/sinh dữ liệu trước)
python3 danh_gia.py --bai-toan ca_hai                 # đánh giá; --thu-nghiem: các thí nghiệm đo độ bền
python3 -m pytest                                     # chạy bộ kiểm thử
```

Mỗi file có `--help`. Dữ liệu, mô hình đã huấn luyện và kết quả có sẵn nên không cần huấn luyện lại để dùng. Hướng dẫn chi tiết: [bao_cao/huong_dan_su_dung.md](bao_cao/huong_dan_su_dung.md).

## Hệ thống làm gì

```
ảnh (vẽ / tải lên / tệp)
  └─ tiền xử lý: giới hạn kích thước → xám → (lọc trung vị) → Otsu → cực tính → thành phần liên thông lớn nhất
                 → hộp bao → thu nhỏ giữ tỉ lệ (trung bình diện tích) → căn giữa 28×28 → kéo giãn độ sáng
       ├─ chế độ chữ số   → CNN 10 lớp (huấn luyện trên MNIST)
       └─ chế độ hình học → CNN 3 lớp (huấn luyện trên 13.500 ảnh tự sinh)
  └─ lớp dự đoán + điểm softmax từng lớp (chưa hiệu chuẩn, không phải độ chính xác)
```

## Cấu trúc

```
BTL/
├── README.md
├── requirements.txt
├── huan_luyen.py          sinh dữ liệu + huấn luyện CNN
├── nhan_dang.py           nhận diện một ảnh
├── danh_gia.py            đánh giá + thí nghiệm đo độ bền
├── ung_dung.py            ứng dụng web (nhận diện, thu ảnh thật)
├── mo_hinh/               trọng số: <bai_toan>_{co_so,cai_tien,tot_nhat}.pth, lua_chon.json
├── loi/                   cau_hinh, cnn, tang_cuong, huan_luyen, danh_gia, du_doan
├── tien_xu_ly/            xử lý ảnh tự cài: anh, loc, nguong, hinh_thai, lien_thong, thu_phong, luong_xu_ly
├── bo_du_lieu/            nạp MNIST, bộ sinh hình, thu ảnh thật
├── giao_dien/             máy chủ web và hai trang HTML
├── thu_nghiem/            script tái tạo các thí nghiệm đo độ bền
├── du_lieu/               mnist/, hinh_hoc_tu_sinh/, chu_so_viet_tay/, hinh_ve_tay/ (101 ảnh thật của một người), anh_mau/
├── ket_qua/               số liệu và biểu đồ (chu_so/, hinh_hoc/, tich_hop/, ra_soat_tien_xu_ly/)
├── kiem_thu/              bộ kiểm thử (mỗi mô-đun một file)
└── bao_cao/               báo cáo Word, hướng dẫn, thiết kế, thực nghiệm; latex/ (bao_cao.tex, chuong/, hinh/)
```

## Kết quả

| Phần | Kết quả | Ghi chú |
|---|---|---|
| CNN chữ số | **9915/10.000 (99,15%)** trên MNIST test | MNIST chuẩn, không phải chữ viết tay của nhóm |
| CNN hình học | **2019/2025 (99,70%)** trên test tự sinh; 270/300 (90,0%) trên stress | Dữ liệu tổng hợp |
| Hình vẽ Pillow có rung tay (960 ảnh/cấu hình) | **99,6%** với thu nhỏ theo diện tích + kéo giãn; 87,3% với nội suy song tuyến tính | Ảnh do chương trình vẽ |
| **Ảnh thật** (58 chữ số, 43 hình; **chỉ một người viết**) | chữ số **50/58 (86,2%)**; hình học **43/43** | Khoảng tin cậy 95%: 75–93% và 92–100%; chữ số thấp hơn nhiều so với MNIST |
| Chữ số nét 1,5% chiều cao (150 ảnh) | **95,3%** so với 52,7% của nội suy song tuyến tính | Ảnh tổng hợp từ MNIST |
| Dữ liệu hình học đa dạng so với đơn giản (hình vẽ Pillow, chữ nhật dài) | **100%** so với 48,8% (cùng 4.500 ảnh) | Ảnh tổng hợp; các họ hình còn lại gần như không đổi |

Phương pháp, bảng đầy đủ và các hạn chế: [bao_cao/thuc_nghiem.md](bao_cao/thuc_nghiem.md). Thiết kế và quy ước đầu vào CNN: [bao_cao/thiet_ke.md](bao_cao/thiet_ke.md).

## Điều cần biết

- **Ảnh viết/vẽ tay thật chỉ có từ một người viết**: 58 chữ số và 43 hình, đã đủ số lượng tối thiểu (trạng thái `ready` chỉ xét số lượng, không xét số người viết). Trên bộ này chữ số **50/58 (86,2%)**, thấp hơn nhiều so với 99,15% trên MNIST; hình học 43/43. Cần thêm ảnh từ người viết khác (`python3 ung_dung.py --thu-anh`, rồi `python3 danh_gia.py --bai-toan ca_hai`).
- Chọn chữ số hay hình học là thao tác tay; hai mạng không có lớp "không phải", nên ảnh sai loại vẫn bị gán một lớp.
- Nét mảnh trên ảnh lớn là điểm yếu của nội suy song tuyến tính (nét bị rách thành chấm rời); hệ thống dùng trung bình theo diện tích nên đã khắc phục, xem thực nghiệm.
- Chữ nhật quá dài (tỉ lệ cạnh ≲ 0,15) và hình tròn méo nặng kèm nhiễu mạnh vẫn dễ sai; Otsu toàn cục kém khi nền loang sáng nặng.
- Báo cáo LaTeX ở `bao_cao/latex/`; PDF cần XeLaTeX (Overleaf) hoặc `python3 bao_cao/latex/bien_dich.py`. Môi trường đã chạy thử: macOS, Python 3.10, torch 2.2.2, numpy 1.26.4; chưa thử trên Windows.
