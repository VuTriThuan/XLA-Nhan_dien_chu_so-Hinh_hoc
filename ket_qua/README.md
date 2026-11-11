# Kết quả

Số liệu và biểu đồ do `python3 danh_gia.py --bai-toan ca_hai` và `python3 danh_gia.py --thu-nghiem` tạo ra; không sửa tay.

- `chu_so/`, `hinh_hoc/`: mỗi cấu hình (`co_so/`, `cai_tien/`) có `config.json`, `history.*`, `test_metrics.json`, `test_predictions.npz`, ma trận nhầm lẫn, ảnh lỗi, cặp nhầm lẫn,
  `external_metrics.json` (ảnh thật; hiện `ready` về số lượng: 58 chữ số và 43 hình, đều của một người viết) và các bộ phụ (chữ số: `diagnostic_metrics.json`, `preprocessing_comparison.json`; hình học: `stress_metrics.json`, `grouped_metrics.json`).
  Ở thư mục bài toán: `comparison.json/csv` (McNemar), biểu đồ so sánh; hình học thêm `rotation_thickness_sweep.png`, `robustness_sweep.json`.
- `tich_hop/`: thí nghiệm đo độ bền cả hệ thống (nét mảnh, hình vẽ Pillow, so sánh dữ liệu hình học) bằng ảnh do chương trình tạo.
- `ra_soat_tien_xu_ly/`: hồ sơ rà soát tiền xử lý đợt đầu (ảnh trước/sau khi sửa lỗi, nhật ký kiểm thử); không do lệnh nào tạo lại.
