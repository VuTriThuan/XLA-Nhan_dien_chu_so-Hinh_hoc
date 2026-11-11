# Hồ sơ rà soát tiền xử lý

Bằng chứng của đợt rà soát đầu tiên (09/11/2026) phần tiền xử lý: ảnh trước/sau khi sửa lỗi và nhật ký kiểm thử.
Không thuộc luồng nhận dạng; chỉ để tham khảo và đưa vào báo cáo. Mô tả các lỗi đã sửa: [nhat_ky_ra_soat.md](nhat_ky_ra_soat.md).

| Nội dung | Ghi chú |
|---|---|
| `comparison.png` | Bảng so sánh lỗi trước/sau sửa và giới hạn nét đứt |
| `*_before_*.png`, `*_after_*.png`, `hole_input.png`, `broken_*.png` | Ảnh minh chứng từng lỗi |
| `test_results*.txt`, `cli_results.json` | Nhật ký kiểm thử và kiểm tra dòng lệnh của bản cũ |
| `real_scan_circle_*.png`, `real_source.json` | Hình tròn in cắt từ đề thi scan (không phải hình vẽ tay) |

Các lỗi này đều có test hồi quy tương ứng trong `kiem_thu/test_tien_xu_ly.py`.
