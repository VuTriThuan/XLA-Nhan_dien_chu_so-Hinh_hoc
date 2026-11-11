# Ảnh chữ số viết tay thật (58 ảnh, một người viết)

Đặt ảnh vào thư mục con theo chữ số `0/` … `9/` (PNG hoặc JPEG, một chữ số mỗi ảnh), hoặc dùng `python3 ung_dung.py --thu-anh`.
Mỗi ảnh nên có nguồn, mã người viết và nhãn được ghi **trước** khi dự đoán (trang thu ảnh tự ghi vào tệp `.json` cạnh ảnh).

Hiện có 58 ảnh của **một người viết**, cắt từ ảnh chụp ba tờ giấy (tên `trangNN_SS.png` = trang NN, ký hiệu thứ SS theo thứ tự đọc): chữ số 3 có 9 ảnh; 0, 1, 2, 9 mỗi chữ số 6 ảnh; 4, 5, 6, 7, 8 mỗi chữ số 5 ảnh.
Đã đủ số lượng tối thiểu, nhưng chỉ có một người viết nên nên bổ sung ảnh của người khác.

Cần tối thiểu 50 ảnh khác nhau, ít nhất 5 ảnh mỗi chữ số, càng nhiều người viết càng tốt. Chỉ dùng để **đánh giá** (`python3 danh_gia.py --bai-toan chu_so`),
không dùng để huấn luyện hay chỉnh mô hình đã chốt. Ảnh lấy từ MNIST không được tính vào bộ này.
