> **Lưu ý khi đọc.** Đây là nhật ký rà soát lần đầu của phần tiền xử lý (09/11/2026), viết trước khi gộp mã vào một package. Tên file trong văn bản là tên cũ: `preprocessing.py` nay là `tien_xu_ly/` (mỗi nhóm kỹ thuật một mô-đun), `demo.py` đã bỏ (thay bằng `python3 nhan_dang.py`), các test nay nằm trong `kiem_thu/test_tien_xu_ly.py`. Các script `audit_*.py` không còn trong dự án; ảnh và nhật ký minh chứng giữ nguyên ở thư mục này. Các lỗi được mô tả đều đã sửa và có test hồi quy.

# Báo cáo rà soát phần tiền xử lý

Ngày kiểm tra: 09/11/2026. Phạm vi thay đổi: chỉ thư mục tien_xu_ly. Đã đọc preprocessing.py, chương trình dòng lệnh, kiem_thu/test_preprocessing.py, README và kiểm tra ảnh kết quả.

## Kết luận

### Bổ sung sau đối chiếu bản rà soát bên ngoài

Đã thêm `polarity` và `edge_dark_fraction` vào Result và metadata.json; auto phát UserWarning khi tỷ lệ mẫu viền tối nằm trong [0.35,0.65]. Giữ nguyên luật chọn nền/nét và các sửa lỗi closing/median trước đó. Ngưỡng cảnh báo là kinh nghiệm, không phải cam kết phát hiện mọi ảnh sai cực tính. Test mới xác nhận ảnh viền mơ hồ vẫn có thể chọn sai nhưng có cảnh báo, ép cực tính cho hộp bao đúng, ảnh sạch không đổi pixel, JSON và cảnh báo CLI được ghi đúng kênh.

Bộ kiểm thử hiện có 27 phương thức (24 trước đó + 3 mới trong kiem_thu/test_polarity_metadata.py). Các con số 24 và log cũ bên dưới mô tả đợt rà soát trước bổ sung; log cập nhật được lưu tại ket_qua/ra_soat_tien_xu_ly/test_results_polarity.txt.

24 phương thức kiểm thử đã chạy thành công: 8 kiểm thử cũ và 16 kiểm thử mới, một số gồm nhiều trường hợp con. CLI digit/shape đều chạy thành công. Phần tiền xử lý có thể bàn giao theo quy ước bên dưới, nhưng chưa thể kết luận độ chính xác nhận dạng do chưa có CNN. Ba lỗi xử lý ảnh được tái hiện và sửa; thêm kiểm tra tham số và thông báo lỗi đọc file.

## Lỗi đã xác nhận và sửa

| Vị trí trong preprocessing.py | Trước sửa và bằng chứng | Cách sửa |
|---|---|---|
| preprocess, khối foreground khoảng dòng 184–187 | Ảnh khối đen có một chấm trắng: median đã đưa pixel ở giữa về 0 nhưng crop vẫn có cường độ tiền cảnh 0 thay vì 255. Ảnh cuối còn lỗ nhiễu. | Lấy mức xám từ ảnh sau median thay vì ảnh gốc. |
| preprocess, cùng khối foreground | Closing có mask ở giữa bằng 255 nhưng crop vẫn bằng 0: vùng mới tạo trên mask không hiện trong đầu ra. | Khi bật hình thái, dùng mask nhị phân làm cường độ đầu ra; công khai thay đổi này trong README. |
| binary_morphology, từ dòng 81 | Closing mask 5×5 toàn 1 chỉ giữ 9/25 pixel do cắt phần giãn ngoài biên trước khi co. | Đệm đủ miền cho phép ghép, chỉ cắt về kích thước gốc sau cả hai bước. Sau sửa giữ đủ 25/25 pixel. |
| Kiểm tra tham số, từ dòng 7 và preprocess | median_size=3.0 gây TypeError nội bộ của NumPy; output_size=True từng được hiểu như số nguyên; morphology rỗng bị lặng lẽ bỏ qua. | Báo ValueError rõ ràng, từ chối bool ở tham số kích thước, kiểm tra tên phép và kiểu keep_largest. |
| chương trình dòng lệnh | Đọc đường dẫn không tồn tại hoặc ảnh hỏng ngoài khối xử lý lỗi. | Bắt OSError khi đọc ảnh và báo lỗi ngắn qua CLI. |

Chương trình cũng từ chối đầu ra hoàn toàn bằng 0 sau thu nhỏ. Đây là biện pháp phát hiện thất bại, không khắc phục toàn bộ hiện tượng mất nét khi giảm kích thước.

Ảnh trước sửa được lưu trước khi thay mã; không chạy lại ket_qua/ra_soat_tien_xu_ly/audit_before.py. Dùng ket_qua/ra_soat_tien_xu_ly/audit_examples.py để tái tạo phần sau sửa. Xem ket_qua/ra_soat_tien_xu_ly/comparison.png và các PNG có tên before/after. Log cuối: ket_qua/ra_soat_tien_xu_ly/test_results.txt.

## Cách kiểm tra độc lập

- RGB đỏ/lục/lam có kết quả xám tính tay 76/150/29; alpha 0/255/128 cho nền trắng, đen, xám 127.
- Median được đối chiếu với vòng lặp lấy từng lân cận, kẹp tọa độ ở biên và sắp xếp; cửa sổ 1,3,5.
- Otsu được đối chiếu với cách chia mảng thành hai lớp, tính trực tiếp trung bình và điểm số cho cả 256 ngưỡng trên 12 ảnh ngẫu nhiên seed cố định. Kiểm tra giá trị mục tiêu tối ưu để tránh nhầm các ngưỡng đồng hạng là lỗi.
- Bốn phép hình thái được đối chiếu với vòng lặp Boolean độc lập, SE 1,3,5, trên ảnh toàn 0, toàn 1 và ảnh ngẫu nhiên.
- Liên thông: đường chéo phải là một vùng theo 8-lân cận; hai vùng bằng diện tích chọn vùng gặp trước khi quét từ trên xuống; ảnh rỗng tiền cảnh trả mask rỗng.
- Nội suy: tính từng tọa độ và tổng bốn trọng số bằng vòng lặp, so sánh khi thu/phóng, chiều bằng 1 và giữ nguyên kích thước.
- Bounding box/căn giữa: hình chữ nhật 10×20 phải thành 10×20 trong vùng nội dung 20×20, đặt tại hàng 9:19, cột 4:24 trên canvas 28×28; kiểm tra toàn bộ ma trận bằng đáp án dựng trước.
- Kiểm tra ảnh đảo màu, float [0,1]/[0,255], RGBA, dịch vị trí, nhiễu muối tiêu, ảnh sát viền, ảnh 2×2 và 1×1, nét một pixel, nét đứt, ảnh NaN/Inf/phức/rỗng/sai số kênh và tham số sai.
- Đầu ra được kiểm tra dtype float32, giá trị, nền đen và thứ tự chiều NHWC/NCHW cho cả hai kích thước.

## Chạy chương trình và ảnh có sẵn

Digit: vi_du/digit_light/input.png, đầu ra ket_qua/ra_soat_tien_xu_ly/cli_digit; kích thước 28×28, miền [0,1]. Đây là chữ số 7 tổng hợp bằng nét thẳng, không phải mẫu viết tay thực.

Shape: ket_qua/ra_soat_tien_xu_ly/real_scan_circle_input.png, đầu ra ket_qua/ra_soat_tien_xu_ly/cli_shape; kích thước 64×64, miền [0,1], Otsu 128. Ảnh được cắt từ hình tròn B trong tài liệu scan tài liệu/Image_Processing-main/Thi/IMG_1187.JPG, hộp (375,1250,470,1335). Đã xem cả nguồn và kết quả: vùng tròn được giữ nhưng độ sáng còn biến thiên do hình gốc có gradient. Đây là hình in trong bản scan, không phải hình vẽ tay.

Đã xem bảng ảnh JPG có sẵn: chủ yếu đề thi scan/chụp, một ảnh chân dung và các trang có ghi chú tay nhiều chữ. Không tìm thấy mẫu chữ số viết tay đơn lẻ được gán nhãn sẵn phù hợp để đánh giá nhận dạng; chưa báo cáo kết quả trên tập chữ số thật. Chưa ghép CNN hoặc đo accuracy.

## Giới hạn còn lại

1. keep_largest=True làm mất đoạn nét đứt nhỏ hơn. Kiểm thử tái hiện: hộp bao chỉ giữ đoạn dưới; keep_largest=False giữ cả hai. Minh họa broken_largest.png/broken_all.png. Không tự đoán phần nào là nét, phần nào là nhiễu.
2. Median 3×3 xóa nét một pixel: kiểm thử xác nhận pipeline báo ảnh đồng nhất thay vì trả dự đoán. Mặc định không bật lọc.
3. Nền suy ra từ viền có thể sai khi đối tượng chiếm viền. Kiểm thử khối đen có lỗ trắng cho thấy auto chọn lỗ trắng; polarity='dark' chọn khối đen đúng theo ý người dùng. Không có cách suy ra ý nghĩa nền/nét chắc chắn chỉ từ histogram.
4. Otsu toàn cục chưa giải quyết bóng đổ mạnh, nền hoa văn hoặc nhiều đối tượng. Nội suy bilinear chưa có lọc chống alias khi thu nhỏ; một số nét mảnh có thể suy yếu/mất.
5. Căn hộp bao không phải căn trọng tâm MNIST. Cường độ nét xám được giữ, chưa tăng tương phản tự động. Hình học đầu ra phải khớp phân bố lúc train.
6. Float tự suy ra thang đo từ max gây mơ hồ với ảnh float [0,255] chỉ chứa 0,1: truyền uint8 để chỉ rõ thang 255. Bool cần chuyển uint8 0/255. Ảnh RGBA giả định màu RGB thông thường, không phải alpha đã nhân sẵn.
7. Chưa đo tốc độ/RAM với ảnh rất lớn; BFS bằng Python và cửa sổ median có thể tốn thời gian/bộ nhớ. Chưa kiểm thử trên nhiều ảnh chụp điện thoại của nhóm.

## Quy ước tích hợp

Đầu vào: H×W uint8 hoặc RGB/RGBA uint8; RGB không phải BGR. Float chỉ dùng khi rõ thang đo. Một đối tượng chính. Hàm preprocess trả Result.image float32 [0,1], nền 0; digit 28×28, nội dung tối đa 20×20; shape 64×64, nội dung tối đa 52×52. Size và margin chỉnh được. tensor('NHWC') trả (1,H,W,1); tensor('NCHW') trả (1,1,H,W), vẫn là NumPy. PyTorch cần torch.from_numpy; việc chuyển thiết bị và inference thuộc người tích hợp.

Người huấn luyện CNN phải xác nhận kích thước, cực tính, khoảng pixel, độ lớn vật thể, nhị phân hay xám, và có chuẩn hóa tiếp bằng mean/std hay không. Nếu CNN dùng Normalize(mean,std), người tích hợp áp dụng sau đầu ra này, không chia 255 lần hai. Chưa có mô hình để xác nhận tương thích thực tế.

## Câu hỏi bảo vệ cần chuẩn bị

- Tại sao median đã sửa mask mà trước đây ảnh cuối vẫn nhiễu? Vì lấy cường độ từ ảnh gốc, bỏ qua kết quả lọc.
- Tại sao đóng ảnh phải giữ phần giãn ngoài biên? Bước co cần các pixel đó; cắt sớm làm mất thông tin lân cận.
- Giữ thành phần lớn nhất có luôn đúng không? Không; nét đứt có thể là cùng một chữ nhưng thành nhiều vùng.
- Otsu có hiểu đâu là chữ không? Không, chỉ tối ưu tách hai lớp cường độ; phải có quy ước cực tính và giả định nền.
- Vì sao không bật mọi bộ lọc? Chúng có thể xóa nét hoặc lấp lỗ vốn phân biệt các chữ số.
- 24 kiểm thử đạt có chứng minh CNN chính xác không? Không; chỉ là kiểm tra tính đúng trên các trường hợp đã chạy. Cần dữ liệu thật và mô hình để đánh giá nhận dạng.
- Vì sao khi bật hình thái dùng đầu ra nhị phân? Để vùng thêm/bớt có tác dụng rõ; đổi lại mất thông tin xám, nên cần thí nghiệm với CNN.


## Chốt bản 09/11/2026

Đã gom 5 kiểm thử hồi quy vòng 2, mở rộng đối chiếu quy ước biên cho SE 3,5,7, và sửa mô tả tài liệu. Code xử lý và chương trình dòng lệnh giữ nguyên sau bổ sung metadata. Lần chạy cuối: 32 kiểm thử đạt, 0 failure, 0 error; 4 trường hợp CLI đạt, hai tensor lưu ra được kiểm tra về kích thước, dtype, miền giá trị và cùng nội dung. Log: ket_qua/ra_soat_tien_xu_ly/test_results_final.txt và ket_qua/ra_soat_tien_xu_ly/cli_results.json. Báo cáo tai_lieu/BAN_GIAO.md xác định phần đã chốt và phần tích hợp cần mô hình. Không sử dụng các con số audit hoặc tốc độ do bên khác đo làm kết quả của lần kiểm thử này.
