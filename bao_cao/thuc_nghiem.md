# Thực nghiệm: phương pháp và kết quả

Mọi số liệu dưới đây do chính mã của dự án tạo ra và nằm trong `ket_qua/` (JSON, CSV, biểu đồ); lệnh tái tạo ở cuối tài liệu.

## 1. Nguyên tắc đánh giá

- Chọn mô hình **chỉ bằng validation**: accuracy validation cao nhất, hòa thì loss validation thấp nhất; `mo_hinh/lua_chon.json` ghi trước khi đọc test và `danh_gia` từ chối chạy nếu checkpoint đã đổi.
- Tập train/validation/test **tách rời** (kiểm tra không có chỉ số hay ảnh trùng); tập test không bao giờ được đọc trong huấn luyện.
- So sánh hai cấu hình bằng kiểm định McNemar chính xác ghép cặp trên cùng ảnh test; khoảng Wilson 95% mô tả bất định do cỡ mẫu.
- Mỗi cấu hình huấn luyện **một hạt (42)** nên các chênh lệch nhỏ chưa đủ ý nghĩa thống kê.
- **Ảnh viết/vẽ tay thật của nhóm chỉ có từ một người viết**: 58 ảnh chữ số và 43 ảnh hình (bút mực trên giấy, chụp ảnh rồi cắt từng ký hiệu thành một ảnh riêng), đủ số lượng tối thiểu nhưng chưa có người viết khác (mục 5). MNIST, bộ hình tổng hợp, bộ stress, bộ chẩn đoán, bộ quét góc–độ dày và ảnh vẽ Pillow đều không phải ảnh người thật.

## 2. Chữ số (MNIST)

Dữ liệu: 54.000 train / 6.000 validation / 10.000 test chính thức (hạt chia 20261109). Công thức: Adam 1e-3, StepLR(8, 0,3), 12 epoch, batch 128, hạt 42; cấu hình `cai_tien` thêm tăng cường affine (xoay ±10°, dịch ±2 px, co giãn 0,9–1,1).

| Cấu hình | Val | Test | Khoảng Wilson 95% | F1 macro |
|---|---:|---:|---|---:|
| `co_so` | 98,88% | 9916/10.000 = 99,16% | [98,96; 99,32] | 99,15% |
| `cai_tien` (được chọn) | 99,02% | 9915/10.000 = 99,15% | [98,95; 99,31] | 99,14% |

McNemar: cơ sở sai/cải tiến đúng 37 ảnh, ngược lại 38 ảnh, p = 1,0 → không kết luận khác biệt. Cặp nhầm nhiều nhất: 9→7 (7), 5→3 (6), 9→4 (6), 1→2 (4), 4→9 (4).

Chẩn đoán trên cùng 1.000 ảnh test (100/chữ số; **không độc lập với test**):

| Cấu hình | Gốc | Dịch ±3 px | Xoay ±20° | Nhiễu Gauss 0,15 |
|---|---:|---:|---:|---:|
| `co_so` | 99,1% | 89,6% | 98,7% | 99,3% |
| `cai_tien` | 99,0% | 97,7% | 98,3% | 98,4% |

Cùng 1.000 ảnh đi qua tiền xử lý thống nhất (căn theo hộp bao thay vì trọng tâm): 97,6%, so với 99,0% khi đưa thẳng vào mạng.

## 3. Hình học (dữ liệu tổng hợp)

Dữ liệu: 13.500 ảnh (9.450 / 2.025 / 2.025) + bộ stress 300 ảnh; đầu vào CNN do `xu_ly_anh` tạo từ ảnh gốc 28×28. Adam 1e-3, batch 64, 20 epoch, hạt 42.

| Cấu hình | Val | Test | Khoảng Wilson 95% | F1 macro | Stress |
|---|---:|---:|---|---:|---:|
| `co_so` | 99,75% | 2014/2025 = 99,46% | [99,03; 99,70] | 99,46% | 258/300 = 86,0% |
| `cai_tien` (được chọn) | 99,90% | 2019/2025 = 99,70% | [99,36; 99,86] | 99,70% | 270/300 = 90,0% |

McNemar: cơ sở sai/cải tiến đúng 6 ảnh, ngược lại 1 ảnh, p = 0,125. Sáu ảnh test sai: 2 elip, 1 elip lượn sóng, 2 chữ nhật, 1 hình vuông. Trên stress: chữ nhật→tam giác 13, tròn→tam giác 9, tròn→chữ nhật 7. Trong 24 lỗi softmax cao nhất được vẽ lại (`ket_qua/hinh_hoc/cai_tien/stress/failure_cases.png`), phần lớn là hình bị đứt nét do bộ stress xóa một cột pixel: bước "giữ thành phần liên thông lớn nhất" chỉ còn lại một cung hoặc một góc nên mạng đoán sai. Đây là hệ quả của giới hạn đã biết của bước liên thông (xem `thiet_ke.md`), không chỉ của mạng.

Quét ghép cặp 40 đối tượng × 12 góc × 3 độ dày (tam giác, chữ nhật; bộ stress; 480 ảnh mỗi độ dày):

| Độ dày nét | 1 px | 2 px | 3 px |
|---|---:|---:|---:|
| `co_so` | 90,8% | 92,1% | 89,0% |
| `cai_tien` | 94,2% | 94,8% | 95,0% |

Cấu hình `cai_tien` thay đồng thời BatchNorm, Dropout và tăng cường nên không quy được khác biệt cho một yếu tố.

## 4. Cả hệ thống: nét mảnh và hình vẽ Pillow (`ket_qua/tich_hop/`)

Ba cấu hình khâu thu nhỏ/chuẩn hóa: **S.t.t.** (nội suy song tuyến tính, cách cũ), **Diện tích** (trung bình theo diện tích), **D.t.+kéo giãn** (mặc định).

Chữ số nét mảnh (150 ảnh/dòng; MNIST phóng to, làm mỏng nét bằng Zhang–Suen rồi giãn lại):

| Nét / chiều cao chữ số | S.t.t. | Diện tích | D.t.+kéo giãn |
|---|---:|---:|---:|
| 12% | 99,3% | 99,3% | 99,3% |
| 8% | 98,7% | 98,7% | 98,7% |
| 5% | 98,0% | 98,7% | 98,7% |
| 3% | 90,7% | 98,0% | 98,0% |
| 1,5% | 52,7% | 82,0% | 95,3% |

Hình vẽ Pillow có rung tay (6 họ hình × 4 độ dày nét × 40 ảnh = 960 ảnh mỗi cấu hình): trung bình **87,3% (S.t.t.) → 94,2% (diện tích) → 99,6% (diện tích + kéo giãn)**. Hình tròn nét mảnh nhất (1,2% kích thước): 5% → 100%;
chữ nhật dài nét mảnh nhất: 38% (S.t.t.), 5% (chỉ diện tích), 98% (diện tích + kéo giãn). Cần cả hai bước: trung bình diện tích giữ được nét nhưng làm nó mờ, kéo giãn độ sáng đưa nó về mức ảnh huấn luyện.
Ảnh Pillow cùng họ hình với dữ liệu đa dạng nên phép đo này cho biết khả năng tổng quát hóa sang bộ vẽ khác, không phải sang hình người vẽ thật.

## 5. Ảnh viết/vẽ tay thật (một người viết)

Nguồn: một người viết chữ số và vẽ hình bằng bút mực trên giấy, chụp ảnh các tờ giấy bằng điện thoại, rồi cắt từng ký hiệu thành một ảnh riêng
(`du_lieu/chu_so_viet_tay/<chữ số>/`, `du_lieu/hinh_ve_tay/<lớp>/`; tên `trangNN_SS.png` = trang NN, ký hiệu thứ SS theo thứ tự đọc). Có ba tờ: tờ 1 gồm 30 chữ số và 20 hình,
tờ 2 gồm 23 chữ số (một vết bẩn trên giấy đã bị bỏ, không phải chữ số), tờ 3 gồm 5 chữ số 3, 11 hình tròn và 12 tam giác. Nhãn được ghi theo thứ tự đọc **trước** khi chạy dự đoán; sau đó người viết đã rà soát lại toàn bộ nhãn.
Đánh giá bằng `python3 danh_gia.py --bai-toan ca_hai`, tức đúng đường xử lý của ứng dụng. Ảnh chụp có nét chữ mặt sau tờ giấy lộ mờ qua giấy và ánh sáng không đều, nên gần với điều kiện thật hơn ảnh sinh sẵn.

**Đã đủ số lượng tối thiểu** (trạng thái `ready`, chỉ xét số lượng): chữ số 58 ảnh (cần 50, mỗi chữ số ≥ 5: chữ số 3 có 9 ảnh; 0, 1, 2, 9 có 6 ảnh; 4–8 có 5 ảnh), hình 43 ảnh
(16 tròn, 17 tam giác, 10 chữ nhật/vuông; cần 30 và mỗi hình ≥ 10). **Nhưng chỉ có một người viết**, nên đây vẫn không đại diện cho chữ viết tay nói chung.

| Bài toán | `co_so` | `cai_tien` (được chọn) | Khoảng Wilson 95% (`cai_tien`) |
|---|---:|---:|---|
| Chữ số | 49/58 = 84,5% | **50/58 = 86,2%** | [75,1; 92,8] |
| Hình học | 42/43 = 97,7% | **43/43 = 100%** | [91,8; 100] |

- **Hình học**: 43/43 gồm tam giác nhiều hướng (đỉnh lên, đỉnh xuống, nằm ngang, tam giác vuông), hình tròn lớn và nhỏ (kể cả vòng vẽ xoắn), chữ nhật rất dài và 3 hình tô đặc (hệ thống chỉ học hình đường viền).
  Mô hình cơ sở sai 1 ảnh (hình vuông tô đặc bị nhận là tam giác). Mẫu chữ nhật/vuông chỉ có 10 ảnh.
- **Chữ số**: 50/58 thấp hơn nhiều so với 99,15% trên MNIST. Tám ảnh sai: 2→1, 4→2, 4→9, 7→2, 8→3 và ba ảnh 9→3. Recall theo chữ số: 0, 1, 3, 5, 6 đúng hết (5 ảnh chữ số 3 của tờ 3 đều đúng); 2 là 5/6; 7 và 8 là 4/5; 4 là 3/5; chữ số 9 chỉ 3/6.
  Theo từng tờ: tờ 1 đạt 25/30, tờ 2 đạt 20/23, tờ 3 đạt 5/5 (tờ 3 chỉ có chữ số 3).
  Xem ảnh đầu vào CNN của các ảnh sai (quan sát bằng mắt, chưa phân tích định lượng): cả ba chữ số 9 sai được viết kiểu chữ "g" có đuôi móc; chữ số 8 viết hẹp; một chữ số 4 viết đầu kín trông như số 9 có đuôi dài;
  một chữ số 7 kiểu có gạch ngang giữa và móc đầu bị nhận là 2 (độ tin cậy softmax thấp, 0,41); ở một chữ số 4 khác, nét sổ đứng không dính vào nét chéo
  nên bị bước "giữ thành phần liên thông lớn nhất" bỏ mất (đầu vào chỉ còn hình chữ L). Thử tắt bước đó trên 30 ảnh của tờ thứ nhất: số 4 được nhận đúng nhưng hai ảnh khác bị sai thêm, còn **24/30**
  (chưa thử lại trên toàn bộ ảnh), nên đây không phải lỗi sửa được bằng một tùy chọn.
- **Cách đọc kết quả**: các ảnh đều của một người viết nên khoảng tin cậy chỉ nói về độ bất định do cỡ mẫu, không nói về sự khác nhau giữa người viết. Khoảng cách giữa 99,15% (MNIST) và 86,2% là tín hiệu đáng chú ý
  về chênh lệch giữa MNIST và chữ viết tay thật (kiểu viết như số 9 chữ "g" hay số 7 có gạch ngang, cách căn chữ, ảnh chụp), nhưng nguyên nhân chưa được kiểm chứng. Cần thêm ảnh từ nhiều người.
- Các ảnh này chỉ dùng để **đánh giá**, không dùng để huấn luyện hay chỉnh mô hình đã chốt; kết quả ở đây vì vậy không được "tối ưu" cho bộ ảnh này.

## 6. So sánh dữ liệu huấn luyện hình học

Cùng kiến trúc `cai_tien`, công thức, hạt (42) và tiền xử lý; chỉ khác bộ sinh dữ liệu và số ảnh. **Đơn giản** = tam giác đều, chữ nhật tỉ lệ 0,55–1, hình tròn hoàn hảo, không rung đỉnh
(kiểu dữ liệu của bản đầu). Hai mô hình phụ (đơn giản 4.500 ảnh, đa dạng 4.500 ảnh) được huấn luyện trong thư mục tạm; mô hình thứ ba là mô hình chính (đa dạng, 13.500 ảnh). Mọi bộ đo là ảnh tổng hợp.

| Dữ liệu huấn luyện | Test đơn giản (675) | Test đa dạng (2.025) | Stress đa dạng (300) | Hình vẽ Pillow (TB) |
|---|---:|---:|---:|---:|
| Đơn giản, 4.500 ảnh | 100,0% | 96,9% | 80,3% | **91,5%** |
| Đa dạng, 4.500 ảnh | 99,9% | 99,5% | 85,3% | **99,7%** |
| Đa dạng, 13.500 ảnh (mô hình chính) | 100,0% | 99,7% | 90,0% | **99,6%** |

Hình vẽ Pillow theo họ hình (trung bình 4 độ dày nét, 160 ảnh mỗi họ):

| Họ hình (Pillow) | Đơn giản, 4.500 ảnh | Đa dạng, 4.500 ảnh | Đa dạng, 13.500 ảnh (mô hình chính) |
|---|---:|---:|---:|
| Tròn | 100,0% | 100,0% | 100,0% |
| Tam giác đều | 100,0% | 100,0% | 100,0% |
| Tam giác vuông | 100,0% | 100,0% | 99,4% |
| Tam giác bất kỳ | 100,0% | 98,1% | 99,4% |
| Chữ nhật (0,55–1) | 100,0% | 100,0% | 100,0% |
| Chữ nhật dài (0,25–0,45) | 48,8% | 100,0% | 98,8% |

Nhận xét:

- Bộ test của chính dữ liệu đơn giản đã **bão hòa** (100,0%), nên không cho biết mô hình có tổng quát hay không; trên test đa dạng mô hình đó chỉ đạt 96,9%.
- Dữ liệu đa dạng cải thiện rõ nhất ở **chữ nhật dài**: 48,8% → 100% (4.500 ảnh) và 98,8% (13.500 ảnh). Các họ còn lại gần như không đổi: tam giác vuông và bất kỳ đã nhận đúng khi mô hình chỉ học tam giác đều
  (với tiền xử lý hiện tại), nên phần lớn lỗi tam giác trước đây là do thu nhỏ chứ không phải do dữ liệu.
- Gấp 3 số ảnh (4.500 → 13.500) tăng accuracy trên stress (85,3% → 90,0%) nhưng không thấy khác biệt trên hình vẽ Pillow (99,7% so với 99,6%, trong sai số của 960 ảnh).
- Một hạt mỗi cấu hình; bộ Pillow cùng họ hình với dữ liệu đa dạng nên đo khả năng tổng quát hóa sang bộ vẽ khác, không phải sang hình người vẽ thật.

## 7. Tái tạo

```bash
python3 huan_luyen.py --sinh-du-lieu            # MNIST + 13.500 ảnh hình học
python3 huan_luyen.py --bai-toan hinh_hoc       # (chữ số: --bai-toan chu_so; ghi đè mô hình hiện có)
python3 danh_gia.py --bai-toan hinh_hoc
python3 danh_gia.py --thu-nghiem               # net_manh, ve_pil, so_sanh_du_lieu_hinh (vài chục phút)
```

Số liệu huấn luyện có tính lặp lại trên cùng máy (CPU, hạt cố định). Giữa các máy/phiên bản thư viện có thể lệch nhẹ; kết quả đã lưu và checksum của checkpoint nằm trong `mo_hinh/lua_chon.json` và `ket_qua/`.
