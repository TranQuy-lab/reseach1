# Tiền xử lý bốn bộ v2, giữ IP và port

Ngày 2026-09-20. Áp dụng nckh để kiểm tra dữ liệu và thiết kế bước chuẩn bị
có thể truy vết. Nguồn: CSV gộp Kaggle đã kiểm tra SHA-256 và số dòng trong
`results/kaggle_download_manifest.json` và `results/kaggle_data_inventory.json`.

## Đầu ra và cách dùng

Bốn file chính trong `../data/processed_four/`:

- `NF-UNSW-NB15-v2.parquet`
- `NF-BoT-IoT-v2.parquet`
- `NF-ToN-IoT-v2.parquet`
- `NF-CSE-CIC-IDS2018-v2.parquet`

Parquet là bảng dữ liệu có kiểu cột, nén được và đọc theo lô; không cần
chuyển lại CSV để huấn luyện. Mỗi file giữ 46 cột nguồn và thêm hai cột:
`source_row_id` là thứ tự dòng dữ liệu trong CSV gốc, bắt đầu từ 1;
`flow_group_id` là mã nhóm đầu vào phục vụ kiểm tra lặp và chia tập.

Giữ nguyên `IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, `L4_SRC_PORT`, `L4_DST_PORT`.
Port dùng int64, chỉ chấp nhận 0–65535; giữ port 0 vì không mặc định coi
đó là lỗi. Các cột nguyên khác cũng giữ int64; bốn cột vốn có phần thập
phân dùng float64. Không ép số đếm lớn xuống int16/int32 hoặc float32.

Dựng node sau này bằng `(Dataset, IP, port)`. IP/port là thông tin cần để
xây đồ thị; giữ trong file không có nghĩa bắt buộc đưa IP dưới dạng số
vào mô hình. Có thể thử port làm feature trong một thí nghiệm riêng.
Không đưa `Dataset`, `source_row_id`, `flow_group_id`, `Label`, `Attack`
vào ma trận đặc trưng. Hai cột nhãn chỉ là mục tiêu cần dự đoán.

## Chính sách làm sạch

- Đọc tuần tự theo lô, tách theo Dataset; không lấy mẫu và không cân bằng lớp.
- Kiểm tra thiếu dữ liệu, số không hữu hạn, IPv4 hợp lệ, port hợp lệ,
  tên dataset, nhãn rỗng và quan hệ `Label == 0` tương ứng `Attack == Benign`.
- Dòng không đạt được đưa sang `rejected_rows.parquet` kèm lý do, nếu có.
  File này là vùng cách ly để xem xét, không phải bộ dữ liệu huấn luyện thứ năm.
  Số lỗi theo từng lý do có thể chồng lấp; tổng số dòng loại chỉ đếm một lần.
- Không tự điền 0, cắt ngoại lệ, sửa nhãn, đổi cách viết tên attack hoặc
  xóa các flow hợp lệ chỉ vì cùng đặc trưng. Không chuẩn hóa toàn bộ file.
- Giữ các dòng lặp, đồng thời gán cùng `flow_group_id` nếu cùng IP/port,
  đặc trưng và Dataset, không phụ thuộc nhãn. Các flow này có thể là các
  lần liên lạc khác nhau; dữ liệu hiện tại không đủ để khẳng định là lỗi.
  Nếu sau này chọn loại trùng, phải đánh giá tác động đến topology và lớp hiếm.

Mã nhóm ghép hai hash uint64 của pandas với khóa cố định. Đây là công cụ
nhóm thực dụng, không phải chứng minh không có va chạm. Nhãn mâu thuẫn
trong cùng nhóm được báo cáo, không tự chọn một nhãn để thay thế.

**Khi chia train/validation/test, phải giữ cả nhóm ở cùng một split.**
Mã nhóm không loại trừ rò rỉ do cùng host, phiên hoặc đợt tấn công; cần
protocol bổ sung nếu muốn đo khả năng tổng quát sang mạng mới. Chỉ fit
imputer/scaler/encoder trên train. Chưa chia tập ở bước xuất bốn file này.

## Vì sao không chạy nguyên xi mã cũ trong repo?

`src/nids_preprocessing/cleaning.py` bỏ IP trước khi loại trùng, làm mất
thông tin dựng đồ thị và có thể gộp các liên lạc từ endpoint khác nhau.
`schema.py` của nhánh ấy khai báo port int16, không biểu diễn được đầy đủ
0–65535. Nó còn downcast một số cột số trước khi đánh giá miền giá trị.
Bản mới tham khảo cấu trúc kiểm tra nhưng dùng adapter riêng, giữ nguyên
nhánh cũ để không làm thay đổi bằng chứng của pilot lịch sử.

Không so số dòng lặp của bản giữ IP với con số loại trùng sau khi bỏ IP
trong notebook cũ như thể chúng đo cùng một thứ.

## Tái chạy và bằng chứng

Từ thư mục repo, dùng môi trường `.venv-research` đã khóa phiên bản:

```bash
.venv-research/bin/python research/preprocess_four.py \
  --source data/raw/NF-UQ-NIDS-v2/NF-UQ-NIDS-v2.csv \
  --output data/processed_four \
  --report research/results/preprocessing_four.json
.venv-research/bin/python research/verify_preprocessed_four.py
```

Lệnh tạo dữ liệu từ chối thư mục đầu ra đã tồn tại; để tái chạy phải chọn
thư mục mới và điều chỉnh đường dẫn kiểm tra tương ứng. Các file `.partial`
không phải đầu ra hoàn tất. Biên bản gồm chính sách, SHA-256 nguồn/mã/file,
số dòng theo dataset/lớp, số dòng loại và thời gian chạy.

`verify_preprocessed_four.py` đọc lại Parquet, kiểm tra số dòng, nhãn,
miền port, ID dòng duy nhất, phân bố lớp và đếm nhóm đầu vào lặp/xung đột
nhãn xuyên các lô. Bộ nhớ engine giới hạn 1 GB; giới hạn này không phải
mức RAM tối đa của cả tiến trình. Kết quả ở
`results/preprocessing_four_verification.json`.

Nhánh pilot `nids_research.prepare` chưa tự nhận bốn file mới: nó vẫn dành
cho Parquet NF-ToN-IoT-v2 cũ và hai cột endpoint đã ghép. Cần adapter chia
tập/dựng endpoint theo protocol mới trước khi chạy huấn luyện bốn bộ.

## Kết quả chạy thực tế

Đã quét 75.987.976 dòng và giữ đủ số dòng trong bốn file. Không phát hiện
dòng lỗi theo các kiểm tra đã nêu, nên không tạo file cách ly. SHA-256
nguồn khớp manifest tải. Toàn bộ 54 kiểm thử đạt; kiểm tra độc lập trên
bốn file cũng đạt. Tổng dung lượng Parquet: 3.276.040.889 byte (~3,28 GB).

| Dataset | Dòng | Nhóm đầu vào lặp | Nhóm có nhãn tấn công mâu thuẫn |
|---|---:|---:|---:|
| NF-UNSW-NB15-v2 | 2,390,275 | 2,450 | 2,450 |
| NF-BoT-IoT-v2 | 37,763,497 | 120,210 | 0 |
| NF-ToN-IoT-v2 | 16,940,496 | 0 | 0 |
| NF-CSE-CIC-IDS2018-v2 | 18,893,708 | 2,934 | 2,882 |

Các nhóm mâu thuẫn chưa được loại hoặc gán lại nhãn. Phải giữ nhóm cùng
split; khi đánh giá nên báo cáo riêng ảnh hưởng của các nhóm này. Không
có lỗi quan hệ nhãn nhị phân không đồng nghĩa mọi nhãn tấn công đều đúng.

Số NaN/Inf bằng 0 trên file nguồn hiện tại khác số 384 ghi trong notebook
cũ. Chưa có bằng chứng xác định nguyên nhân chênh lệch; không ép xóa 384
dòng để khớp ghi chú cũ. Nhãn trong bản gộp cũng khác nhãn bộ riêng, ví dụ
CSE chứa `Brute Force`, `injection` và cách viết nguồn `Infilteration`;
đã giữ nguyên để bảo toàn nguồn, không âm thầm sửa chính tả.

Biên bản: [tiền xử lý](results/preprocessing_four.json),
[kiểm tra độc lập](results/preprocessing_four_verification.json).
