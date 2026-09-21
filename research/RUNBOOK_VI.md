# Bắt đầu lại E-GraphSAGE từ đây

> Phạm vi nghiên cứu mới: **bốn bộ v2**. Xem [tài liệu chính](RESEARCH_PLAN_VI.md)
> để chọn nguồn tải và máy tiền xử lý. Các lệnh bên dưới chỉ dành cho pilot lịch sử.

## Phạm vi bản chạy này

Đây là baseline thích nghi NF-ToN-IoT-v2 và một pilot ngoại tuyến, không phải
bản tái lập nguyên trạng bài báo. Tất cả ghi vào thư mục mới; dữ liệu và
kết quả cũ vẫn nguyên vẹn. Bản triển khai CPU, không tự chọn CUDA hay FP16.

Điểm bắt đầu là **Parquet đã qua tiền xử lý** trong repo. Không có raw CSV
trong phiên này để kiểm chứng lại bước trích xuất 39 đặc trưng, điền 0,
lọc/lấy mẫu từ NF-UQ-NIDS-v2. Không gọi kiểm toán này là kiểm toán dữ liệu gốc.

## Các phần nên đọc

1. `PROTOCOL.md`: câu hỏi, đối chứng, quy tắc lấy mẫu/chia tập và giới hạn.
2. `../src/nids_research/prepare.py`: kiểm toán toàn bộ và lấy mẫu có ID nguồn.
3. `../src/nids_research/preprocessing.py`: scaler/nhãn fit từ train, lưu JSON.
4. `../src/nids_research/models.py`: dựng đồ thị và hai lớp E-GraphSAGE.
5. `../src/nids_research/experiment.py`: huấn luyện, chọn checkpoint, đánh giá.
6. `REPORT_VI.md`: kết quả thực đo và quyết định tiếp theo (sau khi chạy xong).

## Chạy lại trên máy CPU

Thực hiện từ thư mục gốc repo, dùng Python 3.12 và `uv`. Môi trường khóa ở
`requirements-lock.txt` là môi trường đã chạy thực tế, không phải yêu cầu
của code gốc. Torch CPU lấy từ kho wheel chính thức PyTorch.

```bash
uv venv --python 3.12 .venv-research
uv pip install --python .venv-research/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r research/requirements-lock.txt
PYTHONPATH=src .venv-research/bin/python -m pytest tests -q
```

Nếu `.venv-research` đã tồn tại và dùng được thì bỏ qua lệnh tạo môi trường.
Không cài thêm DGL: encoder mới thực hiện trực tiếp các phép tính bằng Torch.

### 1. Kiểm toán và chuẩn bị mẫu

```bash
PYTHONPATH=src .venv-research/bin/python -m nids_research.prepare \
  --data data/dl_processed \
  --output research/artifacts/pilot_data_rerun \
  --sample-size 50000 --seed 20260919
```

Quét toàn bộ train và validation cũ theo batch 65.536 dòng, không nạp toàn
bộ 13,55 triệu dòng vào RAM. DuckDB giới hạn 600 MB cho bước nhóm fingerprint;
giới hạn đó không bao gồm RAM của Python/Arrow và không bảo đảm mức RAM toàn tiến trình.
Thư mục đầu ra phải mới: chương trình từ chối ghi đè.

Đầu ra:
- `audit.json`: phạm vi quét, checksum nguồn, nhãn, giá trị không hữu hạn,
  endpoint/nhãn sai, khoảng min/max và nhóm trùng.
- `fingerprints.parquet`, `audit.duckdb`: bằng chứng trung gian kiểm toán.
- `train.parquet`, `val.parquet`, `test.parquet`: 35.000/5.000/10.000 flow.
- `splits.json`: checksum, phân bố lớp và danh sách đặc trưng theo thứ tự.

Không dùng target encoding; PROTOCOL/L7_PROTO/TCP_FLAGS vẫn là numeric như
nhánh v2 cũ. Đây là lựa chọn baseline, không khẳng định encoding này tối ưu.

### 2. Huấn luyện thí nghiệm đã chốt

```bash
PYTHONPATH=src .venv-research/bin/python -m nids_research.experiment \
  --data research/artifacts/pilot_data_rerun \
  --output research/artifacts/pilot_runs_rerun \
  --seeds 11 22 33 --epochs 120 --patience 20 --threads 2
```

Chạy 4 mô hình × 3 seed. Neural models chọn checkpoint bằng validation
macro-F1; test không dùng chọn epoch. Random Forest dùng cấu hình cố định.
Vì khác kiến trúc và tối ưu hóa, cùng ngân sách không đồng nghĩa cùng mức hội tụ.

Mỗi run lưu:
- `config.json`, `preprocessor.json`, `history.json`;
- `model.pt` (chỉ state_dict; nạp với weights_only=True), hoặc `forest.npz`
  (mảng số, không pickle);
- `metrics.json`: đủ 10 lớp, kể cả lớp dự đoán không đúng mẫu nào;
- `test_predictions.parquet`: flow ID, nhãn thực, dự đoán, xác suất;
- kiểm tra nạp lại artifact cho cùng xác suất mà không cần cột nhãn.

Thư mục tổng có `provenance.json`, `runs.csv`, `summary.csv`.
Đồ thị song hướng chỉ phục vụ truyền thông điệp, không nhân đôi loss/metrics.
Không gọi file `model.pt` cũ trong `output/` để tránh lẫn kiến trúc và scaler.

### 3. Dùng lại mô hình đã lưu

```bash
PYTHONPATH=src .venv-research/bin/python -m nids_research.predict \
  --run research/artifacts/pilot_runs_rerun/sage_edge_seed11 \
  --input research/artifacts/pilot_data_rerun/test.parquet \
  --output research/artifacts/predictions_rerun.parquet
```

Đầu vào chỉ cần 39 cột đặc trưng theo tên và `src_node`, `dst_node`.
Không cần `Attack`, `Label`, `Dataset`; nếu có, không đưa chúng vào mô hình.
Scaler được nạp từ artifact, tuyệt đối không fit lại trên dữ liệu mới.

**Graph context là toàn bộ file đầu vào.** Chia file thành nhiều mảnh để
predict có thể làm đổi lân cận và dự đoán; không làm vậy rồi gọi là tương
đương. Các luồng đến sau nằm trong cùng file có thể ảnh hưởng biểu diễn:
đây là suy luận ngoại tuyến, không phải cam kết phát hiện online.

### 4. Kiểm chứng lại metric và tạo báo cáo

```bash
PYTHONPATH=src .venv-research/bin/python research/validate_results.py \
  --data research/artifacts/pilot_data_rerun \
  --runs research/artifacts/pilot_runs_rerun \
  --cli research/artifacts/predictions_rerun.parquet \
  --output research/artifacts/verification_rerun.json
.venv-research/bin/python research/diagnose_graphs.py \
  --data research/artifacts/pilot_data_rerun \
  --output research/artifacts/graph_diagnostics.json
MPLCONFIGDIR=/tmp/nids-mpl .venv-research/bin/python research/build_report.py \
  --data research/artifacts/pilot_data_rerun \
  --runs research/artifacts/pilot_runs_rerun \
  --output research/results_rerun --report research/REPORT_RERUN_VI.md
```

Script báo cáo dành cho đúng protocol 4 mô hình × seed 11/22/33, mẫu
50.000 và split đã định nghĩa. Nếu đổi protocol, phải sửa phần diễn giải
cùng với cấu hình, không chỉ thay tên file rồi giữ nguyên kết luận.

## Nếu muốn bắt đầu từ raw CSV

Nhánh pilot này chưa cung cấp adapter raw-to-Parquet đã kiểm thử trên CSV thật.
Không dùng `src/nids_preprocessing/` trực tiếp cho GNN vì nó bỏ IP trước khi
dựng endpoint. Cần bảo toàn IP/port, đọc port bằng int32/uint16 hợp lệ,
ghi ID nguồn, kiểm tra nhãn và thống kê NaN/Inf; thống nhất chính sách làm
sạch trước khi chia tập. Mọi tham số học từ dữ liệu phải fit trên train.

Các giá trị 0 đã có trong Parquet không thể tự phân biệt là số 0 thật hay
giá trị được điền từ NaN/Inf trước đó. Muốn kết luận về chất lượng bước này
phải có raw data và log tiền xử lý.

## Những điều không được suy ra từ pilot

- Nhiều seed cùng split không phải nhiều bộ dữ liệu độc lập.
- Macro-F1 lớp hiếm có thể biến động lớn vì rất ít mẫu test.
- Random split giữ nhiều endpoint chung, không đánh giá host/network mới.
- Lấy mẫu flow làm đổi cấu trúc lân cận; không ngoại suy sang full graph.
- Baseline này dùng v2, 39 features và quy tắc khác notebook gốc; không so
  chênh lệch điểm trực tiếp với bảng bài báo như bằng chứng cải tiến.
- Chưa có temporal/group holdout, cross-dataset, tuning tương đương hoặc
  đánh giá hội tụ toàn diện. Không dùng kết quả để cam kết hiệu quả triển khai.
