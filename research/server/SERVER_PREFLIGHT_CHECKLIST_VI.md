# Checklist trước khi chạy full-data

Không chạy thẳng module huấn luyện. Dùng pipeline để mọi cổng dưới đây được
cưỡng chế và lưu bằng chứng.

## 1. Máy và dữ liệu

- Repository ở đúng commit đã phát hành, không sửa file tracked trên server.
- Python 3.12; môi trường được tạo bằng `bootstrap_server.sh`.
- RAM tối thiểu 120 GiB; GPU NVIDIA tối thiểu 24 GiB VRAM và hỗ trợ BF16.
- Bốn Parquet thật đã được `git lfs pull` và hard-link vào
  `data/processed_four/`; không còn file pointer LFS.
- `check_server.py` phải xác nhận đúng SHA-256 và tổng 75.987.976 dòng.

## 2. Chuẩn bị và benchmark

```bash
PYTHONPATH=src .venv-server/bin/python research/server/run_full_pipeline.py \
  --stage prepare --threads 12
```

Chỉ tiếp tục khi:

- `full_prepare.json` có `complete: true`, bảo toàn dòng và không trùng
  `flow_group_id` giữa split;
- manifest có thống kê nhóm nhãn mâu thuẫn và IP overlap theo split;
- `full_benchmark_estimate.json` có `safe_to_launch_72: true`;
- `training_budget` đúng 20.000 step và validation mỗi 1.000 step;
- log benchmark không có NaN/Inf/OOM.

## 3. Run chính

```bash
PYTHONPATH=src .venv-server/bin/python research/server/run_full_pipeline.py \
  --stage train --threads 12
```

Không đổi batch, fanout, seed, patience hoặc ngân sách sau khi xem test. Có thể
chạy lại đúng lệnh để resume; pipeline chỉ bỏ qua run đã đủ artifact và có
provenance khớp.

## 4. Điều kiện hoàn tất

Chạy lần lượt stage `verify`, `report`, `test`. Chỉ coi là hoàn tất khi:

- `runs.csv` có đúng 72 tổ hợp duy nhất;
- 72 checkpoint được nạp lại và metric được tính lại;
- verification lưu SHA-256 của sáu artifact bắt buộc cho từng run;
- protocol và mọi file mã vẫn khớp SHA-256 lúc bắt đầu;
- 46+ test đều đạt;
- báo cáo có paired-seed deltas, cảnh báo lớp hiếm và provenance hình.

Random Forest/GBDT và sampled-neighbor evaluation là thí nghiệm xác nhận riêng.
Không thêm chúng vào giữa ma trận 72 run chính.
