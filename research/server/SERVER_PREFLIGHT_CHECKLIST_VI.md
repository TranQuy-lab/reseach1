# Checklist trước khi chạy full-data

Không chạy thẳng module huấn luyện. Dùng pipeline để mọi cổng dưới đây được
cưỡng chế và lưu bằng chứng.

## 1. Máy và dữ liệu

- Repository ở đúng commit đã phát hành, không sửa file tracked trên server.
- Python 3.12; môi trường được tạo bằng `bootstrap_server.sh`.
- RAM tối thiểu 120 GiB; GPU NVIDIA tối thiểu 23 GiB VRAM khả dụng và hỗ trợ
  BF16. RTX 3090 24 GB thường báo khoảng 23,56 GiB qua PyTorch.
- Bốn Parquet thật đã được `git lfs pull` và hard-link vào
  `data/processed_four/`; không còn file pointer LFS.
- `check_server.py` phải xác nhận đúng SHA-256 và tổng 75.987.976 dòng.

## 2. Chuẩn bị và benchmark

```bash
PYTHONPATH=src .venv-server/bin/python research/server/run_full_pipeline.py \
  --stage prepare --threads 12 --num-workers 4 \
  --budget-usd 6 --hourly-price-usd <gia-thue-GPU-moi-gio>
```

Chỉ tiếp tục khi:

- `full_prepare.json` có `complete: true`, bảo toàn dòng và không trùng
  `flow_group_id` giữa split;
- manifest có thống kê nhóm nhãn mâu thuẫn và IP overlap theo split;
- `full_benchmark_estimate.json` có `safe_to_launch_72: true`;
- `training_budget` đúng hai lượt train, tối thiểu 1.500 step, bốn lần
  validation mục tiêu mỗi lượt và khoảng validation tối thiểu 500 step;
- benchmark và train dùng cùng `num_workers` (mặc định 4);
- CUDA peak dưới 90% VRAM khả dụng và RSS tiến trình chính dưới 80% RAM host;
  lưu ý `peak_rss_kib_process` là `RUSAGE_SELF` nên **không** gồm bốn DataLoader
  worker. Đo BoT-IoT thực tế: tiến trình chính báo 53,4 GiB trong khi cgroup
  `memory.current` cao hơn hẳn; kernel này không có `memory.peak`, nên hãy lấy
  mẫu `memory.max`/`memory.current` của cgroup (hoặc RSS cả cây tiến trình)
  trong lúc benchmark thay vì tin tuyệt đối vào con số của estimator;
- `cost_estimate.cost_gate_evaluated: true` khi thuê GPU và
  `planning_cost_usd <= 6`; nếu giá bằng 0 thì phải tự so giá thuê với
  `max_hourly_price_for_budget_usd` trước khi khởi chạy;
- log benchmark không có NaN/Inf/OOM.

## 3. Run chính

```bash
PYTHONPATH=src .venv-server/bin/python research/server/run_full_pipeline.py \
  --stage train --threads 12 --num-workers 4
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
