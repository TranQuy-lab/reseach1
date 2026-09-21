# Gói chạy server E-GraphSAGE

Gói này tái tạo benchmark đã kiểm chứng: bốn dataset v2, hai task, ba mô
hình, ba seed, tổng cộng 72 run. Notebook chỉ điều phối các module trong
`src/nids_minibatch`; không sao chép logic huấn luyện sang cell riêng.

## Cấu hình server

- Ubuntu 22.04/24.04 x86-64 và Python 3.12.
- Pilot đã kiểm chứng: tối thiểu 16 GB RAM, khuyến nghị 32 GB.
- Từ đầu gồm tải raw: ít nhất 40 GB trống, khuyến nghị 60 GB.
- Nếu đã chuyển đủ bốn Parquet đã kiểm chứng: cần tối thiểu 12 GiB trống
  sau khi cài môi trường để tạo split, checkpoint và báo cáo; nên có 18 GiB.
- Pilot: GPU NVIDIA 16–24 GB VRAM. Full-data: RTX 5090 32 GB VRAM hoặc
  tương đương, driver/runtime hỗ trợ CUDA 12.8; benchmark sẽ chặn nếu thiếu.
- Pipeline full-data 75.987.976 flow nằm trong các notebook `10_FULL_*` đến
  `12_FULL_*`. Notebook 10 bắt buộc benchmark giới hạn trước và không tự chạy
  72 cấu hình. Notebook 11 chỉ chạy khi cổng ETA/RAM/VRAM/ổ đĩa đạt.

## Các file phải chuyển lên server

### Tải repository hoàn chỉnh từ GitHub

Repository `reseach1` chứa cả code, kế hoạch và bốn Parquet Git LFS. Trên
server mới chỉ cần clone một repository. Tạo hard-link vào thư mục dữ liệu mà
pipeline sử dụng để không nhân đôi 3,28 GB Parquet:

```bash
cd /workspace
git lfs install --skip-repo
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
  https://github.com/TranQuy-lab/reseach1.git
cd /workspace/reseach1
git lfs pull

mkdir -p data/processed_four
ln ./*.parquet data/processed_four/
```

Nếu lệnh `ln` báo hai thư mục thuộc hai ổ đĩa khác nhau, thay bằng `cp` và
bảo đảm còn ít nhất 18 GiB trống. Không đưa token GitHub hay Kaggle vào
notebook, file cấu hình hoặc lịch sử shell. Sau đó cài môi trường và chạy bước
kiểm tra ở phần dưới; pipeline sẽ xác minh lại tên file, checksum và số dòng.

Chuyển toàn bộ repository, ngoại trừ `.venv-*`, `research/artifacts/`, cache
Python và dữ liệu nếu chọn tải lại trên server. Các thành phần bắt buộc:

- `src/nids_minibatch/`
- `src/nids_research/schema.py`
- `research/server/`
- `research/download_kaggle.py`, `research/verify_kaggle_data.py`
- `research/preprocess_four.py`, `research/verify_preprocessed_four.py`
- `research/validate_minibatch_results.py`, `research/build_minibatch_report.py`
- `research/PROTOCOL_MINIBATCH_VI.md`
- `tests/`
- `notebooks/server/`

Không cần mang raw CSV nếu server có Internet. `data_manifest.json` ghi URL,
kích thước, SHA-256 và số dòng kỳ vọng. Kaggle API key không cần vì URL này
công khai; không chép token vào server hoặc notebook.

Nếu muốn bỏ qua tải và tiền xử lý, có thể chuyển bốn file trong
`data/processed_four/`; pipeline sẽ kiểm tra lại trước khi chia tập. Tổng dung
lượng bốn Parquet khoảng 3,28 GB. Phải giữ nguyên tên file; checksum phải
khớp `data_manifest.json`. Khi đủ bốn file hợp lệ, stage download tự bỏ qua.

## Cài môi trường

GPU CUDA 12.8:

```bash
cd /workspace/reseach1
ACCELERATOR=cu128 bash research/server/bootstrap_server.sh
```

CPU:

```bash
ACCELERATOR=cpu bash research/server/bootstrap_server.sh
```

Script tạo `.venv-server`, cài phiên bản khóa, đăng ký kernel Jupyter và chạy
kiểm tra tài nguyên. PyTorch 2.8.0 CUDA 12.8 và wheel PyG tương ứng tồn tại
trên các index chính thức của PyTorch/PyG.

## Chạy tự động pilot

```bash
DEVICE=auto THREADS=8 bash research/server/run_all.sh
```

Mỗi stage có log trong `research/artifacts/server_logs/`. Download theo
byte-range có thể tiếp tục; train dùng `--resume`, kiểm tra provenance và bỏ
qua run đã đủ artifact. Lệnh kết thúc chỉ sau khi checkpoint verification,
báo cáo và test đều đạt.

Full-data không dùng `run_all.sh`. Chuẩn bị và benchmark giới hạn trước:

```bash
PYTHONPATH=src .venv-server/bin/python research/server/run_full_pipeline.py \
  --stage prepare --threads 12
```

Chỉ khi `research/results/full_benchmark_estimate.json` có
`safe_to_launch_72: true` mới chạy:

```bash
PYTHONPATH=src .venv-server/bin/python research/server/run_full_pipeline.py \
  --stage train --threads 12
```

## Chạy bằng JupyterLab

```bash
.venv-server/bin/jupyter lab --no-browser --ip=0.0.0.0
```

Mở và chạy theo thứ tự:

1. `notebooks/server/00_CHECK_SERVER.ipynb`
2. `notebooks/server/01_DOWNLOAD_PREPROCESS.ipynb`
3. `notebooks/server/02_SPLIT_TUNE.ipynb`
4. `notebooks/server/03_TRAIN_72_RUNS.ipynb`
5. `notebooks/server/04_VERIFY_REPORT.ipynb`

Các notebook trên giữ benchmark pilot lịch sử. Kết quả chính full-data chạy:

1. `notebooks/server/10_FULL_PREPARE_BENCHMARK.ipynb`
2. xem `research/results/full_benchmark_estimate.json`
3. `notebooks/server/11_FULL_TRAIN_72.ipynb`
4. `notebooks/server/12_FULL_VERIFY_REPORT.ipynb`

Mặc định `NIDS_DEVICE=auto` và `NIDS_THREADS=8`. Có thể đặt trước khi mở
Jupyter, ví dụ `NIDS_DEVICE=cuda NIDS_THREADS=12 ...`.

## Đầu ra hoàn tất

- `research/artifacts/minibatch_runs/runs.csv` có đúng 72 dòng.
- `research/results/minibatch_verification.json` có `passed: true` và
  `runs_checked: 72`.
- `research/results/minibatch/summary.csv` và biểu đồ PNG/SVG.
- `research/MINIBATCH_REPORT_VI.md`.

Đối với full-data:

- `research/artifacts/full_runs/runs.csv` có đúng 72 dòng.
- `research/results/full_verification.json` có `passed: true`.
- `research/results/full/summary.csv` và biểu đồ PNG/SVG.
- `research/FULL_DATA_REPORT_VI.md`.

Đây là điều kiện hoàn tất máy kiểm tra được; không dựa vào việc notebook chỉ
chạy hết cell mà không báo lỗi.
