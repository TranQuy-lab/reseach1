# reseach1 — E-GraphSAGE trên NF-UQ-NIDS-v2

Repository này chỉ chứa sản phẩm nghiên cứu đã chốt trong quá trình xây dựng
pipeline E-GraphSAGE: bốn dataset v2 đã tiền xử lý, protocol, mã nguồn,
notebook server, kiểm thử và kết quả kiểm chứng gọn. Các báo cáo “Đột phá”,
model, output và notebook lịch sử từ repository `reseach` cũ không thuộc
phạm vi này.

## Dữ liệu

| Dataset | Số dòng |
|---|---:|
| NF-UNSW-NB15-v2 | 2.390.275 |
| NF-BoT-IoT-v2 | 37.763.497 |
| NF-ToN-IoT-v2 | 16.940.496 |
| NF-CSE-CIC-IDS2018-v2 | 18.893.708 |
| **Tổng** | **75.987.976** |

Bốn file Parquet ở thư mục gốc dùng Git LFS. Dữ liệu giữ IP và port để định
danh node theo `(IP, port)`.

## Tài liệu chính

- `research/RESEARCH_PLAN_VI.md`: kế hoạch nghiên cứu.
- `research/PREPROCESSING_FOUR_VI.md`: quy trình tiền xử lý bốn dataset.
- `research/PROTOCOL_FULL_DATA_VI.md`: protocol kết quả chính trên toàn bộ dữ liệu.
- `research/server/README_SERVER_VI.md`: cài đặt và vận hành trên server.
- `research/EVIDENCE_NOTES_VI.md`: nguồn chứng cứ và giới hạn diễn giải.

## Chạy trên server

> **Tạm khóa run chính (kiểm toán 2026-09-21):** chưa chạy
> `11_FULL_TRAIN_72.ipynb`. Cổng 500 batch hiện tại ngoại suy sang 10/30/60
> epoch, trong khi một epoch full-data có số bước lớn hơn pilot rất nhiều.
> Phải cập nhật estimator và ngân sách huấn luyện theo số bước trước khi thuê
> GPU cho 72 run. Chi tiết và số đo nằm trong
> `research/PIPELINE_AUDIT_2026-09-21_VI.md`.

```bash
git lfs install --skip-repo
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
  https://github.com/TranQuy-lab/reseach1.git
cd reseach1
git lfs pull
mkdir -p data/processed_four
ln ./*.parquet data/processed_four/
ACCELERATOR=cu128 bash research/server/bootstrap_server.sh
```

Sau đó chạy lần lượt:

1. `notebooks/server/10_FULL_PREPARE_BENCHMARK.ipynb`
2. kiểm tra `research/results/full_benchmark_estimate.json`
3. chưa chạy `notebooks/server/11_FULL_TRAIN_72.ipynb` cho đến khi khóa tạm
   ở trên được gỡ bằng protocol và estimator phiên bản mới
4. `notebooks/server/12_FULL_VERIFY_REPORT.ipynb`

Pipeline dùng toàn bộ 75.987.976 flow cho kết quả chính. Mini-batch là cơ chế
nạp và tối ưu mô hình; nó không có nghĩa là lấy mẫu bỏ bớt dữ liệu.

## Kiểm thử

```bash
PYTHONPATH=src python -m pytest tests -q
```

Các thư mục dữ liệu trung gian, checkpoint và artifact huấn luyện không được
đưa lên Git. Báo cáo full-data chỉ được tạo sau khi server huấn luyện và bước
xác minh hoàn tất.
