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

> **Run chính vẫn khóa cho tới khi benchmark server đạt:** pipeline hiện dùng
> ngân sách hai lượt qua train/dataset, tối thiểu 1.500 optimizer step/run và
> chỉ cho checkpoint hợp lệ sau một lượt đầy đủ. Estimator đo nhiều cửa sổ
> sau warm-up và tự từ chối khởi chạy nếu ETA/RAM/VRAM/ổ đĩa không đạt.

Ma trận vẫn là 72 run. Với split hiện tại, chính sách hai lượt dự kiến dùng
479.916 optimizer step, giảm 66,7% so với 20.000 step cố định cho từng run.

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
2. kiểm tra `research/results/full_benchmark_estimate.json`; chỉ tiếp tục khi
   `safe_to_launch_72` là `true` và ngân sách step khớp protocol
3. `notebooks/server/11_FULL_TRAIN_72.ipynb`
4. `notebooks/server/12_FULL_VERIFY_REPORT.ipynb`

Thiết kế full-data sẽ dùng toàn bộ 75.987.976 flow; kết quả full-data chưa được
chạy. Mini-batch là cơ chế nạp và tối ưu mô hình, không có nghĩa là lấy mẫu bỏ
bớt dữ liệu.

## Kiểm thử

```bash
PYTHONPATH=src .venv-server/bin/python -m pytest tests -q
```

Không cài trực tiếp `requirements-minibatch*.txt`: wheel Torch/PyG nằm trên
index riêng. Dùng duy nhất `research/server/bootstrap_server.sh` để tránh cài
sai bản CPU/CUDA. Checklist trước run nằm trong
`research/server/SERVER_PREFLIGHT_CHECKLIST_VI.md`.

Các thư mục dữ liệu trung gian, checkpoint và artifact huấn luyện không được
đưa lên Git. Báo cáo full-data chỉ được tạo sau khi server huấn luyện và bước
xác minh hoàn tất.
