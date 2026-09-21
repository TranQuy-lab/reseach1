# Hướng dẫn chạy E-GraphSAGE mini-batch

Tài liệu này là lệnh chạy lại pipeline đã dùng cho bốn bộ v2. Chạy từ thư
mục gốc repository. Dữ liệu lớn và artifact mô hình được bỏ khỏi Git.

## 1. Tạo môi trường

```bash
uv venv .venv-minibatch --python 3.12
uv pip sync research/requirements-minibatch-lock.txt \
  --python .venv-minibatch/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --find-links https://data.pyg.org/whl/torch-2.8.0+cpu.html \
  --index-strategy unsafe-best-match
export PYTHONPATH=src
```

Máy CPU hiện tại đủ cho tiền xử lý và benchmark pilot. GPU không bắt buộc.
Huấn luyện toàn bộ 75,99 triệu flow nên dùng server tối thiểu 64 GB RAM,
GPU 24 GB VRAM và 100 GB SSD trống, sau khi benchmark loader trên 250 nghìn
và một triệu flow.

## 2. Tạo split và pilot

Đầu vào là bốn Parquet trong `data/processed_four/`. Lệnh này chia nhóm flow
70/10/20 và tạo mẫu phân tầng theo protocol:

```bash
python -m nids_minibatch.prepare \
  --input data/processed_four \
  --output data/minibatch_splits \
  --manifest research/results/minibatch_prepare.json
```

Không fit scaler ở bước này. Scaler chỉ fit trên pilot train khi bắt đầu mỗi
dataset/task. IP và port được giữ để dựng endpoint nhưng không đi vào vector
đặc trưng.

## 3. Chọn cấu hình trên validation

```bash
python -m nids_minibatch.tune \
  --data data/minibatch_splits \
  --output research/artifacts/minibatch_tuning \
  --threads 4
```

Bước này chỉ dùng UNSW multiclass validation. Kết quả đã chọn batch 2048 và
fanout hai hop `[10, 5]` theo quy tắc ghi trước trong protocol.

## 4. Chạy 72 thí nghiệm

```bash
python -m nids_minibatch.training \
  --data data/minibatch_splits \
  --output research/artifacts/minibatch_runs \
  --datasets NF-UNSW-NB15-v2 NF-BoT-IoT-v2 NF-ToN-IoT-v2 NF-CSE-CIC-IDS2018-v2 \
  --tasks multiclass binary \
  --models edge_mlp sage sage_edge \
  --seeds 11 22 33 \
  --epochs 60 --patience 10 \
  --batch-size 2048 --fanout 10 5 --threads 4
```

Chương trình từ chối ghi đè thư mục output đã tồn tại. Muốn chạy lại, chọn
tên output mới để giữ nguyên bằng chứng của lần trước. Nếu tiến trình bị
gián đoạn, chạy lại đúng lệnh với `--resume`; chương trình kiểm tra provenance,
bỏ qua run đủ artifact và chỉ làm lại run dở dang.

## 5. Xác minh và tạo báo cáo

```bash
python research/validate_minibatch_results.py \
  --data data/minibatch_splits \
  --runs research/artifacts/minibatch_runs \
  --output research/results/minibatch_verification.json \
  --threads 4

python research/build_minibatch_report.py \
  --runs research/artifacts/minibatch_runs \
  --tuning research/artifacts/minibatch_tuning \
  --prepare research/results/minibatch_prepare.json \
  --verification research/results/minibatch_verification.json \
  --output research/results/minibatch \
  --report research/MINIBATCH_REPORT_VI.md
```

Validator kiểm tra đủ 72 tổ hợp, SHA-256 protocol, scaler giữa các seed,
ID dòng, tổng xác suất, checkpoint replay và metric tính lại. Chỉ dùng báo
cáo sau khi `passed` bằng `true`.

## 6. Kiểm thử mã

```bash
python -m pytest tests -q
```

Test quan trọng nhất so logits mini-batch lấy toàn bộ lân cận với full graph
trên đồ thị tổng hợp, đồng thời kiểm tra cả ba mô hình và hai task có loss,
gradient hữu hạn.
