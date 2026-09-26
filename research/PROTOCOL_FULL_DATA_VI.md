# Protocol full-data E-GraphSAGE trên bốn bộ NF-UQ-NIDS-v2

Trạng thái: protocol step-based đã sửa sau kiểm toán 2026-09-21; 72 run vẫn
khóa cho đến khi benchmark trên đúng server sinh `safe_to_launch_72=true`.
Xem `PIPELINE_AUDIT_2026-09-21_VI.md`.

## Mục tiêu và dữ liệu

- Dùng toàn bộ 75.987.976 flow thuộc NF-UNSW-NB15-v2, NF-BoT-IoT-v2,
  NF-ToN-IoT-v2 và NF-CSE-CIC-IDS2018-v2; không lấy mẫu theo lớp.
- Một flow là một cạnh có hướng từ `(Dataset, src_ip, src_port)` đến
  `(Dataset, dst_ip, dst_port)`. IP và port chỉ định nghĩa node, không được đưa
  trực tiếp vào vector đặc trưng cạnh.
- Giữ 39 đặc trưng NetFlow. Giữ `source_row_id` và `flow_group_id` để truy vết.
- Gán toàn bộ một `flow_group_id` vào train/validation/test bằng
  `hash(flow_group_id, 20260920) % 10`: 70% train, 10% validation, 20% test.
  Không đổi seed hoặc split sau khi xem metric.
- Scaler, phép ổn định trường có độ lớn cực trị và ánh xạ lớp chỉ fit trên
  train rồi áp dụng nguyên trạng cho validation/test.

## Mô hình và cách đưa full-data qua GPU

- `edge_mlp`: baseline chỉ dùng 39 edge feature.
- `sage`: E-GraphSAGE hai lớp mean aggregation; classifier dùng embedding hai
  endpoint.
- `sage_edge`: cùng encoder và thêm trực tiếp edge feature vào classifier.
- Trước khi checkpoint đầu tiên được phép lưu, cả ba mô hình phải đi qua mọi
  cạnh train ít nhất một lượt. E-GraphSAGE lấy lân cận
  hai hop cho từng nhóm seed edge để graph lớn hơn VRAM vẫn chạy được. Đây là
  huấn luyện full-data theo batch, không phải lấy mẫu bỏ bớt tập dữ liệu.
- Cấu hình khóa trước run chính: hidden 128, dropout 0,2, Adam learning rate
  0,001, batch 4096, fanout `[15, 10]`. Edge feature của graph full-data được
  lưu FP16 để giảm RAM/VRAM; forward/loss dùng BF16 autocast trên CUDA và
  logits được đổi về FP32 trước khi tính xác suất/validation loss.
- Ngân sách khóa trước run chính là hai lượt qua train cho từng dataset, với
  tối thiểu 1.500 optimizer step/run. Cụ thể,
  `max_steps=max(1500, 2*ceil(train_rows/4096))`. Checkpoint đầu tiên chỉ hợp
  lệ sau một lượt đầy đủ. Validation chạy ở cuối lượt đầu, sau đó theo khoảng
  `max(500, ceil(steps_per_pass/4))` và tại step cuối. Cách này giữ đủ 72 run
  nhưng không buộc dataset nhỏ dùng cùng 20.000 step như dataset lớn.
- Data loader dùng bốn worker, pinned host memory, prefetch hai batch/worker và
  chuyển bất đồng bộ khi chạy CUDA. Giá trị worker phải giống giữa benchmark
  và run chính; có thể đổi bằng tham số server nhưng phải benchmark lại từ đầu.
- Benchmark phải nhận ngân sách $6 và giá thuê GPU/giờ thực tế, nhân ETA với hệ
  số dự phòng 1,35 rồi chặn khởi chạy nếu `planning_cost_usd > 6`. Khi chạy trên
  máy sở hữu sẵn có thể đặt giá 0; báo cáo vẫn xuất mức giá/giờ tối đa để không
  vượt ngân sách nếu chuyển sang máy thuê.
- Binary và multiclass dùng cùng split, feature scaler, topology và tensor đặc
  trưng; pipeline dựng chúng một lần/dataset rồi chỉ thay tensor nhãn. Mỗi task
  vẫn có preprocessor, class mapping, checkpoint và metric riêng.

## Ma trận thí nghiệm và đánh giá

- Hai task: multiclass dùng `Attack`; binary dùng `Label`.
- Bốn dataset × hai task × ba model × ba seed `{11,22,33}` = 72 run.
- Với split đã khóa và batch 4096, ngân sách dự kiến là UNSW 1.500,
  BoT-IoT 12.910, ToN-IoT 5.792 và CSE-CIC 6.460 step/run. Tổng ma trận là
  479.916 optimizer step, giảm 66,7% so với phương án 20.000 step × 72 nhưng
  vẫn giữ nguyên đủ 72 tổ hợp.
- Loss là balanced cross-entropy với trọng số chỉ tính từ train.
- Chọn checkpoint theo validation macro-F1. Test chỉ được tính sau khi đã chọn
  checkpoint và không dùng để đổi cấu hình.
- Báo accuracy, macro-F1, weighted-F1, precision/recall/F1 từng lớp, support và
  confusion matrix. Tổng hợp ba seed bằng trung bình và độ lệch chuẩn.
- Mỗi run giữ checkpoint, scaler, cấu hình, lịch sử, metric full-test và tối đa
  100.000 dòng dự đoán kiểm toán được chọn theo vị trí cách đều. Giới hạn này
  chỉ giảm dung lượng artifact; metric luôn được tính trên toàn bộ test split.
- Train dùng directional neighbor sampling `[15,10]`. Validation/test truyền
  thông điệp trên **toàn bộ cạnh của graph** theo từng lớp, nhưng cộng dồn theo
  chunk cạnh (`EVALUATION_CHUNK_EDGES`) để tensor message bị chặn theo chunk
  thay vì theo cả split. Đây là phép tính tương đương, không phải xấp xỉ:
  kiểm chứng trên `tests/test_chunked_evaluation.py` cho chênh lệch logit so với
  forward toàn graph ≤ 1,2e-7 (float32) và không đổi theo kích thước chunk. Lý do
  bắt buộc: forward trọn split làm NF-BoT-IoT-v2 xin một buffer 14,41 GiB và OOM
  trên card 24 GB. Đánh giá sampled-neighbor là thí nghiệm độ nhạy riêng, không
  được trộn vào 72 run chính.
- `edge_mlp` là baseline nội bộ của ma trận chính, không đại diện cho mọi mô
  hình bảng. Random Forest/GBDT và baseline cùng capacity phải chạy ở protocol
  xác nhận riêng trên đúng split trước khi tuyên bố GNN hơn baseline nói chung.

## Cổng kiểm chứng

1. Tổng số dòng bốn nguồn phải là 75.987.976 và checksum phải khớp manifest.
2. Không có `flow_group_id` xuất hiện ở hơn một split; số dòng phải được bảo toàn.
   Báo thêm nhóm nhãn mâu thuẫn theo split và tỉ lệ IP validation/test đã xuất
   hiện trong train để giới hạn diễn giải về host chưa thấy.
3. Benchmark giới hạn 500 batch cho từng dataset/model phải có loss hữu hạn,
   checkpoint nạp lại được và CUDA không OOM; estimator tính ngân sách riêng
   cho từng dataset theo đúng công thức hai lượt train.
4. Mỗi run phải lưu đủ artifact; checkpoint replay phải khớp mẫu dự đoán đã lưu.
5. File `runs.csv` cuối cùng phải có đúng 72 tổ hợp duy nhất.
6. Cổng tài nguyên phải có ít nhất ba cửa sổ 50 batch sau 50 batch warm-up,
   báo median/MAD/p90, tách nạp dữ liệu, dựng graph, train, validation và đánh
   giá cuối. ETA dùng p90 nhân hệ số an toàn 1,35 và không được vượt 14 ngày.
7. File cổng phải khóa đúng `train_passes=2`, `min_train_steps=1500`,
   `evals_per_pass=4`, khoảng validation tối thiểu 500 step và cùng số worker;
   notebook 11 từ chối ngân sách hoặc cấu hình thực thi khác.
8. Preflight phải xác minh Git commit sạch, SHA-256 và số dòng của bốn Parquet,
   đúng phiên bản thư viện, RAM ≥120 GiB, CUDA/BF16 và VRAM ≥24 GiB.
9. Provenance của run phải chứa Git commit, SHA-256 protocol và SHA-256 từng
   file mã thí nghiệm; validator từ chối nếu mã đã thay đổi.
10. Validator ghi SHA-256 của checkpoint, preprocessor, cấu hình, history,
    metric và mẫu prediction cho từng run để đóng gói artifact về sau.

## Phạm vi kết luận

Kết quả cho phép so sánh mô hình trên toàn bộ bốn bộ dữ liệu với split nhóm cố
định. Thiết kế chưa chứng minh khả năng tổng quát sang một dataset chưa thấy,
khả năng phát hiện zero-day hoặc hiệu năng thời gian thực; các câu hỏi đó cần
cross-dataset, temporal/host holdout và benchmark triển khai riêng.

## Công cụ hỗ trợ phương pháp

Quy trình kiểm toán thiết kế và cổng tài nguyên sử dụng Scientific Agent
Skills: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
*Scientific Agent Skills: A Library of Procedural Knowledge for Research
Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
