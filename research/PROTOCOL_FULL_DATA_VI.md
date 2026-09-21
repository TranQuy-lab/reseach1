# Protocol full-data E-GraphSAGE trên bốn bộ NF-UQ-NIDS-v2

Trạng thái: **tạm khóa khởi chạy**, kiểm toán ngày 2026-09-21. Ma trận thí
nghiệm và split vẫn được giữ, nhưng ngân sách 60 epoch và cổng thời gian hiện
tại phải được thay bằng ngân sách theo số bước trước khi chạy full-data. Xem
`PIPELINE_AUDIT_2026-09-21_VI.md`.

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
- Cả ba mô hình đều đi qua mọi cạnh train ở mỗi epoch. E-GraphSAGE lấy lân cận
  hai hop cho từng nhóm seed edge để graph lớn hơn VRAM vẫn chạy được. Đây là
  huấn luyện full-data theo batch, không phải lấy mẫu bỏ bớt tập dữ liệu.
- Cấu hình khóa trước run chính: hidden 128, dropout 0,2, Adam learning rate
  0,001, batch 4096, fanout `[15, 10]`, BF16 autocast trên CUDA.
- Cấu hình cũ validation mỗi 3 epoch, tối đa 60 epoch và patience 10 **không
  còn được phép dùng cho run chính**. Một epoch full-data có 409–6.455 bước
  tùy dataset, nên đơn vị epoch không tương đương pilot. Protocol sửa đổi phải
  khóa số bước tối đa, nhịp validation theo bước và điều kiện đã đi qua toàn
  bộ cạnh train ít nhất một lần.

## Ma trận thí nghiệm và đánh giá

- Hai task: multiclass dùng `Attack`; binary dùng `Label`.
- Bốn dataset × hai task × ba model × ba seed `{11,22,33}` = 72 run.
- Loss là balanced cross-entropy với trọng số chỉ tính từ train.
- Chọn checkpoint theo validation macro-F1. Test chỉ được tính sau khi đã chọn
  checkpoint và không dùng để đổi cấu hình.
- Báo accuracy, macro-F1, weighted-F1, precision/recall/F1 từng lớp, support và
  confusion matrix. Tổng hợp ba seed bằng trung bình và độ lệch chuẩn.
- Mỗi run giữ checkpoint, scaler, cấu hình, lịch sử, metric full-test và tối đa
  100.000 dòng dự đoán kiểm toán được chọn theo vị trí cách đều. Giới hạn này
  chỉ giảm dung lượng artifact; metric luôn được tính trên toàn bộ test split.

## Cổng kiểm chứng

1. Tổng số dòng bốn nguồn phải là 75.987.976 và checksum phải khớp manifest.
2. Không có `flow_group_id` xuất hiện ở hơn một split; số dòng phải được bảo toàn.
3. Preflight full-data một epoch trên NF-UNSW-NB15-v2 phải có loss hữu hạn,
   checkpoint nạp lại được và CUDA không OOM.
4. Mỗi run phải lưu đủ artifact; checkpoint replay phải khớp mẫu dự đoán đã lưu.
5. File `runs.csv` cuối cùng phải có đúng 72 tổ hợp duy nhất.
6. Cổng tài nguyên phải dùng median của nhiều cửa sổ sau warm-up, tách thời
   gian nạp dữ liệu, dựng graph, train và validation; kết quả từ estimator cũ
   không được mở khóa notebook 11.

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
