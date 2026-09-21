# Protocol E-GraphSAGE mini-batch trên bốn bộ NetFlow v2

Trạng thái: chốt trước khi chạy kết quả mini-batch, ngày 2026-09-20.
Protocol pilot lịch sử trong `PROTOCOL.md` được giữ nguyên.

## Mục tiêu

Xác minh E-GraphSAGE có thể huấn luyện bằng mini-batch trên cả bốn bộ
NF-UNSW-NB15-v2, NF-BoT-IoT-v2, NF-ToN-IoT-v2 và
NF-CSE-CIC-IDS2018-v2, đồng thời đo ảnh hưởng của việc đưa trực tiếp đặc
trưng cạnh vào classifier. Đây là benchmark cục bộ trước khi chạy toàn
quy mô trên server, không phải kết quả cuối cùng của đề tài.

## Đơn vị dữ liệu và chia tập

- Một flow là một cạnh có hướng từ `(Dataset, src_ip, src_port)` đến
  `(Dataset, dst_ip, dst_port)`; thêm Dataset vào định danh để không nối
  nhầm các mạng thí nghiệm độc lập.
- Giữ 39 đặc trưng NetFlow làm edge feature. Không dùng IP, port, Dataset,
  `source_row_id`, `flow_group_id`, Label hoặc Attack làm feature số.
- Gán cả `flow_group_id` vào một split bằng hash có seed `20260920`:
  70% train, 10% validation, 20% test. Không thay seed sau khi xem metric.
- Các nhóm có nhãn mâu thuẫn vẫn ở cùng một split và được báo cáo riêng;
  không tự sửa hoặc xóa nhãn.
- Local benchmark lấy mẫu có seed `20260921` trong từng `(split, Attack)`:
  tối đa 20.000 train, 5.000 validation và 10.000 test cho mỗi lớp. Lớp
  dưới ngưỡng được giữ toàn bộ. Kết quả phản ánh benchmark phân tầng này,
  không phải phân bố tự nhiên toàn bộ dataset.
- Scaler và ánh xạ lớp chỉ fit trên train, rồi áp dụng cho validation/test.

## Mô hình và mini-batch

- `edge_mlp`: baseline không dùng topology.
- `sage`: E-GraphSAGE hai lớp, hidden 128, mean aggregation, dropout 0,2;
  decoder nhận embedding hai endpoint.
- `sage_edge`: cùng encoder, decoder nhận thêm edge feature đã chuẩn hóa.
- Message graph thêm cạnh ngược để truyền thông điệp; mỗi flow nguồn chỉ
  tạo một nhãn, một loss và một dự đoán.
- Mini-batch chỉ dùng cho train. Mỗi batch chọn seed edges và lấy lân cận
  hai hop. Cạnh seed vẫn thuộc message graph như công thức upstream; model
  không nhìn thấy Attack/Label của cạnh lân cận.
- Các cấu hình ứng viên được chọn với `sage_edge` bằng validation multiclass
  trên NF-UNSW-NB15-v2, seed mô hình 11, tối đa 30 epoch, patience 6:
  batch `{2048, 4096}` × fanout `{[10,5], [15,10]}`.
- Chọn validation macro-F1 cao nhất; nếu chênh dưới 0,005, ưu tiên cấu hình
  nhanh hơn, sau đó ít peak RSS hơn. Khóa cấu hình trước các run chính.

## Huấn luyện và đánh giá

- Hai task: multiclass dùng `Attack`; binary dùng `Label`.
- Run chính: bốn dataset × ba model × seed `{11,22,33}` cho mỗi task.
- Adam, learning rate 0,001; balanced cross-entropy tính từ train; tối đa
  60 epoch, early stopping patience 10 theo validation macro-F1.
- Validation và test pilot chạy full-graph riêng theo split, không lấy mẫu,
  để dự đoán xác định. Test không dùng chọn batch, fanout, epoch hoặc model.
- Báo accuracy, macro-F1, weighted-F1, per-class precision/recall/F1 và
  support; lưu confusion matrix, thời gian, peak RSS, cấu hình, checkpoint,
  scaler, dự đoán và checksum nguồn.
- So sánh ba seed bằng trung bình và độ lệch chuẩn. Không xem các seed là
  bốn dataset độc lập và không dùng test để diễn giải tuning.

## Cổng kiểm chứng trước run chính

1. Unit test bảo toàn IP/port, group split và scaler train-only.
2. Trên graph tổng hợp nhỏ, mini-batch `fanout=[-1,-1]` phải khớp logits
   full-batch trong sai số số học khi model ở evaluation mode.
3. Một epoch smoke test cho ba model và hai task phải có loss/gradient hữu hạn.
4. Artifact nạp lại phải tái tạo xác suất test trong dung sai đã ghi.
5. Nếu một batch vượt bộ nhớ, giảm batch size; không thay fanout dựa trên test.

## Phạm vi kết luận

Local benchmark dùng dữ liệu phân tầng có giới hạn mỗi lớp và không đại
diện hoàn toàn cho prevalence thực. Full-data chỉ chạy sau khi benchmark
1 epoch ở 250k và 1M flow xác nhận RAM/thời gian. Cross-dataset,
temporal/host holdout và triển khai online là các thí nghiệm riêng.
