# Báo cáo full-data E-GraphSAGE trên bốn bộ v2

Pipeline đã xử lý toàn bộ 75.987.976 flow, gồm bốn dataset, hai task, ba mô
hình và 5 seed, tổng cộng 120 run. Full-data ở đây có
nghĩa là mọi flow trong split đều được sử dụng; huấn luyện vẫn dùng neighbor
mini-batch và đánh giá dùng layer-wise full-graph inference.

- Tổng thời gian fit và đánh giá cộng dồn: 8.69 giờ.
- Một verifier tách khỏi training đã nạp 120 checkpoint,
  chạy full-test inference, tính lại metric trên toàn bộ test và đối chiếu xác
  suất trên các dòng dự đoán được lưu.
- Benchmark trước khi chạy dùng hệ số an toàn 1.35.
- Bảng tổng hợp: `research/results/full_5seed/summary.csv`.
- Chỉ số từng lớp: `research/results/full_5seed/per_class.csv`.
- Chênh lệch ghép cặp theo cùng seed: `research/results/full_5seed/paired_seed_deltas.csv`.
- Cảnh báo lớp có support dưới 1.000: `research/results/full_5seed/rare_class_warning.csv`.
- Biểu đồ: `research/results/full_5seed/macro_f1_full.png` và `.svg`.

Split được kiểm tra không trùng `flow_group_id`. Tỉ lệ IP validation/test đã
thấy trong train được ghi trong `prepare_manifest.json`; đây là phép đo nguy cơ
rò rỉ danh tính host để giới hạn diễn giải, không phải lỗi chia nhóm flow.
Số nhóm có nhãn mâu thuẫn và số dòng liên quan cũng được báo riêng theo split.

Kết luận khoa học phải dựa trên macro-F1, chỉ số từng lớp và độ lệch chuẩn giữa
5 seed. Seed là lặp tính toán trên cùng dataset, không phải mạng độc
lập. Bảng paired delta mang tính mô tả; với 5 cặp, không dùng khoảng
tin cậy chuẩn hoặc p-value như bằng chứng duy nhất. Các lớp trong
`rare_class_warning.csv` chỉ được diễn giải như kết quả khám phá.
