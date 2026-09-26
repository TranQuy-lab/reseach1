# Báo cáo full-data E-GraphSAGE trên bốn bộ v2

Pipeline đã xử lý toàn bộ 75.987.976 flow, gồm bốn dataset, hai task, ba mô
hình và ba seed, tổng cộng 72 run. Không có bước lấy mẫu bỏ bớt flow train,
validation hoặc test. Batch chỉ là cơ chế đưa toàn bộ cạnh qua GPU.

- Tổng thời gian fit và đánh giá cộng dồn: 5.45 giờ.
- Validator độc lập đã nạp và kiểm tra 72 checkpoint.
- Benchmark trước khi chạy dùng hệ số an toàn 1.35.
- Bảng tổng hợp: `research/results/full/summary.csv`.
- Chỉ số từng lớp: `research/results/full/per_class.csv`.
- Chênh lệch ghép cặp theo cùng seed: `paired_seed_deltas.csv`.
- Cảnh báo lớp có support dưới 1.000: `rare_class_warning.csv`.
- Biểu đồ: `research/results/full/macro_f1_full.png` và `.svg`.

Split được kiểm tra không trùng `flow_group_id`. Tỉ lệ IP validation/test đã
thấy trong train được ghi trong `prepare_manifest.json`; đây là phép đo nguy cơ
rò rỉ danh tính host để giới hạn diễn giải, không phải lỗi chia nhóm flow.
Số nhóm có nhãn mâu thuẫn và số dòng liên quan cũng được báo riêng theo split.

Kết luận khoa học phải dựa trên macro-F1, chỉ số từng lớp và độ lệch chuẩn giữa
ba seed. Bảng paired delta chỉ mang tính mô tả; ba seed không đủ cho khoảng tin
cậy bootstrap ổn định. Các lớp trong `rare_class_warning.csv` không được dùng
để đưa ra kết luận chắc chắn.
