# Protocol pilot E-GraphSAGE — 2026-09-19

Được viết trước khi huấn luyện hoặc xem điểm số pilot mới. Mục đích là
khôi phục nền nghiên cứu và kiểm chứng tính thực thi, chưa xác nhận tính mới.

## Câu hỏi và phạm vi

Trên một mẫu NF-ToN-IoT-v2, quan hệ đồ thị có giúp phân loại đa lớp hơn
đặc trưng từng luồng? Nhánh đưa trực tiếp đặc trưng cạnh vào bộ phân loại
đóng góp bao nhiêu? Đơn vị dự đoán là flow gốc, không phải cạnh ngược.
Đây là phân tích ngoại tuyến; không tuyên bố hiệu quả thời gian thực,
zero-day hoặc tổng quát liên mạng.

## Dữ liệu và chia tập

- Chỉ đọc bản sao dữ liệu đã qua tiền xử lý trong data/dl_processed.
  Không có CSV nguyên bản, nên không thể kiểm chứng các biến đổi trước đó.
- Kiểm toán toàn bộ train và validation cũ: schema, nhãn, NaN/Inf,
  phân bố lớp, dấu vân tay trùng bản ghi trong/giữa các tập.
- Mẫu mới lấy chỉ từ train cũ; validation cũ không dùng chọn mô hình.
  Toàn bộ kho dữ liệu đã từng được dùng trong dự án, do đó test mới chỉ
  là test nội bộ giữ lại cho pilot, không phải bộ đánh giá ngoài độc lập.
- Dấu vân tay gồm 39 đặc trưng và hai endpoint, không chứa nhãn.
  Nhóm có nhãn mâu thuẫn bị loại khỏi mẫu và được đếm. Chọn một bản ghi
  đại diện cho mỗi nhóm còn lại. Đây là thí nghiệm trên bản ghi duy nhất,
  không đại diện cho phân phối tần suất flow ban đầu.
- Lấy tối đa 50.000 nhóm bằng thứ tự băm có seed 20260919; quét toàn bộ
  file train, không lấy riêng các dòng đầu. Không dùng kết quả mô hình để
  thay đổi mẫu. Nếu lớp quá ít để chia, báo rõ và dừng thay vì giấu lớp.
- Chia có phân tầng theo Attack: 70/10/20, seed 20260919. Lưu ID,
  nguồn file/dòng, dấu vân tay và split. Không trùng nhóm giữa các split.
- Chỉ fit StandardScaler trên train. Numeric encoding giữ như repo v2;
  không dùng target encoding ở pilot. Không đưa Label/Attack/Dataset vào X.
  Từ chối NaN/Inf trong pilot thay vì thêm chính sách điền thiếu ngầm.
- Đỉnh IP:port, bảo toàn endpoint; cạnh song song được giữ. Mỗi split
  có đồ thị riêng; graph test có thể dùng các đặc trưng test để truyền
  thông điệp nhưng tuyệt đối không dùng nhãn test.

## Thiết kế đối chứng cố định

1. edge_mlp: chỉ đặc trưng cạnh; MLP hai tầng 128, dropout 0,2.
2. sage: port PyTorch của phép tính SAGELayer gốc; 2 tầng, hidden 128,
   thông điệp tuyến tính, mean, ReLU khi cập nhật đỉnh; linear([h_u,h_v]).
3. sage_edge: cùng encoder như sage, linear([h_u,h_v,e]). So với sage
   chỉ thêm đường đặc trưng cạnh, không đồng thời đổi sang MLP sâu hơn.
4. random_forest: 100 cây, class_weight=balanced, min_samples_leaf=2.

GNN dùng cạnh hai chiều để truyền thông điệp, nhưng loss và đánh giá chỉ
trên từng flow gốc một lần. Không nhân đôi self-loop. Khởi tạo đỉnh bằng
vector một có kích thước bằng số đặc trưng. Tích lũy FP32. Bảo toàn IP
thay vì random IP từng dòng: đây là baseline thích nghi v2 có sửa protocol,
KHÔNG phải tái lập chính xác số liệu bài báo hoặc toàn bộ notebook gốc.

Ba seed mô hình: 11, 22, 33 trên cùng split. Neural models: Adam lr=0,001,
cross-entropy với trọng số N/(C*n_c) chỉ tính từ train, tối đa 120 epoch,
early stopping patience=20 theo macro-F1 validation, không tune trên test.
Full batch cho pilot để kiểm soát graph context; 1 epoch = 1 optimizer step.
Ngân sách này không khẳng định mọi mô hình đã hội tụ hoặc tối ưu ngang nhau.
Không dùng test để chọn "mô hình thắng" rồi báo như xác nhận ngoài mẫu.

## Đánh giá và tiêu chí hoàn thành

Chính: macro-F1 cố định trên toàn bộ lớp train. Phụ: weighted-F1, accuracy,
precision/recall/F1/support từng lớp, confusion matrix, thời gian, tham số,
số đỉnh/cạnh. Báo trung bình/SD giữa 3 seed; đây là độ biến động khởi tạo,
không phải khoảng tin cậy tổng quát hóa. Lớp ít mẫu được ghi số hỗ trợ.

Kiểm thử: scaler không học val/test; nhóm trùng không lọt qua split;
nhãn lạ gây lỗi; aggregation khớp phép tính thủ công; hướng cạnh/reverse
không tăng số dự đoán; gradient hữu hạn; checkpoint nạp lại cho cùng logits;
predict không cần nhãn và không phụ thuộc thứ tự tên đỉnh.

Hoàn thành khi kiểm toán, tests, 12 lượt pilot, artifacts tái lập và báo cáo
tiếng Việt tồn tại. Chưa gọi là nghiên cứu hoàn chỉnh toàn quy mô; nghiên
cứu tiếp cần raw data, đánh giá nhóm/thời gian phù hợp, nhiều tập dữ liệu,
ngân sách hội tụ, kiểm tra encoding và lấy mẫu lân cận.

## Nguồn và hỗ trợ phương pháp

- Lo et al. (2022), E-GraphSAGE, https://arxiv.org/abs/2103.16329;
  repo waimorris/E-GraphSAGE commit e05eb74e586891ec7821a363fc42d6944615b945.
- Repo người dùng: TranQuy-lab/reseach, commit 710f3624aba850fbe91a13cc4777f3254a038072.
- Skill nckh đã sử dụng: scientific-critical-thinking, experimental-design,
  scikit-learn, exploratory-data-analysis (hướng dẫn; Parquet đọc bằng PyArrow).
- Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
  Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents.
  https://doi.org/10.48550/arXiv.2609.00065. Hỗ trợ quy trình, không phải
  bằng chứng thực nghiệm cho E-GraphSAGE.
