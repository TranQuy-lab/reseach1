# Bằng chứng nền và giới hạn diễn giải

Đối chiếu ngày 2026-09-19. Đây là ghi chú nguồn chọn lọc, không phải tổng
quan hệ thống hay xác nhận tính mới toàn bộ lĩnh vực.

| Nguồn | Đã đọc gì | Điều hỗ trợ | Điều không suy ra |
|---|---|---|---|
| Lo et al., E-GraphSAGE, NOMS 2022, [arXiv](https://arxiv.org/abs/2103.16329) | PDF v8 do người dùng cung cấp, nhất là phương pháp trang 4–6 và bảng V trang 7; notebook NF-ToN-IoT đa lớp, commit e05eb74 | Edge classification; thông điệp sử dụng đặc trưng cạnh; test graph riêng; baseline có thể port | Không mặc định số liệu NF-ToN-IoT áp dụng cho NF-ToN-IoT-v2 |
| Venturi et al., Practical Evaluation of Graph Neural Networks in Network Intrusion Detection, ITASEC 2023, [bản ghi của cơ sở tác giả](https://iris.unimore.it/handle/11380/1322647) | Abstract và metadata, chưa đọc toàn văn ở vòng này | Có nghiên cứu trước về bất cập đánh giá ngoại tuyến khi áp dụng online và đánh đổi độ trễ/phát hiện | Không được gọi vấn đề offline/online là khám phá đầu tiên của dự án này |
| Guerra et al., Self-Supervised Learning of Graph Representations for Network Intrusion Detection, NeurIPS 2025, [proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/9ddb13ae9150f99298065d889f951014-Abstract-Conference.html) | Abstract chính thức, chưa đọc toàn văn ở vòng này | GraphIDS kết hợp biểu diễn đồ thị và masked autoencoder để phát hiện bất thường | Không dùng điểm số của họ làm so sánh trực tiếp với multiclass supervised pilot này |
| scikit-learn, [Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) | Tài liệu phương pháp tiền xử lý | Fit transformation từ train, áp dụng lại trên dữ liệu giữ lại | Không chứng minh một split cụ thể đã tránh mọi loại rò rỉ |

## Các sửa đổi diễn giải so với báo cáo cũ

- Bản PDF v8 là bài NOMS 2022; bảng đa lớp liên quan là bảng V trang 7.
  Không dùng tham chiếu TNSM/Table 13 trong báo cáo cũ cho chính bản PDF này.
- NF-ToN-IoT trong bài có weighted-F1 0,63 đa lớp; F1 1,00 là số nhị phân
  làm tròn trong bảng II. Macro-F1, weighted-F1 và binary-F1 không thay thế nhau.
- DoS và XSS F1=0 đã được bài báo công bố. Việc một số lớp thất bại không
  tự nó là một hiện tượng mới của repo.
- Đổi tập dữ liệu, feature set, topology, classifier và ngân sách huấn luyện
  cùng lúc không cho phép quy nguyên nhân cải thiện cho một thành phần.
- Code công bố random IP nguồn theo từng dòng, không bảo toàn ánh xạ host;
  notebook NF-ToN-IoT đa lớp dùng split 60/40 và không chuyển eval trước
  test. Những nhận xét này chỉ áp dụng cho code commit đã kiểm tra.
- Script biểu đồ cũ của repo fit mean/std trên validation, khác chuẩn hóa
  khi train. Không dùng các biểu đồ đó như đánh giá đã tái kiểm chứng.

## Cách chọn đóng góp tiếp theo

Trước tiên cần baseline có thể tái lập. Sau đó chọn một câu hỏi có đối chứng:
vai trò của đặc trưng cạnh trực tiếp; độ nhạy với graph context/lấy mẫu;
hoặc hiệu quả lớp hiếm dưới split nhóm/thời gian. Kết quả pilot và abstract
các bài trên chưa đủ để khẳng định tính mới của bất cứ hướng nào.

## Hỗ trợ quy trình

Bộ nckh được sử dụng để phân biệt quan sát với diễn giải, chốt protocol
trước điểm số, bảo toàn dữ liệu nguồn, và báo giới hạn. Nguồn skill:
Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026),
[Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065).
Không xem hướng dẫn của skill là kết quả nghiên cứu về NIDS.
