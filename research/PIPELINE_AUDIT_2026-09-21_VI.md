# Kiểm toán pipeline full-data ngày 2026-09-21

Tài liệu này ghi các phát hiện phải xử lý trước khi chạy 72 thí nghiệm trên
server. Đây là kiểm toán thiết kế và tài nguyên; chưa phải kết quả full-data.

## Kết luận khởi chạy

Không chạy `notebooks/server/11_FULL_TRAIN_72.ipynb` với phiên bản hiện tại.
Notebook chuẩn bị 10 chỉ được dùng để kiểm tra dữ liệu và tài nguyên có giới
hạn; giá trị `safe_to_launch_72` do estimator cũ sinh ra chưa đủ để mở khóa
run chính.

## Ngân sách cập nhật mô hình

Train split có tổng cộng 53.199.630 flow. Với batch 4.096 đã khóa trong
protocol, một lượt đi qua dữ liệu cần:

| Dataset | Flow train | Batch/lượt |
|---|---:|---:|
| NF-UNSW-NB15-v2 | 1.673.104 | 409 |
| NF-BoT-IoT-v2 | 26.438.303 | 6.455 |
| NF-ToN-IoT-v2 | 11.861.039 | 2.896 |
| NF-CSE-CIC-IDS2018-v2 | 13.227.184 | 3.230 |
| **Tổng cho một model/task/seed** | **53.199.630** | **12.990** |

Qua đủ 72 run, một epoch tương ứng 233.820 bước tối ưu; 60 epoch tương ứng
khoảng 14.029.200 bước. Cổng hiện tại chạy tối đa 500 batch cho mỗi cặp
dataset/model rồi lấy thời gian của lần đo duy nhất để ngoại suy cả epoch và
các kịch bản 10/30/60 epoch. Cách này chưa loại warm-up và chưa biểu diễn độ
dao động I/O, neighbor sampling hoặc validation.

Estimator thay thế phải dùng nhiều cửa sổ sau warm-up, báo median cùng độ
phân tán, và tách thời gian nạp dữ liệu, dựng graph, train, validation cùng
đánh giá cuối. Ngân sách run chính phải khóa theo số bước và nhịp validation
theo bước, đồng thời bảo đảm checkpoint hợp lệ đã đi qua toàn bộ train split
ít nhất một lần.

## Seed 22 của BoT-IoT nhị phân

Edge MLP seed 22 đạt validation macro-F1 0,923887 và test macro-F1 0,923671;
best epoch 7 và dừng ở epoch 17. Hai seed 11 và 33 đạt test macro-F1 lần lượt
0,993271 và 0,993207. Vì validation và test cùng thấp, đây không phải bằng
chứng cho một test split đặc biệt khó. Lịch sử cho thấy loss tiếp tục giảm
nhưng macro-F1 dao động mạnh, phù hợp với bất ổn tối ưu hoặc quyết định dưới
balanced cross-entropy. Không được loại seed 22 hay chỉnh riêng cấu hình dựa
trên test; mọi biện pháp ổn định phải áp dụng trước và đồng nhất cho mọi seed
và mô hình.

Báo cáo dùng độ lệch chuẩn mẫu (`pandas.std`, `ddof=1`). Giá trị 0,9700 ±
0,0402 của ba seed vì vậy được tính đúng.

## Độ trùng IP giữa train và holdout

Audit trên full split đo tỷ lệ flow mà cả IP nguồn và IP đích đã xuất hiện
trong train:

| Dataset | Validation | Test |
|---|---:|---:|
| NF-UNSW-NB15-v2 | 100,0000% | 99,9998% |
| NF-BoT-IoT-v2 | 99,9997% | 99,9997% |
| NF-ToN-IoT-v2 | 99,8517% | 99,8510% |
| NF-CSE-CIC-IDS2018-v2 | 99,2176% | 99,2148% |

Con số khoảng 25% trước đây dùng node `(IP, port)` và vì vậy che mức trùng
host do cổng client thay đổi. Split `flow_group_id` hiện tại vẫn hợp lệ cho
câu hỏi phân loại flow trong cùng môi trường và không tự nó chứng minh rò rỉ
nhãn. Tuy nhiên, kết quả từ split này không được diễn giải thành khả năng tổng
quát sang host chưa thấy. Phải bổ sung thí nghiệm host-holdout theo IP nếu bài
báo muốn đưa ra kết luận đó; trước khi chạy cần kiểm tra từng lớp còn hiện
diện trong train/validation/test vì UNSW và BoT-IoT có rất ít IP duy nhất.

## Đối chứng cần bổ sung

- Train graph thật, rewire graph khi test: đo độ phụ thuộc topology lúc suy
  luận.
- Train và test đều rewire, huấn luyện lại từ đầu với cùng seed/ngân sách:
  kiểm tra mô hình có cần topology thật hay không.
- GBDT dùng đặc trưng tổng hợp 1-hop và 2-hop từ đúng graph của từng split,
  tuyệt đối không dùng nhãn holdout để tạo đặc trưng. Cách này cho baseline
  cùng quyền truy cập topology với GNN hai lớp.

Các đối chứng là nhánh xác nhận sau khi pipeline chính ổn định. Chúng không
được dùng để thay đổi split hoặc cấu hình sau khi đã xem test chính.
