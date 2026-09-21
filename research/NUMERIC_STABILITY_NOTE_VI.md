# Biên bản ổn định số học ToN-IoT

Trong lần chạy chính ngày 2026-09-20/21, pipeline dừng tại run ToN-IoT đầu
tiên vì loss không hữu hạn. Dữ liệu không chứa NaN/Inf, nhưng hai trường rate
có giá trị hữu hạn cực lớn:

| Cột | Train max tuyệt đối | Validation max tuyệt đối | Test max tuyệt đối |
|---|---:|---:|---:|
| `SRC_TO_DST_SECOND_BYTES` | 3,36e165 | 1,12e44 | 1,12e261 |
| `DST_TO_SRC_SECOND_BYTES` | 5,38e66 | 1,87e257 | 2,06e223 |

Bình phương các giá trị này khi ước lượng phương sai làm tràn `float64`.
Cách phục hồi được chốt từ train: mọi cột có `max(abs(x_train)) > 1e20` dùng
`sign(x) * log1p(abs(x))`, sau đó StandardScaler vẫn chỉ fit trên train.
Danh sách cột được lưu trong `preprocessor.json` và validator dùng lại đúng
danh sách đó. Trên ToN-IoT, đúng hai cột trên được chọn; cả train, validation
và test sau biến đổi đều hữu hạn.

Mã tiếp tục kiểm tra provenance, đủ artifact và danh sách run trong
`runs.csv`. Nó bỏ qua 36 run UNSW/BoT-IoT đã hoàn tất, xóa duy nhất thư mục
ToN-IoT dở dang rồi chạy tiếp. Unit test mới tạo giá trị đến 1e200, kiểm tra
biến đổi hữu hạn và round-trip preprocessor. Toàn bộ 72 checkpoint sau cùng
đã được xác minh lại độc lập.
