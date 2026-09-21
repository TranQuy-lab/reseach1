# Tài liệu chính — E-GraphSAGE trên bốn bộ NetFlow v2

Cập nhật 2026-09-20 theo xác nhận của chủ đề tài. Tài liệu này quyết định
phạm vi nghiên cứu tiếp theo; `PROTOCOL.md` và `REPORT_VI.md` lưu lại pilot
đã hoàn thành, không được hiểu là kết quả trên cả bốn bộ dữ liệu.
Áp dụng bộ skill nckh và nguyên tắc thiết kế thí nghiệm: tách quyết định
trước thí nghiệm khỏi kết luận sau khi xem kết quả.

> **Cập nhật hoàn tất 2026-09-21:** benchmark mini-batch cục bộ đã chạy đủ
> 72/72 tổ hợp trên bốn dataset, hai task, ba mô hình và ba seed. Cả 72
> checkpoint đã được nạp lại, xác suất tái tạo sai lệch tối đa 0 và sai lệch
> metric tối đa 1,11e-16. Xem [báo cáo](MINIBATCH_REPORT_VI.md),
> [protocol khóa trước](PROTOCOL_MINIBATCH_VI.md),
> [runbook](MINIBATCH_RUNBOOK_VI.md) và
> [biên bản ổn định số học](NUMERIC_STABILITY_NOTE_VI.md). Các đoạn trạng
> thái cũ phía dưới được giữ làm lịch sử và không còn là trạng thái hiện tại.

## 1. Phạm vi đã chốt

Chạy và báo cáo riêng trên **NF-UNSW-NB15-v2, NF-BoT-IoT-v2,
NF-ToN-IoT-v2, NF-CSE-CIC-IDS2018-v2** — bốn thành phần của NF-UQ-NIDS-v2.
Nguồn chính thức và cửa tải: [UQ — NetFlow V2 Datasets](https://www.cyber.uq.edu.au/node/824).
Không chọn bản v1, v3 hoặc bộ CICFlowMeter thay thế.

Đây là đánh giá E-GraphSAGE được thích nghi cho v2; không gọi là tái lập
nguyên trạng bốn dataset của bài E-GraphSAGE. Tham chiếu phương pháp:
[bài báo](https://arxiv.org/abs/2103.16329),
[code tác giả](https://github.com/waimorris/E-GraphSAGE).

Lựa chọn tải đã cập nhật: dùng file gộp Kaggle theo yêu cầu người dùng,
sau đó kiểm toán và tách bốn bộ để chạy riêng. Nếu dùng file
gộp NF-UQ-NIDS-v2, phải tách theo cột Dataset, ghi lại bảng ánh xạ nhãn;
không coi nhãn của bản gộp và bản riêng mặc nhiên giống nhau. Không tải
cả file gộp lẫn bốn file riêng nếu không có nhu cầu đối chiếu cụ thể.

## 2. Trạng thái hiện tại

Ngày 2026-09-20: người dùng yêu cầu tự động tải từ Kaggle. Đã tải và kiểm tra bản CSV
gộp từ `aryashah2k/nfuqnidsv2-network-intrusion-detection-dataset`; API công
khai liệt kê CSV 13.729.330.230 byte, gói tải 2.185.336.021 byte. Giữ nguồn
trong `data/raw/NF-UQ-NIDS-v2/`; trạng thái ở `download_status.json`. Chỉ
coi tải hoàn tất khi trạng thái là `complete` và có manifest SHA-256.
Không dùng hoặc lưu API key vì nguồn cho phép tải công khai.

Kết quả tải: CSV 13.729.330.230 byte, kiểm tra CRC gói nén đạt, đã lưu
SHA-256 trong [manifest](results/kaggle_download_manifest.json). Quét CSV
xác nhận **75.987.976 dòng**, đủ bốn bộ và số dòng khớp nguồn UQ:

| Dataset | Số dòng đã đếm |
|---|---:|
| NF-UNSW-NB15-v2 | 2.390.275 |
| NF-BoT-IoT-v2 | 37.763.497 |
| NF-ToN-IoT-v2 | 16.940.496 |
| NF-CSE-CIC-IDS2018-v2 | 18.893.708 |

Chi tiết: [biên bản kiểm tra](results/kaggle_data_inventory.json).
Biên bản tải chỉ kiểm tra định dạng/số dòng. Sau đó đã xuất bốn file
Parquet riêng, giữ IP/port và ID nguồn tại `data/processed_four/`; kiểm tra
thiếu, NaN/Inf, IP/port và quan hệ nhãn không phát hiện dòng lỗi.
Giữ flow lặp với mã nhóm để chia tập, chưa chuẩn hóa hoặc cân bằng lớp.
Kiểm tra độc lập đã đạt; phát hiện 5.332 nhóm đầu vào có nhãn tấn công
mâu thuẫn, giữ nguyên và ghi nhận để xử lý theo protocol chia tập.
Chi tiết và giới hạn: [tiền xử lý bốn bộ](PREPROCESSING_FOUR_VI.md).
Ổ hiện tại còn khoảng 33 GiB sau tải và giải nén.

Repo có Parquet đã xử lý từ
NF-ToN-IoT-v2, đã dùng cho pilot 50.000 flow. Các kết quả đó chỉ chứng minh
quy trình thử nghiệm chạy được trên đầu vào ấy; chưa kiểm chứng raw CSV.

Nhánh `nids_research.prepare` hiện chỉ hỗ trợ Parquet NF-ToN-IoT-v2 của
pilot. Đã có adapter raw CSV bốn bộ tại `research/preprocess_four.py`, được kiểm
thử và chạy trên toàn bộ CSV. Xem [hướng dẫn tiền xử lý](PREPROCESSING_FOUR_VI.md). Các lệnh trong
[runbook pilot](RUNBOOK_VI.md) chưa phải lệnh chạy toàn bộ nghiên cứu mới.
Không đổi protocol đã băm trong provenance để làm như pilot vốn có phạm
vi bốn bộ. Protocol thực nghiệm mới phải được chốt riêng trước khi chạy.

## 3. Có cần server để tiền xử lý?

**Không cần server cho tiền xử lý.** Điều này đã được kiểm chứng trên máy
hiện tại, không còn là ước lượng. Máy có Ryzen 7 8845H (8 core/16 thread),
13 GiB RAM, 4 GiB swap, SSD NVMe và không có GPU NVIDIA. Ngày 2026-09-20,
pipeline đọc theo lô đã quét 75.987.976 dòng, giữ IP/port và xuất bốn
Parquet trong **449,34 giây (7 phút 29 giây)**. Kiểm tra độc lập mất thêm
**48,01 giây**. Tốc độ xuất trung bình đo được khoảng 169.110 dòng/giây;
đây là số đo của lần chạy này, không phải bảo đảm cho lần chạy khác.

Dung lượng thực tế: gói tải 2,19 GB, CSV 13,73 GB, bốn Parquet 3,28 GB.
Sau toàn bộ thao tác, máy còn khoảng 27 GiB trống. Mức này đủ để tiếp tục
tạo split, scaler và các manifest nếu ghi tuần tự. Nên giữ ít nhất 10 GiB
trống cho file tạm; nếu lưu dự đoán xác suất của nhiều run trên toàn bộ
test thì 27 GiB có thể nhanh chóng không đủ. Không nhân bản CSV/Parquet
không cần thiết và không xóa dữ liệu gốc khi chưa có bản sao đã kiểm tra.

Server trở nên cần thiết ở **huấn luyện toàn dữ liệu**, không phải ở bước
CSV → Parquet. Riêng ma trận 39 đặc trưng float32 của NF-BoT-IoT-v2 đã
khoảng 5,89 GB. Tensor thông điệp 128 chiều cho 37,76 triệu cạnh khoảng
19,3 GB cho một chiều, chưa tính cạnh ngược, gradient, optimizer và node.
Vì vậy cách full-batch hiện tại không phù hợp với RAM 13 GiB và có thể vẫn
không vừa GPU 24 GB. Phải chuyển sang edge/neighbor mini-batch trước khi
thuê GPU; thuê máy lớn mà giữ full-batch sẽ tốn tiền nhưng chưa giải quyết
được thiết kế bộ nhớ.

Phương án máy chủ sau khi benchmark: **64 GB RAM, GPU 24 GB VRAM, SSD/NVMe
còn ít nhất 100 GB**. Cấu hình 32 GB RAM có thể dùng khi mini-batch tốt,
nhưng 64 GB an toàn hơn cho loader, cache và đánh giá. GPU 80 GB chỉ đáng
xem xét nếu cố chạy graph rất lớn gần full-batch; không phải lựa chọn đầu.

### Dự toán thời gian từ trạng thái hiện tại

| Công việc | Máy hiện tại | Server/GPU | Cơ sở |
|---|---:|---:|---|
| Tái tạo 4 Parquet | 7 phút 29 giây | Không cần | Đã đo |
| Kiểm tra lại 4 Parquet | 48 giây | Không cần | Đã đo |
| Viết, kiểm thử và chạy chia group train/val/test | 2–4 giờ phát triển + 15–45 phút chạy | Không cần | Ước lượng; chưa triển khai |
| Pilot 50k–250k flow/dataset, 3 seed | 1–4 giờ | Không cần | Ước lượng từ pilot 50k |
| Benchmark đến 1 triệu flow/dataset | 6–16 giờ | Tùy chọn | Ngoại suy gần tuyến tính; phải đo lại |
| Full 4 dataset, multiclass, 2 GNN × 3 seed × 120 epoch | Tối thiểu khoảng 6,3 ngày CPU | Khoảng 1–4 ngày | Ngoại suy lạc quan từ pilot; chưa tính lỗi bộ nhớ/I/O |
| Thêm toàn bộ thí nghiệm binary như bài gốc | Khoảng gấp đôi | Khoảng gấp đôi | Số lượt huấn luyện tăng gấp đôi |

Mốc 6,3 ngày chỉ là cận lạc quan: pilot 50.000 flow mất khoảng 50–60 giây
cho một GNN đến 120 epoch; ngoại suy theo số lần duyệt cạnh trên 75,99 triệu
flow. Full-batch hiện tại sẽ hết bộ nhớ trước khi đạt thời gian này. Sau
khi có mini-batch, phải benchmark 1 epoch trên 250k và 1M flow rồi dùng
`thời gian/epoch × số epoch thực chạy × số seed`; không dùng bảng trên để
đặt deadline cứng.

## 4. Trình tự chuẩn bị dữ liệu từ đầu

1. Theo yêu cầu cập nhật, tải CSV gộp từ [Kaggle](https://www.kaggle.com/datasets/aryashah2k/nfuqnidsv2-network-intrusion-detection-dataset).
   Giữ bản nguồn trong `data/raw/NF-UQ-NIDS-v2/`; đầu ra sau tách có thư
   mục riêng cho từng dataset. Lưu URL, ngày tải, tên file, số byte và SHA-256.
   Chỉ tải CSV NetFlow đã trích xuất, không cần bắt đầu từ PCAP.
2. Đọc header và một lô nhỏ: xác nhận kiểu cột, IP/port, các đặc trưng,
   Label và Attack. Ghi số dòng và phân bố nhãn; so với mô tả nguồn.
   Không sử dụng con số của Parquet cũ làm số dòng kỳ vọng cho CSV mới.
3. Giữ IP/port để dựng endpoint; giữ Dataset và ID dòng nguồn để truy vết.
   Không đưa nhãn, ID dòng hoặc tên dataset vào đặc trưng mô hình. Nếu ghép
   dữ liệu, thêm namespace dataset vào endpoint để tránh nối nhầm địa chỉ
   trùng nhau giữa các mạng thí nghiệm độc lập.
4. Quét CSV theo lô, ghi Parquet. Ghi thống kê NaN/Inf, IP/port sai,
   nhãn bất nhất, trùng và xung đột nhãn. Chốt cách xử lý có log; không
   mặc định điền mọi giá trị lỗi bằng 0. Loại trùng phải kiểm tra xuyên lô,
   không chỉ gọi loại trùng trong từng lô.
5. Chốt đơn vị chia tập và nhóm chống rò rỉ trước khi lấy mẫu. Không để
   bản sao cùng flow đi vào nhiều split. Không suy ra thời gian từ thứ tự
   dòng nếu không có timestamp đáng tin cậy. Ghi rõ giới hạn random split.
6. Fit imputer/scaler/encoder bằng train; dùng lại cho validation/test.
   Lấy mẫu hoặc cân bằng lớp ở train theo protocol, không cân bằng test
   để làm đẹp điểm. Lưu support từng lớp, kể cả lớp quá hiếm để kết luận.
7. Tạo đồ thị riêng theo chính sách split đã chốt; giữ cạnh song song.
   Cạnh ngược phục vụ truyền thông điệp không được nhân đôi số mẫu đánh giá.
   Không tùy ý chia lô cạnh khi suy luận vì có thể đổi lân cận.
8. Xuất manifest, thống kê tài nguyên, split và cấu hình; sau đó áp dụng
   cùng adapter đã kiểm thử cho ba bộ còn lại. Mỗi bộ có đầu ra độc lập.

Trạng thái: đã tải, kiểm tra cấu trúc và xuất bốn bảng riêng theo
`PREPROCESSING_FOUR_VI.md`. Chia train/validation/test, fit scaler và dựng
đồ thị vẫn là giai đoạn tiếp theo; chưa thực hiện trên bốn bảng này.

## 5. Điều kiện hoàn tất nghiên cứu bốn bộ

Mỗi bộ cần có kiểm toán raw, split cố định, baseline bảng và E-GraphSAGE
cùng đầu vào hợp lệ, nhiều seed, checkpoint và dự đoán tái lập được.
Báo cáo macro-F1, weighted-F1, precision/recall/F1 cùng support theo lớp,
ma trận nhầm lẫn, thời gian và bộ nhớ. So sánh biến thể có/không đưa trực
tiếp đặc trưng cạnh vào bộ phân loại với cùng ngân sách lựa chọn mô hình.

Báo cáo riêng từng bộ trước khi tổng hợp; bốn bộ gộp thành một lần chạy
không thay thế bốn lần đánh giá. Không dùng test chọn siêu tham số.
Cross-dataset là thí nghiệm riêng, cần thống nhất nhãn và lớp chưa thấy;
không mặc định dùng chung bảng nhãn chỉ vì các bộ chung định dạng NetFlow.

## 6. Thông tin còn thiếu

Chưa có cấu hình server và chưa có benchmark mini-batch ở 250k/1M flow.
Hai thông tin này quyết định thời gian/chi phí huấn luyện toàn quy mô.
Không cần chờ server để làm bước tiếp theo: chốt protocol chia group,
tạo train/validation/test cho từng dataset, rồi chạy pilot tăng dần trên
máy hiện tại. Chỉ thuê server sau khi loader mini-batch vượt kiểm thử và
có số đo thời gian, RAM cùng dung lượng artifact của một epoch.
