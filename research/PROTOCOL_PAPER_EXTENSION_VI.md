# Protocol khóa trước — phần mở rộng phục vụ bài báo E-GraphSAGE

Trạng thái: **khóa trước khi chạy các thí nghiệm trong tài liệu này**.

Ngày khóa: 2026-09-28. Phần full-data 120 run đã được quan sát trước khi
protocol này được viết, vì vậy các kết quả cũ là bằng chứng nền và mọi phân
tích phát sinh từ chúng phải được gọi là mô tả hoặc khám phá. Các thí nghiệm
bên dưới là phần mở rộng có đối chứng; không sửa hoặc ghi đè
`data/full_splits`, `research/artifacts/full_runs` hay
`research/artifacts/full_runs_5seed`.

## 1. Câu hỏi và giới hạn claim

Đơn vị ghép cặp là cùng dataset, task và seed. Năm seed là lặp tính toán trên
cùng dữ liệu, không phải năm mạng độc lập. Bốn dataset được báo riêng; không
coi 120 hay tổng số run mới là cỡ mẫu độc lập.

Các claim được phép sau từng cổng:

1. Baseline bảng cho phép claim so sánh dự đoán với RF, ExtraTrees và HGB dưới
   cùng split/preprocessing.
2. Rewiring cho phép đánh giá sự phụ thuộc của dự đoán vào relational
   structure. Chênh lệch `sage - edge_mlp` tự nó không cô lập topology.
3. Endpoint holdout cho phép claim giới hạn về host-IP chưa thấy trên phần flow
   được giữ lại; không được gọi là cross-network generalization.
4. Budget sensitivity kiểm tra rival explanation rằng kết quả BoT của `sage`
   chỉ do ngân sách hai pass.

Rewiring được định nghĩa là **confirmatory replication/extension**, không phải
một thuật toán mới và không được claim là structural-intervention study đầu
tiên trong GNN-IDS. Mốc chặn overclaim là Chan et al. (2026),
https://doi.org/10.1109/ACCESS.2026.3703442. Điểm khác cần được kiểm chứng bằng
kết quả là quy mô bốn bộ NF-UQ-NIDS-v2, năm seed, negative control,
RR/RW/WW/WR, baseline bảng và endpoint holdout trong cùng protocol.

Outcome chính là **test macro-F1**. Outcome phụ: weighted-F1, accuracy,
per-class precision/recall/F1, thời gian, RAM/VRAM. PR-AUC/FPR không được suy
ra nếu artefact hiện tại chưa lưu đủ probability của toàn bộ test.

## 2. Baseline bảng full-data

- Dữ liệu: `data/full_splits`, không chia lại.
- Preprocessor: dùng đúng preprocessor đã persist từ `full_runs_5seed`; không
  fit lại trên validation/test.
- Dataset: đủ bốn bộ; task: binary và multiclass.
- Comparator: Random Forest, ExtraTrees, HistGradientBoosting.
- Seed: 11, 22, 33, 44, 55. Nếu một estimator là deterministic dưới cấu hình
  khóa, các lần lặp trùng nhau vẫn được báo là trùng, không dùng SD bằng 0 để
  phóng đại độ chắc chắn.
- Cấu hình khóa: 100 tree/iteration, depth tối đa 32, class balancing chỉ từ
  train, không internal early stopping, không dùng test để chọn tham số.
- Mỗi model phải được reload và tính lại metric trước khi coi là hoàn tất.

Contrast chính báo riêng, không chọn model thắng sau khi xem test:
`sage - RF`, `sage - ExtraTrees`, `sage - HGB`, `sage_edge - RF`,
`sage_edge - ExtraTrees`, `sage_edge - HGB`.

## 3. Graph-rewiring falsification

Rewiring seed chính: **20260928**. Destination endpoint tuple `(IP, port)`
được hoán vị trong từng dataset/split; 39 flow feature, label, source endpoint,
`source_row_id` và `flow_group_id` giữ nguyên. Phép hoán vị giữ destination
marginal và source marginal, nhưng phá quan hệ row-to-destination. Manifest
phải xác minh số dòng, feature/label identity, endpoint marginal và tỷ lệ cạnh
thực sự bị đổi trước khi train.

Bốn điều kiện:

| Mã | Train topology | Evaluation topology | Vai trò |
|---|---|---|---|
| RR | thật | thật | kết quả 5-seed hiện có |
| RW | thật | rewired | kiểm tra phụ thuộc topology khi inference |
| WW | rewired | rewired | kiểm tra khả năng học trên topology giả |
| WR | rewired | thật | kiểm tra transfer ngược |

- `sage` và `sage_edge`: đủ bốn dataset, hai task, năm seed cho WW; cùng
  checkpoint được cross-evaluate để tạo RW/WR.
- `edge_mlp`: negative control RW; probability và metric phải bất biến trong
  tolerance vì model này không đọc endpoint topology.
- Estimand chính: paired macro-F1 delta `RR - RW` theo cùng seed.
- Estimand phụ: `RR - WW`, `RR - WR` và per-class delta.
- Hai rewire seed bổ sung chỉ là sensitivity evaluation nếu chi phí cho phép;
  không thay seed chính sau khi xem kết quả.

## 4. Endpoint-disjoint holdout

IP, không phải `(IP, port)`, là đơn vị gán split. Một flow chỉ được giữ khi hai
IP đầu cuối được gán cùng split; flow nối hai split bị loại và phải báo số
lượng/phân bố lớp. Mọi IP trong output chỉ được xuất hiện ở đúng một split.

Feasibility gate theo từng dataset/task:

- binary: cả hai lớp có ít nhất 30 flow trong train, validation và test;
- multiclass: mọi lớp nguồn có ít nhất 30 flow trong cả ba split;
- dataset/task không đạt bị loại có lý do, không gộp lớp hoặc đổi seed sau khi
  xem hiệu năng;
- tỷ lệ giữ lại và class support trước/sau split phải xuất trong manifest.

Model: `edge_mlp`, `sage`, `sage_edge`; seed 11, 22, 33, 44, 55; budget và
checkpoint rule giống full-data chính. Kết quả chỉ áp dụng cho retained
endpoint-disjoint subgraph.

## 5. Budget sensitivity cho BoT-IoT

- Dataset: NF-BoT-IoT-v2; task: binary và multiclass.
- Model: `sage`, `sage_edge`; seed: 11, 22, 33, 44, 55.
- Budget: 2 pass đã có, chạy mới 4 và 8 pass; minimum 1.500 step, bốn lần
  validation mỗi pass.
- Không chọn budget bằng test. Learning curve và best validation macro-F1 là
  bằng chứng chính cho underfitting; test macro-F1 của cả ba budget được báo
  sau khi hoàn tất ma trận đã khóa.
- Nếu 4/8 pass không cải thiện validation một cách nhất quán, giả thuyết
  “hai pass là nguyên nhân chính” không được ủng hộ. Kết quả âm vẫn phải báo.

## 6. Phân tích và multiplicity

- Luôn công bố năm giá trị seed, mean, SD, median, min/max và số delta dương.
- Dùng paired delta theo cùng seed; không coi các run là độc lập.
- Exact sign-flip/permutation với năm cặp chỉ là phân tích hỗ trợ vì độ phân
  giải rất thấp. Không dùng p-value làm bằng chứng duy nhất.
- Các contrast chính được nhóm theo bốn family ở trên; nếu báo nhiều p-value,
  dùng Benjamini-Hochberg trong từng family và ghi cả giá trị chưa hiệu chỉnh.
- Lớp hiếm và phân tích được chọn sau khi xem 120 run cũ phải ghi exploratory.

## 7. Thứ tự chạy và cổng dừng

1. Sửa tài liệu 5-seed và chạy toàn bộ test code.
2. Tabular baseline và verifier.
3. Rewire prepare, hard gate, RW, WW, WR và negative control.
4. Endpoint prepare; chỉ train dataset/task qua feasibility gate.
5. Budget sensitivity BoT.
6. Sinh bảng effect, seed-level appendix và threats-to-validity.

Mỗi stage ghi output riêng, hỗ trợ resume và không ghi đè thư mục đã có dở.
Lỗi gate, thiếu lớp, mismatch provenance/checkpoint hoặc negative control đổi
quá tolerance phải dừng stage. Không sửa protocol này sau khi xem kết quả;
mọi thay đổi được ghi thành amendment có ngày, lý do và tác động dự kiến.

## 8. Dấu vết hỗ trợ thiết kế

Thiết kế này sử dụng các nguyên tắc randomization, replication, blocking,
falsification control và phân biệt reproducibility/replicability từ Scientific
Agent Skills: Kassis et al. (2026), https://doi.org/10.48550/arXiv.2609.00065.
