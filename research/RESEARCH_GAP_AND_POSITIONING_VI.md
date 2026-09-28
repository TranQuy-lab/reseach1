# Khoảng trống nghiên cứu và định vị lĩnh vực

## 1. Đề tài thuộc lĩnh vực nào?

Đối tượng nghiên cứu vẫn là **an toàn thông tin**, cụ thể là network intrusion
detection trên dữ liệu NetFlow. Graph neural network là công cụ mô hình hóa,
không phải đối tượng cuối cùng. Bài phù hợp nhất với định vị:

> An toàn thông tin thực nghiệm có sử dụng graph machine learning.

Không định vị là bài đề xuất thuật toán AI mới, vì repo chưa có kiến trúc,
loss, cơ chế sampling hoặc learning objective mới. Đóng góp AI nằm ở thiết kế
ablation và bằng chứng về điều kiện sử dụng relational modeling; đóng góp an
toàn thông tin nằm ở threat setting, loại tấn công, false-positive/rare-class
behavior và khả năng chuyển sang endpoint chưa thấy.

## 2. Khoảng trống được chính repo chứng minh

### Gap A — Chưa cô lập giá trị của topology

Kết quả hiện có cho thấy relational variant và flow-only model khác nhau,
nhưng phép so sánh còn đồng thời thay đổi kiến trúc, số tham số và đường truyền
edge feature. Vì vậy chưa biết phần cải thiện đến từ topology thật hay từ model
capacity/optimization. Graph rewiring RR/RW/WW/WR là phép falsification chính.

### Gap B — Chưa biết hiệu quả trên host chưa thấy

Trong split chính, 99,21–100% flow validation/test có cả hai IP đã xuất hiện
trong train. Split vẫn hợp lệ cho within-environment flow classification, nhưng
không trả lời unseen-host detection. Endpoint-disjoint holdout đóng khoảng
trống này và phải báo cả số flow bị loại cùng class coverage.

### Gap C — Chưa có đối chứng tabular đủ mạnh

`edge_mlp` là flow-only neural ablation, không đại diện toàn bộ classical ML.
Nếu thiếu RF, ExtraTrees và HGB trên cùng split/preprocessing, chưa thể biết GNN
có hơn các mô hình bảng mạnh hay không.

### Gap D — Optimization có thể bị nhầm thành giới hạn kiến trúc

BoT-IoT cho thấy `sage` thấp bất thường dưới ngân sách hai pass. Chưa thể kết
luận topology không hữu ích nếu chưa loại rival explanation là underfitting.
Sensitivity 2/4/8 pass, chọn checkpoint bằng validation, giải quyết câu hỏi này.

### Gap E — Reproducibility mạnh nhưng operational validity còn thiếu

120/120 checkpoint verification chứng minh computational reproducibility trên
cùng data/code. Nó không tự chứng minh khả năng triển khai IDS. Một bài hướng
ứng dụng sau này vẫn cần PR-AUC, FPR tại TPR cố định, calibration, throughput,
latency và kiểm thử dưới distribution shift.

## 3. Khoảng trống trung tâm nên dùng trong bài

> Existing results in this repository do not yet establish whether the
> observed performance differences arise from network relational structure,
> direct flow features, optimization budget, or endpoint familiarity. This
> study closes that evidential gap through strong tabular comparators,
> topology-rewiring controls, endpoint-disjoint evaluation, and a prespecified
> training-budget sensitivity analysis on a reproducible full-scale pipeline.

Đây là **evidential/mechanistic gap trong một bài toán cybersecurity**, không
phải tuyên bố rằng toàn bộ văn liệu trước đây chưa từng nghiên cứu topology.
Đặc biệt, một kiểm tra văn liệu cập nhật ngày 2026-09-28 đã tìm thấy nghiên cứu
can thiệp cấu trúc GNN-IDS năm 2026, trong đó có degree-preserving rewiring trên
ToN-IoT và CIC-IDS-2017. Vì vậy bài này **không được claim là nghiên cứu đầu
tiên dùng rewiring hoặc structural intervention cho IDS**. Rewiring ở đây là
phép kiểm chứng xác nhận và mở rộng trên protocol/dữ liệu của repo.

Novelty có thể bảo vệ, nếu các thí nghiệm hoàn tất, là **tổ hợp bằng chứng**:

1. full-scale trên bốn bộ cấu thành NF-UQ-NIDS-v2;
2. năm seed với checkpoint/probability/metric được kiểm chứng độc lập;
3. cùng lúc có tabular comparator mạnh, bốn điều kiện RR/RW/WW/WR và negative
   control không dùng topology;
4. tách within-environment flow classification khỏi endpoint-disjoint
   generalization;
5. khóa trước kiểm tra rival explanation về training budget.

Đây mới là định vị “controlled replication and extension”, không phải thuật
toán GNN mới. Muốn khẳng định đây là literature gap đầy đủ vẫn phải làm
systematic/bounded literature search và lưu bảng source-to-claim; không được
suy novelty chỉ từ repo hoặc từ lần tìm kiếm sơ bộ này.

## 4. Research questions và bằng chứng quyết định

| Research question | Thí nghiệm quyết định | Claim tối đa nếu đạt |
|---|---|---|
| Relational structure có giúp không? | RR so với RW, WW và WR | Prediction phụ thuộc topology trong setting đã đo |
| GNN có hơn tabular ML không? | RF/ExtraTrees/HGB cùng split | Comparative performance dưới protocol khóa |
| Có phát hiện endpoint chưa thấy không? | IP-disjoint holdout | Generalization trên retained unseen-host subgraph |
| BoT `sage` kém vì underfitting? | 2/4/8 pass | Budget sensitivity, không phải causal proof duy nhất |
| Hiệu quả có đồng nhất không? | Paired delta theo dataset/task/seed | Dataset/task-dependent association |

## 5. Câu chuyện bài báo nên giữ

1. Bài toán an toàn thông tin: phát hiện xâm nhập từ NetFlow ở quy mô lớn.
2. Vấn đề mô hình: flow feature có thể chưa khai thác quan hệ endpoint.
3. Vấn đề bằng chứng: relational model thắng không đồng nghĩa topology hữu ích.
4. Phương pháp: full-scale verified pipeline cộng các falsification controls.
5. Kết quả cần báo cả trường hợp topology giúp, không giúp và bị giới hạn.
6. Kết luận thực hành: chỉ dùng GNN khi relational gain vượt baseline và còn
   tồn tại dưới rewiring/holdout controls phù hợp.

## 6. Tiêu đề theo trạng thái bằng chứng

Trước khi có rewiring:

> Dataset- and Task-Dependent Performance of E-GraphSAGE on NF-UQ-NIDS-v2

Sau khi rewiring và tabular baseline hoàn tất:

> When Does Relational Structure Help NetFlow Intrusion Detection? A
> Reproducible Controlled Study of E-GraphSAGE on NF-UQ-NIDS-v2

Chỉ thêm “unseen-host generalization” vào tiêu đề/abstract nếu endpoint holdout
qua feasibility gate và có kết quả được báo đầy đủ.

## 7. Mốc văn liệu dùng để chặn overclaim

- Lo et al., *E-GraphSAGE: A Graph Neural Network based Intrusion Detection
  System for IoT*, https://arxiv.org/abs/2103.16329 — công trình nền đã claim
  khai thác cả edge feature và topology trên bốn benchmark NIDS.
- Zhong et al., *A survey on graph neural networks for intrusion detection
  systems: Methods, trends and challenges*,
  https://doi.org/10.1016/j.cose.2024.103821 — xác nhận GNN-IDS đã là một dòng
  nghiên cứu với vấn đề graph construction, model design và deployment.
- Chan et al., *Structural Sensitivity of Graph Neural Networks for Intrusion
  Detection: An Intervention Study*,
  https://doi.org/10.1109/ACCESS.2026.3703442 — bằng chứng rằng structural
  intervention/rewiring trong GNN-IDS đã có trước; công trình hiện tại phải tự
  định vị là replication/extension có kiểm soát trên setting khác.

Danh sách này là mốc định vị tối thiểu, chưa thay thế systematic review.
