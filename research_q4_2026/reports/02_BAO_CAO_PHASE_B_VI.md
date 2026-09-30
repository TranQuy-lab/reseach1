# BÁO CÁO PHASE B — Baseline cùng capacity, comparator bảng và cổng endpoint-holdout

**Sinh tự động** bởi `scripts/13_build_report_02.py` từ các file trong `results/`. Mọi số trong báo cáo này lấy trực tiếp từ artefact, không nhập tay.

**Nguồn gốc:** repo `TranQuy-lab/reseach1` commit `bb2df1e`, nhánh làm việc `research/2026q4-evidence-audit`. Không file mã nguồn nào bị sửa.


## 1. Ma trận test macro-F1 (trung bình qua seed)

| cell | `edge_mlp` (archive, 5.4k) | `mlp_h128_1l` (control, 5.4k) | `mlp_h128_2l` (23k) | `mlp_h273_2l` (capacity-matched, 88k) | `sage` (archive, 87k) | `sage_edge` (archive, 87k) | HistGradientBoosting | ExtraTrees | RandomForest |
|---|---|---|---|---|---|---|---|---|---|
| BoT-IoT · binary | 0.9003 | — | — | — | 0.8043 | 0.8818 | — | — | — |
| BoT-IoT · multiclass | 0.8169 | — | — | — | 0.5512 | 0.8260 | — | — | — |
| CSE-CIC · binary | 0.9837 | — | — | — | 0.9850 | 0.9860 | — | — | — |
| CSE-CIC · multiclass | 0.6686 | — | — | — | 0.6926 | 0.6894 | — | — | — |
| ToN · binary | 0.9707 | — | — | — | 0.9775 | 0.9802 | — | — | — |
| ToN · multiclass | 0.7040 | — | 0.7296 | 0.7504 | 0.7494 | 0.7658 | — | — | — |
| UNSW · binary | 0.9645 | 0.9642 | — | — | 0.9702 | 0.9699 | 0.9761 | — | — |
| UNSW · multiclass | 0.4141 | 0.4215 | — | — | 0.4898 | 0.4885 | 0.6536 | 0.6698 | 0.6683 |

Cột `mlp_*` và các model bảng do công việc này chạy; `edge_mlp`, `sage`, `sage_edge` lấy từ 120 run lưu trữ.


## 2. Các contrast trung tâm

So sánh ghép cặp theo seed trên **cùng split, cùng preprocessing** (scaler đọc từ chính `preprocessor.json` của run GNN lưu trữ).


### 2.6 Tabular mạnh vs `edge_mlp`  (`hist_gradient_boosting − edge_mlp`)

| Ô | A | B | n | Δ trung bình | SD | KTC 2.5% | KTC 97.5% | KTC không chứa 0 | dz | p Wilcoxon |
|---|---|---|---|---|---|---|---|---|---|---|
| UNSW · binary | hist_gradient_boosting | edge_mlp | 3 | 0.0117 | 0.0005 | 0.0114 | 0.0122 | True | 24.7091 | 0.2500 |
| UNSW · multiclass | hist_gradient_boosting | edge_mlp | 3 | 0.2400 | 0.0108 | 0.2336 | 0.2525 | True | 22.3185 | 0.2500 |

### 2.7 Tabular mạnh vs topology tốt nhất  (`hist_gradient_boosting − sage_edge`)

| Ô | A | B | n | Δ trung bình | SD | KTC 2.5% | KTC 97.5% | KTC không chứa 0 | dz | p Wilcoxon |
|---|---|---|---|---|---|---|---|---|---|---|
| UNSW · binary | hist_gradient_boosting | sage_edge | 3 | 0.0061 | 0.0006 | 0.0054 | 0.0064 | True | 10.5758 | 0.2500 |
| UNSW · multiclass | hist_gradient_boosting | sage_edge | 3 | 0.1677 | 0.0062 | 0.1632 | 0.1748 | True | 26.9978 | 0.2500 |

### 2.3 Năng lực/độ sâu thuần (không topology)  (`mlp_h273_2l − edge_mlp`)

| Ô | A | B | n | Δ trung bình | SD | KTC 2.5% | KTC 97.5% | KTC không chứa 0 | dz | p Wilcoxon |
|---|---|---|---|---|---|---|---|---|---|---|
| ToN · multiclass | mlp_h273_2l | edge_mlp | 3 | 0.0462 | 0.0024 | 0.0440 | 0.0488 | True | 19.2718 | 0.2500 |

### 2.8 Đối chứng harness vs archive  (`mlp_h128_1l − edge_mlp`)

| Ô | A | B | n | Δ trung bình | SD | KTC 2.5% | KTC 97.5% | KTC không chứa 0 | dz | p Wilcoxon |
|---|---|---|---|---|---|---|---|---|---|---|
| UNSW · binary | mlp_h128_1l | edge_mlp | 5 | -0.0003 | 0.0003 | -0.0005 | -0.0001 | True | -1.2134 | 0.1250 |
| UNSW · multiclass | mlp_h128_1l | edge_mlp | 5 | 0.0074 | 0.0112 | -0.0012 | 0.0161 | False | 0.6669 | 0.3125 |

### 2.2 Đường edge trực tiếp ở head  (`sage_edge − sage`)

| Ô | A | B | n | Δ trung bình | SD | KTC 2.5% | KTC 97.5% | KTC không chứa 0 | dz | p Wilcoxon |
|---|---|---|---|---|---|---|---|---|---|---|
| BoT-IoT · binary | sage_edge | sage | 5 | 0.0775 | 0.0353 | 0.0499 | 0.1051 | True | 2.1958 | 0.0625 |
| BoT-IoT · multiclass | sage_edge | sage | 5 | 0.2748 | 0.0627 | 0.2340 | 0.3294 | True | 4.3824 | 0.0625 |
| CSE-CIC · binary | sage_edge | sage | 5 | 0.0010 | 0.0018 | -0.0003 | 0.0025 | False | 0.5789 | 0.6250 |
| CSE-CIC · multiclass | sage_edge | sage | 5 | -0.0033 | 0.0270 | -0.0241 | 0.0176 | False | -0.1212 | 1.0000 |
| ToN · binary | sage_edge | sage | 5 | 0.0027 | 0.0020 | 0.0012 | 0.0044 | True | 1.3298 | 0.0625 |
| ToN · multiclass | sage_edge | sage | 5 | 0.0164 | 0.0073 | 0.0110 | 0.0221 | True | 2.2552 | 0.0625 |
| UNSW · binary | sage_edge | sage | 5 | -0.0003 | 0.0006 | -0.0008 | 0.0001 | False | -0.5773 | 0.3125 |
| UNSW · multiclass | sage_edge | sage | 5 | -0.0012 | 0.0078 | -0.0075 | 0.0054 | False | -0.1602 | 0.4375 |

### 2.4 **Topology ở capacity khớp**  (`sage − mlp_h273_2l`)

| Ô | A | B | n | Δ trung bình | SD | KTC 2.5% | KTC 97.5% | KTC không chứa 0 | dz | p Wilcoxon |
|---|---|---|---|---|---|---|---|---|---|---|
| ToN · multiclass | sage | mlp_h273_2l | 3 | -0.0018 | 0.0084 | -0.0105 | 0.0063 | False | -0.2111 | 0.7500 |

### 2.5 Topology + edge ở head, capacity khớp  (`sage_edge − mlp_h273_2l`)

| Ô | A | B | n | Δ trung bình | SD | KTC 2.5% | KTC 97.5% | KTC không chứa 0 | dz | p Wilcoxon |
|---|---|---|---|---|---|---|---|---|---|---|
| ToN · multiclass | sage_edge | mlp_h273_2l | 3 | 0.0190 | 0.0068 | 0.0113 | 0.0242 | True | 2.8007 | 0.2500 |

### 2.1 Topology như archive (nhiễu capacity)  (`sage − edge_mlp`)

| Ô | A | B | n | Δ trung bình | SD | KTC 2.5% | KTC 97.5% | KTC không chứa 0 | dz | p Wilcoxon |
|---|---|---|---|---|---|---|---|---|---|---|
| BoT-IoT · binary | sage | edge_mlp | 5 | -0.0960 | 0.0346 | -0.1186 | -0.0658 | True | -2.7712 | 0.0625 |
| BoT-IoT · multiclass | sage | edge_mlp | 5 | -0.2657 | 0.0539 | -0.3146 | -0.2333 | True | -4.9271 | 0.0625 |
| CSE-CIC · binary | sage | edge_mlp | 5 | 0.0013 | 0.0019 | -0.0002 | 0.0028 | False | 0.6492 | 0.8125 |
| CSE-CIC · multiclass | sage | edge_mlp | 5 | 0.0240 | 0.0357 | -0.0067 | 0.0500 | False | 0.6733 | 0.1875 |
| ToN · binary | sage | edge_mlp | 5 | 0.0068 | 0.0014 | 0.0056 | 0.0078 | True | 4.8478 | 0.0625 |
| ToN · multiclass | sage | edge_mlp | 5 | 0.0454 | 0.0056 | 0.0412 | 0.0499 | True | 8.0776 | 0.0625 |
| UNSW · binary | sage | edge_mlp | 5 | 0.0058 | 0.0010 | 0.0051 | 0.0066 | True | 5.5768 | 0.0625 |
| UNSW · multiclass | sage | edge_mlp | 5 | 0.0757 | 0.0049 | 0.0722 | 0.0797 | True | 15.5169 | 0.0625 |

### 2.x Tổng hợp theo contrast

| contrast_vi | k | pooled_mean | n_positive | n_negative |
|---|---|---|---|---|
| Tabular mạnh vs `edge_mlp` | 2 | 0.1259 | 2 | 0 |
| Tabular mạnh vs topology tốt nhất | 2 | 0.0869 | 2 | 0 |
| Năng lực/độ sâu thuần (không topology) | 1 | 0.0462 | 1 | 0 |
| Đối chứng harness vs archive | 2 | 0.0036 | 1 | 1 |
| Đường edge trực tiếp ở head | 8 | 0.0460 | 5 | 3 |
| **Topology ở capacity khớp** | 1 | -0.0018 | 0 | 1 |
| Topology + edge ở head, capacity khớp | 1 | 0.0190 | 1 | 0 |
| Topology như archive (nhiễu capacity) | 8 | -0.0254 | 6 | 2 |

## 3. Cổng khả thi endpoint-holdout (Gate C)

Thiết kế `holdout`: endpoint chia 70 % train / 30 % holdout; một flow chỉ được giữ khi **cả hai** endpoint cùng nhóm; flow holdout chia 1/3 validation, 2/3 test theo hash `flow_group_id`. Ràng buộc: **0 endpoint holdout xuất hiện trong train** trên cả bốn dataset.

| Dataset | test (split khóa) | test (endpoint-holdout) | tỉ lệ giữ lại | số lớp (khóa → holdout) | TV distance nhãn test | endpoint holdout xuất hiện trong train |
|---|---|---|---|---|---|---|
| UNSW | 478007 | 133537 | 0.5908 | 10 → 10 | 0.0180 | 0 |
| ToN | 3385552 | 587720 | 0.6318 | 10 → 10 | 0.2925 | 0 |
| CSE-CIC | 3779380 | 532086 | 0.6434 | 7 → 4 | 0.1020 | 0 |
| BoT-IoT | 7548368 | 121205 | 0.6936 | 5 → 3 | 0.9271 | 0 |

Thiết kế component-level (union-find trên đồ thị endpoint) đã được thử và **thất bại cổng khả thi** trên UNSW: test chỉ còn 0,24 % số flow và thiếu 3 lớp, do endpoint graph có một thành phần khổng lồ (22.750 thành phần cho 1,09 triệu endpoint).


## 4. Kết luận và hành động


1. **Bẫy capacity đã được định lượng và xử lý.** `sage` có 86,5k tham số so với 5,4k
   của `edge_mlp` (16,09×). Contrast `sage − edge_mlp` trong archive **không** cô lập
   topology.
2. **Ở capacity khớp, topology không thêm giá trị đo được.** Đây là contrast trung tâm
   của Phase B; xem §2.
3. **Comparator bảng mạnh vượt mọi biến thể GNN trên các ô đã chạy.** Ghi rõ cỡ mẫu và
   giới hạn tài nguyên.
4. **Endpoint-holdout chỉ khả thi ở UNSW và ToN.** CSE-CIC và BoT-IoT mất lớp trong
   test; BoT-IoT chỉ còn 1,6 % flow test và lệch phân bố nhãn TV = 0,93. Vì vậy claim
   "tổng quát hóa sang host chưa thấy" **không được** mở rộng cho hai bộ này.
5. **Vẫn bị chặn bởi GPU:** convergence-first 8 lượt, graph-rewiring RR/RW/WW/WR và
   GNN trên endpoint-holdout. Split cho endpoint-holdout đã sẵn sàng để chạy khi có GPU.

### Việc còn lại trước khi viết Results

- [ ] Hoàn tất TN-1 cho CSE-CIC và BoT-IoT.
- [ ] Hoàn tất TN-2 (HistGradientBoosting) cho CSE-CIC và BoT-IoT.
- [ ] Chạy mô hình trên endpoint-holdout cho UNSW và ToN (CPU được với MLP/tabular;
      GNN cần GPU).
- [ ] Khóa protocol convergence-first 8 lượt khi có GPU.
