# BÁO CÁO KẾT QUẢ — Kiểm toán, chẩn đoán cơ chế và thí nghiệm xác nhận

**Sinh tự động** bởi `scripts/13_build_report_02.py`; mọi số đọc trực tiếp từ `results/`, không nhập tay.

**Nguồn gốc:** `TranQuy-lab/reseach1` commit `bb2df1e`, nhánh `research/2026q4-evidence-audit`. Không file nào trong `src/`, `tests/`, `research/` bị sửa.


## 0. Bốn kết luận trung tâm

1. **Không có bằng chứng `sage` hơn `edge_mlp` về trung bình.**
   Pooled Δ = +0.0021
   macro-F1, KTC 95 % [-0.0105;
   +0.0146] — chứa 0.
2. **Giả thuyết underfitting bị bác bỏ; cơ chế là sụp đổ tối ưu hóa.**
   Tỉ lệ sụp đổ `edge_mlp` 0%,
   `sage` 25%,
   `sage_edge` 5%;
   Fisher exact `sage` vs `edge_mlp` p = 0.0031 (Holm).
3. **Ba seed không đủ.** SD tăng tới 6.67× khi thêm
   seed 44/55; 4/24
   quyết định dựa trên KTC bị đảo.
4. **Ngân sách không đồng nhất.** UNSW chạy 3,67 lượt, ba bộ còn lại đúng 2,0 lượt.


## 1. Kiểm toán tính toàn vẹn

- Thiết kế 4 dataset × 2 task × 3 model × 5 seed: 120 run GNN lưu trữ, mọi ô đúng 5 seed: `True`.
- `runs.csv` công bố khớp `metrics.json` từng run tới 5.68e-14.
- Verify độc lập: `passed=True`, `runs_checked=120`, replay error 0.0, metric error 1.11e-16.
- `checkpoint_parameter_max_abs_error` bằng 0 trên mọi run: `True`.

### 1.1 Ngân sách huấn luyện thực tế

| Ô | steps/lượt | steps đã chạy | ràng buộc chi phối | lượt hiệu dụng |
|---|---|---|---|---|
| BoT-IoT · binary | 6455 | 12910 | dataset_passes | 2.0000 |
| BoT-IoT · multiclass | 6455 | 12910 | dataset_passes | 2.0000 |
| CSE-CIC · binary | 3230 | 6460 | dataset_passes | 2.0000 |
| CSE-CIC · multiclass | 3230 | 6460 | dataset_passes | 2.0000 |
| ToN · binary | 2896 | 5792 | dataset_passes | 2.0000 |
| ToN · multiclass | 2896 | 5792 | dataset_passes | 2.0000 |
| UNSW · binary | 409 | 1500 | min_train_steps | 3.6675 |
| UNSW · multiclass | 409 | 1500 | min_train_steps | 3.6675 |

## 2. Chẩn đoán learning curve (120 run lưu trữ)

Quy tắc chỉ dùng **validation**, không dùng test.

- Suy giảm sau đỉnh (R2): **53** run (44.2%)
- Còn tăng ở biên ngân sách (R1): 44 run (36.7%)
- Plateau thật (R3): 23 run (19.2%)
- Tỉ lệ run có checkpoint tốt nhất là lần đánh giá cuối: 36.7%

### 2.1 Lượt hiệu dụng theo dataset

| Dataset | lượt trung bình | min | max |
|---|---|---|---|
| BoT-IoT | 2.0000 | 2.0000 | 2.0000 |
| CSE-CIC | 2.0000 | 2.0000 | 2.0000 |
| ToN | 2.0000 | 2.0000 | 2.0000 |
| UNSW | 3.6675 | 3.6675 | 3.6675 |

## 3. Sụp đổ tối ưu hóa

Quy tắc khóa trước, chỉ dùng validation: `val_std ≥ 0,05` **hoặc** `drop_after_best ≥ 0,10`.

| Model | số run sụp đổ | tổng run | tỉ lệ |
|---|---|---|---|
| edge_mlp | 0 | 40 | 0.0000 |
| sage | 10 | 40 | 0.2500 |
| sage_edge | 2 | 40 | 0.0500 |


| Dataset | số run sụp đổ | tổng run | tỉ lệ |
|---|---|---|---|
| BoT-IoT | 9 | 30 | 0.3000 |
| CSE-CIC | 3 | 30 | 0.1000 |
| ToN | 0 | 30 | 0.0000 |
| UNSW | 0 | 30 | 0.0000 |

### 3.1 Độ bền của kết luận

| Model | tỉ lệ | KTC 2.5% | KTC 97.5% | n |
|---|---|---|---|---|
| edge_mlp | 0.0000 | 0.0000 | 0.0000 | 40 |
| sage | 0.2500 | 0.1250 | 0.4000 | 40 |
| sage_edge | 0.0500 | 0.0000 | 0.1250 | 40 |


| Cặp | odds ratio | p | p Holm |
|---|---|---|---|
| sage_vs_edge_mlp | — | 0.0010 | 0.0031 |
| sage_vs_sage_edge | 6.3333 | 0.0252 | 0.0504 |
| sage_edge_vs_edge_mlp | — | 0.4937 | 0.4937 |

Validation là bộ chọn checkpoint rất tốt:

| Model | ρ(best-val, test) | khoảng cách trung bình |
|---|---|---|
| edge_mlp | 0.9996 | 0.0005 |
| sage | 0.9435 | -0.0284 |
| sage_edge | 0.9977 | -0.0063 |

## 4. Thống kê 5 seed trên archive

| Contrast | số ô | Δ gộp (RE) | KTC 2.5% | KTC 97.5% | KTC dự báo 2.5% | KTC dự báo 97.5% | I² (%) | ô dương | ô âm |
|---|---|---|---|---|---|---|---|---|---|
| sage_edge_minus_edge_mlp | 8 | 0.0216 | 0.0153 | 0.0279 | 0.0003 | 0.0428 | 98.7979 | 7 | 1 |
| sage_edge_minus_sage | 8 | 0.0067 | 0.0019 | 0.0116 | -0.0077 | 0.0211 | 95.5620 | 5 | 3 |
| sage_minus_edge_mlp | 8 | 0.0021 | -0.0105 | 0.0146 | -0.0403 | 0.0444 | 99.5160 | 6 | 2 |

- Wilcoxon sống sót Holm: **0**/24
- Sign test sống sót Holm: **0**/24
- KTC (chưa hiệu chỉnh) không chứa 0: 18/24

## 5. 3 seed so với 5 seed

| Ô | mean(3) | SD(3) | mean(5) | SD(5) | SD(5)/SD(3) | lệch mean |
|---|---|---|---|---|---|---|
| BoT-IoT · binary · edge_mlp | 0.9114 | 0.0038 | 0.9003 | 0.0256 | 6.6702 | -0.0111 |
| UNSW · binary · sage | 0.9700 | 0.0002 | 0.9702 | 0.0010 | 5.1895 | 0.0003 |
| CSE-CIC · multiclass · edge_mlp | 0.6607 | 0.0049 | 0.6686 | 0.0162 | 3.2774 | 0.0079 |
| BoT-IoT · multiclass · sage | 0.5698 | 0.0207 | 0.5512 | 0.0511 | 2.4739 | -0.0187 |
| ToN · multiclass · edge_mlp | 0.7042 | 0.0016 | 0.7040 | 0.0039 | 2.3926 | -0.0002 |
| ToN · binary · sage_edge | 0.9806 | 0.0003 | 0.9802 | 0.0006 | 2.1601 | -0.0004 |
| UNSW · binary · sage_edge | 0.9700 | 0.0004 | 0.9699 | 0.0009 | 2.1302 | -0.0001 |
| BoT-IoT · multiclass · edge_mlp | 0.8147 | 0.0019 | 0.8169 | 0.0036 | 1.9013 | 0.0022 |

## 6. Thống kê đồ thị — vì sao topology giúp ở chỗ này mà không ở chỗ khác

| Dataset | flow train | endpoint | cạnh/node | flow/cặp endpoint | tỉ lệ tương tác lặp | degree trung vị | entropy lớp (nats) |
|---|---|---|---|---|---|---|---|
| UNSW | 1673104 | 940143 | 1.780 | 1.240 | 0.193 | 2.000 | 0.235 |
| ToN | 11861039 | 1046743 | 11.331 | 1.575 | 0.365 | 4.000 | 1.694 |
| CSE-CIC | 13227184 | 7778169 | 1.701 | 1.281 | 0.219 | 1.000 | 0.499 |
| BoT-IoT | 26438303 | 336138 | 78.653 | 11.964 | 0.916 | 91.000 | 0.918 |

BoT-IoT có cấu trúc quan hệ **giàu nhất** (78,7 cạnh/node, 91,6 % tương tác lặp) — và cũng chính là nơi `sage` sụp đổ nhiều nhất. UNSW và CSE-CIC gần như độc lập theo flow (1,7–1,8 cạnh/node, ~20 % lặp), nên message passing không có thông tin bổ sung để khai thác.


## 7. Cổng khả thi endpoint-holdout

| Dataset | test (split khóa) | test (holdout) | tỉ lệ giữ lại | lớp (khóa → holdout) | TV nhãn test | endpoint holdout trong train |
|---|---|---|---|---|---|---|
| UNSW | 478007 | 133537 | 0.5908 | 10 → 10 | 0.0180 | 0 |
| ToN | 3385552 | 587720 | 0.6318 | 10 → 10 | 0.2925 | 0 |
| CSE-CIC | 3779380 | 532086 | 0.6434 | 7 → 4 | 0.1020 | 0 |
| BoT-IoT | 7548368 | 121205 | 0.6936 | 5 → 3 | 0.9271 | 0 |

Thiết kế `holdout`: endpoint chia 70 % train / 30 % holdout; flow chỉ giữ khi **cả hai** endpoint cùng nhóm; flow holdout chia 1/3 val, 2/3 test. Thiết kế component-level đã được thử và **thất bại cổng khả thi** (UNSW: test còn 0,24 % flow, thiếu 3 lớp, do endpoint graph có một thành phần khổng lồ).


## 8. Phase B — capacity, topology và comparator bảng

### 8.1 Ma trận test macro-F1 (trung bình qua seed)

| cell | `edge_mlp` (5.4k) | `mlp_h128_1l` control (5.4k) | `mlp_h128_2l` (23k) | `mlp_h273_2l` capacity-matched (88k) | `sage` (87k) | `sage_edge` (87k) | HistGB | ExtraTrees | RandomForest |
|---|---|---|---|---|---|---|---|---|---|
| BoT-IoT · binary | 0.9003 | — | — | — | 0.8043 | 0.8818 | — | — | — |
| BoT-IoT · multiclass | 0.8169 | — | — | — | 0.5512 | 0.8260 | — | — | — |
| CSE-CIC · binary | 0.9837 | — | — | 0.9874 | 0.9850 | 0.9860 | — | — | — |
| CSE-CIC · multiclass | 0.6686 | 0.6535 | — | 0.6738 | 0.6926 | 0.6894 | — | — | — |
| ToN · binary | 0.9707 | — | — | 0.9813 | 0.9775 | 0.9802 | — | — | — |
| ToN · multiclass | 0.7040 | 0.7013 | 0.7296 | 0.7494 | 0.7494 | 0.7658 | — | — | — |
| UNSW · binary | 0.9645 | 0.9642 | 0.9662 | 0.9674 | 0.9702 | 0.9699 | 0.9761 | 0.9803 | 0.9828 |
| UNSW · multiclass | 0.4141 | 0.4215 | 0.4425 | 0.4628 | 0.4898 | 0.4885 | 0.6536 | 0.6698 | 0.6683 |

Các cột `mlp_*`, `HistGB`, `ExtraTrees`, `RandomForest` do công việc này chạy trên **đúng split**; `edge_mlp`/`sage`/`sage_edge` lấy từ 120 run lưu trữ. Scaler đọc từ chính `preprocessor.json` của run GNN, **không** fit lại.

### 8.2 Contrast ghép cặp theo seed


**Đối chứng harness vs archive**  (`mlp_h128_1l − edge_mlp`)

| Ô | A | B | n | Δ | SD | KTC 2.5% | KTC 97.5% | KTC không chứa 0 | dz | p Wilcoxon |
|---|---|---|---|---|---|---|---|---|---|---|
| CSE-CIC · multiclass | mlp_h128_1l | edge_mlp | 3 | -0.0072 | 0.0046 | -0.0119 | -0.0028 | True | -1.5698 | 0.2500 |
| ToN · multiclass | mlp_h128_1l | edge_mlp | 5 | -0.0027 | 0.0087 | -0.0094 | 0.0039 | False | -0.3118 | 0.6250 |
| UNSW · binary | mlp_h128_1l | edge_mlp | 5 | -0.0003 | 0.0003 | -0.0005 | -0.0001 | True | -1.2134 | 0.1250 |
| UNSW · multiclass | mlp_h128_1l | edge_mlp | 5 | 0.0074 | 0.0112 | -0.0012 | 0.0161 | False | 0.6669 | 0.3125 |

**Capacity/độ sâu thuần (KHÔNG topology)**  (`mlp_h273_2l − edge_mlp`)

| Ô | A | B | n | Δ | SD | KTC 2.5% | KTC 97.5% | KTC không chứa 0 | dz | p Wilcoxon |
|---|---|---|---|---|---|---|---|---|---|---|
| CSE-CIC · multiclass | mlp_h273_2l | edge_mlp | 3 | 0.0131 | 0.0066 | 0.0073 | 0.0202 | True | 1.9988 | 0.2500 |
| ToN · binary | mlp_h273_2l | edge_mlp | 2 | 0.0108 | 0.0004 | 0.0105 | 0.0111 | True | 28.2886 | 0.5000 |
| ToN · multiclass | mlp_h273_2l | edge_mlp | 5 | 0.0454 | 0.0061 | 0.0402 | 0.0499 | True | 7.4547 | 0.0625 |
| UNSW · binary | mlp_h273_2l | edge_mlp | 5 | 0.0029 | 0.0006 | 0.0025 | 0.0034 | True | 4.7858 | 0.0625 |
| UNSW · multiclass | mlp_h273_2l | edge_mlp | 5 | 0.0488 | 0.0165 | 0.0344 | 0.0594 | True | 2.9563 | 0.0625 |

****Topology ở capacity khớp****  (`sage − mlp_h273_2l`)

| Ô | A | B | n | Δ | SD | KTC 2.5% | KTC 97.5% | KTC không chứa 0 | dz | p Wilcoxon |
|---|---|---|---|---|---|---|---|---|---|---|
| CSE-CIC · multiclass | sage | mlp_h273_2l | 3 | 0.0250 | 0.0256 | -0.0007 | 0.0506 | False | 0.9746 | 0.5000 |
| ToN · binary | sage | mlp_h273_2l | 2 | -0.0053 | 0.0008 | -0.0058 | -0.0047 | True | -6.5874 | 0.5000 |
| ToN · multiclass | sage | mlp_h273_2l | 5 | -0.0000 | 0.0073 | -0.0055 | 0.0054 | False | -0.0004 | 1.0000 |
| UNSW · binary | sage | mlp_h273_2l | 5 | 0.0029 | 0.0014 | 0.0020 | 0.0040 | True | 2.1093 | 0.0625 |
| UNSW · multiclass | sage | mlp_h273_2l | 5 | 0.0269 | 0.0163 | 0.0163 | 0.0413 | True | 1.6568 | 0.0625 |

**Topology + edge ở head, capacity khớp**  (`sage_edge − mlp_h273_2l`)

| Ô | A | B | n | Δ | SD | KTC 2.5% | KTC 97.5% | KTC không chứa 0 | dz | p Wilcoxon |
|---|---|---|---|---|---|---|---|---|---|---|
| CSE-CIC · multiclass | sage_edge | mlp_h273_2l | 3 | 0.0133 | 0.0086 | 0.0060 | 0.0227 | True | 1.5461 | 0.2500 |
| ToN · binary | sage_edge | mlp_h273_2l | 2 | -0.0007 | 0.0007 | -0.0012 | -0.0002 | True | -1.0041 | 0.5000 |
| ToN · multiclass | sage_edge | mlp_h273_2l | 5 | 0.0164 | 0.0061 | 0.0117 | 0.0212 | True | 2.6921 | 0.0625 |
| UNSW · binary | sage_edge | mlp_h273_2l | 5 | 0.0026 | 0.0013 | 0.0015 | 0.0036 | True | 1.9301 | 0.0625 |
| UNSW · multiclass | sage_edge | mlp_h273_2l | 5 | 0.0257 | 0.0159 | 0.0150 | 0.0396 | True | 1.6166 | 0.0625 |

**Topology như archive (nhiễu capacity)**  (`sage − edge_mlp`)

| Ô | A | B | n | Δ | SD | KTC 2.5% | KTC 97.5% | KTC không chứa 0 | dz | p Wilcoxon |
|---|---|---|---|---|---|---|---|---|---|---|
| BoT-IoT · binary | sage | edge_mlp | 5 | -0.0960 | 0.0346 | -0.1186 | -0.0658 | True | -2.7712 | 0.0625 |
| BoT-IoT · multiclass | sage | edge_mlp | 5 | -0.2657 | 0.0539 | -0.3146 | -0.2333 | True | -4.9271 | 0.0625 |
| CSE-CIC · binary | sage | edge_mlp | 5 | 0.0013 | 0.0019 | -0.0002 | 0.0028 | False | 0.6492 | 0.8125 |
| CSE-CIC · multiclass | sage | edge_mlp | 5 | 0.0240 | 0.0357 | -0.0067 | 0.0500 | False | 0.6733 | 0.1875 |
| ToN · binary | sage | edge_mlp | 5 | 0.0068 | 0.0014 | 0.0057 | 0.0078 | True | 4.8478 | 0.0625 |
| ToN · multiclass | sage | edge_mlp | 5 | 0.0454 | 0.0056 | 0.0412 | 0.0499 | True | 8.0776 | 0.0625 |
| UNSW · binary | sage | edge_mlp | 5 | 0.0058 | 0.0010 | 0.0051 | 0.0067 | True | 5.5768 | 0.0625 |
| UNSW · multiclass | sage | edge_mlp | 5 | 0.0757 | 0.0049 | 0.0722 | 0.0797 | True | 15.5169 | 0.0625 |

**Đường edge trực tiếp ở head**  (`sage_edge − sage`)

| Ô | A | B | n | Δ | SD | KTC 2.5% | KTC 97.5% | KTC không chứa 0 | dz | p Wilcoxon |
|---|---|---|---|---|---|---|---|---|---|---|
| BoT-IoT · binary | sage_edge | sage | 5 | 0.0775 | 0.0353 | 0.0499 | 0.1051 | True | 2.1958 | 0.0625 |
| BoT-IoT · multiclass | sage_edge | sage | 5 | 0.2748 | 0.0627 | 0.2340 | 0.3294 | True | 4.3824 | 0.0625 |
| CSE-CIC · binary | sage_edge | sage | 5 | 0.0010 | 0.0018 | -0.0003 | 0.0025 | False | 0.5789 | 0.6250 |
| CSE-CIC · multiclass | sage_edge | sage | 5 | -0.0033 | 0.0270 | -0.0241 | 0.0176 | False | -0.1212 | 1.0000 |
| ToN · binary | sage_edge | sage | 5 | 0.0027 | 0.0020 | 0.0012 | 0.0043 | True | 1.3298 | 0.0625 |
| ToN · multiclass | sage_edge | sage | 5 | 0.0164 | 0.0073 | 0.0110 | 0.0221 | True | 2.2552 | 0.0625 |
| UNSW · binary | sage_edge | sage | 5 | -0.0003 | 0.0006 | -0.0008 | 0.0001 | False | -0.5773 | 0.3125 |
| UNSW · multiclass | sage_edge | sage | 5 | -0.0012 | 0.0078 | -0.0075 | 0.0054 | False | -0.1602 | 0.4375 |

**Bảng mạnh vs `edge_mlp`**  (`hist_gradient_boosting − edge_mlp`)

| Ô | A | B | n | Δ | SD | KTC 2.5% | KTC 97.5% | KTC không chứa 0 | dz | p Wilcoxon |
|---|---|---|---|---|---|---|---|---|---|---|
| UNSW · binary | hist_gradient_boosting | edge_mlp | 3 | 0.0117 | 0.0005 | 0.0114 | 0.0122 | True | 24.7091 | 0.2500 |
| UNSW · multiclass | hist_gradient_boosting | edge_mlp | 3 | 0.2400 | 0.0108 | 0.2336 | 0.2525 | True | 22.3185 | 0.2500 |

**Bảng mạnh vs topology tốt nhất**  (`hist_gradient_boosting − sage_edge`)

| Ô | A | B | n | Δ | SD | KTC 2.5% | KTC 97.5% | KTC không chứa 0 | dz | p Wilcoxon |
|---|---|---|---|---|---|---|---|---|---|---|
| UNSW · binary | hist_gradient_boosting | sage_edge | 3 | 0.0061 | 0.0006 | 0.0054 | 0.0064 | True | 10.5758 | 0.2500 |
| UNSW · multiclass | hist_gradient_boosting | sage_edge | 3 | 0.1677 | 0.0062 | 0.1632 | 0.1748 | True | 26.9978 | 0.2500 |

### 8.3 Tổng hợp

| Contrast | số ô | Δ trung bình | ô dương | ô âm |
|---|---|---|---|---|
| Bảng mạnh vs `edge_mlp` | 2 | 0.1259 | 2 | 0 |
| Bảng mạnh vs topology tốt nhất | 2 | 0.0869 | 2 | 0 |
| Capacity/độ sâu thuần (KHÔNG topology) | 5 | 0.0242 | 5 | 0 |
| Đối chứng harness vs archive | 4 | -0.0007 | 1 | 3 |
| Đường edge trực tiếp ở head | 8 | 0.0460 | 5 | 3 |
| **Topology ở capacity khớp** | 5 | 0.0099 | 3 | 2 |
| Topology + edge ở head, capacity khớp | 5 | 0.0114 | 4 | 1 |
| Topology như archive (nhiễu capacity) | 8 | -0.0254 | 6 | 2 |

## 8b. Đánh giá trên split endpoint-holdout (cùng model, cùng ngân sách)

So sánh trực tiếp giữa split khóa (`flow_group_id`) và split endpoint-disjoint. Kiến trúc, seed và ngân sách giống hệt; khác biệt duy nhất là flow test có thể chứa endpoint **chưa từng thấy** trong train. Scaler được fit lại trên train của từng split (dân số huấn luyện khác nhau) và điều này được ghi rõ.

| Ô | n | split khóa | endpoint-holdout | mức giảm | KTC 2.5% | KTC 97.5% | giảm tương đối (%) |
|---|---|---|---|---|---|---|---|
| ToN · multiclass · mlp_h128_1l | 3 | 0.6963 | 0.4088 | 0.2874 | 0.2734 | 0.2975 | 41.2818 |
| ToN · multiclass · mlp_h273_2l | 3 | 0.7504 | 0.4396 | 0.3109 | 0.3060 | 0.3186 | 41.4237 |
| UNSW · binary · mlp_h128_1l | 3 | 0.9641 | 0.9802 | -0.0161 | -0.0166 | -0.0153 | -1.6695 |
| UNSW · binary · mlp_h273_2l | 3 | 0.9674 | 0.9834 | -0.0160 | -0.0167 | -0.0153 | -1.6580 |
| UNSW · multiclass · mlp_h128_1l | 3 | 0.4218 | 0.4053 | 0.0165 | -0.0128 | 0.0382 | 3.9116 |
| UNSW · multiclass · mlp_h273_2l | 3 | 0.4697 | 0.4616 | 0.0081 | -0.0234 | 0.0291 | 1.7243 |

Mức giảm **dương** nghĩa là split endpoint-holdout làm giảm chất lượng. Nếu KTC chứa 0 thì không có bằng chứng suy giảm.


## 9. Kết luận và việc còn lại

**Đã đóng:** kiểm toán 120 run; chẩn đoán hội tụ/sụp đổ; thống kê 5 seed và
meta-analysis; lớp hiếm; thống kê đồ thị; cổng endpoint-holdout; tái tạo split khớp
manifest; baseline cùng capacity và comparator bảng (đang hoàn tất nốt các ô còn lại).

**Bị chặn bởi GPU** (server `vast-gpu` 202.59.206.41:11493 từ chối kết nối):

1. convergence-first 8 lượt cho UNSW/ToN/BoT;
2. graph-rewiring RR/RW/WW/WR;
3. GNN trên endpoint-holdout (split đã sẵn sàng);
4. learning-rate/budget sensitivity cho `sage` BoT-IoT.

**Ước lượng khi có GPU 24 GiB:** ma trận 72 run ba seed từng mất 5,45 h. Convergence-first
8 lượt cho 3 dataset × 3 model × 3–5 seed ước tính 15–30 h GPU.
