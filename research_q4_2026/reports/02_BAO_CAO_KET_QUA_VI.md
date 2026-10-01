# BÁO CÁO KẾT QUẢ — Kiểm toán, chẩn đoán cơ chế và thí nghiệm xác nhận

**Sinh tự động** bởi `scripts/13_build_report_02.py`; mọi số đọc trực tiếp từ `results/`, không nhập tay.

**Nguồn gốc:** `TranQuy-lab/reseach1` commit `bb2df1e`, nhánh `research/2026q4-evidence-audit`. Không file nào trong `src/`, `tests/`, `research/` bị sửa.


## 0. Tám kết luận trung tâm

1. **Không có bằng chứng `sage` hơn `edge_mlp` về trung bình.**
   Pooled Δ = +0.0021
   macro-F1, KTC 95 % [-0.0105;
   +0.0146] — chứa 0.
2. **Giả thuyết underfitting bị bác bỏ; cơ chế là sụp đổ tối ưu hóa.**
   Tỉ lệ sụp đổ `edge_mlp` 0%,
   `sage` 25%,
   `sage_edge` 5%;
   Fisher exact `sage` vs `edge_mlp` p = 0.0031 (Holm).
3. **Capacity giải thích một phần có ý nghĩa ở CẢ 8 ô; topology còn lại thì không tự
   biện minh.** Hiệu ứng capacity dương và KTC không chứa 0 ở cả 8 ô (+0,003 → +0,049).
   Sau khi khớp capacity, `sage` **kém hơn** có ý nghĩa ở 4 ô — BoT mc **−0,267**,
   BoT bin **−0,132**, ToN bin −0,004, CSE bin −0,003 — không phân biệt được ở 2 ô
   (ToN mc −0,000; CSE mc **+0,010** với KTC chứa 0), và chỉ hơn nhẹ ở 2 ô (UNSW mc
   +0,027; UNSW bin +0,003). **Không ô nào topology dương mạnh.** Lưu ý: CSE mc co từ
   +0,025 (3 seed) xuống +0,010 (5 seed) — thêm một ví dụ 3 seed phóng đại hiệu ứng.
4. **Phụ thuộc endpoint: đã đo và thấy KHÔNG đáng kể trên split khóa — và một kết luận
   cũ đã bị bác bỏ.** Split endpoint-disjoint từng cho thấy ToN multiclass mất 0,311,
   nhưng đó là **split bị nhiễu**: `HistGradientBoosting` (không hề dùng danh tính
   endpoint) cũng mất 0,866 → **0,472** trên chính split đó. Đo sạch trên split khóa:
   chỉ **621/3.385.552 (0,018 %)** flow test có cả hai endpoint chưa thấy, và trên nhóm
   đó điểm chỉ giảm 0,039. Vậy tỉ lệ chồng lấp endpoint **không** thổi phồng kết quả,
   nhưng split này cũng **không** dùng được cho claim unseen-host.
5. **Tóm tắt cấu trúc thô thắng message passing — lặp lại trên hai bộ.**
   `mlp_struct` (chỉ đếm lân cận, tính từ train) đạt 0.5003
   trên UNSW mc (vs `sage` 0,4898) và 0,8188 trên ToN mc (vs `sage_edge` 0,7658). Thêm
   tỉ lệ nhãn train (`mlp_struct_lab`) giúp trên ToN (0,8262) nhưng hại trên UNSW (0,4708).
   Ngược lại, **trung bình đặc trưng lân cận thì thất bại**: `mlp_nbr_mean` =
   0.4266 (n=3) — *thấp hơn* cả flow-only.
6. **Bảng mạnh thắng 5/6 ô, thua 1 ô.** +0,168 (UNSW mc), +0,097 (ToN mc), +0,012
   (ToN bin), +0,006 (UNSW bin), +0,004 (CSE bin); **−0,016 trên CSE mc** (HGB 0,6712
   < `sage` 0,6926). Không gộp thành một claim duy nhất.
7. **Biến thể sụp đổ không triển khai được**: `sage` trên BoT-IoT có tỉ lệ báo động giả
   **25,02 %** (250.189/triệu flow benign, recall benign 0,75), `edge_mlp` 0,19 % và
   `sage_edge` 0,32 %.
8. **Model flow-only không chuyển giao xuyên mạng**: nội bộ 5/5 lần vượt bộ dự đoán hằng
   số, **xuyên mạng chỉ 4/15**; ToN→BoT 0,035 và ToN→UNSW 0,199 so với baseline 0,499
   và 0,490.


### 0.1 Bảng mức phụ thuộc endpoint (cùng model, cùng ngân sách)

| Ô | n | split khóa | endpoint-holdout | mức giảm | KTC 2.5% | KTC 97.5% |
|---|---|---|---|---|---|---|
| ToN · binary · mlp_h128_1l | 3 | 0.9712 | 0.9708 | 0.0004 | -0.0011 | 0.0018 |
| ToN · binary · mlp_h273_2l | 3 | 0.9817 | 0.9807 | 0.0010 | 0.0002 | 0.0020 |
| ToN · multiclass · mlp_h128_1l | 3 | 0.6963 | 0.4088 | 0.2874 | 0.2734 | 0.2975 |
| ToN · multiclass · mlp_h273_2l | 3 | 0.7504 | 0.4396 | 0.3109 | 0.3060 | 0.3186 |
| UNSW · binary · mlp_h128_1l | 3 | 0.9641 | 0.9802 | -0.0161 | -0.0166 | -0.0153 |
| UNSW · binary · mlp_h273_2l | 3 | 0.9674 | 0.9834 | -0.0160 | -0.0167 | -0.0153 |
| UNSW · multiclass · mlp_h128_1l | 3 | 0.4218 | 0.4053 | 0.0165 | -0.0128 | 0.0382 |
| UNSW · multiclass · mlp_h273_2l | 3 | 0.4697 | 0.4616 | 0.0081 | -0.0234 | 0.0291 |

### 0.2 Tỉ lệ báo động giả của detector binary

| Dataset | Model | tỉ lệ báo động giả | FP/triệu flow | recall Benign |
|---|---|---|---|---|
| BoT-IoT | edge_mlp | 0.0019 | 1853.4182 | 0.9981 |
| BoT-IoT | sage | 0.2502 | 250189.3870 | 0.7498 |
| BoT-IoT | sage_edge | 0.0032 | 3236.1269 | 0.9968 |
| CSE-CIC | edge_mlp | 0.0019 | 1876.6604 | 0.9981 |
| CSE-CIC | sage | 0.0020 | 1957.8883 | 0.9980 |
| CSE-CIC | sage_edge | 0.0012 | 1235.1339 | 0.9988 |
| ToN | edge_mlp | 0.0326 | 32560.4022 | 0.9674 |
| ToN | sage | 0.0155 | 15496.3593 | 0.9845 |
| ToN | sage_edge | 0.0147 | 14707.8415 | 0.9853 |
| UNSW | edge_mlp | 0.0060 | 5945.5104 | 0.9940 |
| UNSW | sage | 0.0050 | 4947.1873 | 0.9950 |
| UNSW | sage_edge | 0.0050 | 5005.1179 | 0.9950 |

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


Thiết kế `holdout`: endpoint chia 70 % train / 30 % holdout; flow chỉ giữ khi **cả hai** endpoint cùng nhóm; flow holdout chia 1/3 val, 2/3 test. Thiết kế component-level đã được thử và **thất bại cổng khả thi** (UNSW: test còn 0,24 % flow, thiếu 3 lớp, do endpoint graph có một thành phần khổng lồ).


## 8. Phase B — capacity, topology và comparator bảng

### 8.1 Ma trận test macro-F1 (trung bình qua seed)

| cell | `edge_mlp` (5.4k) | `mlp_h128_1l` control (5.4k) | `mlp_h128_2l` (23k) | `mlp_h273_2l` capacity-matched (88k) | `sage` (87k) | `sage_edge` (87k) | HistGB | ExtraTrees | RandomForest |
|---|---|---|---|---|---|---|---|---|---|
| BoT-IoT · binary | 0.9003 | 0.9113 | — | 0.9354 | 0.8043 | 0.8818 | — | — | — |
| BoT-IoT · multiclass | 0.8169 | 0.8288 | — | 0.8364 | 0.5512 | 0.8260 | — | — | — |
| CSE-CIC · binary | 0.9837 | 0.9840 | — | 0.9876 | 0.9850 | 0.9860 | 0.9894 | — | — |
| CSE-CIC · multiclass | 0.6686 | 0.6585 | — | 0.6828 | 0.6926 | 0.6894 | 0.6712 | — | — |
| ToN · binary | 0.9707 | 0.9712 | — | 0.9818 | 0.9775 | 0.9802 | 0.9928 | — | — |
| ToN · multiclass | 0.7040 | 0.7013 | 0.7296 | 0.7494 | 0.7494 | 0.7658 | 0.8664 | — | — |
| UNSW · binary | 0.9645 | 0.9642 | 0.9662 | 0.9674 | 0.9702 | 0.9699 | 0.9761 | 0.9803 | 0.9828 |
| UNSW · multiclass | 0.4141 | 0.4215 | 0.4425 | 0.4628 | 0.4898 | 0.4885 | 0.6536 | 0.6698 | 0.6683 |

Các cột `mlp_*`, `HistGB`, `ExtraTrees`, `RandomForest` do công việc này chạy trên **đúng split**; `edge_mlp`/`sage`/`sage_edge` lấy từ 120 run lưu trữ. Scaler đọc từ chính `preprocessor.json` của run GNN, **không** fit lại.

### 8.2 Contrast ghép cặp theo seed


**Đối chứng harness vs archive**  (`mlp_h128_1l − edge_mlp`)

| Ô | A | B | n | Δ | SD | KTC 2.5% | KTC 97.5% | KTC không chứa 0 | dz | p Wilcoxon |
|---|---|---|---|---|---|---|---|---|---|---|
| BoT-IoT · binary | mlp_h128_1l | edge_mlp | 3 | -0.0001 | 0.0071 | -0.0082 | 0.0051 | False | -0.0094 | 1.0000 |
| BoT-IoT · multiclass | mlp_h128_1l | edge_mlp | 3 | 0.0141 | 0.0090 | 0.0052 | 0.0231 | True | 1.5713 | 0.2500 |
| CSE-CIC · binary | mlp_h128_1l | edge_mlp | 5 | 0.0003 | 0.0011 | -0.0003 | 0.0013 | False | 0.2922 | 1.0000 |
| CSE-CIC · multiclass | mlp_h128_1l | edge_mlp | 5 | -0.0101 | 0.0141 | -0.0218 | -0.0005 | True | -0.7168 | 0.1875 |
| ToN · binary | mlp_h128_1l | edge_mlp | 5 | 0.0005 | 0.0005 | 0.0001 | 0.0008 | True | 0.9648 | 0.1250 |
| ToN · multiclass | mlp_h128_1l | edge_mlp | 5 | -0.0027 | 0.0087 | -0.0094 | 0.0039 | False | -0.3118 | 0.6250 |
| UNSW · binary | mlp_h128_1l | edge_mlp | 5 | -0.0003 | 0.0003 | -0.0005 | -0.0001 | True | -1.2134 | 0.1250 |
| UNSW · multiclass | mlp_h128_1l | edge_mlp | 5 | 0.0074 | 0.0112 | -0.0012 | 0.0161 | False | 0.6669 | 0.3125 |

**Capacity/độ sâu thuần (KHÔNG topology)**  (`mlp_h273_2l − edge_mlp`)

| Ô | A | B | n | Δ | SD | KTC 2.5% | KTC 97.5% | KTC không chứa 0 | dz | p Wilcoxon |
|---|---|---|---|---|---|---|---|---|---|---|
| BoT-IoT · binary | mlp_h273_2l | edge_mlp | 3 | 0.0240 | 0.0113 | 0.0140 | 0.0362 | True | 2.1263 | 0.2500 |
| BoT-IoT · multiclass | mlp_h273_2l | edge_mlp | 3 | 0.0217 | 0.0105 | 0.0105 | 0.0313 | True | 2.0721 | 0.2500 |
| CSE-CIC · binary | mlp_h273_2l | edge_mlp | 5 | 0.0038 | 0.0010 | 0.0031 | 0.0046 | True | 3.8576 | 0.0625 |
| CSE-CIC · multiclass | mlp_h273_2l | edge_mlp | 5 | 0.0142 | 0.0095 | 0.0070 | 0.0220 | True | 1.4960 | 0.0625 |
| ToN · binary | mlp_h273_2l | edge_mlp | 5 | 0.0111 | 0.0007 | 0.0106 | 0.0116 | True | 16.2188 | 0.0625 |
| ToN · multiclass | mlp_h273_2l | edge_mlp | 5 | 0.0454 | 0.0061 | 0.0402 | 0.0497 | True | 7.4547 | 0.0625 |
| UNSW · binary | mlp_h273_2l | edge_mlp | 5 | 0.0029 | 0.0006 | 0.0025 | 0.0034 | True | 4.7858 | 0.0625 |
| UNSW · multiclass | mlp_h273_2l | edge_mlp | 5 | 0.0488 | 0.0165 | 0.0344 | 0.0594 | True | 2.9563 | 0.0625 |

****Topology ở capacity khớp****  (`sage − mlp_h273_2l`)

| Ô | A | B | n | Δ | SD | KTC 2.5% | KTC 97.5% | KTC không chứa 0 | dz | p Wilcoxon |
|---|---|---|---|---|---|---|---|---|---|---|
| BoT-IoT · binary | sage | mlp_h273_2l | 3 | -0.1317 | 0.0227 | -0.1461 | -0.1056 | True | -5.8101 | 0.2500 |
| BoT-IoT · multiclass | sage | mlp_h273_2l | 3 | -0.2666 | 0.0115 | -0.2798 | -0.2593 | True | -23.2506 | 0.2500 |
| CSE-CIC · binary | sage | mlp_h273_2l | 5 | -0.0026 | 0.0014 | -0.0036 | -0.0014 | True | -1.8563 | 0.0625 |
| CSE-CIC · multiclass | sage | mlp_h273_2l | 5 | 0.0098 | 0.0410 | -0.0236 | 0.0372 | False | 0.2388 | 0.8125 |
| ToN · binary | sage | mlp_h273_2l | 5 | -0.0043 | 0.0013 | -0.0052 | -0.0031 | True | -3.2551 | 0.0625 |
| ToN · multiclass | sage | mlp_h273_2l | 5 | -0.0000 | 0.0073 | -0.0055 | 0.0054 | False | -0.0004 | 1.0000 |
| UNSW · binary | sage | mlp_h273_2l | 5 | 0.0029 | 0.0014 | 0.0020 | 0.0041 | True | 2.1093 | 0.0625 |
| UNSW · multiclass | sage | mlp_h273_2l | 5 | 0.0269 | 0.0163 | 0.0163 | 0.0413 | True | 1.6568 | 0.0625 |

**Topology + edge ở head, capacity khớp**  (`sage_edge − mlp_h273_2l`)

| Ô | A | B | n | Δ | SD | KTC 2.5% | KTC 97.5% | KTC không chứa 0 | dz | p Wilcoxon |
|---|---|---|---|---|---|---|---|---|---|---|
| BoT-IoT · binary | sage_edge | mlp_h273_2l | 3 | -0.0507 | 0.0140 | -0.0609 | -0.0347 | True | -3.6124 | 0.2500 |
| BoT-IoT · multiclass | sage_edge | mlp_h273_2l | 3 | -0.0193 | 0.0203 | -0.0357 | 0.0034 | False | -0.9492 | 0.5000 |
| CSE-CIC · binary | sage_edge | mlp_h273_2l | 5 | -0.0016 | 0.0014 | -0.0028 | -0.0007 | True | -1.1519 | 0.0625 |
| CSE-CIC · multiclass | sage_edge | mlp_h273_2l | 5 | 0.0065 | 0.0240 | -0.0145 | 0.0219 | False | 0.2719 | 0.6250 |
| ToN · binary | sage_edge | mlp_h273_2l | 5 | -0.0016 | 0.0009 | -0.0022 | -0.0008 | True | -1.7791 | 0.0625 |
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
| ToN · binary | sage | edge_mlp | 5 | 0.0068 | 0.0014 | 0.0056 | 0.0078 | True | 4.8478 | 0.0625 |
| ToN · multiclass | sage | edge_mlp | 5 | 0.0454 | 0.0056 | 0.0412 | 0.0499 | True | 8.0776 | 0.0625 |
| UNSW · binary | sage | edge_mlp | 5 | 0.0058 | 0.0010 | 0.0051 | 0.0066 | True | 5.5768 | 0.0625 |
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
| UNSW · multiclass | sage_edge | sage | 5 | -0.0012 | 0.0078 | -0.0077 | 0.0052 | False | -0.1602 | 0.4375 |

**Bảng mạnh vs `edge_mlp`**  (`hist_gradient_boosting − edge_mlp`)

| Ô | A | B | n | Δ | SD | KTC 2.5% | KTC 97.5% | KTC không chứa 0 | dz | p Wilcoxon |
|---|---|---|---|---|---|---|---|---|---|---|
| CSE-CIC · binary | hist_gradient_boosting | edge_mlp | 3 | 0.0059 | 0.0009 | 0.0050 | 0.0068 | True | 6.5656 | 0.2500 |
| CSE-CIC · multiclass | hist_gradient_boosting | edge_mlp | 3 | 0.0105 | 0.0046 | 0.0073 | 0.0157 | True | 2.3003 | 0.2500 |
| ToN · binary | hist_gradient_boosting | edge_mlp | 2 | 0.0223 | 0.0002 | 0.0222 | 0.0224 | True | 136.3543 | 0.5000 |
| ToN · multiclass | hist_gradient_boosting | edge_mlp | 3 | 0.1621 | 0.0019 | 0.1599 | 0.1635 | True | 84.4880 | 0.2500 |
| UNSW · binary | hist_gradient_boosting | edge_mlp | 3 | 0.0117 | 0.0005 | 0.0114 | 0.0122 | True | 24.7091 | 0.2500 |
| UNSW · multiclass | hist_gradient_boosting | edge_mlp | 3 | 0.2400 | 0.0108 | 0.2336 | 0.2525 | True | 22.3185 | 0.2500 |

**Bảng mạnh vs topology tốt nhất**  (`hist_gradient_boosting − sage_edge`)

| Ô | A | B | n | Δ | SD | KTC 2.5% | KTC 97.5% | KTC không chứa 0 | dz | p Wilcoxon |
|---|---|---|---|---|---|---|---|---|---|---|
| CSE-CIC · binary | hist_gradient_boosting | sage_edge | 3 | 0.0039 | 0.0017 | 0.0027 | 0.0059 | True | 2.3330 | 0.2500 |
| CSE-CIC · multiclass | hist_gradient_boosting | sage_edge | 3 | -0.0159 | 0.0076 | -0.0215 | -0.0073 | True | -2.0961 | 0.2500 |
| ToN · binary | hist_gradient_boosting | sage_edge | 2 | 0.0122 | 0.0002 | 0.0121 | 0.0124 | True | 73.6813 | 0.5000 |
| ToN · multiclass | hist_gradient_boosting | sage_edge | 3 | 0.0969 | 0.0056 | 0.0918 | 0.1028 | True | 17.4153 | 0.2500 |
| UNSW · binary | hist_gradient_boosting | sage_edge | 3 | 0.0061 | 0.0006 | 0.0054 | 0.0064 | True | 10.5758 | 0.2500 |
| UNSW · multiclass | hist_gradient_boosting | sage_edge | 3 | 0.1677 | 0.0062 | 0.1632 | 0.1748 | True | 26.9978 | 0.2500 |

### 8.3 Tổng hợp

| Contrast | số ô | Δ trung bình | ô dương | ô âm |
|---|---|---|---|---|
| Bảng mạnh vs `edge_mlp` | 6 | 0.0754 | 6 | 0 |
| Bảng mạnh vs topology tốt nhất | 6 | 0.0452 | 5 | 1 |
| Capacity/độ sâu thuần (KHÔNG topology) | 8 | 0.0215 | 8 | 0 |
| Đối chứng harness vs archive | 8 | 0.0011 | 4 | 4 |
| Đường edge trực tiếp ở head | 8 | 0.0460 | 5 | 3 |
| **Topology ở capacity khớp** | 8 | -0.0457 | 3 | 5 |
| Topology + edge ở head, capacity khớp | 8 | -0.0027 | 4 | 4 |
| Topology như archive (nhiễu capacity) | 8 | -0.0254 | 6 | 2 |

## 8b. Đánh giá trên split endpoint-holdout (cùng model, cùng ngân sách)

So sánh trực tiếp giữa split khóa (`flow_group_id`) và split endpoint-disjoint. Kiến trúc, seed và ngân sách giống hệt; khác biệt duy nhất là flow test có thể chứa endpoint **chưa từng thấy** trong train. Scaler được fit lại trên train của từng split (dân số huấn luyện khác nhau) và điều này được ghi rõ.

| Ô | n | split khóa | endpoint-holdout | mức giảm | KTC 2.5% | KTC 97.5% | giảm tương đối (%) |
|---|---|---|---|---|---|---|---|
| ToN · binary · mlp_h128_1l | 3 | 0.9712 | 0.9708 | 0.0004 | -0.0011 | 0.0018 | 0.0373 |
| ToN · binary · mlp_h273_2l | 3 | 0.9817 | 0.9807 | 0.0010 | 0.0002 | 0.0020 | 0.1010 |
| ToN · multiclass · mlp_h128_1l | 3 | 0.6963 | 0.4088 | 0.2874 | 0.2734 | 0.2975 | 41.2818 |
| ToN · multiclass · mlp_h273_2l | 3 | 0.7504 | 0.4396 | 0.3109 | 0.3060 | 0.3186 | 41.4237 |
| UNSW · binary · mlp_h128_1l | 3 | 0.9641 | 0.9802 | -0.0161 | -0.0166 | -0.0153 | -1.6695 |
| UNSW · binary · mlp_h273_2l | 3 | 0.9674 | 0.9834 | -0.0160 | -0.0167 | -0.0153 | -1.6580 |
| UNSW · multiclass · mlp_h128_1l | 3 | 0.4218 | 0.4053 | 0.0165 | -0.0128 | 0.0382 | 3.9116 |
| UNSW · multiclass · mlp_h273_2l | 3 | 0.4697 | 0.4616 | 0.0081 | -0.0234 | 0.0291 | 1.7243 |

Mức giảm **dương** nghĩa là split endpoint-holdout làm giảm chất lượng. Nếu KTC chứa 0 thì không có bằng chứng suy giảm.


## 8c. Thống kê cấu trúc cục bộ thay cho message passing (TN-5)

Đặc trưng cấu trúc **chỉ tính từ train**: số flow theo endpoint nguồn/đích, số đối tác phân biệt, số lần lặp đúng cặp endpoint, và (biến thể `_lab`) tỉ lệ tấn công theo nhãn train của endpoint. Endpoint chưa thấy nhận 0 / prior.

| Ô | Model | n | tham số | test macro-F1 | SD | best val macro-F1 |
|---|---|---|---|---|---|---|
| ToN · multiclass | mlp_struct | 3 | 89827 | 0.8188 | 0.0042 | 0.8191 |
| ToN · multiclass | mlp_struct_lab | 3 | 90373 | 0.8262 | 0.0079 | 0.8263 |
| UNSW · binary | mlp_struct | 3 | 87635 | 0.9697 | 0.0005 | 0.9725 |
| UNSW · binary | mlp_struct_lab | 3 | 88181 | 0.8367 | 0.0234 | 0.8419 |
| UNSW · multiclass | mlp_struct | 3 | 89827 | 0.5003 | 0.0038 | 0.5023 |
| UNSW · multiclass | mlp_struct_lab | 3 | 90373 | 0.4708 | 0.0150 | 0.4670 |

Đối chiếu: trên cùng ô, `mlp_h273_2l` (chỉ flow, capacity khớp) và `sage`/`sage_edge` nằm ở §8.1.


## 8d. Hồ sơ báo động giả (từ confusion matrix test đầy đủ)

Repo không lưu probability của GNN nên **không** tính lại được PR-AUC; confusion matrix đầy đủ thì có, đủ để tính tỉ lệ báo động giả tại chính operating point argmax của từng model.

| Dataset | Model | tỉ lệ báo động giả | báo động giả/triệu flow | recall Benign |
|---|---|---|---|---|
| BoT-IoT | edge_mlp | 0.0019 | 1853.4182 | 0.9981 |
| BoT-IoT | sage | 0.2502 | 250189.3870 | 0.7498 |
| BoT-IoT | sage_edge | 0.0032 | 3236.1269 | 0.9968 |
| CSE-CIC | edge_mlp | 0.0019 | 1876.6604 | 0.9981 |
| CSE-CIC | sage | 0.0020 | 1957.8883 | 0.9980 |
| CSE-CIC | sage_edge | 0.0012 | 1235.1339 | 0.9988 |
| ToN | edge_mlp | 0.0326 | 32560.4022 | 0.9674 |
| ToN | sage | 0.0155 | 15496.3593 | 0.9845 |
| ToN | sage_edge | 0.0147 | 14707.8415 | 0.9853 |
| UNSW | edge_mlp | 0.0060 | 5945.5104 | 0.9940 |
| UNSW | sage | 0.0050 | 4947.1873 | 0.9950 |
| UNSW | sage_edge | 0.0050 | 5005.1179 | 0.9950 |

Các lớp có FPR xấu nhất:

| Ô | Model | Lớp | support | FPR | FP/triệu |
|---|---|---|---|---|---|
| BoT-IoT | sage | DoS | 3333638.0000 | 0.3727 | 372660.7873 |
| CSE-CIC | edge_mlp | Infilteration | 23465.0000 | 0.3172 | 317165.0583 |
| BoT-IoT | sage | Attack | 7521175.0000 | 0.2502 | 250189.3870 |
| BoT-IoT | sage | DDoS | 3664129.0000 | 0.2336 | 233579.9625 |
| CSE-CIC | sage_edge | Infilteration | 23465.0000 | 0.1061 | 106124.9256 |
| CSE-CIC | sage | Infilteration | 23465.0000 | 0.0652 | 65188.0035 |
| BoT-IoT | sage | Reconnaissance | 522928.0000 | 0.0478 | 47803.1839 |
| CSE-CIC | edge_mlp | Benign | 3326441.0000 | 0.0428 | 42841.0890 |
| CSE-CIC | sage_edge | Benign | 3326441.0000 | 0.0396 | 39646.8399 |
| CSE-CIC | sage | Benign | 3326441.0000 | 0.0380 | 38036.0269 |
| ToN | edge_mlp | Attack | 2166556.0000 | 0.0326 | 32560.4022 |
| CSE-CIC | sage | Benign | 3326441.0000 | 0.0307 | 30693.7579 |

## 8e. Cô lập mức phụ thuộc endpoint trên split khóa (TN-10)

Giữ **nguyên** model, dữ liệu train, preprocessing và toàn bộ tập test; chỉ phân nhóm tập test theo việc endpoint của flow có xuất hiện trong train hay không. Vì mọi thứ khác không đổi, khác biệt giữa các nhóm chỉ có thể do mức quen thuộc endpoint.

| Ô | Nhóm test | số seed | số flow | macro-F1 | SD | lệch so với toàn bộ test |
|---|---|---|---|---|---|---|
| ToN · multiclass | all_test | 3 | 3385552 | 0.8664 | 0.0032 | 0.0000 |
| ToN · multiclass | both_endpoints_seen | 3 | 3319739 | 0.8663 | 0.0031 | -0.0000 |
| ToN · multiclass | neither_endpoint_seen | 3 | 621 | 0.8219 | 0.0016 | -0.0445 |
| ToN · multiclass | one_endpoint_seen | 3 | 65192 | 0.8678 | 0.0043 | 0.0014 |
| UNSW · multiclass | all_test | 1 | 478007 | 0.6522 | — | 0.0000 |
| UNSW · multiclass | both_endpoints_seen | 1 | 362724 | 0.6554 | — | 0.0033 |
| UNSW · multiclass | neither_endpoint_seen | 1 | 9533 | 0.6435 | — | -0.0087 |
| UNSW · multiclass | one_endpoint_seen | 1 | 105750 | 0.6410 | — | -0.0112 |

Kèm theo đó: split endpoint-disjoint **không** phải công cụ hợp lệ để đo mức phụ thuộc endpoint. `HistGradientBoosting` — mô hình không dùng danh tính endpoint — cũng giảm từ 0,866 xuống **0,472** trên split đó, nên phần lớn mức giảm là dịch chuyển phân bố do cách dựng split.


### 8e.1 Đối chứng bác bỏ split endpoint-disjoint

| Ô | số seed | HGB trên split endpoint-disjoint |
|---|---|---|
| ToN · multiclass | 1 | 0.4724 |

So sánh: HGB trên **split khóa** cùng ô đạt 0,8664 (§8.1).


## 8f. Chuyển giao xuyên mạng của model flow-only (TN-7)

Huấn luyện trên train của bộ nguồn (scaler của bộ nguồn), chọn checkpoint trên validation của bộ nguồn, đánh giá trên test của bộ đích sau khi đưa đặc trưng về chuẩn hoá của bộ nguồn. Baseline là bộ dự đoán hằng số; macro-F1 của nó với tỉ lệ lớp đa số *p* là *p*/(1+*p*) — **không** phải accuracy.

| Hướng | n | macro-F1 | majority macro-F1 | số lần vượt baseline |
|---|---|---|---|---|
| ToN → BoT-IoT | 3 | 0.0356 | 0.4991 | 0 |
| ToN → CSE-CIC | 3 | 0.5094 | 0.4681 | 2 |
| ToN → ToN | 3 | 0.9817 | 0.3902 | 2 |
| ToN → UNSW | 3 | 0.1781 | 0.4900 | 0 |
| UNSW → BoT-IoT | 3 | 0.3035 | 0.4991 | 1 |
| UNSW → CSE-CIC | 3 | 0.4286 | 0.4681 | 0 |
| UNSW → ToN | 3 | 0.3810 | 0.3902 | 1 |
| UNSW → UNSW | 3 | 0.9674 | 0.4900 | 3 |

**Nội bộ: 5/6 lần chạy vượt baseline hằng số. Xuyên mạng: chỉ 4/18 lần.**


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
