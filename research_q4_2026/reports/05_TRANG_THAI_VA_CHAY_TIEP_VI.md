# LOG ĐÓNG GÓI — TRẠNG THÁI & HƯỚNG DẪN CHẠY TIẾP

**Lần đóng gói:** 2026-10-01 (lần 2, sau khi hoàn tất Gate B và toàn bộ ma trận)  
**Lý do:** người dùng yêu cầu dừng máy và đóng gói để chạy tiếp sau.  
**Nhánh đã đẩy:** `research/2026q4-evidence-audit` trên `TranQuy-lab/reseach1`  
**Thư mục làm việc:** `/home/noble-tran/nghiencuu/egs-nfuq-2026Q4`

---

## 0. Tóm tắt một câu

Đã kiểm toán 120 run lưu trữ, bác bỏ giả thuyết underfitting, đóng **Gate B** (tabular) cho
**3/4 bộ dữ liệu**, đo sạch **Gate C** (endpoint) và **rút lại một claim sai của chính mình**,
hoàn tất **Gate D**, và chạy **8 ô × 2 baseline cùng capacity** — tất cả trên CPU, không sửa
một dòng mã nguồn nào.

---

## 1. Ma trận kết quả cuối (test macro-F1, trung bình qua seed)

| Ô | `edge_mlp` (5,4k) | `mlp_h128_1l` | `mlp_h128_2l` | **`mlp_h273_2l` (khớp capacity)** | `sage` | `sage_edge` | HistGB | ExtraTrees | RF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UNSW · mc | 0,4141 | 0,4215 | 0,4425 | **0,4628** | 0,4898 | 0,4885 | **0,6536** | **0,6698** | **0,6683** |
| UNSW · bin | 0,9645 | 0,9642 | 0,9662 | **0,9674** | 0,9702 | 0,9699 | 0,9761 | 0,9803 | **0,9828** |
| ToN · mc | 0,7040 | 0,7013 | 0,7296 | **0,7494** | 0,7494 | 0,7658 | **0,8664** | — | — |
| ToN · bin | 0,9707 | 0,9712 | — | **0,9818** | 0,9775 | 0,9802 | **0,9928** | — | — |
| CSE-CIC · mc | 0,6686 | 0,6585 | — | **0,6828** | **0,6926** | **0,6894** | 0,6712 | — | — |
| CSE-CIC · bin | 0,9837 | 0,9840 | — | 0,9876 | 0,9850 | 0,9860 | **0,9894** | — | — |
| BoT-IoT · mc | 0,8169 | 0,8288 | — | **0,8400** | **0,5512** | 0,8260 | không khả thi | — | — |
| BoT-IoT · bin | 0,9003 | 0,9113 | — | **0,9328** | **0,8043** | 0,8818 | không khả thi | — | — |

## 2. Ba contrast then chốt (ghép cặp theo seed)

### 2.1 Hiệu ứng capacity (`mlp_h273_2l − edge_mlp`) — **dương ở CẢ 8 ô, n=5 mỗi ô**
+0,0488 (UNSW mc) · +0,0454 (ToN mc) · +0,0325 (BoT bin) · +0,0231 (BoT mc) · +0,0142
(CSE mc) · +0,0111 (ToN bin) · +0,0038 (CSE bin) · +0,0029 (UNSW bin). Mọi KTC 95 %
không chứa 0.

### 2.1b Số seed cuối
`mlp_h273_2l` (contrast trung tâm) đạt **n=5 ở 7/8 ô** (BoT và CSE được nâng từ 3 lên 5
trong phiên này); `mlp_h128_1l` n=5 ở 6/8 ô, n=3 ở BoT. Việc nâng seed đã **làm co** hai
contrast: CSE mc +0,025 → **+0,010**; BoT mc −0,267 → **−0,289**.

### 2.2 Topology ở capacity khớp (`sage − mlp_h273_2l`)

| Ô | Δ | KTC 95 % | Đọc |
|---|---:|---|---|
| BoT · mc | **−0,2888** | [−0,345; −0,254] | kém hơn mạnh (n=5) |
| BoT · bin | **−0,1285** | [−0,142; −0,115] | kém hơn mạnh (n=5) |
| ToN · bin | −0,0043 | [−0,005; −0,003] | kém hơn |
| CSE-CIC · bin | −0,0026 | [−0,004; −0,001] | kém hơn (n=5) |
| ToN · mc | −0,0000 | [−0,006; +0,005] | không phân biệt được |
| CSE-CIC · mc | +0,0098 | [−0,024; +0,037] | không phân biệt được (n=5) |
| UNSW · mc | +0,0269 | [+0,016; +0,041] | hơn nhẹ |
| UNSW · bin | +0,0029 | [+0,002; +0,004] | hơn nhẹ |

**Không ô nào topology dương mạnh.**

### 2.3 Bảng mạnh so với GNN tốt nhất
+0,168 (UNSW mc) · +0,097 (ToN mc) · +0,012 (ToN bin) · +0,006 (UNSW bin) · +0,004
(CSE-CIC bin) · **−0,016 (CSE-CIC mc)** → thắng 5/6 ô, **thua 1 ô**.

## 3. Các phát hiện cơ chế khác

| Thí nghiệm | Kết quả | Ý nghĩa |
|---|---|---|
| Sụp đổ tối ưu hóa (120 run) | `edge_mlp` 0/40 · `sage` 10/40 (25 %) · `sage_edge` 2/40 (5 %); Fisher exact Holm p = 0,0031 | Underfitting bị bác bỏ; BoT binary 5/5 seed sụp |
| Báo động giả (TN-6) | `sage` BoT-IoT **25,02 %** (250.189 FP/triệu); `edge_mlp` 0,19 % | Biến thể sụp đổ **không triển khai được** |
| Thống kê đồ thị | BoT 78,65 cạnh/node & 91,6 % lặp; UNSW/CSE 1,70–1,78 & ~20 % | Giải thích vì sao topology chỉ có thể giúp ở BoT |
| TN-5 đặc trưng cấu trúc | UNSW mc 0,5003 > `sage` 0,4898; ToN mc 0,8188 > `sage_edge` 0,7658 | **Lặp lại trên 2 bộ**: đếm lân cận thắng message passing |
| TN-8 trung bình lân cận | 0,4266 < flow-only 0,4628 (UNSW mc) | **Kết quả âm**: không được nói "message passing chỉ là trung bình lân cận" |
| TN-7 chuyển giao xuyên mạng | nội bộ **5/5** vượt baseline hằng số; xuyên mạng **4/15**; ToN→BoT 0,035 vs baseline 0,499 | Model flow-only **không** chuyển giao được |
| TN-10 exposure endpoint | BoT **0 %** · ToN 0,018 % · CSE 0,136 % · UNSW 2,0 % | Chồng lấp endpoint **không** thổi phồng kết quả |
| TN-10 hiệu ứng | ToN −0,044 (621 flow) · UNSW −0,009 (9.533 flow) | Phụ thuộc endpoint **không đáng kể** |
| TN-9 đối chứng | HGB 0,866 → **0,472** trên split endpoint-disjoint | Split đó **bị nhiễu** — đã **rút lại** claim cũ |

## 4. Sự cố toàn vẹn dữ liệu đã phát hiện & xử lý

| Sự cố | Ảnh hưởng | Xử lý |
|---|---|---|
| `scripts/23` ghi **đè** file raw TN-10 mỗi lần chạy → mất số raw của ToN khi chạy UNSW | Mất bằng chứng raw cho các số ToN đã báo | ✅ Đã sửa sang **append**; ✅ đang **tái tạo** TN-10 ToN; bản UNSW tách riêng ở `results/tn10_endpoint_isolation_unsw_snapshot.csv` |
| `scripts/09` ghi **đè** `tn2_tuning.csv` mỗi lần chạy → mất log cấu hình đã thử trước đó | Mất log "cấu hình thua" của UNSW/ToN | ✅ Đã sửa sang **append**. Cấu hình **thắng** vẫn nguyên trong `tn2_tabular_runs.csv` |
| Thêm cột `peak_rss_gib` giữa chừng → CSV 16→17 cột, pandas không đọc được | TN-2 tạm thời không đọc được | ✅ Đã sửa; ghi nhớ: **thêm cột thì phải ghi ra file mới** |
| Đĩa đầy 99 % | Nguy cơ hỏng khi ghi file | ✅ Đã xóa bản đệm tái tạo được (§6) |

## 5. Việc còn lại (ưu tiên giảm dần)

| # | Việc | Lệnh | Ước tính |
|---|---|---|---|
| 1 | **Tái tạo raw TN-10 cho ToN** (đang chạy), rồi gộp với snapshot UNSW | `scripts/23_tn10_endpoint_isolation.py --datasets NF-ToN-IoT-v2 --tasks multiclass --seeds 11 22 33` | ~30 phút |
| 2 | TN-10 cho **CSE-CIC** (5.138 flow chưa thấy endpoint) | cùng script, `--datasets NF-CSE-CIC-IDS2018-v2` | ~1,5 h, một mình |
| 3 | TN-8 cho **ToN** — mở rộng kết quả âm sang bộ thứ hai | `scripts/21_tn8_neighbour_mean_smoothing.py --datasets NF-ToN-IoT-v2 --tasks multiclass --seeds 11 22 33` | ~2 h |
| 4 | TN-5 nốt: UNSW binary (2 seed), ToN binary | `scripts/18_tn5_structural_features.py --datasets NF-UNSW-NB15-v2 NF-ToN-IoT-v2 --tasks binary ...` | ~2 h |
| 5 | TN-9 nốt seed 22/33 (cần tái tạo `work/holdout_features`) | `scripts/16_holdout_cpu_evaluation.py` rồi `scripts/22_tn9_holdout_tabular.py` | ~1,5 h |
| 6 | TN-2 thêm ExtraTrees/RandomForest cho **ToN** | `scripts/09_tn2_tabular_comparators.py --datasets NF-ToN-IoT-v2 --models extra_trees random_forest` | ~2 h, cần RAM |
| 6b | ~~CSE-CIC lên 5 seed~~ | — | ✅ **đã xong**: contrast topology co từ +0,025 xuống +0,010 |
| 7 | **BoT-IoT tabular** | — | ❌ Không khả thi ở 13 GiB (cần ~14 GiB) |
| 8 | Convergence-first 8 lượt · rewiring RR/RW/WW/WR · GNN trên endpoint-holdout | — | ⛔ **Cần GPU ≥ 24 GiB VRAM** |

## 6. Dữ liệu: cái gì trong Git, cái gì không

**Trong Git** (`research_q4_2026/`): toàn bộ script, **58 file kết quả**, 10 hình, 5 báo cáo,
1 protocol — **nguồn duy nhất của mọi con số**; `reports/02` sinh tự động từ đó.

**Không trong Git** (tái tạo được):

| Đường dẫn | Dung lượng | Trạng thái |
|---|---:|---|
| `data/processed_four` | 3,1 GB | còn |
| `data/full_splits` | 2,9 GB | còn |
| `work/features` | 12 GB | còn |
| `work/struct` | 0,15 GB | còn |
| `work/nbr` | 0,7 GB | còn |
| `work/holdout_features` | 3,6 GB | ❌ **đã xóa** → tái tạo bằng `scripts/16` |
| `data/endpoint_splits` | 1,9 GB | ❌ **đã xóa** → tái tạo bằng `scripts/10 --strategy holdout` |

**Tái tạo từ đầu** nếu đổi máy: tải 4 Parquet từ
`https://media.githubusercontent.com/media/TranQuy-lab/reseach1/main/<tên-file>.parquet`,
rồi `nids_minibatch.prepare` để tạo split, rồi `scripts/07` + `scripts/07b_fix_binary_labels.py`
(**07b bắt buộc**).

## 7. Ghi chú tài nguyên (đo được, không ước lượng)

| Việc | RAM đỉnh | Ghi chú |
|---|---:|---|
| HGB trên ToN-IoT (11,9 M dòng) | **10,3 GiB** | phải chạy một mình |
| HGB trên CSE-CIC (13,2 M dòng) | **10,5 GiB** | phải chạy một mình |
| HGB trên BoT-IoT (26,4 M dòng) | ~14 GiB | **không chạy được** trên máy 13 GiB |
| ExtraTrees/RF trên ToN-IoT (11,9 M dòng) | ước tính 15–30 GiB cho rừng cây | **không khả thi**; chỉ chạy được trên UNSW (1,67 M). Ghi là giới hạn tài nguyên, **không** subsample ngầm (quy tắc 7 của protocol) |
| TN-1 BoT (MLP) | ~2,5 GiB | 500–1.050 s/run |
| MLP trên UNSW | ~0,5 GiB | 10–45 s/run |

Máy: 16 luồng, 13 GiB RAM, **không GPU**. Đĩa còn **6,9 GB** sau khi dọn.

## 8. Cam kết tuân thủ

```bash
git diff --stat main..HEAD -- src/ tests/ research/ notebooks/
# → rỗng: KHÔNG một file mã nguồn nào bị sửa
```

Toàn bộ công việc nằm trong `research_q4_2026/`, đúng yêu cầu đề bài.

## 9. Danh mục script

| Script | Vai trò |
|---|---|
| `01_inventory_audit.py` | Kiểm kê + kiểm chứng 120 run |
| `02_convergence_diagnostics.py` | Phân loại learning curve (plateau / decay / still rising) |
| `03_five_seed_stats.py` | Thống kê ghép cặp + meta-analysis + hiệu chỉnh Holm |
| `04_rare_class_and_instability.py` | Lớp hiếm + bất ổn định |
| `05_collapse_sensitivity.py` | Sụp đổ tối ưu hóa + độ nhạy xếp hạng |
| `06_figures.py` | Hình Phase A |
| `07` + `07b` | Ma trận đặc trưng (**07b bắt buộc**: nhãn nhị phân là số nguyên) |
| `08_tn1_capacity_matched_mlp.py` | **TN-1** baseline cùng capacity |
| `09_tn2_tabular_comparators.py` | **TN-2** comparator bảng (Gate B) |
| `10_endpoint_disjoint_split.py` | Split endpoint-disjoint (3 chiến lược) |
| `11_phaseB_analysis.py` | Tổng hợp Phase B + hình 5/6/9 |
| `12_endpoint_feasibility.py` | Cổng khả thi endpoint |
| `13_build_report_02.py` | **Sinh báo cáo kết quả tự động** |
| `14_collapse_robustness.py` | Độ bền kết luận sụp đổ |
| `15_graph_statistics.py` | Thống kê đồ thị |
| `16_holdout_cpu_evaluation.py` | **TN-4** đánh giá trên split endpoint |
| `17_holdout_comparison.py` | So sánh split khóa vs endpoint |
| `18_tn5_structural_features.py` | **TN-5** đặc trưng cấu trúc |
| `19_tn6_operational_metrics.py` | **TN-6** tỉ lệ báo động giả |
| `20_tn7_cross_dataset_transfer.py` | **TN-7** chuyển giao xuyên mạng |
| `21_tn8_neighbour_mean_smoothing.py` | **TN-8** trung bình đặc trưng lân cận |
| `22_tn9_holdout_tabular.py` | **TN-9** HGB trên split endpoint |
| `23_tn10_endpoint_isolation.py` | **TN-10** cô lập endpoint trên split khóa |
