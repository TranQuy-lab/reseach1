# ĐỊNH VỊ LẠI VÀ SỔ ĐĂNG KÝ CLAIM (cập nhật 2026-10-01)

**Cơ sở:** kiểm toán 120 run lưu trữ + Phase B (baseline cùng capacity, comparator bảng,
endpoint-holdout). Xem `01_BAO_CAO_KIEM_TRA_DINH_VI_LAI_VI.md` và
`02_BAO_CAO_PHASE_B_VI.md`.

---

## 1. Câu chuyện bài báo sau kiểm toán

Cách kể cũ — "E-GraphSAGE đạt macro-F1 X trên bốn bộ NF-UQ-NIDS-v2, có lợi thế phụ thuộc
dataset/task" — **không còn đứng vững** sau khi kiểm toán, vì ba lý do định lượng:

1. Biến thể topology thuần (`sage`) **không hơn** MLP chỉ dùng flow feature về trung bình
   (pooled Δ = +0,0021; KTC 95 % [−0,0105; +0,0146]).
2. `sage` có **16,09× tham số** của `edge_mlp`. Khi so ở capacity khớp, **khoảng một nửa**
   lợi thế biến mất ở cả hai ô đã hoàn tất (UNSW +0,049; ToN +0,046), và phần còn lại
   **phụ thuộc dataset**: UNSW multiclass còn +0,027 (KTC [+0,016; +0,041]), ToN multiclass
   còn −0,002 (KTC [−0,011; +0,006]).
3. Comparator bảng mạnh **vượt mọi biến thể GNN** trên các ô đã chạy (UNSW multiclass:
   HistGradientBoosting 0,654 so với `sage_edge` 0,489).

Câu chuyện mới có thể bảo vệ được:

> **Phần lớn lợi thế được báo cáo của relational model trên NetFlow IDS là hiệu ứng
> capacity: khoảng một nửa tái tạo được bằng mô hình cùng capacity không hề truyền thông
> điệp. Phần còn lại sau khi khớp capacity phụ thuộc dataset (UNSW multiclass +0,027;
> ToN multiclass −0,002), nhỏ hơn nhiều so với khoảng cách tới mô hình bảng mạnh, trong
> khi độ ổn định tối ưu hóa theo seed là yếu tố phân biệt mạnh nhất giữa các biến thể.**

Đây là **bằng chứng âm có kiểm soát** — loại đóng góp mà lĩnh vực GNN-IDS đang thiếu,
và nó nhất quán với cảnh báo trong khảo sát của Zhong et al. về khoảng cách giữa kết quả
tự báo cáo và đối chứng công bằng.

## 2. Tiêu đề đề xuất

**Khuyến nghị chính:**

> **Apparent Gains of Graph Neural Networks for NetFlow Intrusion Detection Are Explained by
> Model Capacity and Optimisation Stability, Not Relational Structure**

**Phương án phụ:**

> **When Does Relational Structure Help NetFlow Intrusion Detection? A Capacity-Matched,
> Five-Seed Re-examination of E-GraphSAGE on NF-UQ-NIDS-v2**

**Không dùng:** mọi tiêu đề có "outperforms traditional machine learning", "superior GNN",
"generalizable across networks", "zero-day", "real-time".

## 3. Đóng góp có thể bảo vệ

| # | Đóng góp | Bằng chứng | Trạng thái |
|---|---|---|---|
| G1 | Protocol full-scale tái lập được: 4 dataset, 2 task, 3 model, 5 seed, 120 run, verify độc lập 120/120, replay error 0,0 | `full_verification_5seed.json` | Đã có |
| G2 | Bác bỏ giả thuyết underfitting cho kết quả âm BoT-IoT; thay bằng chẩn đoán sụp đổ tối ưu hóa theo seed | `convergence_*.csv`, `collapse_*.csv` | Đã có |
| G3 | Tách capacity khỏi topology bằng baseline cùng capacity trên đúng split | `tn1_mlp_runs.csv`, `phaseB_contrasts.csv` | Đang hoàn tất |
| G4 | Comparator bảng mạnh trên đúng split, vượt mọi biến thể GNN | `tn2_tabular_runs.csv` | Đang hoàn tất |
| G5 | Cổng khả thi cho endpoint-holdout: khả thi ở UNSW/ToN, không khả thi ở CSE-CIC/BoT-IoT | `endpoint_feasibility.json` | Đã có |
| G6 | Định lượng rằng 3 seed không đủ (SD tăng tới 6,67×; 4/24 quyết định KTC đảo) | `seed_inflation_3_vs_5.csv` | Đã có |

## 4. Sổ đăng ký claim (bản thay thế)

Quy ước trạng thái: **OK** = có artefact hỗ trợ và phát biểu không vượt bằng chứng;
**SỬA** = phải đổi cách phát biểu; **RÚT** = không được dùng.

| ID | Claim | Trạng thái | Phát biểu được phép |
|---|---|---|---|
| N01 | Topology thuần không hơn flow-only về trung bình | **OK** | "Across the eight dataset × task cells the topology-only variant showed no average macro-F1 advantage over the flow-only ablation (pooled Δ = +0.002, 95 % CI [−0.010, +0.015])." |
| N02 | Ở capacity khớp, lợi thế còn lại **phụ thuộc dataset**: UNSW multiclass +0,027 (KTC [+0,016; +0,041]); ToN multiclass −0,002 (KTC [−0,011; +0,006]) | **OK** (đang hoàn tất cho CSE/BoT) | Báo từng ô bằng `phaseB_contrasts.csv`, nêu rõ n; **không** gộp hai dataset thành một claim duy nhất. |
| N03 | BoT-IoT `sage` kém là do sụp đổ tối ưu hóa, không phải underfitting | **OK** | "In 10 of 40 `sage` runs (25 %; 5/5 on BoT-IoT binary) validation macro-F1 fell after the best checkpoint by up to 0.43, which rejects a pure underfitting account at this budget." |
| N04 | Đường edge trực tiếp ở head là yếu tố ổn định | **OK** | "Collapse rate: 0 % (`edge_mlp`), 25 % (`sage`), 5 % (`sage_edge`)." |
| N05 | Bảng mạnh vượt GNN trên các ô đã chạy | **OK** (giới hạn theo ô đã chạy) | Nêu rõ dataset/task đã chạy và cỡ mẫu; không khái quát cho bộ chưa chạy. |
| N06 | Bất ổn định dự báo điểm cuối | **OK** | Spearman ρ = −0,539 (n = 120, p = 2,2 × 10⁻¹⁰). |
| N07 | 3 seed không đủ | **OK** | "SD increased by up to 6.67× when two further seeds were added; 4 of 24 bootstrap-CI decisions changed." |
| N08 | UNSW chạy 3,67 lượt, ba bộ còn lại 2,0 | **OK** | Bắt buộc nêu như threat to validity. |
| N09 | Endpoint-holdout **khả thi về mặt lớp nhãn** ở UNSW/ToN, không ở CSE-CIC/BoT-IoT | **SỬA** — xem N21: thiết kế này **không** đo được mức phụ thuộc endpoint | Nêu TV distance nhãn và số lớp bị mất; **không** dùng nó để suy ra "unseen host" |
| N19 | ~~Phụ thuộc endpoint giải thích mức giảm 0,311 trên split endpoint-holdout~~ | **RÚT — ĐÃ BỊ BÁC BỎ bởi TN-9 và TN-10** | Xem N21 và N22. Kết luận cũ dựa trên một split bị nhiễu; phải rút khỏi mọi bản thảo. |
| N21 | **Split endpoint-disjoint không phải công cụ hợp lệ để đo mức phụ thuộc endpoint**: `HistGradientBoosting` — mô hình **không hề dùng danh tính endpoint** — cũng giảm từ 0,866 xuống **0,472** trên split đó. Vậy phần lớn mức giảm là **dịch chuyển phân bố** do cách dựng split (mất 37 % flow, TV nhãn 0,293), không phải do endpoint chưa thấy | **OK** (n=1 cho HGB, hiệu ứng rất lớn) | Phải báo kèm; **cấm** dùng TN-4/TN-9 để claim unseen-host |
| N22 | **Trên split khóa, phụ thuộc endpoint gần như không tồn tại**: chỉ **621/3.385.552 (0,018 %)** flow test ToN multiclass có **cả hai** endpoint chưa thấy trong train; trên nhóm đó HGB đạt 0,8236 so với 0,8628 toàn bộ test (−0,039, n=621, 2 seed). Nhóm "một endpoint chưa thấy" (1,9 %) không khác biệt (0,8629). Vậy tỉ lệ 99,21–100 % đã nêu trong tài liệu là **đúng ở cấp endpoint**, và nó **không** gây ra khác biệt hiệu năng đo được | **OK** | Đây là cách đóng "threat to validity" bằng **đo lường**, thay vì suy đoán |
| N23 | **Model flow-only không chuyển giao được xuyên mạng**: nội bộ đạt 0,967 (UNSW) và 0,981 (ToN), nhưng trong **15 lần chạy xuyên mạng chỉ 4 lần vượt được bộ dự đoán hằng số**; hai ô thảm họa là ToN→BoT 0,035 và ToN→UNSW 0,199 (baseline hằng số 0,499 và 0,490) | **OK** (2 nguồn × 3 đích, 2–3 seed) | Baseline majority tính bằng macro-F1 = p/(1+p), **không** phải accuracy; đã sửa nhãn metric sai ban đầu |
| N20 | **Nội dung aggregation không tái tạo được bằng tay**: cho model trung bình đặc trưng cạnh của lân cận (78 chiều, 87.356 tham số, khớp `sage`) chỉ đạt **0,4218**, *thấp hơn* cả baseline flow-only (0,4628) và `sage` (0,4898) | **OK** (n=2 UNSW mc, đang bổ sung) | Đây là **kết quả âm** phải báo; nghĩa là trọng số aggregation học được có vai trò thật, không được nói "message passing chỉ là trung bình lân cận" |
| N17 | **Thống kê cấu trúc cục bộ thay được message passing**: MLP flow-only + 5 đặc trưng cấu trúc (degree nguồn/đích, số đối tác phân biệt, số lần lặp cặp endpoint, tính từ **train only**) đạt 0,497–0,505 trên UNSW multiclass, **cao hơn** `sage` 0,490 và `sage_edge` 0,489 ở cùng mức tham số | **OK** (UNSW; ToN đang chạy) | Nêu rõ đặc trưng chỉ tính từ train, endpoint chưa thấy nhận giá trị 0 |
| N18 | **Biến thể sụp đổ cũng không dùng được trong vận hành**: `sage` trên BoT-IoT có tỉ lệ báo động giả **25,02%** (250.189 báo động giả/triệu flow benign), recall benign 0,75; `edge_mlp` 0,19%, `sage_edge` 0,32%. FPR từng lớp xấu nhất: DoS 37,3%, DDoS 23,4% | **OK** | Tính từ confusion matrix test đầy đủ tại operating point argmax của từng model; **không** claim PR-AUC vì repo không lưu probability của GNN |
| N16 | **Mức phụ thuộc endpoint phụ thuộc độ giàu đồ thị**: UNSW không suy giảm (KTC chứa 0; binary còn tốt hơn), ToN multiclass giảm **0,293** macro-F1 (0,701 → 0,409) trên split endpoint-holdout với **cùng model, cùng ngân sách** | **OK** | Đây là kết quả mạnh nhất ủng hộ cơ chế; phải báo kèm việc scaler được fit lại trên train của từng split |
| N10 | Tổng quát hóa sang host/mạng mới | **RÚT** | Chỉ nói "we construct and release an endpoint-disjoint split; model evaluation on it is future work" trừ khi chạy xong. |
| N11 | GNN vượt baseline truyền thống | **RÚT** | Bằng chứng hiện tại đi theo hướng ngược lại. |
| N12 | Tái lập trực tiếp bài báo gốc | **RÚT** | Giữ "controlled adaptation". |
| N13 | Topology là nguyên nhân nhân quả | **RÚT** | Không có can thiệp cấu trúc (rewiring) nào được chạy. |
| N14 | Zero-day / thời gian thực / cross-dataset | **RÚT** | Ngoài phạm vi protocol. |
| N15 | "Có ý nghĩa thống kê" ở cấp ô | **SỬA** | Với n = 3–5, không contrast cấp ô nào sống sót sau hiệu chỉnh Holm; chỉ báo effect size và KTC. |

## 5. Threat to validity phải ghi trong bài

1. **Ngân sách không đồng nhất:** UNSW 3,67 lượt so với 2,0 ở ba bộ còn lại.
2. **Sụp đổ tối ưu hóa theo seed** chưa được loại bỏ bằng early stopping/learning-rate
   schedule chính thức (đây là việc của protocol convergence-first).
3. **Chỉ 5 seed**, không đủ để suy luận thống kê cấp ô.
4. **Endpoint overlap** trong split chính (92,7–100 % IP holdout đã có trong train). Đã định
   lượng: gần như không ảnh hưởng ở UNSW, nhưng làm mất **0,293** macro-F1 ở ToN
   multiclass; không đo được ở CSE-CIC và BoT-IoT vì split endpoint-disjoint mất lớp.
5. **Split theo `flow_group_id`** là within-environment, không phải temporal.
6. **Tabular comparator** dùng chính đặc trưng flow mà GNN dùng; không có leakage
   IP/port, nhưng cũng không mô hình hóa được quan hệ.
7. **Giới hạn tài nguyên:** một số ô comparator chưa chạy đủ do RAM CPU.
8. **Numerical:** scaler giữa nhóm seed 11/22/33 và 44/55 lệch tương đối ≤ 5,7 × 10⁻¹¹
   (nhiễu tích lũy dấu phẩy động, không phải thay đổi tiền xử lý).

## 6. Việc bị chặn và điều kiện gỡ chặn

| Việc | Chặn bởi | Điều kiện gỡ |
|---|---|---|
| Convergence-first 8 lượt (Phase A/B của protocol hội tụ) | Không có GPU; server `vast-gpu` (202.59.206.41:11493) từ chối kết nối | Thuê/khởi động GPU ≥ 24 GiB VRAM, RAM ≥ 64 GiB |
| Graph-rewiring RR/RW/WW/WR | Cần GPU và cần model đã chứng minh hội tụ | Sau convergence-first |
| GNN trên endpoint-holdout | Cần GPU | Split đã sẵn sàng tại `data/endpoint_splits/*__ipport__holdout` |
| Learning-rate/budget sensitivity `sage` BoT-IoT | Cần GPU | Sau convergence-first |

**Ước lượng chi phí nếu thuê GPU:** ma trận 72 run ba seed mất 5,45 h trên GPU 24 GiB.
Convergence-first 8 lượt cho UNSW + ToN + BoT với 3 model × 3–5 seed ước tính
15–30 h GPU, tức khoảng **$8–20** ở mức $0,6/giờ. Đây là con số cần xác nhận bằng
benchmark một epoch trước khi cam kết.
