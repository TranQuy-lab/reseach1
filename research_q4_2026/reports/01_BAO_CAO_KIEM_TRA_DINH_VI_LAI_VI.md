# BÁO CÁO KIỂM TRA, CHẨN ĐOÁN VÀ ĐỊNH VỊ LẠI
## Đề tài: E-GraphSAGE trên bốn bộ NetFlow NF-UQ-NIDS-v2

**Ngày:** 2026-10-01  
**Cơ sở:** `TranQuy-lab/reseach1`, commit `bb2df1e` (`main`)  
**Thư mục tính toán:** `egs-nfuq-2026Q4/` (tách biệt hoàn toàn, không sửa file mã nguồn nào)  
**Trạng thái:** phân tích hậu nghiệm (exploratory/post-hoc) trên 120 run đã lưu trữ; **không thay thế** protocol xác nhận khóa trước.

---

## 0. Tóm tắt điều hành

Tôi đã kiểm toán lại toàn bộ bằng chứng lưu trữ bằng tính toán độc lập, khai thác một nguồn dữ liệu mà các báo cáo trước **chưa dùng**: 120 file `history.json` chứa **learning curve validation theo từng bước** của mọi run. Kết quả làm thay đổi đáng kể cách diễn giải đề tài.

**Bốn kết luận điều chỉnh quan trọng nhất:**

| # | Kết luận cũ trong repo | Kết luận sau kiểm toán |
|---|---|---|
| 1 | "Topology (`sage`) có lợi thế dataset/task-dependent" | **Không có bằng chứng `sage` hơn `edge_mlp` về trung bình.** Pooled Δ = +0,0021 macro-F1, KTC 95% [−0,0105; +0,0146] — chứa 0. Trong 8 ô, `sage` chỉ hơn ở 6 ô với biên độ nhỏ. |
| 2 | "`sage` BoT-IoT kém do underfitting / ngân sách hai lượt" | **Bác bỏ giả thuyết underfitting.** 10/40 run `sage` (25%) bị **sụp đổ tối ưu hóa** (optimization collapse), BoT-IoT binary là **5/5 = 100%**. Validation macro-F1 *giảm* sau checkpoint tốt nhất (ví dụ 0,602 → 0,175 ở seed 11), không phải còn tăng. |
| 3 | "`sage_edge` là ablation mở rộng phụ" | **`sage_edge` là biến thể duy nhất ổn định.** Tỉ lệ sụp đổ 5% so với 25% của `sage`; đây là **cơ chế**, không phải chi tiết phụ: đường truyền đặc trưng flow trực tiếp vào classifier head hoạt động như một neo ổn định. |
| 4 | "3 seed là đủ cho mô tả; 5 seed là khuyến nghị" | **3 seed không đủ và đã tạo kết luận sai.** SD tăng tới **6,67×** khi thêm seed 44/55; **4/24 quyết định dựa trên KTC bị đảo** giữa 3 và 5 seed. |

**Phát hiện phương pháp luận nghiêm trọng:** UNSW-NB15 được huấn luyện **3,67 lượt** trong khi ba bộ còn lại đúng **2,0 lượt**, do ràng buộc `min_train_steps=1500` ghi đè ngân sách hai lượt. **30/30 run UNSW có checkpoint tốt nhất ở lượt ≥ 3.** So sánh xuyên bộ theo "protocol hai lượt" vì vậy **không hợp lệ** cho UNSW.

**Hướng tiếp theo được chọn:** chuyển trọng tâm từ "topology có giúp không" sang **"điều gì quyết định kết quả quan sát được"** — cụ thể là đóng ba lỗ hổng bằng chứng có thể chạy trên CPU: (a) baseline **cùng capacity** và (b) **comparator bảng mạnh** trên đúng split, và (c) **chẩn đoán sụp đổ tối ưu hóa** làm kết quả chính.

---

## 1. Kiểm tra: những gì đã xác minh được

### 1.1 Tính toàn vẹn thiết kế

| Hạng mục | Kết quả |
|---|---|
| Số run GNN lưu trữ | 120 (72 ở `full_runs`, 48 ở `full_runs_seeds44_55`) |
| Thiết kế 4 dataset × 2 task × 3 model × 5 seed | **Đầy đủ**, mọi ô đúng 5 seed |
| Artifact đủ (`config`, `history`, `metrics`, `preprocessor`, `model.pt`) | 120/120, **không thiếu file nào** |
| `runs.csv` công bố vs `metrics.json` từng run | Khớp tới **1,11 × 10⁻¹⁶** trên mọi cột metric |
| `summary.csv` (mean/std) tính lại từ `runs.csv` | Khớp tới **1,14 × 10⁻¹⁶** |
| `checkpoint_parameter_max_abs_error` | **0,0** trên cả 120 run |
| Verifier độc lập | `passed=true`, `runs_checked=120`, replay error `0,0`, metric error `1,11 × 10⁻¹⁶` |
| Số artifact có SHA-256 trong biên bản verify | 120 |

Đây là mức tái lập tính toán **rất mạnh** và là tài sản thực sự của đề tài. Không có sai lệch số liệu nào giữa văn bản và artifact.

### 1.2 Ngân sách huấn luyện thực tế (phát hiện chưa được ghi nhận)

Từ `config.json` của từng run:

| Dataset | `steps_per_full_pass` | `steps_ran` | Lượt hiệu dụng | Ràng buộc chi phối |
|---|---:|---:|---:|---|
| NF-UNSW-NB15-v2 | 409 | 1.500 | **3,667** | `min_train_steps` |
| NF-ToN-IoT-v2 | 2.896 | 5.792 | 2,000 | `dataset_passes` |
| NF-CSE-CIC-IDS2018-v2 | 3.230 | 6.460 | 2,000 | `dataset_passes` |
| NF-BoT-IoT-v2 | 6.455 | 12.910 | 2,000 | `dataset_passes` |

Hệ quả định lượng: **30/30 run UNSW có `best_pass` ≥ 3** (trung bình 3,6–4,0), và **60% run UNSW vẫn đang tăng ở biên ngân sách**. Nghĩa là UNSW không chỉ được huấn luyện nhiều hơn 83% về số lượt, mà phần ngân sách thêm đó **thực sự được dùng** (không phải huấn luyện thừa).

> **Điều chỉnh bắt buộc:** mọi so sánh đặt UNSW cạnh ba bộ còn lại dưới nhãn "protocol hai lượt" phải được ghi rõ là **không đồng nhất ngân sách**, hoặc tách UNSW ra khỏi bảng so sánh chính.

### 1.3 Điểm cần xác minh còn mở (provenance)

SHA-256 của bốn file Parquet trong Git LFS **không khớp** trường `source.sha256` trong `results/full_5seed/prepare_manifest.json`:

| Dataset | SHA-256 LFS (repo) | SHA-256 trong manifest |
|---|---|---|
| UNSW | `ec59445b…` | `377f6027…` |
| BoT-IoT | `3181bc41…` | `8cf253ef…` |
| ToN-IoT | `71b1c929…` | `a453e47e…` |
| CSE-CIC | `63d42756…` | `5eb5627b…` |

Số dòng đã kiểm tra khớp manifest (UNSW = 2.390.275). Hai khả năng: (a) file Parquet được tái xuất sau khi chạy 120 run nhưng giữ nguyên nội dung; (b) manifest ghi checksum của bản sao khác. **Đang xử lý:** tái tạo split cục bộ từ chính bốn file LFS này và so khớp số dòng từng split cùng tỉ lệ chồng lấp IP với manifest lưu trữ — nếu khớp, nội dung được chứng minh tương đương.

---

## 2. Chẩn đoán learning curve: underfitting hay sụp đổ tối ưu hóa?

### 2.1 Phương pháp

Với mỗi run, tôi phân loại quỹ đạo validation macro-F1 theo một quy tắc **chỉ dùng validation, không dùng test**:

- **R1 — còn tăng ở biên:** checkpoint tốt nhất là lần đánh giá cuối và độ dốc cuối dương ⇒ nguy cơ underfitting.
- **R2 — suy giảm sau đỉnh:** validation macro-F1 giảm > 0,005 so với đỉnh ⇒ bất ổn định/overshoot, **không phải** underfitting.
- **R3 — plateau:** đạt dải phẳng ±0,005 và giữ được tới hết ngân sách.

### 2.2 Kết quả trên 120 run

| Phân loại | Số run | Tỉ lệ |
|---|---:|---:|
| **R2 — suy giảm sau đỉnh** | **53** | **44,2%** |
| R1 — còn tăng ở biên ngân sách | 44 | 36,7% |
| R3 — plateau thật | 23 | 19,2% |

**Không có mô hình nào "hội tụ" theo nghĩa chặt.** Nhưng cơ chế chi phối là **suy giảm**, không phải underfitting:

- `sage` BoT-IoT binary seed 11: validation macro-F1 **0,602 → 0,175** (mất 0,426).
- `sage` BoT-IoT multiclass seed 33: **0,548 → 0,295** (mất 0,253).
- `edge_mlp` cùng ô: dao động chỉ ±0,03.

### 2.3 Bằng chứng cơ chế: tỉ lệ sụp đổ theo biến thể

Quy tắc khóa trước (chỉ dùng validation): `collapse := val_std ≥ 0,05 OR drop_after_best ≥ 0,10`.

| Model | Sụp đổ / tổng | Tỉ lệ |
|---|---:|---:|
| `edge_mlp` (chỉ flow) | 0/40 | **0,0%** |
| `sage` (topology, không có edge ở head) | 10/40 | **25,0%** |
| `sage_edge` (topology + edge trực tiếp ở head) | 2/40 | **5,0%** |

Phân bố theo ô:

| Ô | `sage` sụp đổ | `sage_edge` |
|---|---:|---:|
| BoT-IoT binary | **5/5 (100%)** | 0/5 |
| BoT-IoT multiclass | **4/5 (80%)** | 0/5 |
| CSE-CIC multiclass | 1/5 | 2/5 |
| Tất cả ô còn lại | 0 | 0 |

**Đọc kết quả này:** BoT-IoT `sage` binary không phải "topology không phù hợp với BoT-IoT". Đó là **kiến trúc này không huấn luyện được ổn định** dưới optimizer/ngân sách đã khóa, trên graph này, với mọi seed. Việc thêm đường đặc trưng flow trực tiếp vào head (`sage_edge`) gần như xóa bỏ hiện tượng đó.

### 2.4 Bất ổn định dự báo điểm cuối

Tương quan Spearman giữa độ bất ổn định của đường validation và test macro-F1 (n = 120):

| Phạm vi | ρ | p |
|---|---:|---:|
| Toàn bộ run | **−0,539** | 2,2 × 10⁻¹⁰ |
| Riêng `edge_mlp` | −0,698 | 5,5 × 10⁻⁷ |
| Riêng `sage` | −0,365 | 0,021 |
| Riêng `sage_edge` | −0,505 | 8,9 × 10⁻⁴ |

Bất ổn định tối ưu hóa là **yếu tố dự báo hạng nhất** của điểm cuối, mạnh hơn cả danh tính mô hình.

### 2.5 Bimodality theo seed ở cấp lớp (bằng chứng quyết định)

CSE-CIC multiclass, F1 lớp `DDoS` (support 278.547, lớp lớn nhất):

| Model | seed 11 | seed 22 | seed 33 | seed 44 | seed 55 |
|---|---:|---:|---:|---:|---:|
| `edge_mlp` | 0,988 | 0,980 | 0,970 | 0,995 | 0,993 |
| `sage` | 0,991 | **0,641** | 0,973 | **0,662** | 0,992 |
| `sage_edge` | **0,780** | 0,983 | **0,691** | **0,647** | 0,997 |

Cùng split, cùng preprocessing, cùng siêu tham số — chỉ khác seed. Kết quả **lưỡng cực**: hoặc ~0,99, hoặc ~0,65. Đây không phải nhiễu thống kê; đây là hai chế độ tối ưu hóa khác nhau.

BoT-IoT multiclass, F1 lớp `Benign` với `sage`: 0,446 / 0,823 / 0,424 / 0,688 / **0,139** — trong khi `sage_edge` ổn định 0,857–0,883 và `edge_mlp` 0,907–0,919.

---

## 3. Thống kê 5 seed: điều gì thực sự đứng vững

### 3.1 Thiết kế phân tích

Đơn vị dự báo là flow/cạnh; đơn vị lặp là **seed** (n = 5). Seed được ghép cặp giữa các model trong cùng ô. Ba contrast khóa trước: `sage − edge_mlp`, `sage_edge − edge_mlp`, `sage_edge − sage`. 8 ô × 3 contrast = 24 so sánh.

### 3.2 Kết quả từng ô

| Dataset | Task | `sage − edge_mlp` | `sage_edge − edge_mlp` | `sage_edge − sage` |
|---|---|---:|---:|---:|
| BoT-IoT | binary | **−0,0960** | −0,0185 | +0,0775 |
| BoT-IoT | multiclass | **−0,2657** | +0,0091 | **+0,2748** |
| CSE-CIC | binary | +0,0013 | +0,0023 | +0,0010 |
| CSE-CIC | multiclass | +0,0240 | +0,0207 | −0,0033 |
| ToN-IoT | binary | +0,0068 | +0,0095 | +0,0027 |
| ToN-IoT | multiclass | +0,0454 | +0,0618 | +0,0164 |
| UNSW | binary | +0,0058 | +0,0054 | −0,0003 |
| UNSW | multiclass | **+0,0757** | **+0,0745** | −0,0012 |

(Giá trị là trung bình chênh lệch ghép cặp theo seed.)

### 3.3 Gộp bằng mô hình hiệu ứng ngẫu nhiên (DerSimonian-Laird)

| Contrast | Δ gộp | KTC 95% | KTC dự báo 95% | I² | Ô dương/âm |
|---|---:|---|---|---:|---:|
| `sage_edge − edge_mlp` | **+0,0216** | [+0,0153; +0,0279] | [+0,0003; +0,0428] | 98,8% | 7/1 |
| `sage_edge − sage` | +0,0067 | [+0,0019; +0,0116] | [−0,0077; +0,0211] | 95,6% | 5/3 |
| `sage − edge_mlp` | +0,0021 | [−0,0105; +0,0146] | [−0,0403; +0,0444] | 99,5% | 6/2 |

**Kết luận có thể bảo vệ:**

1. **`sage_edge` hơn `edge_mlp` một lượng nhỏ nhưng nhất quán** (7/8 ô dương; cả KTC gộp và KTC dự báo đều dương). Đây là claim mạnh nhất mà dữ liệu hiện có hỗ trợ.
2. **`sage` KHÔNG hơn `edge_mlp` về trung bình** (KTC gộp chứa 0; KTC dự báo trải rộng cả hai phía). Một mô hình có **16,1× số tham số** (86.530 so với 5.378) kèm message passing **không** tạo ra lợi ích trung bình đo được so với MLP nhỏ chỉ dùng flow feature.
3. **I² 95–99%** xác nhận mức phụ thuộc dataset/task rất lớn — nhưng đây là **dị biệt**, không phải bằng chứng về lợi ích topology.
4. **Không một contrast cấp ô nào sống sót sau hiệu chỉnh đa so sánh** (Wilcoxon Holm: 0/24; sign test Holm: 0/24). Với n = 5 seed, không được tuyên bố "ý nghĩa thống kê" ở cấp ô.

### 3.4 Hiệu chuẩn capacity — lỗ hổng chưa được đóng

`PROTOCOL_FULL_DATA_VI.md` đã yêu cầu: *"Random Forest/GBDT và baseline cùng capacity phải chạy ở protocol xác nhận riêng trên đúng split trước khi tuyên bố GNN hơn baseline nói chung."* **Yêu cầu này chưa được thực hiện.**

| Model | Tham số | Kiến trúc |
|---|---:|---|
| `edge_mlp` | **5.378** | 1 lớp ẩn 128 |
| `sage` | **86.530** | 2 lớp message passing, ẩn 128 |
| `sage_edge` | 86.608 | như `sage` + 39 edge feature ở head |

Tỉ lệ **16,09×**. Vì vậy contrast `sage − edge_mlp` **đồng thời** thay đổi topology, độ sâu và capacity — đúng như Gap A mà khung nghiên cứu đã tự nêu. Kết quả pooled ≈ 0 ở trên khiến cách diễn giải "lợi ích topology" càng khó bảo vệ: nếu 16× capacity + message passing không tạo ra lợi ích trung bình, thì giả thuyết tiết kiệm nhất là **cấu trúc quan hệ không đóng góp gì đo được trong setting này**.

### 3.5 Số seed: 3 không đủ — bằng chứng định lượng

| Dataset | Task | Model | mean(3 seed) | SD(3) | mean(5 seed) | SD(5) | SD(5)/SD(3) |
|---|---|---:|---:|---:|---:|---:|---:|
| BoT-IoT | binary | `edge_mlp` | 0,9114 | 0,0038 | 0,9003 | 0,0256 | **6,67×** |
| UNSW | binary | `sage` | 0,9700 | 0,0002 | 0,9702 | 0,0010 | **5,19×** |
| CSE-CIC | multiclass | `edge_mlp` | 0,6607 | 0,0049 | 0,6686 | 0,0162 | **3,28×** |
| BoT-IoT | multiclass | `sage` | 0,5698 | 0,0207 | 0,5512 | 0,0511 | **2,47×** |

- **9/24 ô** có SD tăng > 1,5× khi thêm hai seed.
- **4/24 quyết định dựa trên KTC bị đảo** (ví dụ CSE-CIC multiclass `sage − edge_mlp`: KTC 3 seed [0,0066; 0,0624] dương ⇒ 5 seed [−0,0034; 0,0500] chứa 0).
- **1/24 đảo dấu** trung bình.

Kết luận: các bảng 3 seed trong tài liệu khung cũ **đã đánh giá thấp biến thiên** và trong một số ô dẫn tới kết luận ngược. Mọi suy luận phải dùng 5 seed.

---

## 4. Lớp hiếm: hai chiều, không phải một chiều

Sau 5 seed (support < 1.000), F1 test:

| Dataset | Lớp | Support | `edge_mlp` | `sage` | `sage_edge` | Đọc |
|---|---|---:|---:|---:|---:|---|
| ToN-IoT | ransomware | 674 | 0,073 ± 0,006 | **0,631 ± 0,050** | 0,588 ± 0,068 | Topology giúp **rất mạnh** |
| UNSW | Shellcode | 296 | 0,230 ± 0,047 | **0,415 ± 0,103** | 0,365 ± 0,054 | Topology giúp |
| UNSW | Analysis | 448 | 0,150 ± 0,034 | 0,159 ± 0,073 | **0,182 ± 0,052** | Giúp nhẹ |
| BoT-IoT | Theft | 480 | 0,270 ± 0,018 | 0,346 ± 0,161 | **0,402 ± 0,079** | Giúp, `sage` rất bất ổn |
| UNSW | Backdoor | 432 | **0,191 ± 0,017** | 0,128 ± 0,074 | 0,138 ± 0,067 | Topology **làm hại** |
| CSE-CIC | injection | 85 | 0,0006 | 0,014 | 0,015 | Support quá nhỏ, không kết luận |
| UNSW | Worms | 37 | 0,055 | 0,054 | 0,058 | Support quá nhỏ, không kết luận |

**Điều chỉnh quan trọng:** trên **cùng một bộ UNSW**, topology giúp `Shellcode` (+0,185) nhưng làm hại `Backdoor` (−0,063). Vì vậy claim phải ở dạng **"phụ thuộc lớp cụ thể"**, không phải "topology giúp lớp hiếm". Đồng thời `sage` trên BoT-IoT `Theft` có SD 0,161 — lớn hơn cả hiệu ứng trung bình, nên hiệu ứng đó chưa thể coi là ổn định.

---

## 5. Điều chỉnh: danh sách claim phải sửa hoặc rút

| # | Claim trong tài liệu hiện hành | Hành động |
|---|---|---|
| C1 | "Trong full-scale multiclass, model topology tốt nhất có mean macro-F1 cao hơn `edge_mlp` trên 4/4 dataset" | **Giữ nhưng đổi nhãn** thành mô tả `sage_edge`; nêu rõ `sage` (biến thể gần bài gốc nhất) không hơn. |
| C2 | "Mini-batch: topology thắng `edge_mlp` ở multiclass trên 4/4 dataset" | **Hạ mức**: mini-batch là protocol khác (lấy mẫu phân tầng), không dùng để suy luận cạnh tranh. |
| C3 | "Full-scale train dùng hai lượt qua train với tối thiểu 1.500 step" | **Sửa**: thêm rằng UNSW thực tế chạy 3,67 lượt và 30/30 checkpoint tốt nhất ở lượt ≥ 3. |
| C4 | "`sage` BoT-IoT suy giảm — cần thí nghiệm phân biệt underfitting" | **Cập nhật**: underfitting bị bác bỏ bằng learning curve; cơ chế là sụp đổ tối ưu hóa (5/5 seed). |
| C5 | "Topology giúp phát hiện lớp hiếm" | **Sửa thành phụ thuộc lớp**: giúp ToN ransomware/UNSW Shellcode, hại UNSW Backdoor. |
| C6 | "SD qua 3 seed là mô tả hữu ích" | **Thêm cảnh báo định lượng**: SD tăng tới 6,67×; 4/24 quyết định KTC đảo. |
| C7 | Bất kỳ hàm ý nào rằng contrast `sage − edge_mlp` cô lập topology | **Rút**: bị nhiễu bởi 16,09× capacity và độ sâu. Cần baseline cùng capacity. |

---

## 6. Lựa chọn hướng tiếp theo

### 6.1 Đánh giá các hướng đã đề xuất

| Hướng | Giá trị khoa học | Khả thi cục bộ (CPU, 16 luồng, 13 GiB RAM) | Quyết định |
|---|---|---|---|
| Convergence-first pilot 8 lượt (UNSW, ToN, BoT) | Cao | **Không** — cần GPU 24 GB | **Chờ tài nguyên** |
| Graph-rewiring RR/RW/WW/WR | Cao | **Không** — cần GPU | **Chờ tài nguyên** |
| Endpoint-disjoint holdout (GNN) | Cao | **Không** — cần GPU | Tách: dựng split trên CPU, train chờ GPU |
| RF/GBDT full-scale (Gate B) | **Rất cao** | **Có** | **THỰC HIỆN NGAY** |
| Baseline **cùng capacity** | **Rất cao** | **Có** (chỉ cần MLP) | **THỰC HIỆN NGAY** |
| Chẩn đoán sụp đổ tối ưu hóa | **Rất cao** | **Đã xong** | **Đã có kết quả** |

### 6.2 Hướng được chọn: "điều gì quyết định kết quả?"

Dữ liệu lưu trữ đã trả lời dứt khoát câu hỏi *"topology có giúp không"* ở dạng hiện tại: **không có lợi ích trung bình đo được, và phần lớn phương sai quan sát được là do bất ổn định tối ưu hóa theo seed.** Vì vậy câu hỏi nghiên cứu nên được nâng cấp từ so sánh mô hình sang **phân rã nguyên nhân**.

**Câu hỏi trung tâm mới:**

> Trong điều kiện phân bố lớp tự nhiên của NF-UQ-NIDS-v2, phần nào của hiệu năng quan sát được quy cho (i) capacity mô hình, (ii) cấu trúc quan hệ, (iii) đường truyền đặc trưng trực tiếp, và (iv) độ ổn định tối ưu hóa theo seed?

**Ba thí nghiệm đóng cổng, chạy được trên CPU:**

**TN-1 — Baseline cùng capacity (`edge_mlp` sâu, khớp tham số).**
Huấn luyện MLP chỉ dùng flow feature với 2 lớp ẩn và số tham số ≈ 86,5k để khớp `sage`. Nếu MLP cùng capacity đạt ngang `sage`, thì "lợi ích topology" biến mất và nguyên nhân là capacity. Đây là phép thử mà protocol đã yêu cầu nhưng chưa làm. **Chi phí: thấp (không cần message passing).**

**TN-2 — Comparator bảng mạnh full-scale (Gate B).**
Random Forest / ExtraTrees / HistGradientBoosting trên **đúng split, đúng preprocessing đã khóa**, tuning chỉ trên validation. Trả lời câu hỏi mà repo chưa trả lời được: classical ML đứng ở đâu so với cả ba biến thể GNN. **Chi phí: trung bình.**

**TN-3 — Tái tạo split và xác minh nguồn.**
Tái tạo quy tắc `hash(flow_group_id, 20260920) % 10` từ bốn Parquet LFS, so khớp số dòng từng split và tỉ lệ chồng lấp IP với `prepare_manifest.json` lưu trữ. Đóng câu hỏi provenance ở §1.3 và tạo điều kiện cho TN-1/TN-2 dùng **đúng split**. **Chi phí: thấp.**

### 6.3 Tiêu đề và định vị đề xuất (cập nhật)

> **When Does Relational Structure Help? Separating Model Capacity, Optimisation Stability and Direct Flow Evidence in Full-Scale NetFlow Intrusion Detection**

Tiếng Việt:

> **Khi nào cấu trúc quan hệ giúp ích? Phân tách capacity, độ ổn định tối ưu hóa và bằng chứng flow trực tiếp trong phát hiện xâm nhập NetFlow quy mô đầy đủ**

Định vị: **an toàn thông tin thực nghiệm + phương pháp luận đánh giá có kiểm soát**. Đóng góp không phải kiến trúc mới, mà là:
1. phát hiện và định lượng rằng phần lớn "lợi ích GNN" trong tài liệu tự báo cáo có thể là **hiệu ứng ổn định tối ưu hóa và capacity**, không phải cấu trúc;
2. một protocol tái lập được ở quy mô 76 triệu flow với 5 seed và kiểm chứng độc lập 120/120;
3. bằng chứng âm rõ ràng: biến thể topology thuần không hơn MLP flow-only **dù có 16× tham số**.

### 6.4 Cổng quyết định trước khi viết Results

- [ ] TN-3 khớp manifest ⇒ split tái lập được, dùng được cho TN-1/TN-2.
- [ ] TN-1 hoàn tất cho cả 4 dataset × 2 task × ≥3 seed.
- [ ] TN-2 hoàn tất tối thiểu trên UNSW, ToN, CSE-CIC (BoT tùy RAM), ghi rõ dấu chân tài nguyên.
- [ ] Chẩn đoán sụp đổ được viết thành mục Results có quy tắc khóa trước.
- [ ] Mọi claim cấp ô hoặc gộp có KTC và ghi rõ không hiệu chỉnh được đa so sánh ⇒ báo effect size.
- [ ] Không claim nào về host chưa thấy, zero-day hoặc thời gian thực.

---

## 7. Phụ lục — artifact tái tạo được

Mọi script nằm trong `egs-nfuq-2026Q4/scripts/`, chỉ đọc repo nguồn.

| Script | Vai trò | Output |
|---|---|---|
| `01_inventory_audit.py` | Kiểm kê 120 run + 20 RF, kiểm chứng số liệu | `run_registry.csv`, `audit_inventory.json` |
| `02_convergence_diagnostics.py` | Chẩn đoán learning curve | `convergence_per_run.csv`, `convergence_summary.json` |
| `03_five_seed_stats.py` | Thống kê ghép cặp + meta-analysis | `paired_contrasts.csv`, `pooled_meta.csv` |
| `04_rare_class_and_instability.py` | Lớp hiếm + bất ổn định | `per_class_summary.csv`, `rare_class_5seed.csv` |
| `05_collapse_sensitivity.py` | Sụp đổ tối ưu hóa + độ nhạy | `collapse_flags.csv`, `ranking_sensitivity.csv` |
| `06_figures.py` | Hình | `figures/fig1..fig4` |

Hình: `fig1_contrast_forest.png`, `fig2_instability_and_budget.png`, `fig3_collapse_rates.png`, `fig4_learning_curves.png`.
