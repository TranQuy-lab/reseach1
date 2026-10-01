# TRẠNG THÁI & HƯỚNG DẪN CHẠY TIẾP

**Ngày dừng:** 2026-10-01 ~07:50 (giờ VN)  
**Lý do dừng:** theo yêu cầu người dùng — máy không chạy tiếp được lúc này.  
**Nhánh đã đẩy:** `research/2026q4-evidence-audit` trên `TranQuy-lab/reseach1`  
**Thư mục làm việc cục bộ:** `/home/noble-tran/nghiencuu/egs-nfuq-2026Q4`

> ⚠️ **Điều quan trọng nhất cần biết:** kết quả phân tích và toàn bộ script **đã nằm trong Git**
> (`research_q4_2026/`). Nhưng **dữ liệu và ma trận đặc trưng thì không** (~24 GB), vì quá lớn.
> Chúng **tái tạo được hoàn toàn** bằng các lệnh ở §3. Nếu giữ được máy hiện tại thì không cần
> tái tạo gì cả.

---

## 1. Đã hoàn thành

| Hạng mục | Kết quả chính | Trạng thái |
|---|---|---|
| Kiểm toán 120 run lưu trữ | 120/120 verify, replay error 0, metric error 1,1e−16, `runs.csv` khớp `metrics.json` tới 5,7e−14 | ✅ |
| Tái tạo split từ Parquet Git LFS | Khớp manifest lưu trữ **tuyệt đối** (số dòng + tỉ lệ chồng lấp IP tới 6 chữ số) | ✅ |
| Chẩn đoán hội tụ từ 120 learning curve | 44 % run suy giảm sau đỉnh; underfitting **bị bác bỏ** | ✅ |
| Sụp đổ tối ưu hóa + độ bền | `edge_mlp` 0/40, `sage` 10/40 (25 %), `sage_edge` 2/40; Fisher Holm p = 0,0031 | ✅ |
| Thống kê 5 seed + meta-analysis | `sage − edge_mlp` = +0,0021 [−0,0105; +0,0146] — chứa 0; 0/24 sống sót Holm | ✅ |
| 3 seed vs 5 seed | SD tăng tới 6,67×; 4/24 quyết định KTC bị đảo | ✅ |
| Thống kê đồ thị | BoT 78,65 cạnh/node & 91,6 % lặp vs UNSW/CSE 1,7–1,8 & ~20 % | ✅ |
| **Gate B** tabular (UNSW 2 task, ToN mc+bin) | UNSW mc: HGB 0,654 vs `sage_edge` 0,489; ToN mc: **0,8664** vs 0,7658; ToN bin: **0,9928** vs 0,9802 | 🟡 CSE chưa |
| **Gate C** endpoint | **Đo sạch**: 0,018 % (ToN) và 2,0 % (UNSW) flow chưa thấy endpoint, hiệu ứng ≤ 0,044. **Bác bỏ** split endpoint-disjoint (HGB 0,866 → 0,472) | ✅ |
| **Gate D** thống kê | Xong | ✅ |
| TN-1 baseline cùng capacity | 7/8 ô; BoT mc 0,8364 **thắng mọi biến thể GNN** | 🟡 BoT binary |
| TN-5 đặc trưng cấu trúc | `mlp_struct` 0,5003 > `sage` 0,4898 (UNSW mc) | ✅ |
| TN-6 báo động giả | `sage` BoT 25,02 % (250.189 FP/triệu) vs `edge_mlp` 0,19 % | ✅ |
| TN-7 chuyển giao xuyên mạng | nội bộ 5/5 vượt baseline hằng số; **xuyên mạng 4/15** | ✅ |
| TN-8 trung bình đặc trưng lân cận | **Kết quả âm**: 0,4266 < flow-only 0,4628 | ✅ |
| TN-9/TN-10 cô lập endpoint | Xong, đã sửa lại một claim sai của chính tôi | ✅ |

## 2. Việc còn lại (theo thứ tự ưu tiên)

| # | Việc | Lệnh | Ước tính |
|---|---|---|---|
| 1 | **TN-1 BoT-IoT binary** (ô cuối ma trận) | §4.1 | ~1,5–2 h |
| 2 | **Gate B: CSE-CIC** (HGB cả 2 task) | §4.2 | ~1,5–2 h, **cần ~11,5 GiB RAM, phải chạy một mình** |
| 3 | TN-5 phần còn lại (ToN; UNSW binary) | §4.3 | ~2 h |
| 4 | TN-8 phần còn lại (ToN) | §4.4 | ~2 h |
| 5 | TN-10 cho CSE/ToN binary (tùy chọn) | §4.5 | ~30 min |
| 6 | ~~BoT-IoT tabular~~ | — | ❌ **Không khả thi ở 13 GiB** (HGB cần ~14 GiB) |
| 7 | Convergence-first 8 lượt, rewiring RR/RW/WW/WR, GNN trên endpoint-holdout | — | ⛔ **Cần GPU ≥ 24 GiB VRAM** |

## 3. Tái tạo dữ liệu cục bộ từ đầu (nếu đổi máy)

```bash
# 0) Môi trường (~2 phút)
cd /home/noble-tran/nghiencuu && mkdir -p egs-nfuq-2026Q4 && cd egs-nfuq-2026Q4
uv venv --python 3.12 .venv && . .venv/bin/activate
uv pip install numpy pandas pyarrow duckdb scikit-learn scipy matplotlib
uv pip install --index-url https://download.pytorch.org/whl/cpu torch

# 1) Tải 4 Parquet (3,28 GB, ~10 phút; đây là Git LFS nhưng tải trực tiếp được)
mkdir -p data/processed_four && cd data/processed_four
for f in NF-UNSW-NB15-v2 NF-BoT-IoT-v2 NF-ToN-IoT-v2 NF-CSE-CIC-IDS2018-v2; do
  curl -sSL --retry 5 -o $f.parquet \
   "https://media.githubusercontent.com/media/TranQuy-lab/reseach1/main/$f.parquet"
done
cd ../..   # SHA-256 kỳ vọng ghi trong results/full_prepare_reproduction.json

# 2) Tái tạo split đã khóa (~80 giây) — dùng chính module của repo, chỉ đọc
PYTHONPATH=/home/noble-tran/nghiencuu/repo_reseach1/src python -m nids_minibatch.prepare \
  --source data/processed_four --output data/full_splits \
  --report results/full_prepare_reproduction.json \
  --threads 12 --memory-limit 8GB --protocol PROTOCOL_FULL_DATA_VI.md
# Kiểm tra: phải khớp prepare_manifest.json lưu trữ về số dòng và ip_overlap

# 3) Ma trận đặc trưng (~10 phút, 12 GB)
python scripts/07_materialise_features.py
python scripts/07b_fix_binary_labels.py          # BẮT BUỘC: Label nhị phân là số nguyên

# 4) Split endpoint-disjoint (nếu cần Gate C)
python scripts/10_endpoint_disjoint_split.py --strategy holdout --threads 6
```

## 4. Lệnh chạy tiếp chính xác

### 4.1 TN-1 BoT-IoT binary (ưu tiên 1)
```bash
cd /home/noble-tran/nghiencuu/egs-nfuq-2026Q4 && . .venv/bin/activate
python scripts/08_tn1_capacity_matched_mlp.py \
  --datasets NF-BoT-IoT-v2 --tasks binary \
  --models mlp_h273_2l mlp_h128_1l --seeds 11 22 33
```
Script **tự bỏ qua run đã có** nên chạy lại an toàn.

### 4.2 Gate B cho CSE-CIC (ưu tiên 2 — PHẢI chạy một mình)
```bash
# Dừng mọi job khác trước; HGB trên 13,2 M dòng cần ~11,5 GiB
python scripts/09_tn2_tabular_comparators.py \
  --datasets NF-CSE-CIC-IDS2018-v2 --tasks multiclass binary \
  --models hist_gradient_boosting --seeds 11 22 33 --n-jobs 14
```
Nếu OOM: chấp nhận và ghi rõ **giới hạn tài nguyên** (quy tắc 7 của protocol), **không** subsample ngầm.

### 4.3–4.5 Các phần còn lại
```bash
python scripts/18_tn5_structural_features.py --datasets NF-ToN-IoT-v2 --tasks multiclass binary \
  --models mlp_struct mlp_struct_lab --seeds 11 22 33
python scripts/21_tn8_neighbour_mean_smoothing.py --datasets NF-ToN-IoT-v2 --tasks multiclass binary \
  --seeds 11 22 33 --threads 6
python scripts/23_tn10_endpoint_isolation.py --datasets NF-BoT-IoT-v2 NF-CSE-CIC-IDS2018-v2 \
  --tasks multiclass --seeds 11 22 33 --n-jobs 14
```

Sau **mỗi** lần chạy, làm mới báo cáo và hình:
```bash
for s in 11 12 14 17 19 13; do python scripts/$s*.py; done
python scripts/06_figures.py
```

## 5. Ghi chú tài nguyên & lỗi đã gặp (để lần sau tránh)

| Vấn đề | Chi tiết | Cách xử lý |
|---|---|---|
| **RAM là nút cổ chai** | Máy 13 GiB. HGB đo được: ToN-IoT **10,3 GiB**, CSE-CIC ~11,5 GiB, BoT-IoT ~14 GiB (không chạy được) | Chạy TN-2 **một mình**; dùng `scripts/run_tn2_when_free.sh` để chờ đủ RAM |
| **OOM đã bị một lần** | TN-2 ToN bị kill khi chạy song song 4 job | Chỉ chạy TN-2 khi các job khác đã dừng |
| **Schema drift** | Thêm cột `peak_rss_gib` giữa chừng làm CSV 16→17 cột, pandas không đọc được | Đã sửa; **nếu thêm cột, phải đổi tên file kết quả** |
| **`Label` nhị phân là số nguyên** | Không phải tên lớp; phải dùng trực tiếp, thứ tự lớp `["Benign","Attack"]` | `scripts/07b_fix_binary_labels.py` bắt buộc chạy sau `07` |
| **`history.json` chỉ 4–5 điểm** | Validation chỉ chạy sau mỗi lượt đầy đủ | Đủ để phân loại plateau/decay, không đủ để ước lượng learning curve mịn |
| **BoT-IoT TN-1 rất chậm** | ~900–1500 s/run do 12.910 step | Chạy qua đêm; script có thể resume |
| **UNSW ngân sách khác** | 3,667 lượt so với 2,0 ở ba bộ còn lại | **Luôn** nêu như threat to validity |
| **Git LFS SHA khác manifest** | SHA file LFS ≠ `source.sha256` trong manifest | Đã chứng minh tương đương bằng tái tạo split; ghi rõ khi công bố |

## 6. Bản đồ file

| Đường dẫn | Nội dung |
|---|---|
| `reports/01_..._KIEM_TRA_DINH_VI_LAI_VI.md` | Kiểm tra, chẩn đoán, định vị lại, chọn hướng |
| `reports/02_BAO_CAO_KET_QUA_VI.md` | **Báo cáo kết quả tổng hợp, sinh tự động từ `results/`** |
| `reports/03_DINH_VI_LAI_VA_CLAIM_VI.md` | Sổ claim (N01–N23) + threat to validity |
| `reports/04_MANUSCRIPT_DRAFT_EN.md` | Bản thảo tiếng Anh (DRAFT) |
| `reports/05_TRANG_THAI_VA_CHAY_TIEP_VI.md` | **File này** |
| `protocols/PROTOCOL_PHASE_B_VI.md` | Protocol khóa trước + sổ amendment |
| `scripts/01..06` | Phase A: kiểm toán, hội tụ, thống kê, lớp hiếm, sụp đổ, hình |
| `scripts/07..10` | Tái tạo dữ liệu, split, endpoint split |
| `scripts/11..23` | Phase B: TN-1…TN-10 |
| `results/*.csv,*.json` | 58 file kết quả — **nguồn duy nhất của mọi con số** |

## 7. Trạng thái goal

Goal **tạm dừng** theo yêu cầu người dùng. Khi tiếp tục, chạy lại đúng thứ tự §4 và bật lại goal.
