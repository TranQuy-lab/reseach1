# research_q4_2026 — Kiểm toán bằng chứng, chẩn đoán cơ chế và protocol xác nhận

Thư mục này chứa **công việc bổ sung 2026-10-01** trên repo `TranQuy-lab/reseach1`.
Mọi thứ ở đây **chỉ đọc** mã nguồn gốc; không file nào trong `src/`, `tests/`,
`research/` bị sửa đổi.

## Vì sao có thư mục này

Kiểm toán độc lập 120 run lưu trữ phát hiện bốn điều làm thay đổi cách diễn giải
đề tài:

1. **Không có bằng chứng `sage` hơn `edge_mlp` về trung bình.** Pooled Δ =
   +0,0021 macro-F1, KTC 95% [−0,0105; +0,0146] — chứa 0. Biến thể gần bài báo
   gốc nhất không tạo ra lợi ích trung bình đo được.
2. **Giả thuyết underfitting cho BoT-IoT bị bác bỏ.** 10/40 run `sage` (25 %) bị
   sụp đổ tối ưu hóa, BoT-IoT binary là **5/5 seed**. Validation macro-F1 *giảm*
   sau đỉnh (ví dụ 0,602 → 0,175), không phải còn tăng.
3. **Đường edge trực tiếp ở head là yếu tố ổn định.** Tỉ lệ sụp đổ: `edge_mlp`
   0 %, `sage` 25 %, `sage_edge` 5 %.
4. **3 seed không đủ.** SD tăng tới 6,67× khi thêm seed 44/55; 4/24 quyết định
   dựa trên KTC bị đảo.

5. **Phụ thuộc endpoint là hiện tượng của multiclass, không phải binary.** Cùng
   kiến trúc, cùng ngân sách, cùng seed: NF-ToN-IoT-v2 multiclass mất **0,311**
   macro-F1 trên split endpoint-disjoint, còn binary mất **0,001**. Tức endpoint
   quen thuộc thay thế cho *phân biệt loại tấn công*, không phải cho tách
   benign/attack.
6. **Tóm tắt cấu trúc thô thắng message passing, nhưng nội dung aggregation thì
   không tái tạo được.** `mlp_struct` (5 đặc trưng đếm) đạt 0,5003 > `sage` 0,4898;
   trong khi cho model trung bình đặc trưng lân cận (78 chiều) chỉ đạt 0,4218 —
   *thấp hơn* cả baseline flow-only 0,4628. Kết quả âm này phải được báo.
7. **Biến thể sụp đổ không triển khai được.** `sage` trên BoT-IoT có tỉ lệ báo động
   giả **25,02 %** (250.189 báo động giả/triệu flow benign, recall benign 0,75);
   `edge_mlp` 0,19 % và `sage_edge` 0,32 %.

Thêm một phát hiện phương pháp luận: UNSW-NB15 chạy **3,67 lượt** trong khi ba bộ
còn lại đúng **2,0 lượt**, do `min_train_steps=1500` ghi đè ngân sách hai lượt;
**30/30** checkpoint tốt nhất của UNSW ở lượt ≥ 3. So sánh xuyên bộ theo nhãn
"protocol hai lượt" vì vậy không hợp lệ cho UNSW.

## Nội dung

| Thư mục | Nội dung |
|---|---|
| `reports/01_BAO_CAO_KIEM_TRA_DINH_VI_LAI_VI.md` | Kiểm tra, chẩn đoán, định vị lại, chọn hướng tiếp theo |
| `reports/02_BAO_CAO_KET_QUA_VI.md` | **Báo cáo kết quả tổng hợp, sinh tự động** từ `results/` |
| `reports/03_DINH_VI_LAI_VA_CLAIM_VI.md` | Sổ đăng ký claim thay thế + threat to validity |
| `reports/04_MANUSCRIPT_DRAFT_EN.md` | Bản thảo tiếng Anh theo định vị mới (DRAFT) |
| `reports/05_TRANG_THAI_VA_CHAY_TIEP_VI.md` | **Trạng thái dừng máy + lệnh chạy tiếp + ghi chú tài nguyên** |
| `protocols/` | Protocol khóa trước cho Phase B (baseline cùng capacity + comparator bảng) |
| `scripts/` | Toàn bộ mã phân tích và thí nghiệm, tái chạy được |
| `results/` | Bảng CSV/JSON sinh ra |
| `figures/` | Hình PNG |

## Tái lập

```bash
python scripts/01_inventory_audit.py            # kiểm kê + kiểm chứng 120 run
python scripts/02_convergence_diagnostics.py    # chẩn đoán learning curve
python scripts/03_five_seed_stats.py            # thống kê ghép cặp + meta-analysis
python scripts/04_rare_class_and_instability.py # lớp hiếm + bất ổn định
python scripts/05_collapse_sensitivity.py       # sụp đổ tối ưu hóa + độ nhạy
python scripts/06_figures.py                    # hình Phase A
python scripts/11_phaseB_analysis.py            # tổng hợp Phase B + hình 5,6
python scripts/12_endpoint_feasibility.py       # cổng khả thi endpoint (hình 7)
python scripts/13_build_report_02.py            # sinh báo cáo kết quả tổng hợp
python scripts/14_collapse_robustness.py        # độ bền của kết luận sụp đổ (hình 8)
python scripts/15_graph_statistics.py           # thống kê đồ thị theo dataset
python scripts/16_holdout_cpu_evaluation.py     # huấn luyện trên split endpoint-holdout
python scripts/17_holdout_comparison.py         # so sánh split khóa vs endpoint-holdout
python scripts/18_tn5_structural_features.py    # TN-5: đặc trưng cấu trúc thay message passing
python scripts/19_tn6_operational_metrics.py    # TN-6: tỉ lệ báo động giả từ confusion matrix
python scripts/20_tn7_cross_dataset_transfer.py # TN-7: chuyển giao xuyên bộ dữ liệu (binary)
python scripts/21_tn8_neighbour_mean_smoothing.py # TN-8: trung bình đặc trưng lân cận
```

Ghi chú tài nguyên: `HistGradientBoosting` trên NF-ToN-IoT-v2 đỉnh **9,7 GiB RSS** (đo được),
trên NF-CSE-CIC-IDS2018-v2 cao hơn, và NF-BoT-IoT-v2 (2,2×) **không chạy được** trên máy
13 GiB. Vì vậy TN-2 cho các bộ lớn phải chạy **một mình**, và BoT-IoT được ghi là giới hạn
tài nguyên thay vì lấy mẫu con ngầm. Xem `protocols/PROTOCOL_PHASE_B_VI.md` mục amendment.

Các script Phase B (`07`…`11`) cần dữ liệu Parquet đã hydrate và split tái tạo;
xem `protocols/PROTOCOL_PHASE_B_VI.md`.

## Kiểm chứng tái tạo split

Tái tạo split cục bộ từ bốn Parquet Git LFS bằng chính module
`nids_minibatch.prepare` cho kết quả **khớp hoàn toàn** với
`research/results/full_5seed/prepare_manifest.json`: số dòng nguồn, số dòng từng
split và tỉ lệ chồng lấp IP khớp tới 6 chữ số thập phân cho cả bốn dataset.

Lưu ý provenance: SHA-256 của bốn file Parquet Git LFS **khác**
`source.sha256` trong manifest lưu trữ, nhưng nội dung tương đương về mặt chia
split (đã chứng minh bằng tái tạo ở trên).

## Trạng thái

| Hạng mục | Trạng thái |
|---|---|
| Kiểm toán 120 run | Xong |
| Chẩn đoán hội tụ / sụp đổ | Xong |
| Thống kê 5 seed + meta-analysis | Xong |
| Lớp hiếm + bất ổn định | Xong |
| Tái tạo split, kiểm chứng provenance | Xong |
| TN-1 baseline cùng capacity | Đang chạy (UNSW xong; ToN gần xong; CSE một phần; BoT chưa) |
| TN-2 comparator bảng | UNSW xong; ToN/CSE đang chạy một mình vì giới hạn RAM; BoT không khả thi |
| TN-5 đặc trưng cấu trúc thay message passing | **Xong phần quyết định**: `mlp_struct` 0,5003 > `sage` 0,4898 (UNSW mc, n=3) |
| TN-6 hồ sơ báo động giả | **Xong**: `sage` BoT-IoT 25,02 % báo động giả |
| TN-8 trung bình đặc trưng lân cận | **Kết quả âm**: 0,4218 < flow-only 0,4628 (n=2) |
| TN-7 chuyển giao xuyên bộ dữ liệu | **Xong**: nội bộ 5/5 lần vượt baseline hằng số; **xuyên mạng chỉ 4/15** |
| TN-9 HGB trên split endpoint-disjoint | **Xong**: 0,866 → 0,472 ⇒ split bị nhiễu, dùng để bác bỏ TN-4 |
| TN-10 cô lập endpoint trên split khóa | **Xong**: 0,018 % (ToN) và 2,0 % (UNSW) flow chưa thấy endpoint; hiệu ứng ≤ 0,044 |
| Endpoint-disjoint split (Gate C, phần CPU) | Xong — khả thi ở UNSW/ToN, không ở CSE-CIC/BoT-IoT |
| Thống kê đồ thị giải thích cơ chế | Xong |
| Độ bền kết luận sụp đổ + Fisher/Holm | Xong |
| Convergence-first 8 lượt, rewiring, GNN trên endpoint-holdout | **Chặn: cần GPU** |
