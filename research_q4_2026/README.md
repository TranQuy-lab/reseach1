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
```

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
| TN-1 baseline cùng capacity | Đang chạy (UNSW xong một phần; ToN/CSE/BoT đang chạy) |
| TN-2 comparator bảng | Đang chạy (UNSW xong; ToN/CSE đang chạy) |
| Endpoint-disjoint split (Gate C, phần CPU) | Xong — khả thi ở UNSW/ToN, không ở CSE-CIC/BoT-IoT |
| Thống kê đồ thị giải thích cơ chế | Xong |
| Độ bền kết luận sụp đổ + Fisher/Holm | Xong |
| Convergence-first 8 lượt, rewiring, GNN trên endpoint-holdout | **Chặn: cần GPU** |
