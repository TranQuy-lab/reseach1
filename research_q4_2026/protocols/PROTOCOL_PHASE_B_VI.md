# PROTOCOL PHASE B — Baseline cùng capacity và comparator bảng trên split đã khóa

**Ngày khóa:** 2026-10-01  
**Trạng thái:** ĐÃ KHÓA trước khi chạy TN-1 và TN-2 (cùng ngày).  
**Cơ sở:** `research/PROTOCOL_FULL_DATA_VI.md`, `ke-hoach-nang-cap-bai-bao-quoc-te.md` (Phase 2 và Phase 4).  
**Không sửa:** không file nào trong `src/`, `tests/`, `research/` của repo nguồn bị thay đổi.

---

## 1. Vì sao phase này tồn tại

`PROTOCOL_FULL_DATA_VI.md` đã ghi:

> "`edge_mlp` là baseline nội bộ của ma trận chính, không đại diện cho mọi mô hình bảng. Random Forest/GBDT và **baseline cùng capacity** phải chạy ở protocol xác nhận riêng trên đúng split trước khi tuyên bố GNN hơn baseline nói chung."

Kiểm toán 2026-10-01 xác nhận hai yêu cầu đó **chưa được thực hiện**, đồng thời định lượng mức độ nghiêm trọng của bẫy capacity:

| Biến thể lưu trữ | Tham số (binary) | Tham số (multiclass 5 lớp) |
|---|---:|---:|
| `edge_mlp` | 5.378 | 6.410 |
| `sage` | 86.530 | 87.301 |
| `sage_edge` | 86.608 | 87.379 |

Tỉ lệ **16,09×**. Vì vậy contrast lưu trữ `sage − edge_mlp` đồng thời thay đổi topology, độ sâu và capacity, và **không thể** dùng để suy luận về topology.

## 2. Câu hỏi và contrast khóa trước

| Mã | Câu hỏi | Contrast | Đọc kết quả |
|---|---|---|---|
| Q-cap | Lợi ích quan sát được có phải chỉ do capacity/độ sâu? | `mlp_h273_2l − edge_mlp` | Dương lớn ⇒ capacity giải thích phần lớn |
| Q-topo | Ở **capacity khớp**, topology có thêm giá trị không? | `sage − mlp_h273_2l` | **Contrast trung tâm** |
| Q-topo-edge | Đường edge trực tiếp ở head có thêm giá trị không? | `sage_edge − sage` | |
| Q-tab | Classical ML mạnh đứng ở đâu? | `hgb − edge_mlp`, `hgb − sage_edge` | |
| Q-control | Harness CPU có tái lập được archive không? | `mlp_h128_1l − edge_mlp` | Phải ≈ 0 thì mới dùng được các contrast khác |

## 3. Thiết kế khóa trước

### 3.1 TN-1 — MLP cùng capacity

| Thành phần | Giá trị |
|---|---|
| Biến thể | `mlp_h128_1l` (tái lập `edge_mlp`), `mlp_h128_2l` (đối chứng độ sâu), `mlp_h273_2l` (khớp capacity `sage`) |
| Seed | 11, 22, 33, 44, 55 |
| Task | multiclass và binary |
| Batch | 4096 |
| Optimizer | Adam, lr 1e-3 (giống protocol gốc) |
| Dropout | 0,2 |
| Loss | balanced cross-entropy, trọng số **chỉ tính từ train** |
| `steps_per_pass` | `ceil(train_rows / 4096)` — **giống hệt** archive |
| `max_steps` | `max(1500, 2 × steps_per_pass)` — giống hệt archive |
| `eval_every` | `max(500, ceil(steps_per_pass / 4))` — giống hệt archive |
| Checkpoint hợp lệ | chỉ sau một lượt đầy đủ; chọn bằng **validation macro-F1** |
| Test | chỉ đọc một lần, sau khi checkpoint đã chọn |

Tham số khớp capacity: `h=273`, 2 lớp ẩn ⇒ 86.270 (binary) và 87.092 (multiclass), tương ứng 99,70% và 99,76% tham số của `sage`.

### 3.2 TN-2 — Comparator bảng

| Thành phần | Giá trị |
|---|---|
| Model | `HistGradientBoosting` (chính), `ExtraTrees`, `RandomForest` |
| Seed | 11, 22, 33 (mở rộng 44/55 nếu tài nguyên cho phép) |
| Tuning | lưới nhỏ khóa trước, chọn bằng **validation macro-F1** |
| Cân bằng lớp | `class_weight="balanced"` cho mọi model |
| Đặc trưng | đúng 39 NetFlow feature; **không** IP/port/ID |
| Scaler | **không fit lại** — dùng `preprocessor.json` lưu trữ của chính run GNN |
| Test | chỉ đọc một lần cho cấu hình đã chọn |

Lưới khóa trước:

```text
hist_gradient_boosting: {lr 0.10, max_iter 300, leaves 31}
                        {lr 0.05, max_iter 600, leaves 31}
                        {lr 0.10, max_iter 400, leaves 63}
extra_trees:            {100 cây, max_features sqrt, min_samples_leaf 1}
                        {200 cây, max_features 0.3,  min_samples_leaf 2}
random_forest:          {100 cây, max_features sqrt, min_samples_leaf 1}
```

### 3.3 Dữ liệu và split

- Tái tạo split từ bốn Parquet Git LFS bằng chính module `nids_minibatch.prepare` của repo (chỉ đọc).
- **Kiểm chứng tái tạo:** số dòng nguồn, số dòng từng split, `cross_split_groups`, và tỉ lệ chồng lấp IP phải khớp `results/full_5seed/prepare_manifest.json` lưu trữ.

## 4. Quy tắc diễn giải khóa trước

1. Contrast chỉ được gọi là hỗ trợ topology nếu `sage − mlp_h273_2l` dương và KTC 95% không chứa 0, và hướng giữ ổn định qua các ô.
2. Nếu `mlp_h273_2l` đạt ngang `sage`, kết luận đúng là **capacity/độ sâu**, không phải topology.
3. `mlp_h128_1l − edge_mlp` phải ≈ 0; nếu lệch lớn hơn SD nội bộ của archive thì harness không dùng được và mọi so sánh phải dừng.
4. Không dùng test để chọn số seed, cấu hình, hay để quyết định chạy thêm dataset.
5. Với n = 5, báo mean, SD, KTC bootstrap và effect size; **không** gọi "ý nghĩa thống kê" ở cấp ô.
6. Không claim host chưa thấy, zero-day, thời gian thực hoặc cross-dataset.
7. Nếu một dataset vượt RAM khả dụng cho một model, ghi rõ **giới hạn tài nguyên**; **không** thay bằng subsample ngầm.

## 5. Sổ đăng ký amendment

| Ngày | Thay đổi | Lý do |
|---|---|---|
| 2026-10-01 | Khóa protocol trước khi chạy | Ngăn HARKing |
| 2026-10-01 | TN-2: chọn cấu hình bằng **một** seed tuning (11) trên validation, sau đó refit **đúng cấu hình đã chọn** cho mọi seed | Giảm chi phí mà không đổi tính công bằng: test vẫn chỉ đọc một lần, chọn cấu hình vẫn chỉ bằng validation |
| 2026-10-01 | TN-2: ExtraTrees/RandomForest chạy trên UNSW và ToN; CSE-CIC và BoT-IoT chỉ chạy HistGradientBoosting nếu vượt RAM | Quy tắc 7 của protocol: ghi rõ giới hạn tài nguyên, không subsample ngầm |
| 2026-10-01 | TN-1: `mlp_h128_2l` (đối chứng độ sâu) chỉ chạy trên UNSW; ToN/CSE/BoT chỉ chạy `mlp_h273_2l` (khớp capacity, contrast trung tâm) và `mlp_h128_1l` (đối chứng harness) | Chi phí CPU: ước tính 10 h nếu giữ đủ ba biến thể; cắt biến thể không thiết yếu |
| 2026-10-01 | TN-1: **CSE-CIC và BoT-IoT chỉ chạy 3 seed {11,22,33}**; UNSW và ToN giữ đủ 5 seed | RAM/CPU cục bộ có hạn; 3 seed vẫn đủ cho contrast ghép cặp và phải được **ghi rõ n** trong mọi bảng. Không cắt dataset, không cắt biến thể trung tâm |
| 2026-10-01 | TN-2: **hoãn ToN/CSE/BoT tới khi TN-1 kết thúc**; chỉ chạy `HistGradientBoosting` cho CSE-CIC và BoT-IoT | TN-2 dùng 9,6 GiB RSS cho ToN và làm cạn swap khi chạy song song TN-1; chạy tuần tự để tránh OOM. Quy tắc 7: ghi rõ giới hạn tài nguyên, không subsample ngầm |
