# Protocol RF bounded — external comparator cho bài báo

Trạng thái: **khóa trước khi chạy**. Đây là comparator tiết kiệm tài nguyên, không thay thế kết quả GNN full-scale 120 run.

## Thiết kế

- Dữ liệu: đúng `data/full_splits`, không chia lại và không lấy mẫu.
- Dataset: NF-UNSW-NB15-v2, NF-BoT-IoT-v2, NF-ToN-IoT-v2, NF-CSE-CIC-IDS2018-v2.
- Task: multiclass.
- Model: Random Forest duy nhất.
- Seed: `{11, 22}`.
- Số run: `4 dataset × 1 task × 2 seed = 8`.
- Features/preprocessor: dùng preprocessor đã persist từ `full_runs_5seed`, fit trên train của cùng dataset/task.
- Cây: `n_estimators=25`, `max_depth=16`, `max_features='sqrt'`, `class_weight='balanced'`.
- Test không dùng để chọn cấu hình.
- Primary metric: test macro-F1; secondary metrics: weighted-F1, accuracy, per-class metrics.

## Phạm vi claim

Kết quả chỉ cho phép nói:

> Random Forest bounded comparator được đánh giá trên cùng full-data split và preprocessing ở bốn dataset multiclass.

Không được gọi đây là:

- full RF seed-stability study;
- so sánh với mọi tabular baseline;
- bằng chứng GNN vượt Random Forest nếu chưa chạy cấu hình RF đầy đủ.

## Verification

Mỗi run phải reload `model.joblib`, tính lại metric trên toàn bộ test split, đối chiếu source row ID và probability artifact được lưu bounded. Cổng hoàn tất yêu cầu:

```text
runs_checked = 8
passed = true
```

Kết quả được dùng như comparator chi phí thấp và phải được ghi rõ `n_estimators=25`, `max_depth=16`, seed 11 và 22 trong báo cáo.
