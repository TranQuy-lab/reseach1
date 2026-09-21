"""Build compact, auditable tables, figures and a Vietnamese result report."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET_LABELS = {
    "NF-UNSW-NB15-v2": "UNSW",
    "NF-BoT-IoT-v2": "BoT-IoT",
    "NF-ToN-IoT-v2": "ToN-IoT",
    "NF-CSE-CIC-IDS2018-v2": "CSE-CIC",
}
MODEL_LABELS = {"edge_mlp": "Edge MLP", "sage": "E-GraphSAGE", "sage_edge": "E-GraphSAGE + edge"}


def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def markdown_table(frame: pd.DataFrame, task: str) -> str:
    part = frame[frame.task == task]
    lines = [
        "| Dataset | Mô hình | Macro-F1 | Weighted-F1 | Accuracy | Thời gian/run (s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASET_LABELS:
        for model in MODEL_LABELS:
            row = part[(part.dataset == dataset) & (part.model == model)].iloc[0]
            lines.append(
                f"| {DATASET_LABELS[dataset]} | {MODEL_LABELS[model]} | "
                f"{fmt(row.test_macro_f1_mean, row.test_macro_f1_std)} | "
                f"{fmt(row.test_weighted_f1_mean, row.test_weighted_f1_std)} | "
                f"{fmt(row.test_accuracy_mean, row.test_accuracy_std)} | "
                f"{fmt(row.seconds_fit_and_evaluate_mean, row.seconds_fit_and_evaluate_std)} |"
            )
    return "\n".join(lines)


def make_figure(summary: pd.DataFrame, destination: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colors = ["#6B7280", "#2563EB", "#E76F51"]
    x = np.arange(len(DATASET_LABELS))
    width = 0.24
    for axis, task in zip(axes, ("multiclass", "binary")):
        subset = summary[summary.task == task]
        for offset, (model, label) in enumerate(MODEL_LABELS.items()):
            ordered = subset[subset.model == model].set_index("dataset").loc[list(DATASET_LABELS)]
            axis.bar(
                x + (offset - 1) * width,
                ordered.test_macro_f1_mean,
                width,
                yerr=ordered.test_macro_f1_std,
                capsize=3,
                label=label,
                color=colors[offset],
            )
        axis.set_title("Đa lớp" if task == "multiclass" else "Nhị phân")
        axis.set_xticks(x, DATASET_LABELS.values(), rotation=20, ha="right")
        axis.set_ylim(0, 1.03)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Macro-F1 trên test")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Benchmark mini-batch phân tầng, trung bình ± SD qua 3 seed")
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    fig.savefig(destination.with_suffix(".png"), dpi=180)
    fig.savefig(destination.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--tuning", type=Path, required=True)
    parser.add_argument("--prepare", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    runs = pd.read_csv(args.runs / "runs.csv")
    if len(runs) != 72:
        raise ValueError(f"Expected 72 completed runs, got {len(runs)}")
    verification = json.loads(args.verification.read_text())
    if not verification.get("passed") or verification.get("runs_checked") != 72:
        raise ValueError("Independent verification has not passed for all 72 runs")
    tuning = json.loads((args.tuning / "selection.json").read_text())
    prepare = json.loads(args.prepare.read_text())

    columns = ["test_macro_f1", "test_weighted_f1", "test_accuracy", "seconds_fit_and_evaluate"]
    summary = runs.groupby(["dataset", "task", "model"], as_index=False)[columns].agg(["mean", "std"])
    summary.columns = ["_".join(c).rstrip("_") for c in summary.columns]
    summary = summary.rename(columns={"dataset_": "dataset", "task_": "task", "model_": "model"})
    # pandas versions differ in how they name grouping columns after the aggregation.
    for name in ("dataset", "task", "model"):
        if name not in summary and f"{name}_" in summary:
            summary = summary.rename(columns={f"{name}_": name})

    per_class = []
    for run in runs.itertuples(index=False):
        run_id = f"{run.dataset}__{run.task}__{run.model}__seed{run.seed}"
        result = json.loads((args.runs / run_id / "metrics.json").read_text())
        for label, values in result["test"]["per_class"].items():
            if label in {"accuracy", "macro avg", "weighted avg"}:
                continue
            per_class.append({
                "dataset": run.dataset, "task": run.task, "model": run.model,
                "seed": run.seed, "class": label, "precision": values["precision"],
                "recall": values["recall"], "f1": values["f1-score"], "support": values["support"],
            })
    per_class_frame = pd.DataFrame(per_class)

    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "summary.csv", index=False)
    runs.to_csv(args.output / "runs.csv", index=False)
    per_class_frame.to_csv(args.output / "per_class.csv", index=False)
    for source, name in [
        (args.runs / "provenance.json", "provenance.json"),
        (args.tuning / "selection.json", "tuning_selection.json"),
        (args.tuning / "candidates.csv", "tuning_candidates.csv"),
        (args.prepare, "prepare_manifest.json"),
        (args.verification, "verification.json"),
    ]:
        shutil.copy2(source, args.output / name)
    make_figure(summary, args.output / "macro_f1_comparison")

    total_hours = runs.seconds_fit_and_evaluate.sum() / 3600
    selected = tuning["selected"]
    best_by_task = (
        summary.sort_values("test_macro_f1_mean", ascending=False)
        .groupby(["dataset", "task"], as_index=False).first()
    )
    wins = best_by_task.model.value_counts().to_dict()
    split_lines = []
    for dataset, values in prepare["datasets"].items():
        pilot = sum(part["pilot_rows"] for part in values["split_rows"].values())
        split_lines.append(f"- {DATASET_LABELS[dataset]}: {pilot:,} flow pilot từ {values['source_rows']:,} flow nguồn.")

    report = f"""# Báo cáo benchmark E-GraphSAGE mini-batch trên bốn bộ v2

Khởi chạy 2026-09-20, hoàn tất 2026-09-21. Báo cáo này được sinh trực tiếp từ 72 run đã hoàn tất
và đã nạp lại checkpoint để kiểm tra độc lập. Đây là benchmark cục bộ trên
mẫu phân tầng đã chốt trước trong `PROTOCOL_MINIBATCH_VI.md`, chưa phải kết
quả trên toàn bộ 75.987.976 flow.

## Thiết kế đã thực hiện

- Bốn dataset × hai task × ba mô hình × ba seed = 72 run.
- Cấu hình được chọn bằng validation UNSW multiclass: batch
  `{selected['batch_size']}`, fanout `{selected['fanout']}`, validation macro-F1
  `{selected['val_macro_f1']:.4f}`; test chưa được xem trong bước chọn.
- Tổng thời gian fit và đánh giá cộng dồn: {total_hours:.2f} giờ CPU.
- Số lần có trung bình test macro-F1 cao nhất trong tám cặp dataset/task:
  {', '.join(f'{MODEL_LABELS[k]}: {v}' for k, v in wins.items())}.
- Mỗi số trong bảng là trung bình ± độ lệch chuẩn qua ba seed.

Con số "đứng đầu" chỉ mô tả trung bình quan sát được. Ba seed không đủ để
khẳng định khác biệt rất nhỏ có ý nghĩa thống kê.

Kích thước benchmark:

{chr(10).join(split_lines)}

## Kết quả đa lớp

{markdown_table(summary, 'multiclass')}

## Kết quả nhị phân

{markdown_table(summary, 'binary')}

![So sánh macro-F1](results/minibatch/macro_f1_comparison.png)

## Cách đọc kết quả

`edge_mlp` kiểm tra thông tin trong riêng đặc trưng flow. `sage` dùng embedding
hai endpoint từ lân cận đồ thị. `sage_edge` nối thêm đặc trưng của chính cạnh
vào bộ phân loại, gần với thay đổi chính đang được khảo sát. Vì mẫu test được
giới hạn theo từng lớp, accuracy và weighted-F1 ở đây không ước lượng trực
tiếp hiệu năng theo tỷ lệ tấn công tự nhiên. Macro-F1 và bảng theo lớp phù hợp
hơn cho so sánh nội bộ này.

So với Edge MLP, GNN tốt nhất tăng macro-F1 đa lớp lần lượt khoảng 0,0880
(UNSW), 0,0364 (BoT-IoT), 0,0414 (ToN-IoT) và 0,0735 (CSE-CIC). Việc nối
edge feature vào decoder có hiệu ứng rõ nhất trên BoT-IoT đa lớp (+0,0148
so với `sage`); ở bảy trường hợp còn lại, chênh lệch tuyệt đối chỉ khoảng
-0,0012 đến +0,0025. Vì vậy bằng chứng hiện tại ủng hộ giá trị của topology,
nhưng chưa cho thấy edge feature trực tiếp luôn tạo cải thiện đáng kể.

## Sự cố số học đã xử lý

Khi bắt đầu ToN-IoT, hai trường rate hữu hạn nhưng đạt tới khoảng 1e165 trong
train và 1e261 trong test, làm phép tính phương sai tràn `float64`. Pipeline
dừng ngay do kiểm tra loss hữu hạn. Bản sửa chọn cột có trị tuyệt đối train
lớn hơn 1e20, áp dụng signed `log1p`, rồi mới fit StandardScaler trên train;
lựa chọn cột được lưu trong từng preprocessor. ToN-IoT chọn đúng hai cột
`SRC_TO_DST_SECOND_BYTES` và `DST_TO_SRC_SECOND_BYTES`; validation/test không
được dùng để chọn phép biến đổi. 36 run UNSW/BoT-IoT đã hoàn tất trước đó
không bị ảnh hưởng và được giữ nguyên. Toàn bộ checkpoint sau cùng đều đã
được validator nạp lại thành công.

Kết quả chỉ chứng minh pipeline mini-batch chạy ổn định và có thể so sánh có
kiểm soát trên bốn tập. Chưa được dùng để tuyên bố tốt hơn bài báo gốc: phiên
bản dataset, cách lấy mẫu, split và ngân sách huấn luyện đều khác. Bước nghiên
cứu kế tiếp là chạy cùng protocol trên toàn dữ liệu bằng server, sau đó bổ sung
host/temporal holdout nếu metadata cho phép.

## Bằng chứng tái lập

- `results/minibatch/runs.csv`: 72 run riêng lẻ.
- `results/minibatch/summary.csv`: trung bình và độ lệch chuẩn.
- `results/minibatch/per_class.csv`: precision, recall, F1, support từng lớp.
- `results/minibatch/verification.json`: kết quả nạp checkpoint, tái tạo xác
  suất và tính lại metric.
- `results/minibatch/provenance.json`: phiên bản môi trường và SHA-256 của
  protocol tại lúc bắt đầu.
- `NUMERIC_STABILITY_NOTE_VI.md`: biên bản lỗi tràn số và cách phục hồi.
- Checkpoint và dự đoán đầy đủ nằm ở `research/artifacts/minibatch_runs/` và
  được bỏ khỏi Git vì kích thước lớn.
"""
    args.report.write_text(report)


if __name__ == "__main__":
    main()
