"""Build the compact full-data result tables and Vietnamese report."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASETS = ["NF-UNSW-NB15-v2", "NF-BoT-IoT-v2", "NF-ToN-IoT-v2", "NF-CSE-CIC-IDS2018-v2"]
MODELS = ["edge_mlp", "sage", "sage_edge"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_deltas(runs: pd.DataFrame) -> pd.DataFrame:
    wide = runs.pivot(
        index=["dataset", "task", "seed"], columns="model",
        values="test_macro_f1",
    ).reset_index()
    rows = []
    comparisons = [
        ("sage", "edge_mlp"),
        ("sage_edge", "edge_mlp"),
        ("sage_edge", "sage"),
    ]
    for row in wide.itertuples(index=False):
        for left, right in comparisons:
            rows.append({
                "dataset": row.dataset, "task": row.task, "seed": row.seed,
                "comparison": f"{left}_minus_{right}",
                "macro_f1_delta": float(getattr(row, left) - getattr(row, right)),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--prepare", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    runs = pd.read_csv(args.runs / "runs.csv")
    verification = json.loads(args.verification.read_text())
    benchmark = json.loads(args.benchmark.read_text())
    prepare = json.loads(args.prepare.read_text())
    if len(runs) != 72 or runs[["dataset", "task", "model", "seed"]].duplicated().any():
        raise ValueError("Full-data report requires 72 unique runs")
    if verification.get("passed") is not True or verification.get("runs_checked") != 72:
        raise ValueError("Independent verification has not passed")

    metrics = ["test_macro_f1", "test_weighted_f1", "test_accuracy", "seconds_fit_and_evaluate"]
    summary = runs.groupby(["dataset", "task", "model"], as_index=False)[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(x).rstrip("_") for x in summary.columns]
    per_class = []
    for row in runs.itertuples(index=False):
        run_id = f"{row.dataset}__{row.task}__{row.model}__seed{row.seed}"
        result = json.loads((args.runs / run_id / "metrics.json").read_text())
        for label, values in result["test"]["per_class"].items():
            if label not in {"accuracy", "macro avg", "weighted avg"}:
                per_class.append({"dataset": row.dataset, "task": row.task,
                                  "model": row.model, "seed": row.seed,
                                  "class": label, **values})

    args.output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output / "runs.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    per_class_frame = pd.DataFrame(per_class)
    per_class_frame.to_csv(args.output / "per_class.csv", index=False)
    deltas = paired_deltas(runs)
    deltas.to_csv(args.output / "paired_seed_deltas.csv", index=False)
    rare = (
        per_class_frame[per_class_frame["support"] < 1_000]
        .groupby(["dataset", "task", "model", "class"], as_index=False)
        .agg(support=("support", "first"), f1_mean=("f1-score", "mean"),
             f1_std=("f1-score", "std"))
    )
    rare.to_csv(args.output / "rare_class_warning.csv", index=False)
    for source, name in (
        (args.runs / "provenance.json", "provenance.json"),
        (args.verification, "verification.json"),
        (args.benchmark, "benchmark_estimate.json"),
        (args.prepare, "prepare_manifest.json"),
    ):
        shutil.copy2(source, args.output / name)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    x = np.arange(4)
    for axis, task in zip(axes, ("multiclass", "binary")):
        part = summary[summary.task == task]
        for i, model in enumerate(MODELS):
            ordered = part[part.model == model].set_index("dataset").loc[DATASETS]
            axis.bar(x + (i - 1) * 0.24, ordered.test_macro_f1_mean, 0.24,
                     yerr=ordered.test_macro_f1_std, capsize=3, label=model)
        axis.set_xticks(x, ["UNSW", "BoT", "ToN", "CSE"], rotation=20)
        axis.set_title(task)
        axis.set_ylim(0, 1.03)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Test macro-F1")
    fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center", ncol=3)
    fig.suptitle("Full-data E-GraphSAGE: mean ± SD across three seeds")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(args.output / "macro_f1_full.png", dpi=180)
    fig.savefig(args.output / "macro_f1_full.svg")
    plt.close(fig)

    provenance_inputs = [
        args.runs / "runs.csv", args.runs / "provenance.json",
        args.verification, args.benchmark, args.prepare, Path(__file__),
    ]
    figure_outputs = [
        args.output / "macro_f1_full.png", args.output / "macro_f1_full.svg",
    ]
    figure_provenance = {
        "generator": str(Path(__file__)),
        "inputs_sha256": {str(path): sha256_file(path) for path in provenance_inputs},
        "outputs_sha256": {path.name: sha256_file(path) for path in figure_outputs},
    }
    (args.output / "figure_provenance.json").write_text(
        json.dumps(figure_provenance, indent=2) + "\n"
    )

    total_hours = runs.seconds_fit_and_evaluate.sum() / 3600
    report = f"""# Báo cáo full-data E-GraphSAGE trên bốn bộ v2

Pipeline đã xử lý toàn bộ 75.987.976 flow, gồm bốn dataset, hai task, ba mô
hình và ba seed, tổng cộng 72 run. Không có bước lấy mẫu bỏ bớt flow train,
validation hoặc test. Batch chỉ là cơ chế đưa toàn bộ cạnh qua GPU.

- Tổng thời gian fit và đánh giá cộng dồn: {total_hours:.2f} giờ.
- Validator độc lập đã nạp và kiểm tra {verification['runs_checked']} checkpoint.
- Benchmark trước khi chạy dùng hệ số an toàn {benchmark['safety_factor']}.
- Bảng tổng hợp: `research/results/full/summary.csv`.
- Chỉ số từng lớp: `research/results/full/per_class.csv`.
- Chênh lệch ghép cặp theo cùng seed: `paired_seed_deltas.csv`.
- Cảnh báo lớp có support dưới 1.000: `rare_class_warning.csv`.
- Biểu đồ: `research/results/full/macro_f1_full.png` và `.svg`.

Split được kiểm tra không trùng `flow_group_id`. Tỉ lệ IP validation/test đã
thấy trong train được ghi trong `prepare_manifest.json`; đây là phép đo nguy cơ
rò rỉ danh tính host để giới hạn diễn giải, không phải lỗi chia nhóm flow.
Số nhóm có nhãn mâu thuẫn và số dòng liên quan cũng được báo riêng theo split.

Kết luận khoa học phải dựa trên macro-F1, chỉ số từng lớp và độ lệch chuẩn giữa
ba seed. Bảng paired delta chỉ mang tính mô tả; ba seed không đủ cho khoảng tin
cậy bootstrap ổn định. Các lớp trong `rare_class_warning.csv` không được dùng
để đưa ra kết luận chắc chắn.
"""
    args.report.write_text(report)


if __name__ == "__main__":
    main()
