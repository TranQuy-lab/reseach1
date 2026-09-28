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


def resolve_run_dir(runs_root: Path, provenance: dict, run_id: str) -> Path:
    direct = runs_root / run_id
    if direct.is_dir():
        return direct
    for source in provenance.get("merged_from", []):
        candidate = Path(source) / run_id
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Missing run artifacts for {run_id}; checked merged source directories"
    )


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
    provenance = json.loads((args.runs / "provenance.json").read_text())
    expected_runs = (
        len(provenance["datasets"]) * len(provenance["tasks"])
        * len(provenance["models"]) * len(provenance["seeds"])
    )
    if len(runs) != expected_runs or runs[["dataset", "task", "model", "seed"]].duplicated().any():
        raise ValueError(f"Full-data report requires {expected_runs} unique runs")
    if verification.get("passed") is not True or verification.get("runs_checked") != expected_runs:
        raise ValueError("Independent verification has not passed")
    seed_count = len(provenance["seeds"])

    metrics = ["test_macro_f1", "test_weighted_f1", "test_accuracy", "seconds_fit_and_evaluate"]
    summary = runs.groupby(["dataset", "task", "model"], as_index=False)[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(x).rstrip("_") for x in summary.columns]
    per_class = []
    for row in runs.itertuples(index=False):
        run_id = f"{row.dataset}__{row.task}__{row.model}__seed{row.seed}"
        run_dir = resolve_run_dir(args.runs, provenance, run_id)
        result = json.loads((run_dir / "metrics.json").read_text())
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
        destination = args.output / name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)

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
    fig.suptitle(f"Full-data E-GraphSAGE: mean ± SD across {seed_count} seeds")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(args.output / "macro_f1_full.png", dpi=180)
    fig.savefig(args.output / "macro_f1_full.svg")
    plt.close(fig)
    svg_path = args.output / "macro_f1_full.svg"
    svg_path.write_text("\n".join(
        line.rstrip() for line in svg_path.read_text().splitlines()
    ) + "\n")

    provenance_inputs = [
        args.runs / "runs.csv", args.runs / "provenance.json",
        args.verification, args.benchmark, args.prepare, Path(__file__),
    ]
    figure_outputs = [
        args.output / "macro_f1_full.png", args.output / "macro_f1_full.svg",
    ]
    root = Path.cwd().resolve()
    def display_path(path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            return resolved.as_posix()

    figure_provenance = {
        "generator": display_path(Path(__file__)),
        "inputs_sha256": {display_path(path): sha256_file(path) for path in provenance_inputs},
        "outputs_sha256": {path.name: sha256_file(path) for path in figure_outputs},
    }
    (args.output / "figure_provenance.json").write_text(
        json.dumps(figure_provenance, indent=2) + "\n"
    )

    total_hours = runs.seconds_fit_and_evaluate.sum() / 3600
    output_label = args.output.as_posix()
    report = f"""# Báo cáo full-data E-GraphSAGE trên bốn bộ v2

Pipeline đã xử lý toàn bộ 75.987.976 flow, gồm bốn dataset, hai task, ba mô
hình và {seed_count} seed, tổng cộng {expected_runs} run. Full-data ở đây có
nghĩa là mọi flow trong split đều được sử dụng; huấn luyện vẫn dùng neighbor
mini-batch và đánh giá dùng layer-wise full-graph inference.

- Tổng thời gian fit và đánh giá cộng dồn: {total_hours:.2f} giờ.
- Một verifier tách khỏi training đã nạp {verification['runs_checked']} checkpoint,
  chạy full-test inference, tính lại metric trên toàn bộ test và đối chiếu xác
  suất trên các dòng dự đoán được lưu.
- Benchmark trước khi chạy dùng hệ số an toàn {benchmark['safety_factor']}.
- Bảng tổng hợp: `{output_label}/summary.csv`.
- Chỉ số từng lớp: `{output_label}/per_class.csv`.
- Chênh lệch ghép cặp theo cùng seed: `{output_label}/paired_seed_deltas.csv`.
- Cảnh báo lớp có support dưới 1.000: `{output_label}/rare_class_warning.csv`.
- Biểu đồ: `{output_label}/macro_f1_full.png` và `.svg`.

Split được kiểm tra không trùng `flow_group_id`. Tỉ lệ IP validation/test đã
thấy trong train được ghi trong `prepare_manifest.json`; đây là phép đo nguy cơ
rò rỉ danh tính host để giới hạn diễn giải, không phải lỗi chia nhóm flow.
Số nhóm có nhãn mâu thuẫn và số dòng liên quan cũng được báo riêng theo split.

Kết luận khoa học phải dựa trên macro-F1, chỉ số từng lớp và độ lệch chuẩn giữa
{seed_count} seed. Seed là lặp tính toán trên cùng dataset, không phải mạng độc
lập. Bảng paired delta mang tính mô tả; với {seed_count} cặp, không dùng khoảng
tin cậy chuẩn hoặc p-value như bằng chứng duy nhất. Các lớp trong
`rare_class_warning.csv` chỉ được diễn giải như kết quả khám phá.
"""
    args.report.write_text(report)


if __name__ == "__main__":
    main()
