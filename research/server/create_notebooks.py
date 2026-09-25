"""Generate thin, reviewable notebooks that call the tested CLI pipeline."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks/server"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


SETUP = """from pathlib import Path
import os, subprocess, sys

ROOT = Path.cwd()
if not (ROOT / 'research/server').is_dir():
    raise RuntimeError('Hãy mở JupyterLab từ thư mục gốc repository')
DEVICE = os.environ.get('NIDS_DEVICE', 'auto')
THREADS = os.environ.get('NIDS_THREADS', '8')

def stage(name):
    command = [sys.executable, '-u', 'research/server/run_pipeline.py',
               '--stage', name, '--device', DEVICE, '--threads', THREADS]
    print(' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)

print({'root': str(ROOT), 'device': DEVICE, 'threads': THREADS, 'python': sys.executable})
"""

SETUP_FULL = """from pathlib import Path
import json, os, subprocess, sys

ROOT = Path.cwd()
if not (ROOT / 'research/server').is_dir():
    raise RuntimeError('Hãy mở JupyterLab từ thư mục gốc repository')
THREADS = os.environ.get('NIDS_THREADS', '12')

def full_stage(name):
    command = [sys.executable, '-u', 'research/server/run_full_pipeline.py',
               '--stage', name, '--threads', THREADS]
    print(' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)

print({'root': str(ROOT), 'threads': THREADS, 'python': sys.executable,
       'pipeline': 'full-data benchmark-gated'})
"""


def notebook(title: str, description: str, stages: list[str], tail: str = "") -> dict:
    cells = [
        markdown(f"# {title}\n\n{description}\n\nChọn kernel **NIDS E-GraphSAGE server** trước khi chạy."),
        code(SETUP),
    ]
    for name in stages:
        cells.append(markdown(f"## Stage `{name}`\n\nStage có checkpoint và có thể chạy lại an toàn."))
        cells.append(code(f"stage('{name}')\n"))
    if tail:
        cells.append(code(tail))
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "NIDS E-GraphSAGE server", "language": "python", "name": "nids-server"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def full_notebook(title: str, description: str, cells: list[dict]) -> dict:
    return {
        "cells": [
            markdown(f"# {title}\n\n{description}\n\nChọn kernel **NIDS E-GraphSAGE server** trước khi chạy."),
            code(SETUP_FULL),
            *cells,
        ],
        "metadata": {
            "kernelspec": {"display_name": "NIDS E-GraphSAGE server", "language": "python", "name": "nids-server"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    specs = [
        ("00_CHECK_SERVER.ipynb", "00 — Kiểm tra server",
         "Kiểm tra CPU, RAM, ổ đĩa, CUDA, PyTorch và PyG trước khi tải dữ liệu.", ["check"], ""),
        ("01_DOWNLOAD_PREPROCESS.ipynb", "01 — Tải và tiền xử lý",
         "Tải CSV công khai từ Kaggle theo byte-range, kiểm kê 75.987.976 dòng, tách bốn Parquet và kiểm tra độc lập.",
         ["download", "preprocess"], ""),
        ("02_SPLIT_TUNE.ipynb", "02 — Chia tập và chọn cấu hình",
         "Chia theo flow_group_id, tạo pilot phân tầng và chọn batch/fanout chỉ bằng validation UNSW đa lớp.",
         ["split", "tune"], ""),
        ("03_TRAIN_72_RUNS.ipynb", "03 — Huấn luyện 72 run",
         "Chạy bốn dataset × hai task × ba mô hình × ba seed. Run đã đủ artifact được bỏ qua khi tiếp tục.",
         ["train"], ""),
        ("04_VERIFY_REPORT.ipynb", "04 — Xác minh và báo cáo",
         "Nạp lại 72 checkpoint, tính lại metric, tạo bảng/biểu đồ và chạy toàn bộ test.",
         ["verify", "report", "test"],
         "import pandas as pd\nfrom IPython.display import display, Markdown\ndisplay(pd.read_csv(ROOT / 'research/results/minibatch/summary.csv'))\ndisplay(Markdown((ROOT / 'research/MINIBATCH_REPORT_VI.md').read_text()))\n"),
    ]
    for filename, title, description, stages, tail in specs:
        (OUT / filename).write_text(json.dumps(notebook(title, description, stages, tail), indent=1, ensure_ascii=False) + "\n")
    full_specs = {
        "10_FULL_PREPARE_BENCHMARK.ipynb": full_notebook(
            "10 — Full-data: chia tập và benchmark giới hạn",
            "Tạo split cho toàn bộ 75.987.976 flow rồi đo 500 batch cho từng cặp dataset/mô hình, bỏ warm-up và tổng hợp theo nhiều cửa sổ. Notebook này không khởi chạy 72 run.",
            [
                markdown("## Chạy cổng chuẩn bị\n\nKết quả ghi ETA cho ngân sách 20.000 optimizer step/run và trạng thái `safe_to_launch_72`."),
                code("full_stage('prepare')\n"),
                code("estimate = json.loads((ROOT / 'research/results/full_benchmark_estimate.json').read_text())\nprint(json.dumps(estimate, indent=2))\n"),
            ],
        ),
        "11_FULL_TRAIN_72.ipynb": full_notebook(
            "11 — Full-data: huấn luyện 72 run",
            "Chỉ chạy khi benchmark đã hoàn tất và cổng tài nguyên đạt. Checkpoint chỉ hợp lệ sau một lượt đầy đủ qua train; batch không làm giảm số flow.",
            [
                code("estimate = json.loads((ROOT / 'research/results/full_benchmark_estimate.json').read_text())\nif estimate.get('safe_to_launch_72') is not True:\n    raise RuntimeError('Cổng benchmark chưa đạt; không được chạy 72 run')\nprint(json.dumps(estimate['step_budget_estimate'], indent=2))\n"),
                markdown("## Khởi chạy có resume\n\nRun đã hoàn thành được giữ nguyên khi notebook chạy lại."),
                code("full_stage('train')\n"),
            ],
        ),
        "12_FULL_VERIFY_REPORT.ipynb": full_notebook(
            "12 — Full-data: xác minh và báo cáo",
            "Nạp lại checkpoint, tính lại metric full-test, tạo bảng, biểu đồ và báo cáo riêng cho full-data.",
            [
                code("full_stage('verify')\nfull_stage('report')\nfull_stage('test')\n"),
                code("import pandas as pd\nfrom IPython.display import display, Markdown\ndisplay(pd.read_csv(ROOT / 'research/results/full/summary.csv'))\ndisplay(Markdown((ROOT / 'research/FULL_DATA_REPORT_VI.md').read_text()))\n"),
            ],
        ),
    }
    for filename, value in full_specs.items():
        (OUT / filename).write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n")
    print(f"generated {len(specs) + len(full_specs)} notebooks in {OUT}")


if __name__ == "__main__":
    main()
