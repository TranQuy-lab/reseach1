# reseach1 — E-GraphSAGE trên NF-UQ-NIDS-v2

Repository hoàn chỉnh gồm mã nguồn, kế hoạch nghiên cứu, notebook chạy server
và bốn bộ dữ liệu v2 đã tiền xử lý. Các file Parquet ở thư mục gốc dùng Git
LFS và vẫn giữ IP cùng port để xây dựng định danh node `(IP, port)`.

| Dataset | Rows |
|---|---:|
| NF-UNSW-NB15-v2 | 2,390,275 |
| NF-BoT-IoT-v2 | 37,763,497 |
| NF-ToN-IoT-v2 | 16,940,496 |
| NF-CSE-CIC-IDS2018-v2 | 18,893,708 |
| **Tổng** | **75,987,976** |

Để chạy trên server, đọc
[hướng dẫn server](research/server/README_SERVER_VI.md). Pipeline kết quả
chính chạy lần lượt ba notebook `10_FULL_*`–`12_FULL_*`; notebook 10 đo thời
gian và tài nguyên trước khi notebook 11 được phép chạy đủ 72 cấu hình.

## Pipeline tiền xử lý nền tảng

> **Phạm vi chính đã chốt (2026-09-20):** chạy riêng bốn bộ **v2** của
> NF-UQ-NIDS-v2. Đọc [tài liệu chính và hướng dẫn chuẩn bị dữ liệu](research/RESEARCH_PLAN_VI.md).
> Đã tải raw CSV và [xuất bốn file giữ IP/port](research/PREPROCESSING_FOUR_VI.md).
> Benchmark mini-batch 72 run đã hoàn tất: xem
> [báo cáo kết quả](research/MINIBATCH_REPORT_VI.md),
> [protocol](research/PROTOCOL_MINIBATCH_VI.md) và
> [hướng dẫn chạy lại](research/MINIBATCH_RUNBOOK_VI.md).
> Để chuyển sang máy chủ, dùng [gói server](research/server/README_SERVER_VI.md).
> Ba notebook `10_FULL_*`–`12_FULL_*` là pipeline kết quả chính trên toàn bộ
> 75.987.976 flow; năm notebook `00`–`04` chỉ giữ benchmark pilot lịch sử.
> Pilot dưới đây chỉ dùng Parquet NF-ToN-IoT-v2 lịch sử.

> **Khởi động lại nghiên cứu (2026-09-19):** xem
> [hướng dẫn chạy pilot](research/RUNBOOK_VI.md) và
> [protocol đã chốt trước thí nghiệm](research/PROTOCOL.md).
> Nhánh mới ở `src/nids_research/`, độc lập với các script cũ.
> README phía dưới mô tả **nhánh tiền xử lý dữ liệu bảng**, không phải
> toàn bộ quy trình E-GraphSAGE. Kết quả/báo cáo cũ chưa được tái xác nhận.

This is a working, tested re-implementation of the team's reference cleaning
notebook (`NF-UQ-NIDS-V2-00-Cleaning`, Kaggle), turned into modular,
documented, reusable code, plus a stratified train/val/test split stage so
Person B (baseline models) and Person C (proposed algorithm) can both load
identical, ready-to-train data.

**Scope**: this covers preprocessing only -- loading the raw CSV, cleaning it
exactly like the reference notebook, and splitting it. It does not choose or
train any model; the algorithm/topic decision is still open per the team's
planning doc, and this pipeline's output is deliberately algorithm-agnostic
so it doesn't need to wait on that decision.

## 1. What's verified vs. what's new

Every non-trivial claim/step below traces to one of these:

| Claim | Source | Where in this repo |
|---|---|---|
| Raw dtypes, row/column counts, NaN count (384), duplicate count (13,315,579) | Reference notebook cell outputs (cells 3, 5-8, 11-13) | `schema.RAW_DTYPES`, `schema.REFERENCE_STATS` |
| Post-shrink dtypes (e.g. `Label` -> int8, `IN_BYTES` -> int32) | Reference notebook cell 11 output | `schema.SHRUNK_DTYPES` |
| Cleaning order: drop IPs -> downcast -> drop inf/NaN -> drop duplicates | Reference notebook cell order (9-13) | `cleaning.clean_pipeline` |
| The four merged sub-datasets are NF-UNSW-NB15-v2 / NF-BoT-IoT-v2 / NF-ToN-IoT-v2 / NF-CSE-CIC-IDS2018-v2 | Dataset documentation supplied by the team | `schema.VALID_DATASETS` |
| `Label == 0 <=> Attack == "Benign"` (Label is fully derived from Attack) | **Verified here** by summing every non-Benign `Attack` value_count from the notebook's own cell 8 output: 21,748,351 + 17,875,585 + ... + 164 = 50,822,681, exactly matching `Label==1`'s count from cell 7 | `schema.BENIGN_VALUE`, checked at runtime by `cleaning.validate_label_matches_attack` |
| Dropping IP-based features is a documented practice for NF-UQ-NIDS-style preprocessing, to avoid classifiers keying on host identity rather than traffic behaviour | An earlier related paper by the same author group (Sarhan, Layeghy, Moustafa, Portmann, *Big Data Technologies and Applications*, 2021) is reported elsewhere as excluding IP addresses **and their ports** from its 8-of-12-feature preprocessing | See "Open decision #1" below -- the reference notebook only drops IPs, not ports |

Anything marked **IMPROVEMENT** in the code's docstrings is new relative to
the reference notebook, with the reasoning written next to it. Nothing here
invents dataset statistics, paper claims, or column semantics that weren't
in the supplied sources.

## 2. Repo layout

```
src/nids_preprocessing/
  schema.py      # column dtypes, reference stats, verified facts, open decisions
  cleaning.py    # load_raw_csv, drop_identifier_columns, downcast_dtypes,
                 # remove_invalid_rows, remove_duplicates, validate_label_matches_attack,
                 # clean_pipeline (orchestrates all of the above)
  sample.py      # stratified_subsample: shrinks the cleaned dataset down to a
                 # manageable training subset, stratified by (Dataset, Attack)
                 # jointly, with rare (Dataset, Attack) groups kept whole
                 # instead of sampled away. Satisfies Prompt A ràng buộc #4.
  splitting.py   # filter_by_dataset, stratified_split, cross_dataset_split,
                 # save_splits -- the two split strategies required side by
                 # side by Prompt A ràng buộc #5
  pipeline.py    # CLI: load -> clean -> filter -> sample -> split -> save -> log
tests/
  fixtures/make_fixture.py  # builds sample_raw.csv from 8 REAL sample rows
                             # (copied from the dataset docs) + 3 clearly
                             # labelled SYNTHETIC edge-case rows (a duplicate,
                             # a missing value, an infinite value)
  test_schema.py / test_cleaning.py / test_splitting.py / test_sample.py
                             # 32 tests, all passing
requirements.txt
pyproject.toml
```

## 3. Setup

```bash
pip install -r requirements.txt
# or, to get the `nids_preprocessing` package importable anywhere:
pip install -e .
```

## 4. Running the tests

```bash
python -m pytest tests/ -v
```

All 32 tests pass against the bundled fixture and synthetic in-memory data.
They do **not** require the real 76M-row CSV -- `test_cleaning.py` /
`test_schema.py` run against 8 real sample rows (copied from the dataset
documentation) plus 3 synthetic edge-case rows built specifically to
exercise the NaN/inf/duplicate-removal logic; `test_splitting.py` /
`test_sample.py` run against synthetic in-memory DataFrames sized to exercise
rare-class protection and cross-dataset partitioning. Regenerate the fixture
with `python tests/fixtures/make_fixture.py` if you ever need to change it.

## 5. Running the real pipeline

```bash
# Smoke test on the first 200k RAW rows only (just checks the code runs
# without crashing -- this is NOT the stratified sub-sample from Prompt A
# ràng buộc #4, see below for that):
python -m nids_preprocessing.pipeline \
  --input /path/to/NF-UQ-NIDS-v2.csv \
  --output-dir ./smoke_output \
  --sample-rows 200000

# Full run on the ENTIRE cleaned dataset, stratified train/val/test split,
# plus a cross-check against the reference notebook's own numbers (only
# meaningful on the full, unfiltered, un-sampled CSV):
python -m nids_preprocessing.pipeline \
  --input /path/to/NF-UQ-NIDS-v2.csv \
  --output-dir ./output \
  --test-size 0.2 --val-size 0.1 --seed 42 \
  --compare-to-reference

# Real stratified SUB-SAMPLE (Prompt A ràng buộc #4): shrink the ~76M-row
# cleaned dataset to ~2M rows, stratified jointly by (Dataset, Attack), with
# any (Dataset, Attack) group smaller than --min-rows-per-group kept in full
# rather than sampled away -- then split the sample:
python -m nids_preprocessing.pipeline \
  --input /path/to/NF-UQ-NIDS-v2.csv \
  --output-dir ./output \
  --target-sample-rows 2000000 --min-rows-per-group 50 \
  --test-size 0.2 --val-size 0.1 --seed 42

# Cross-dataset split (Prompt A ràng buộc #5b): train on 3 of the 4 merged
# sub-datasets, test entirely on the 4th, to measure generalisation to a
# genuinely different network rather than to unseen rows of the same one:
python -m nids_preprocessing.pipeline \
  --input /path/to/NF-UQ-NIDS-v2.csv \
  --output-dir ./output \
  --split-strategy cross_dataset --held-out-dataset NF-BoT-IoT-v2 \
  --val-size 0.1 --seed 42
```

Outputs land in `<output-dir>/`:
- `NF-UQ-NIDS-V2_clean.parquet` -- the full cleaned dataset (mirrors the
  reference notebook's own `.to_parquet` output, cell 14)
- `NF-UQ-NIDS-V2_sampled.parquet` -- only written if `--target-sample-rows`
  / `--target-sample-frac` was passed; the stratified sub-sample the splits
  below are then built from
- `splits/train.parquet`, `splits/val.parquet` (if requested), `splits/test.parquet`
- `run_summary.json` -- every parameter and stat from the run, including the
  reference comparison and the full per-(Dataset,Attack) sampling retention
  report if requested
- `results_log.csv` -- one row appended per run (see note below)

`--target-sample-rows` and `--target-sample-frac` are mutually exclusive; if
neither is given the full cleaned dataset is split directly (no
sub-sampling). `--split-strategy` defaults to `stratified`; pass
`cross_dataset` with `--held-out-dataset <name>` to use the other strategy
instead -- `--test-size` is ignored in that mode since the test set is
defined entirely by which sub-dataset is held out.

## 6. What Person B and Person C should load

```python
import pandas as pd
train = pd.read_parquet("output/splits/train.parquet")
test  = pd.read_parquet("output/splits/test.parquet")

X_train = train.drop(columns=["Label", "Attack"])
y_train_binary     = train["Label"]   # for binary benign-vs-attack models
y_train_multiclass = train["Attack"]  # for multi-class attack-type models
```

Both of you should load the **same** `train.parquet` / `test.parquet` (same
`--seed`) so the proposed algorithm and the baselines are compared on
identical rows, per the project's comparison requirement. `Dataset` is kept
as a column (not dropped) so you can filter to specific sub-datasets or
analyse per-sub-dataset performance without re-running preprocessing.

## 7. Open decisions for the team (not defaulted silently)

1. **Drop ports too?** The reference notebook keeps `L4_SRC_PORT` /
   `L4_DST_PORT` and only drops the two IP columns; a related earlier paper
   from the same authors reportedly drops ports as well. Default here
   matches the reference notebook (`--drop-ports` not set). Pass
   `--drop-ports` if the team decides on the stricter variant -- record
   whichever you choose in `FUTURE_WORK.md` / your shared log, since it
   changes the feature count from 42 to 40.
2. **Split ratios.** `--test-size` / `--val-size` default to nothing forced;
   the CLI examples above use 0.2 / 0.1 as a common convention, not a
   verified requirement -- agree on the actual split with the team.
3. **Feature scaling.** Deliberately *not* done in this stage. Fitting a
   scaler on the full dataset before splitting would leak test-set
   statistics into training; fit any scaler (e.g. `StandardScaler`) on
   `train.parquet` only, inside each of your model-training scripts, and
   apply the same fitted scaler to `val`/`test`.
4. **`results_log.csv` schema.** `pipeline._append_results_log` writes a
   generic, self-describing set of columns (run timestamp, row counts,
   split sizes, parameters). If the team's planning document already
   settled on a different shared schema, rename/remap these columns before
   merging logs -- this script doesn't guess at column names it hasn't
   been shown.

## 8. Known limitations / future work

- The full CSV is ~76M rows; this pipeline loads and *cleans* it in memory
  in one pass, matching the reference notebook -- `--target-sample-rows`
  only shrinks the data *after* cleaning, it doesn't avoid the initial full
  load. That's known to work in a Kaggle notebook environment; on a smaller
  machine, consider chunked loading with incremental parquet writes (not
  implemented here -- cross-chunk duplicate detection is non-trivial and was
  left out rather than implemented unverified).
- `sample.stratified_subsample` groups by the exact (Dataset, Attack) pair,
  so the group count is `4 x (number of Attack categories)` at most (~80 for
  the full dataset's ~20 attack categories) -- this is fine at that scale,
  but if you ever stratify on a much higher-cardinality column the current
  implementation's per-group Python loop would need batching for speed.
- `pd.read_csv` with strict integer dtypes will hard-crash (not silently
  produce NaN) if an integer-typed column ever has a genuinely missing
  value -- discovered while building the test fixture. Not observed in the
  real dataset (its only NaNs come from `+-inf` in float columns, per the
  reference notebook), but worth knowing if you ever preprocess a partial
  export.
- No feature engineering / encoding beyond dtype downcasting is done here,
  by design -- that decision depends on the (not-yet-finalized) algorithm,
  and belongs in Person B/C's stage rather than being guessed at here.

## 9. Citation

```bibtex
@article{sarhan2022towards,
  title   = {Towards a Standard Feature Set for Network Intrusion Detection System Datasets},
  author  = {Mohanad Sarhan and Siamak Layeghy and Marius Portmann},
  year    = {2022},
  journal = {Mobile networks and applications},
  pages   = {1--14},
  publisher = {Springer US},
  url     = {https://doi.org/10.1007/s11036-021-01843-0}
}
```
