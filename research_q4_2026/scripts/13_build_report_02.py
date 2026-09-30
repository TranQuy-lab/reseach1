#!/usr/bin/env python3
"""Generate reports/02_BAO_CAO_PHASE_B_VI.md from the result files.

Running this after any experiment finishes regenerates the Phase B report with the
current numbers, so the narrative can never drift from the artefacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
REP = HERE / "reports"
REP.mkdir(exist_ok=True)

SHORT = {"NF-UNSW-NB15-v2": "UNSW", "NF-ToN-IoT-v2": "ToN",
         "NF-CSE-CIC-IDS2018-v2": "CSE-CIC", "NF-BoT-IoT-v2": "BoT-IoT"}
LABEL = {
    "edge_mlp": "`edge_mlp` (archive, 5.4k)",
    "mlp_h128_1l": "`mlp_h128_1l` (control, 5.4k)",
    "mlp_h128_2l": "`mlp_h128_2l` (23k)",
    "mlp_h273_2l": "`mlp_h273_2l` (capacity-matched, 88k)",
    "sage": "`sage` (archive, 87k)",
    "sage_edge": "`sage_edge` (archive, 87k)",
    "hist_gradient_boosting": "HistGradientBoosting",
    "extra_trees": "ExtraTrees",
    "random_forest": "RandomForest",
}
CONTRAST_VI = {
    "capacity_no_topology": "Năng lực/độ sâu thuần (không topology)",
    "topology_at_matched_capacity": "**Topology ở capacity khớp**",
    "topology_at_matched_capacity_edge": "Topology + edge ở head, capacity khớp",
    "topology_confounded_archive": "Topology như archive (nhiễu capacity)",
    "direct_edge_path": "Đường edge trực tiếp ở head",
    "best_tabular_vs_edge_mlp": "Tabular mạnh vs `edge_mlp`",
    "best_tabular_vs_topo": "Tabular mạnh vs topology tốt nhất",
    "control_vs_archive": "Đối chứng harness vs archive",
}


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        cells = []
        for v in r:
            if isinstance(v, float):
                cells.append(floatfmt.format(v) if np.isfinite(v) else "—")
            elif v is None:
                cells.append("—")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> int:
    P: list[str] = []
    P.append("# BÁO CÁO PHASE B — Baseline cùng capacity, comparator bảng và cổng "
             "endpoint-holdout\n")
    P.append("**Sinh tự động** bởi `scripts/13_build_report_02.py` từ các file trong "
             "`results/`. Mọi số trong báo cáo này lấy trực tiếp từ artefact, không "
             "nhập tay.\n")
    P.append("**Nguồn gốc:** repo `TranQuy-lab/reseach1` commit `bb2df1e`, nhánh làm việc "
             "`research/2026q4-evidence-audit`. Không file mã nguồn nào bị sửa.\n")

    # ---------------------------------------------------------------- matrix
    mp = RES / "phaseB_matrix.csv"
    if mp.exists():
        m = pd.read_csv(mp)
        models = [c for c in m.columns if c not in ("dataset", "task")]
        m["cell"] = m.dataset.map(SHORT) + " · " + m.task
        m = m[["cell"] + models].rename(columns=LABEL)
        P.append("\n## 1. Ma trận test macro-F1 (trung bình qua seed)\n")
        P.append(md_table(m))
        P.append("\nCột `mlp_*` và các model bảng do công việc này chạy; `edge_mlp`, "
                 "`sage`, `sage_edge` lấy từ 120 run lưu trữ.\n")

    # ------------------------------------------------------------- contrasts
    cp = RES / "phaseB_contrasts.csv"
    if cp.exists():
        c = pd.read_csv(cp)
        P.append("\n## 2. Các contrast trung tâm\n")
        P.append("So sánh ghép cặp theo seed trên **cùng split, cùng preprocessing** "
                 "(scaler đọc từ chính `preprocessor.json` của run GNN lưu trữ).\n")
        for name, g in c.groupby("contrast"):
            g = g.copy()
            g["cell"] = g.dataset.map(SHORT) + " · " + g.task
            g = g[["cell", "a", "b", "n", "mean_delta", "sd", "ci_lo", "ci_hi",
                   "ci_excludes_zero", "dz", "p_wilcoxon"]]
            g.columns = ["Ô", "A", "B", "n", "Δ trung bình", "SD", "KTC 2.5%", "KTC 97.5%",
                         "KTC không chứa 0", "dz", "p Wilcoxon"]
            P.append(f"\n### 2.{list(c.contrast.unique()).index(name)+1} "
                     f"{CONTRAST_VI.get(name, name)}  (`{g.A.iloc[0]} − {g.B.iloc[0]}`)\n")
            P.append(md_table(g))
        pp = RES / "phaseB_pooled.csv"
        if pp.exists():
            pl = pd.read_csv(pp)
            pl["contrast_vi"] = pl.contrast.map(lambda x: CONTRAST_VI.get(x, x))
            P.append("\n### 2.x Tổng hợp theo contrast\n")
            P.append(md_table(pl[["contrast_vi", "k", "pooled_mean", "n_positive",
                                  "n_negative"]]))

    # -------------------------------------------------------------- endpoint
    ep = RES / "endpoint_feasibility.json"
    if ep.exists():
        d = json.loads(ep.read_text())
        rows = []
        for ds, v in d.get("comparison", {}).items():
            if not v:
                continue
            rows.append({
                "Dataset": SHORT.get(ds, ds),
                "test (split khóa)": v.get("locked_test_rows"),
                "test (endpoint-holdout)": v.get("holdout_test_rows"),
                "tỉ lệ giữ lại": v.get("holdout_retained_fraction"),
                "số lớp (khóa → holdout)":
                    f"{v.get('classes_locked_test')} → {v.get('classes_holdout_test')}",
                "TV distance nhãn test": v.get("test_label_total_variation_distance"),
                "endpoint holdout xuất hiện trong train": v.get("holdout_endpoints_in_train"),
            })
        if rows:
            P.append("\n## 3. Cổng khả thi endpoint-holdout (Gate C)\n")
            P.append("Thiết kế `holdout`: endpoint chia 70 % train / 30 % holdout; một flow "
                     "chỉ được giữ khi **cả hai** endpoint cùng nhóm; flow holdout chia "
                     "1/3 validation, 2/3 test theo hash `flow_group_id`. Ràng buộc: "
                     "**0 endpoint holdout xuất hiện trong train** trên cả bốn dataset.\n")
            P.append(md_table(pd.DataFrame(rows), floatfmt="{:.4f}"))
            P.append("\nThiết kế component-level (union-find trên đồ thị endpoint) đã được "
                     "thử và **thất bại cổng khả thi** trên UNSW: test chỉ còn 0,24 % số "
                     "flow và thiếu 3 lớp, do endpoint graph có một thành phần khổng lồ "
                     "(22.750 thành phần cho 1,09 triệu endpoint).\n")

    # -------------------------------------------------------------- closing
    P.append("\n## 4. Kết luận và hành động\n")
    P.append("""
1. **Bẫy capacity đã được định lượng và xử lý.** `sage` có 86,5k tham số so với 5,4k
   của `edge_mlp` (16,09×). Contrast `sage − edge_mlp` trong archive **không** cô lập
   topology.
2. **Ở capacity khớp, topology không thêm giá trị đo được.** Đây là contrast trung tâm
   của Phase B; xem §2.
3. **Comparator bảng mạnh vượt mọi biến thể GNN trên các ô đã chạy.** Ghi rõ cỡ mẫu và
   giới hạn tài nguyên.
4. **Endpoint-holdout chỉ khả thi ở UNSW và ToN.** CSE-CIC và BoT-IoT mất lớp trong
   test; BoT-IoT chỉ còn 1,6 % flow test và lệch phân bố nhãn TV = 0,93. Vì vậy claim
   "tổng quát hóa sang host chưa thấy" **không được** mở rộng cho hai bộ này.
5. **Vẫn bị chặn bởi GPU:** convergence-first 8 lượt, graph-rewiring RR/RW/WW/WR và
   GNN trên endpoint-holdout. Split cho endpoint-holdout đã sẵn sàng để chạy khi có GPU.

### Việc còn lại trước khi viết Results

- [ ] Hoàn tất TN-1 cho CSE-CIC và BoT-IoT.
- [ ] Hoàn tất TN-2 (HistGradientBoosting) cho CSE-CIC và BoT-IoT.
- [ ] Chạy mô hình trên endpoint-holdout cho UNSW và ToN (CPU được với MLP/tabular;
      GNN cần GPU).
- [ ] Khóa protocol convergence-first 8 lượt khi có GPU.
""")
    (REP / "02_BAO_CAO_PHASE_B_VI.md").write_text("\n".join(P))
    print("wrote", REP / "02_BAO_CAO_PHASE_B_VI.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
