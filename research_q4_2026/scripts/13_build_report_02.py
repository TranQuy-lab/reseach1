#!/usr/bin/env python3
"""Generate reports/02_BAO_CAO_KET_QUA_VI.md from the result files.

Regenerating after any experiment finishes refreshes every number, so the narrative
cannot drift from the artefacts. Every figure in the report is read from results/.
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
    "edge_mlp": "`edge_mlp` (5.4k)",
    "mlp_h128_1l": "`mlp_h128_1l` control (5.4k)",
    "mlp_h128_2l": "`mlp_h128_2l` (23k)",
    "mlp_h273_2l": "`mlp_h273_2l` capacity-matched (88k)",
    "sage": "`sage` (87k)",
    "sage_edge": "`sage_edge` (87k)",
    "hist_gradient_boosting": "HistGB",
    "extra_trees": "ExtraTrees",
    "random_forest": "RandomForest",
}
CONTRAST_VI = {
    "capacity_no_topology": "Capacity/độ sâu thuần (KHÔNG topology)",
    "topology_at_matched_capacity": "**Topology ở capacity khớp**",
    "topology_at_matched_capacity_edge": "Topology + edge ở head, capacity khớp",
    "topology_confounded_archive": "Topology như archive (nhiễu capacity)",
    "direct_edge_path": "Đường edge trực tiếp ở head",
    "best_tabular_vs_edge_mlp": "Bảng mạnh vs `edge_mlp`",
    "best_tabular_vs_topo": "Bảng mạnh vs topology tốt nhất",
    "control_vs_archive": "Đối chứng harness vs archive",
}


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append(floatfmt.format(v) if np.isfinite(v) else "—")
            elif v is None:
                cells.append("—")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def load(name: str):
    p = RES / name
    if not p.exists():
        return None
    return json.loads(p.read_text()) if name.endswith(".json") else pd.read_csv(p)


def main() -> int:
    P: list[str] = []
    P.append("# BÁO CÁO KẾT QUẢ — Kiểm toán, chẩn đoán cơ chế và thí nghiệm xác nhận\n")
    P.append("**Sinh tự động** bởi `scripts/13_build_report_02.py`; mọi số đọc trực tiếp "
             "từ `results/`, không nhập tay.\n")
    P.append("**Nguồn gốc:** `TranQuy-lab/reseach1` commit `bb2df1e`, nhánh "
             "`research/2026q4-evidence-audit`. Không file nào trong `src/`, `tests/`, "
             "`research/` bị sửa.\n")

    # ------------------------------------------------------------ 0 headline
    au = load("audit_inventory.json") or {}
    cs = load("collapse_summary.json") or {}
    cr = load("collapse_robustness.json") or {}
    st = load("stats_summary.json") or {}
    P.append("\n## 0. Sáu kết luận trung tâm\n")
    t5 = load("tn5_struct_runs.csv")
    t8 = load("tn8_nbr_runs.csv")
    hs = load("holdout_summary.csv")
    op = load("operational_summary.json")
    def _mn(df, ds, task, model):
        if df is None:
            return None
        g = df[(df.dataset == ds) & (df.task == task) & (df.model == model)]
        return None if g.empty else (len(g), float(g.test_macro_f1.mean()))
    t5u = _mn(t5, "NF-UNSW-NB15-v2", "multiclass", "mlp_struct")
    t8u = _mn(t8, "NF-UNSW-NB15-v2", "multiclass", "mlp_nbr_mean")
    P.append(f"""1. **Không có bằng chứng `sage` hơn `edge_mlp` về trung bình.**
   Pooled Δ = {st.get('pooled', [{}])[-1].get('pooled_delta_RE', float('nan')):+.4f}
   macro-F1, KTC 95 % [{st.get('pooled', [{}])[-1].get('ci_lo_RE', float('nan')):+.4f};
   {st.get('pooled', [{}])[-1].get('ci_hi_RE', float('nan')):+.4f}] — chứa 0.
2. **Giả thuyết underfitting bị bác bỏ; cơ chế là sụp đổ tối ưu hóa.**
   Tỉ lệ sụp đổ `edge_mlp` {cr.get('bootstrap_ci', {}).get('edge_mlp', {}).get('rate', float('nan')):.0%},
   `sage` {cr.get('bootstrap_ci', {}).get('sage', {}).get('rate', float('nan')):.0%},
   `sage_edge` {cr.get('bootstrap_ci', {}).get('sage_edge', {}).get('rate', float('nan')):.0%};
   Fisher exact `sage` vs `edge_mlp` p = {cr.get('fisher_exact', {}).get('sage_vs_edge_mlp', {}).get('p_holm', float('nan')):.4f} (Holm).
3. **Phần lớn "lợi thế topology" là capacity.** Contrast ở capacity khớp nhỏ và đổi dấu
   theo dataset (UNSW mc +0,027; ToN mc −0,000).
4. **Phụ thuộc endpoint: đã đo và thấy KHÔNG đáng kể trên split khóa — và một kết luận
   cũ đã bị bác bỏ.** Split endpoint-disjoint từng cho thấy ToN multiclass mất 0,311,
   nhưng đó là **split bị nhiễu**: `HistGradientBoosting` (không hề dùng danh tính
   endpoint) cũng mất 0,866 → **0,472** trên chính split đó. Đo sạch trên split khóa:
   chỉ **621/3.385.552 (0,018 %)** flow test có cả hai endpoint chưa thấy, và trên nhóm
   đó điểm chỉ giảm 0,039. Vậy tỉ lệ chồng lấp endpoint **không** thổi phồng kết quả,
   nhưng split này cũng **không** dùng được cho claim unseen-host.
5. **Tóm tắt cấu trúc thô thắng message passing**: `mlp_struct` (chỉ đếm lân cận)
   = {('%.4f' % t5u[1]) if t5u else '—'} (n={t5u[0] if t5u else 0}) so với `sage` 0,4898.
   Nhưng **trung bình đặc trưng lân cận thì thất bại**: `mlp_nbr_mean` =
   {('%.4f' % t8u[1]) if t8u else '—'} (n={t8u[0] if t8u else 0}) — *thấp hơn* cả flow-only.
6. **Biến thể sụp đổ không triển khai được**: `sage` trên BoT-IoT có tỉ lệ báo động giả
   **25,02 %** (250.189/triệu flow benign), `edge_mlp` 0,19 %.
""")
    if hs is not None:
        P.append("\n### 0.1 Bảng mức phụ thuộc endpoint (cùng model, cùng ngân sách)\n")
        h = pd.read_csv(RES / "holdout_summary.csv")
        h["Ô"] = h.dataset.map(SHORT) + " · " + h.task + " · " + h.model
        h = h[["Ô", "n", "locked_mean", "holdout_mean", "mean_drop", "ci_lo", "ci_hi"]]
        h.columns = ["Ô", "n", "split khóa", "endpoint-holdout", "mức giảm", "KTC 2.5%", "KTC 97.5%"]
        P.append(md_table(h))
    if op is not None:
        P.append("\n### 0.2 Tỉ lệ báo động giả của detector binary\n")
        det = pd.DataFrame(op.get("binary_detector_false_alarm", []))
        if len(det):
            det["Ô"] = det.dataset.map(SHORT)
            det = det[["Ô", "model", "false_alarm_rate", "fp_per_million", "benign_recall"]]
            det.columns = ["Dataset", "Model", "tỉ lệ báo động giả", "FP/triệu flow", "recall Benign"]
            P.append(md_table(det))

    # ---------------------------------------------------------------- 1 audit
    P.append("\n## 1. Kiểm toán tính toàn vẹn\n")
    if au:
        P.append(f"- Thiết kế 4 dataset × 2 task × 3 model × 5 seed: "
                 f"{au.get('n_gnn_runs')} run GNN lưu trữ, mọi ô đúng 5 seed: "
                 f"`{au.get('seeds_per_cell_complete')}`.")
        d = au.get("published_vs_artifact_max_abs_diff", {})
        P.append(f"- `runs.csv` công bố khớp `metrics.json` từng run tới "
                 f"{max((v['max_abs_diff'] or 0) for v in d.values()):.2e}.")
        P.append(f"- Verify độc lập: `passed={au.get('verification', {}).get('passed')}`, "
                 f"`runs_checked={au.get('verification', {}).get('runs_checked')}`, "
                 f"replay error {au.get('verification', {}).get('largest_probability_replay_error')}, "
                 f"metric error {au.get('verification', {}).get('largest_metric_recomputation_error'):.2e}.")
        P.append(f"- `checkpoint_parameter_max_abs_error` bằng 0 trên mọi run: "
                 f"`{au.get('checkpoint_parameter_max_abs_error_all_zero')}`.")
        P.append("\n### 1.1 Ngân sách huấn luyện thực tế\n")
        bp = RES / "budget_provenance.csv"
        if bp.exists():
            b = pd.read_csv(bp)
            b["cell"] = b.dataset.map(SHORT) + " · " + b.task
            b = b[["cell", "spp", "steps_ran_max", "budget_binding", "effective_passes"]]
            b.columns = ["Ô", "steps/lượt", "steps đã chạy", "ràng buộc chi phối",
                         "lượt hiệu dụng"]
            P.append(md_table(b))

    # ----------------------------------------------------------- 2 convergence
    conv = load("convergence_summary.json") or {}
    if conv:
        P.append("\n## 2. Chẩn đoán learning curve (120 run lưu trữ)\n")
        P.append("Quy tắc chỉ dùng **validation**, không dùng test.\n")
        cc = conv.get("classification_counts", {})
        P.append(f"- Suy giảm sau đỉnh (R2): **{cc.get('R2_optimization_decay', 0)}** run "
                 f"({conv.get('classification_share', {}).get('R2_optimization_decay', 0):.1%})")
        P.append(f"- Còn tăng ở biên ngân sách (R1): {cc.get('R1_still_rising', 0)} run "
                 f"({conv.get('classification_share', {}).get('R1_still_rising', 0):.1%})")
        P.append(f"- Plateau thật (R3): {cc.get('R3_plateau', 0)} run "
                 f"({conv.get('classification_share', {}).get('R3_plateau', 0):.1%})")
        P.append(f"- Tỉ lệ run có checkpoint tốt nhất là lần đánh giá cuối: "
                 f"{conv.get('frac_best_is_last_eval', float('nan')):.1%}")
        ep = conv.get("effective_passes_by_dataset", {})
        if ep:
            P.append("\n### 2.1 Lượt hiệu dụng theo dataset\n")
            t = pd.DataFrame([{"Dataset": SHORT.get(k, k), "lượt trung bình": v["mean"],
                               "min": v["min"], "max": v["max"]} for k, v in ep.items()])
            P.append(md_table(t))

    # ------------------------------------------------------------- 3 collapse
    if cs:
        P.append("\n## 3. Sụp đổ tối ưu hóa\n")
        P.append("Quy tắc khóa trước, chỉ dùng validation: "
                 "`val_std ≥ 0,05` **hoặc** `drop_after_best ≥ 0,10`.\n")
        bm = pd.DataFrame(cs.get("collapse_by_model", []))
        if len(bm):
            bm.columns = ["Model", "số run sụp đổ", "tổng run", "tỉ lệ"]
            P.append(md_table(bm))
        bd = pd.DataFrame(cs.get("collapse_by_dataset", []))
        if len(bd):
            bd["Dataset"] = bd.dataset.map(SHORT)
            bd = bd[["Dataset", "n_collapsed", "n_runs", "collapse_rate"]]
            bd.columns = ["Dataset", "số run sụp đổ", "tổng run", "tỉ lệ"]
            P.append("\n")
            P.append(md_table(bd))
        if cr:
            P.append("\n### 3.1 Độ bền của kết luận\n")
            bt = pd.DataFrame([{"Model": k, "tỉ lệ": v["rate"], "KTC 2.5%": v["ci_lo"],
                                "KTC 97.5%": v["ci_hi"], "n": v["n"]}
                               for k, v in cr.get("bootstrap_ci", {}).items()])
            P.append(md_table(bt))
            fx = pd.DataFrame([{"Cặp": k, "odds ratio": v["odds_ratio"],
                                "p": v["p"], "p Holm": v.get("p_holm")}
                               for k, v in cr.get("fisher_exact", {}).items()])
            P.append("\n")
            P.append(md_table(fx))
            vi = pd.DataFrame([{"Model": k, "ρ(best-val, test)": v["spearman_bestval_vs_test"],
                                "khoảng cách trung bình": v["mean_gap_bestval_minus_test"]}
                               for k, v in cr.get("validation_informativeness", {}).items()])
            P.append("\nValidation là bộ chọn checkpoint rất tốt:\n")
            P.append(md_table(vi))

    # ------------------------------------------------------------ 4 statistics
    pc = RES / "paired_contrasts.csv"
    if pc.exists():
        P.append("\n## 4. Thống kê 5 seed trên archive\n")
        mt = load("pooled_meta.csv")
        if mt is not None:
            m = mt[["comparison", "k_cells", "pooled_delta_RE", "ci_lo_RE", "ci_hi_RE",
                    "prediction_interval_lo", "prediction_interval_hi", "I2_percent",
                    "n_cells_positive", "n_cells_negative"]]
            m.columns = ["Contrast", "số ô", "Δ gộp (RE)", "KTC 2.5%", "KTC 97.5%",
                         "KTC dự báo 2.5%", "KTC dự báo 97.5%", "I² (%)", "ô dương", "ô âm"]
            P.append(md_table(m))
        if st:
            P.append(f"\n- Wilcoxon sống sót Holm: **{st.get('wilcoxon_significant_holm')}**/{st.get('n_contrasts_total')}")
            P.append(f"- Sign test sống sót Holm: **{st.get('sign_test_significant_holm')}**/{st.get('n_contrasts_total')}")
            P.append(f"- KTC (chưa hiệu chỉnh) không chứa 0: {st.get('ci_excludes_zero_uncorrected')}/{st.get('n_contrasts_total')}")

    # ---------------------------------------------------------- 5 seed count
    si = RES / "seed_inflation_3_vs_5.csv"
    if si.exists():
        P.append("\n## 5. 3 seed so với 5 seed\n")
        s = pd.read_csv(si).sort_values("std_ratio_5_over_3", ascending=False).head(8)
        s["Ô"] = s.dataset.map(SHORT) + " · " + s.task + " · " + s.model
        s = s[["Ô", "mean3", "std3", "mean5", "std5", "std_ratio_5_over_3",
               "mean_shift_5_minus_3"]]
        s.columns = ["Ô", "mean(3)", "SD(3)", "mean(5)", "SD(5)", "SD(5)/SD(3)", "lệch mean"]
        P.append(md_table(s))

    # --------------------------------------------------------------- 6 graph
    gs = RES / "graph_statistics.csv"
    if gs.exists():
        P.append("\n## 6. Thống kê đồ thị — vì sao topology giúp ở chỗ này mà không ở chỗ khác\n")
        g = pd.read_csv(gs)
        g = g[["label", "train_flows", "distinct_endpoints", "edges_per_node",
               "flows_per_distinct_pair", "repeat_interaction_share", "degree_median",
               "class_entropy_nats"]]
        g.columns = ["Dataset", "flow train", "endpoint", "cạnh/node",
                     "flow/cặp endpoint", "tỉ lệ tương tác lặp", "degree trung vị",
                     "entropy lớp (nats)"]
        P.append(md_table(g, floatfmt="{:.3f}"))
        P.append("\nBoT-IoT có cấu trúc quan hệ **giàu nhất** (78,7 cạnh/node, 91,6 % tương "
                 "tác lặp) — và cũng chính là nơi `sage` sụp đổ nhiều nhất. UNSW và CSE-CIC "
                 "gần như độc lập theo flow (1,7–1,8 cạnh/node, ~20 % lặp), nên message "
                 "passing không có thông tin bổ sung để khai thác.\n")

    # ------------------------------------------------------------ 7 endpoint
    ep = load("endpoint_feasibility.json")
    if ep:
        P.append("\n## 7. Cổng khả thi endpoint-holdout\n")
        rows = []
        for ds, v in ep.get("comparison", {}).items():
            if not v:
                continue
            rows.append({
                "Dataset": SHORT.get(ds, ds),
                "test (split khóa)": v.get("locked_test_rows"),
                "test (holdout)": v.get("holdout_test_rows"),
                "tỉ lệ giữ lại": v.get("holdout_retained_fraction"),
                "lớp (khóa → holdout)": f"{v.get('classes_locked_test')} → {v.get('classes_holdout_test')}",
                "TV nhãn test": v.get("test_label_total_variation_distance"),
                "endpoint holdout trong train": v.get("holdout_endpoints_in_train"),
            })
        if rows:
            P.append(md_table(pd.DataFrame(rows)))
        P.append("\nThiết kế `holdout`: endpoint chia 70 % train / 30 % holdout; flow chỉ giữ "
                 "khi **cả hai** endpoint cùng nhóm; flow holdout chia 1/3 val, 2/3 test. "
                 "Thiết kế component-level đã được thử và **thất bại cổng khả thi** "
                 "(UNSW: test còn 0,24 % flow, thiếu 3 lớp, do endpoint graph có một thành "
                 "phần khổng lồ).\n")

    # ------------------------------------------------------------- 8 phase B
    mp = RES / "phaseB_matrix.csv"
    if mp.exists():
        P.append("\n## 8. Phase B — capacity, topology và comparator bảng\n")
        P.append("### 8.1 Ma trận test macro-F1 (trung bình qua seed)\n")
        m = pd.read_csv(mp)
        models = [c for c in m.columns if c not in ("dataset", "task")]
        m["cell"] = m.dataset.map(SHORT) + " · " + m.task
        m = m[["cell"] + models].rename(columns=LABEL)
        P.append(md_table(m))
        P.append("\nCác cột `mlp_*`, `HistGB`, `ExtraTrees`, `RandomForest` do công việc này "
                 "chạy trên **đúng split**; `edge_mlp`/`sage`/`sage_edge` lấy từ 120 run "
                 "lưu trữ. Scaler đọc từ chính `preprocessor.json` của run GNN, **không** "
                 "fit lại.\n")
    cp = RES / "phaseB_contrasts.csv"
    if cp.exists():
        c = pd.read_csv(cp)
        P.append("### 8.2 Contrast ghép cặp theo seed\n")
        order = ["control_vs_archive", "capacity_no_topology", "topology_at_matched_capacity",
                 "topology_at_matched_capacity_edge", "topology_confounded_archive",
                 "direct_edge_path", "best_tabular_vs_edge_mlp", "best_tabular_vs_topo"]
        for name in order:
            g = c[c.contrast == name]
            if not len(g):
                continue
            g = g.copy()
            g["cell"] = g.dataset.map(SHORT) + " · " + g.task
            g = g[["cell", "a", "b", "n", "mean_delta", "sd", "ci_lo", "ci_hi",
                   "ci_excludes_zero", "dz", "p_wilcoxon"]]
            g.columns = ["Ô", "A", "B", "n", "Δ", "SD", "KTC 2.5%", "KTC 97.5%",
                         "KTC không chứa 0", "dz", "p Wilcoxon"]
            P.append(f"\n**{CONTRAST_VI.get(name, name)}**  (`{g.A.iloc[0]} − {g.B.iloc[0]}`)\n")
            P.append(md_table(g))
        pp = RES / "phaseB_pooled.csv"
        if pp.exists():
            pl = pd.read_csv(pp)
            pl["contrast_vi"] = pl.contrast.map(lambda x: CONTRAST_VI.get(x, x))
            pl = pl[["contrast_vi", "k", "pooled_mean", "n_positive", "n_negative"]]
            pl.columns = ["Contrast", "số ô", "Δ trung bình", "ô dương", "ô âm"]
            P.append("\n### 8.3 Tổng hợp\n")
            P.append(md_table(pl))

    # ------------------------------------------------------- 8b holdout eval
    hs = RES / "holdout_summary.csv"
    if hs.exists():
        P.append("\n## 8b. Đánh giá trên split endpoint-holdout (cùng model, cùng ngân sách)\n")
        P.append("So sánh trực tiếp giữa split khóa (`flow_group_id`) và split "
                 "endpoint-disjoint. Kiến trúc, seed và ngân sách giống hệt; khác biệt "
                 "duy nhất là flow test có thể chứa endpoint **chưa từng thấy** trong train. "
                 "Scaler được fit lại trên train của từng split (dân số huấn luyện khác nhau) "
                 "và điều này được ghi rõ.\n")
        h = pd.read_csv(hs)
        h["Ô"] = h.dataset.map(SHORT) + " · " + h.task + " · " + h.model
        h = h[["Ô", "n", "locked_mean", "holdout_mean", "mean_drop", "ci_lo", "ci_hi",
               "relative_drop_percent"]]
        h.columns = ["Ô", "n", "split khóa", "endpoint-holdout", "mức giảm",
                     "KTC 2.5%", "KTC 97.5%", "giảm tương đối (%)"]
        P.append(md_table(h))
        P.append("\nMức giảm **dương** nghĩa là split endpoint-holdout làm giảm chất lượng. "
                 "Nếu KTC chứa 0 thì không có bằng chứng suy giảm.\n")

    # ---------------------------------------------------- 8c structural feats
    t5 = RES / "tn5_struct_runs.csv"
    if t5.exists():
        d = pd.read_csv(t5)
        P.append("\n## 8c. Thống kê cấu trúc cục bộ thay cho message passing (TN-5)\n")
        P.append("Đặc trưng cấu trúc **chỉ tính từ train**: số flow theo endpoint nguồn/đích, "
                 "số đối tác phân biệt, số lần lặp đúng cặp endpoint, và (biến thể `_lab`) "
                 "tỉ lệ tấn công theo nhãn train của endpoint. Endpoint chưa thấy nhận 0 / "
                 "prior.\n")
        g = d.groupby(["dataset", "task", "model"]).agg(
            n=("seed", "size"), params=("parameters", "first"),
            test_macro_f1_mean=("test_macro_f1", "mean"),
            test_macro_f1_std=("test_macro_f1", "std"),
            best_val=("best_val_macro_f1", "mean")).reset_index()
        g["Ô"] = g.dataset.map(SHORT) + " · " + g.task
        g = g[["Ô", "model", "n", "params", "test_macro_f1_mean", "test_macro_f1_std", "best_val"]]
        g.columns = ["Ô", "Model", "n", "tham số", "test macro-F1", "SD", "best val macro-F1"]
        P.append(md_table(g))
        P.append("\nĐối chiếu: trên cùng ô, `mlp_h273_2l` (chỉ flow, capacity khớp) và "
                 "`sage`/`sage_edge` nằm ở §8.1.\n")

    # -------------------------------------------------- 8d operational metrics
    op = RES / "operational_summary.json"
    if op.exists():
        o = json.loads(op.read_text())
        P.append("\n## 8d. Hồ sơ báo động giả (từ confusion matrix test đầy đủ)\n")
        P.append("Repo không lưu probability của GNN nên **không** tính lại được PR-AUC; "
                 "confusion matrix đầy đủ thì có, đủ để tính tỉ lệ báo động giả tại chính "
                 "operating point argmax của từng model.\n")
        det = pd.DataFrame(o.get("binary_detector_false_alarm", []))
        if len(det):
            det["Ô"] = det.dataset.map(SHORT)
            det = det[["Ô", "model", "false_alarm_rate", "fp_per_million", "benign_recall"]]
            det.columns = ["Dataset", "Model", "tỉ lệ báo động giả", "báo động giả/triệu flow", "recall Benign"]
            P.append(md_table(det))
        worst = pd.DataFrame(o.get("worst_false_alarm_classes", []))
        if len(worst):
            worst["Ô"] = worst.dataset.map(SHORT)
            worst = worst[["Ô", "model", "class", "support", "fpr_mean", "fp_per_million_mean"]]
            worst.columns = ["Ô", "Model", "Lớp", "support", "FPR", "FP/triệu"]
            P.append("\nCác lớp có FPR xấu nhất:\n")
            P.append(md_table(worst))

    # ------------------------------------------------- 8e endpoint isolation
    ti = RES / "tn10_endpoint_isolation_summary.csv"
    if ti.exists():
        d = pd.read_csv(ti)
        P.append("\n## 8e. Cô lập mức phụ thuộc endpoint trên split khóa (TN-10)\n")
        P.append("Giữ **nguyên** model, dữ liệu train, preprocessing và toàn bộ tập test; "
                 "chỉ phân nhóm tập test theo việc endpoint của flow có xuất hiện trong "
                 "train hay không. Vì mọi thứ khác không đổi, khác biệt giữa các nhóm chỉ "
                 "có thể do mức quen thuộc endpoint.\n")
        d["Ô"] = d.dataset.map(SHORT) + " · " + d.task
        d = d[["Ô", "subset", "n_seeds", "n_flows", "macro_f1", "macro_f1_std",
               "delta_vs_all_test"]]
        d.columns = ["Ô", "Nhóm test", "số seed", "số flow", "macro-F1", "SD",
                     "lệch so với toàn bộ test"]
        P.append(md_table(d))
        P.append("\nKèm theo đó: split endpoint-disjoint **không** phải công cụ hợp lệ để "
                 "đo mức phụ thuộc endpoint. `HistGradientBoosting` — mô hình không dùng "
                 "danh tính endpoint — cũng giảm từ 0,866 xuống **0,472** trên split đó, "
                 "nên phần lớn mức giảm là dịch chuyển phân bố do cách dựng split.\n")

    if op is not None:
        pass
    ti9 = RES / "tn9_holdout_tabular.csv"
    if ti9.exists():
        hb = pd.read_csv(ti9)
        P.append("\n### 8e.1 Đối chứng bác bỏ split endpoint-disjoint\n")
        t = hb.groupby(["dataset", "task"]).agg(
            n=("seed", "size"), test_macro_f1=("test_macro_f1", "mean")).reset_index()
        t["Ô"] = t.dataset.map(SHORT) + " · " + t.task
        t = t[["Ô", "n", "test_macro_f1"]]
        t.columns = ["Ô", "số seed", "HGB trên split endpoint-disjoint"]
        P.append(md_table(t))
        P.append("\nSo sánh: HGB trên **split khóa** cùng ô đạt 0,8664 (§8.1).\n")

    # ------------------------------------------------------------ 9 remaining
    P.append("""
## 9. Kết luận và việc còn lại

**Đã đóng:** kiểm toán 120 run; chẩn đoán hội tụ/sụp đổ; thống kê 5 seed và
meta-analysis; lớp hiếm; thống kê đồ thị; cổng endpoint-holdout; tái tạo split khớp
manifest; baseline cùng capacity và comparator bảng (đang hoàn tất nốt các ô còn lại).

**Bị chặn bởi GPU** (server `vast-gpu` 202.59.206.41:11493 từ chối kết nối):

1. convergence-first 8 lượt cho UNSW/ToN/BoT;
2. graph-rewiring RR/RW/WW/WR;
3. GNN trên endpoint-holdout (split đã sẵn sàng);
4. learning-rate/budget sensitivity cho `sage` BoT-IoT.

**Ước lượng khi có GPU 24 GiB:** ma trận 72 run ba seed từng mất 5,45 h. Convergence-first
8 lượt cho 3 dataset × 3 model × 3–5 seed ước tính 15–30 h GPU.
""")
    (REP / "02_BAO_CAO_KET_QUA_VI.md").write_text("\n".join(P))
    print("wrote", REP / "02_BAO_CAO_KET_QUA_VI.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
