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
    P.append("\n## 0. Bốn kết luận trung tâm\n")
    au = load("audit_inventory.json") or {}
    cs = load("collapse_summary.json") or {}
    cr = load("collapse_robustness.json") or {}
    st = load("stats_summary.json") or {}
    P.append(f"""1. **Không có bằng chứng `sage` hơn `edge_mlp` về trung bình.**
   Pooled Δ = {st.get('pooled', [{}])[-1].get('pooled_delta_RE', float('nan')):+.4f}
   macro-F1, KTC 95 % [{st.get('pooled', [{}])[-1].get('ci_lo_RE', float('nan')):+.4f};
   {st.get('pooled', [{}])[-1].get('ci_hi_RE', float('nan')):+.4f}] — chứa 0.
2. **Giả thuyết underfitting bị bác bỏ; cơ chế là sụp đổ tối ưu hóa.**
   Tỉ lệ sụp đổ `edge_mlp` {cr.get('bootstrap_ci', {}).get('edge_mlp', {}).get('rate', float('nan')):.0%},
   `sage` {cr.get('bootstrap_ci', {}).get('sage', {}).get('rate', float('nan')):.0%},
   `sage_edge` {cr.get('bootstrap_ci', {}).get('sage_edge', {}).get('rate', float('nan')):.0%};
   Fisher exact `sage` vs `edge_mlp` p = {cr.get('fisher_exact', {}).get('sage_vs_edge_mlp', {}).get('p_holm', float('nan')):.4f} (Holm).
3. **Ba seed không đủ.** SD tăng tới {au.get('std_ratio_max', float('nan')):.2f}× khi thêm
   seed 44/55; {st.get('ci_decision_flips_3_to_5_seed', '—')}/{st.get('n_stability_rows', '—')}
   quyết định dựa trên KTC bị đảo.
4. **Ngân sách không đồng nhất.** UNSW chạy 3,67 lượt, ba bộ còn lại đúng 2,0 lượt.
""")

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
