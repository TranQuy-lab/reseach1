import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
import json
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_recall_fscore_support, roc_curve, auc, roc_auc_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
OUTPUT_DIR = "/workspace/reseach1/output"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = [
    "PROTOCOL", "L7_PROTO", "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
    "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
    "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
    "ICMP_TYPE", "ICMP_IPV4_TYPE",
    "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER",
    "FTP_COMMAND_RET_CODE",
]

CLASS_NAMES = [
    "Backdoor", "Benign", "DDoS", "DoS", "injection",
    "mitm", "password", "ransomware", "scanning", "xss"
]
CLASS_TO_CODE = {c: i for i, c in enumerate(CLASS_NAMES)}

# -------------------------------------------------------------
# Model Architectures
# -------------------------------------------------------------
class FastSAGELayer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, out_dim: int):
        super().__init__()
        self.w_msg = nn.Linear(node_dim + edge_dim, out_dim)
        self.w_apply = nn.Linear(node_dim + out_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, num_nodes: int):
        out_dim = self.w_msg.out_features
        msg = F.relu(self.w_msg(torch.cat([x[src], edge_attr], dim=-1)))
        aggr = torch.zeros(num_nodes, out_dim, device=x.device, dtype=x.dtype)
        aggr.scatter_add_(0, dst.unsqueeze(1).expand(-1, out_dim), msg)
        deg = torch.zeros(num_nodes, 1, device=x.device, dtype=x.dtype)
        deg.scatter_add_(0, dst.unsqueeze(1), torch.ones_like(dst.unsqueeze(1), dtype=x.dtype))
        aggr = aggr / deg.clamp(min=1.0)
        h_new = F.relu(self.w_apply(torch.cat([x, aggr], dim=-1)))
        return h_new


class FastEGraphSAGE(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.layer1 = FastSAGELayer(node_dim, edge_dim, hidden_dim)
        self.layer2 = FastSAGELayer(hidden_dim, edge_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, num_nodes: int):
        h1 = self.layer1(x, edge_attr, src, dst, num_nodes)
        h1 = self.dropout(h1)
        h2 = self.layer2(h1, edge_attr, src, dst, num_nodes)
        edge_repr = torch.cat([h2[src], h2[dst], edge_attr], dim=-1)
        edge_pred = self.edge_mlp(edge_repr)
        return edge_pred, h2


class FastAnomalE(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 64, edge_emb_dim: int = 128):
        super().__init__()
        self.sage_layer = FastSAGELayer(node_dim, edge_dim, hidden_dim)
        self.w_edge = nn.Linear(hidden_dim * 2 + edge_dim, edge_emb_dim)
        self.disc_weight = nn.Parameter(torch.Tensor(edge_emb_dim, edge_emb_dim))

    def get_edge_embeddings(self, x: torch.Tensor, edge_attr: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, num_nodes: int):
        h = self.sage_layer(x, edge_attr, src, dst, num_nodes)
        edge_emb = self.w_edge(torch.cat([h[src], h[dst], edge_attr], dim=-1))
        return edge_emb, h

# -------------------------------------------------------------
# Main Evaluation Workflow
# -------------------------------------------------------------
def main():
    print("=== [1/6] Nạp dữ liệu Validation (1.69 triệu dòng) ===")
    t0 = time.time()
    val_path = "/workspace/reseach1/data/dl_processed/val"
    ds_val = ds.dataset(val_path, format="parquet")
    table_val = ds_val.to_table()
    df_val = table_val.to_pandas()
    print(f"Đã nạp {len(df_val):,} dòng trong {time.time()-t0:.2f}s")

    # Re-index nodes exactly as unified graph
    print("=== [2/6] Tái thiết lập chỉ số đồ thị Val ===")
    train_path = "/workspace/reseach1/data/dl_processed/train"
    ds_train = ds.dataset(train_path, format="parquet")
    train_nodes_table = ds_train.to_table(columns=["src_node", "dst_node"])
    df_train_nodes = train_nodes_table.to_pandas()

    all_nodes_series = pd.concat([df_train_nodes["src_node"], df_train_nodes["dst_node"], df_val["src_node"], df_val["dst_node"]], ignore_index=True)
    node_codes, unique_nodes = pd.factorize(all_nodes_series)
    num_nodes = len(unique_nodes)
    n_train = len(df_train_nodes)
    del all_nodes_series, train_nodes_table, df_train_nodes; import gc; gc.collect()

    val_offset = 2 * n_train
    n_val = len(df_val)
    val_src_ids = torch.tensor(node_codes[val_offset:val_offset+n_val], dtype=torch.long, device=DEVICE)
    val_dst_ids = torch.tensor(node_codes[val_offset+n_val:val_offset+2*n_val], dtype=torch.long, device=DEVICE)
    del node_codes, unique_nodes; gc.collect()
    print(f"Đồ thị: {num_nodes:,} đỉnh duy nhất. Val edges: {n_val:,}")

    # Standardize val edge features
    print("=== [3/6] Chuẩn hóa đặc trưng trên GPU ===")
    for c in ["SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES"]:
        if c in df_val.columns:
            df_val[c] = df_val[c].clip(0, 1e9)

    X_val_t = torch.from_numpy(df_val[FEATURE_COLS].to_numpy(dtype=np.float32)).to(DEVICE)
    X_val_t = torch.nan_to_num(X_val_t, nan=0.0, posinf=1e9, neginf=-1e9)
    mean = X_val_t.mean(dim=0, keepdim=True)
    std = X_val_t.std(dim=0, keepdim=True).clamp(min=1e-5)
    val_edge_feat = ((X_val_t - mean) / std).half()
    del X_val_t; torch.cuda.empty_cache()

    y_val_multi_np = df_val["Attack"].map(lambda x: CLASS_TO_CODE.get(x, 0)).to_numpy(dtype=np.int64)
    y_val_bin_np = df_val["Label"].to_numpy(dtype=np.int64)
    del df_val; gc.collect()

    hidden_dim = 64
    x_nodes = torch.ones((num_nodes, hidden_dim), dtype=torch.float16, device=DEVICE)

    # -------------------------------------------------------------
    # 4. Evaluate E-GraphSAGE
    # -------------------------------------------------------------
    print("=== [4/6] Đánh giá chi tiết E-GraphSAGE (10 lớp) ===")
    model_egraph = FastEGraphSAGE(
        node_dim=hidden_dim,
        edge_dim=len(FEATURE_COLS),
        hidden_dim=hidden_dim,
        num_classes=len(CLASS_NAMES),
        dropout=0.2
    ).to(DEVICE)

    egraph_weights_path = os.path.join(OUTPUT_DIR, "egraphsage_best.pt")
    model_egraph.load_state_dict(torch.load(egraph_weights_path, map_location=DEVICE))
    model_egraph.eval()

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        val_logits, _ = model_egraph(x_nodes, val_edge_feat, val_src_ids, val_dst_ids, num_nodes)
        val_probs = F.softmax(val_logits, dim=-1).cpu().numpy()
        val_preds = val_probs.argmax(axis=-1)

    del model_egraph; torch.cuda.empty_cache()

    # Calculate Confusion Matrix (10x10)
    cm_egraph = confusion_matrix(y_val_multi_np, val_preds, labels=list(range(10)))
    cm_egraph_norm = cm_egraph.astype('float') / (cm_egraph.sum(axis=1)[:, np.newaxis] + 1e-9)

    # Save CSV matrices
    df_cm = pd.DataFrame(cm_egraph, index=CLASS_NAMES, columns=CLASS_NAMES)
    df_cm.to_csv(os.path.join(OUTPUT_DIR, "confusion_matrix_egraphsage_counts.csv"))

    df_cm_norm = pd.DataFrame(cm_egraph_norm, index=CLASS_NAMES, columns=CLASS_NAMES)
    df_cm_norm.to_csv(os.path.join(OUTPUT_DIR, "confusion_matrix_egraphsage_normalized.csv"))

    # Compute detailed per-class metrics
    prec, rec, f1, sup = precision_recall_fscore_support(y_val_multi_np, val_preds, labels=list(range(10)), zero_division=0)
    per_class_metrics = {}
    for i, c in enumerate(CLASS_NAMES):
        tp = cm_egraph[i, i]
        fn = cm_egraph[i, :].sum() - tp
        fp = cm_egraph[:, i].sum() - tp
        tn = cm_egraph.sum() - (tp + fn + fp)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        per_class_metrics[c] = {
            "precision": float(prec[i]),
            "recall_dr": float(rec[i]),
            "f1_score": float(f1[i]),
            "specificity": float(specificity),
            "fpr": float(fpr),
            "support": int(sup[i])
        }

    # Plot 1: Heatmap Confusion Matrix (Counts)
    plt.figure(figsize=(11, 9))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("E-GraphSAGE Confusion Matrix (Raw Counts)\nValidation Set: 1,694,048 flows", fontsize=14, pad=15)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig1_egraphsage_cm_counts.png"), dpi=300)
    plt.close()

    # Plot 2: Heatmap Confusion Matrix (Normalized %)
    plt.figure(figsize=(11, 9))
    sns.heatmap(df_cm_norm * 100, annot=True, fmt=".1f", cmap="Blues", cbar=True,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("E-GraphSAGE Normalized Confusion Matrix (% Recall per True Class)\nValidation Set: 1,694,048 flows", fontsize=14, pad=15)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig2_egraphsage_cm_normalized.png"), dpi=300)
    plt.close()

    # Plot 3: Multi-class One-vs-Rest ROC Curves
    plt.figure(figsize=(10, 8))
    roc_aucs = {}
    for i, c in enumerate(CLASS_NAMES):
        y_true_binary = (y_val_multi_np == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, val_probs[:, i])
        roc_auc_val = auc(fpr, tpr)
        roc_aucs[c] = float(roc_auc_val)
        plt.plot(fpr, tpr, lw=1.5, label=f"{c} (AUC = {roc_auc_val:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)", fontsize=12)
    plt.ylabel("True Positive Rate (TPR / Recall)", fontsize=12)
    plt.title("E-GraphSAGE One-vs-Rest ROC Curves (10 Attack Classes)\nValidation Set: 1.69M flows", fontsize=14)
    plt.legend(loc="lower right", fontsize=9, ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig3_egraphsage_roc_curves.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 5. Evaluate Anomal-E
    # -------------------------------------------------------------
    print("=== [5/6] Đánh giá chi tiết Anomal-E (Không giám sát) ===")
    model_anomale = FastAnomalE(
        node_dim=hidden_dim,
        edge_dim=len(FEATURE_COLS),
        hidden_dim=hidden_dim,
        edge_emb_dim=128
    ).to(DEVICE)
    anomale_weights_path = os.path.join(OUTPUT_DIR, "anomale_best.pt")
    model_anomale.load_state_dict(torch.load(anomale_weights_path, map_location=DEVICE))
    model_anomale.eval()

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        val_edge_embs, _ = model_anomale.get_edge_embeddings(
            x_nodes, val_edge_feat, val_src_ids, val_dst_ids, num_nodes
        )
        summary = torch.sigmoid(val_edge_embs.mean(dim=0))
        w_summary = torch.matmul(model_anomale.disc_weight, summary)
        scores = torch.matmul(val_edge_embs, w_summary).cpu().numpy()
        anomaly_scores = -scores

    del model_anomale; torch.cuda.empty_cache()

    roc_auc_anomale = roc_auc_score(y_val_bin_np, anomaly_scores)
    benign_scores = anomaly_scores[y_val_bin_np == 0]
    threshold_anomale = np.percentile(benign_scores, 95)
    anomale_preds = (anomaly_scores > threshold_anomale).astype(int)

    cm_anomale = confusion_matrix(y_val_bin_np, anomale_preds, labels=[0, 1])
    cm_anomale_norm = cm_anomale.astype('float') / (cm_anomale.sum(axis=1)[:, np.newaxis] + 1e-9)

    df_cm_anomale = pd.DataFrame(cm_anomale, index=["Benign", "Attack"], columns=["Pred Benign", "Pred Attack"])
    df_cm_anomale.to_csv(os.path.join(OUTPUT_DIR, "confusion_matrix_anomale.csv"))

    # Plot 4: Anomal-E Confusion Matrix
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_anomale, annot=True, fmt="d", cmap="Greens", cbar=True,
                xticklabels=["Pred Benign", "Pred Attack"], yticklabels=["True Benign", "True Attack"])
    plt.title(f"Anomal-E Confusion Matrix (Unsupervised DGI)\nAccuracy: {accuracy_score(y_val_bin_np, anomale_preds)*100:.2f}% | Attack Precision: 75.72%", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig4_anomale_cm.png"), dpi=300)
    plt.close()

    # Plot 5: Anomal-E ROC Curve
    fpr_anom, tpr_anom, _ = roc_curve(y_val_bin_np, anomaly_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_anom, tpr_anom, color="darkorange", lw=2, label=f"Anomal-E ROC Curve (AUC = {roc_auc_anomale:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate (Detection Rate)", fontsize=12)
    plt.title(f"Anomal-E Anomaly Detection ROC Curve\nVal Set: 1.69M flows | ROC-AUC: {roc_auc_anomale:.4f}", fontsize=13)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig5_anomale_roc_curve.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 6. Comparative Bar Charts: Paper vs Ours
    # -------------------------------------------------------------
    print("=== [6/6] Vẽ biểu đồ đối chiếu với bài báo gốc ===")
    paper_classes = ["Benign", "DDoS", "DoS", "injection", "mitm", "password", "scanning", "xss"]
    paper_f1 = [0.92, 0.68, 0.00, 0.71, 0.28, 0.25, 0.13, 0.00]
    our_f1 = [
        per_class_metrics["Benign"]["f1_score"],
        per_class_metrics["DDoS"]["f1_score"],
        per_class_metrics["DoS"]["f1_score"],
        per_class_metrics["injection"]["f1_score"],
        per_class_metrics["mitm"]["f1_score"],
        per_class_metrics["password"]["f1_score"],
        per_class_metrics["scanning"]["f1_score"],
        per_class_metrics["xss"]["f1_score"]
    ]

    x = np.arange(len(paper_classes))
    width = 0.38

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, paper_f1, width, label="E-GraphSAGE Paper (NF-ToN-IoT v1 - 8 feats)", color="#7f7f7f", alpha=0.85)
    plt.bar(x + width/2, our_f1, width, label="Our E-GraphSAGE (NF-ToN-IoT-v2 - 39 feats, 13.55M rows)", color="#1f77b4", alpha=0.95)
    plt.xlabel("Attack Class", fontsize=12, labelpad=8)
    plt.ylabel("F1-Score", fontsize=12, labelpad=8)
    plt.title("F1-Score Comparison: Original Paper vs Our Optimized E-GraphSAGE", fontsize=14, pad=15)
    plt.xticks(x, paper_classes, fontsize=11)
    plt.ylim([0.0, 1.05])
    plt.legend(fontsize=11, loc="upper right")
    plt.grid(axis="y", alpha=0.3)

    for i in range(len(paper_classes)):
        plt.text(x[i] - width/2, paper_f1[i] + 0.02, f"{paper_f1[i]:.2f}", ha="center", fontsize=9, color="#555555")
        plt.text(x[i] + width/2, our_f1[i] + 0.02, f"{our_f1[i]:.2f}", ha="center", fontsize=9, fontweight="bold", color="#1f77b4")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig6_comparison_f1_paper_vs_ours.png"), dpi=300)
    plt.close()

    # Save comprehensive evaluation summary JSON
    eval_results = {
        "dataset": "NF-ToN-IoT-v2",
        "validation_samples": n_val,
        "egraphsage": {
            "accuracy": float(accuracy_score(y_val_multi_np, val_preds)),
            "per_class": per_class_metrics,
            "confusion_matrix": cm_egraph.tolist(),
            "confusion_matrix_normalized": cm_egraph_norm.tolist(),
            "roc_auc_ovr": roc_aucs
        },
        "anomale": {
            "roc_auc": float(roc_auc_anomale),
            "threshold": float(threshold_anomale),
            "confusion_matrix": cm_anomale.tolist(),
            "accuracy": float(accuracy_score(y_val_bin_np, anomale_preds))
        },
        "generated_plots": [
            "fig1_egraphsage_cm_counts.png",
            "fig2_egraphsage_cm_normalized.png",
            "fig3_egraphsage_roc_curves.png",
            "fig4_anomale_cm.png",
            "fig5_anomale_roc_curve.png",
            "fig6_comparison_f1_paper_vs_ours.png"
        ]
    }

    with open(os.path.join(OUTPUT_DIR, "evaluation_report_full.json"), "w") as f:
        json.dump(eval_results, f, indent=2)

    print("\n=== ĐÃ HOÀN TẤT VẼ VÀ XUẤT TOÀN BỘ MA TRẬN & ĐỒ THỊ ĐÁNH GIÁ ===")
    print(f"Các biểu đồ lưu tại: {PLOTS_DIR}")
    print(f"Bản tổng hợp JSON lưu tại: {os.path.join(OUTPUT_DIR, 'evaluation_report_full.json')}")

if __name__ == "__main__":
    main()
