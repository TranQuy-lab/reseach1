"""
====================================================================================================
BÀI 2: ANOMAL-E SELF-SUPERVISED DEEP GRAPH INFOMAX (DGI) ANOMALY DETECTION
QUY MÔ TOÀN BỘ 13.55 TRIỆU LUỒNG MẠNG (NF-ToN-IoT-v2) TRÊN 1 GPU RTX 3090
====================================================================================================
Kiến trúc:
- FastSAGELayer kết hợp FastAnomalE (Deep Graph Infomax)
- Negative Edge Corruption + Bilinear Discriminator
- Trích xuất Anomaly Score không giám sát trên toàn bộ 1.69 triệu luồng Validation
- Hạ nguồn: Kiểm chuẩn ngưỡng Percentile (95% Benign Recall) + Isolation Forest CPU
====================================================================================================
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
import json
import logging
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from sklearn.metrics import classification_report, f1_score, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# 1. Logging Setup with Immediate Flush
# -----------------------------------------------------------------------------
class FlushingFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

OUTPUT_DIR = os.path.abspath("output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)

logger = logging.getLogger("anomale_13m")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

fh = FlushingFileHandler(os.path.join(OUTPUT_DIR, "anomale_run.log"), mode="w", encoding="utf-8")
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)

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

# -----------------------------------------------------------------------------
# 2. Fast Native PyTorch Modules
# -----------------------------------------------------------------------------
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


class FastAnomalE(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 64, edge_emb_dim: int = 128):
        super().__init__()
        self.sage_layer = FastSAGELayer(node_dim, edge_dim, hidden_dim)
        self.w_edge = nn.Linear(hidden_dim * 2 + edge_dim, edge_emb_dim)
        self.disc_weight = nn.Parameter(torch.Tensor(edge_emb_dim, edge_emb_dim))
        nn.init.xavier_uniform_(self.disc_weight)

    def get_edge_embeddings(self, x: torch.Tensor, edge_attr: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, num_nodes: int):
        h = self.sage_layer(x, edge_attr, src, dst, num_nodes)
        edge_emb = self.w_edge(torch.cat([h[src], h[dst], edge_attr], dim=-1))
        return edge_emb, h

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, num_nodes: int):
        if self.training:
            h_pos = checkpoint(self.sage_layer, x, edge_attr, src, dst, num_nodes, use_reentrant=False)
        else:
            h_pos = self.sage_layer(x, edge_attr, src, dst, num_nodes)
        pos_edge_emb = self.w_edge(torch.cat([h_pos[src], h_pos[dst], edge_attr], dim=-1))

        # Corrupt edge features (negative samples)
        e_perm = torch.randperm(edge_attr.size(0), device=x.device)
        neg_attr = edge_attr[e_perm]
        neg_edge_emb = self.w_edge(torch.cat([h_pos[src], h_pos[dst], neg_attr], dim=-1))

        # Global graph summary vector
        summary = torch.sigmoid(pos_edge_emb.mean(dim=0))

        # Bilinear discriminator scores
        w_summary = torch.matmul(self.disc_weight, summary)
        pos_scores = torch.matmul(pos_edge_emb, w_summary)
        neg_scores = torch.matmul(neg_edge_emb, w_summary)

        loss_pos = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
        loss_neg = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
        return loss_pos + loss_neg, w_summary, pos_edge_emb


def resolve_data_path():
    candidates = [
        ("data/dl_processed/train", "data/dl_processed/val"),
        ("nids_preprocessing_pipeline_updated/data/dl_processed/train", "nids_preprocessing_pipeline_updated/data/dl_processed/val"),
        ("/workspace/reseach1/data/dl_processed/train", "/workspace/reseach1/data/dl_processed/val"),
    ]
    for t_path, v_path in candidates:
        if os.path.exists(t_path) and os.path.exists(v_path):
            return t_path, v_path
    raise FileNotFoundError("Không tìm thấy thư mục dữ liệu train/val parquet!")


def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0

    logger.info("=" * 80)
    logger.info("=== BẮT ĐẦU THỰC THI ANOMAL-E (DGI) TRÊN PHÂN TẬP NF-ToN-IoT-v2 (13.55M DÒNG) ===")
    logger.info(f"Phần cứng: {gpu_name} ({total_vram_gb:.1f} GB VRAM) | CPU Cores: {os.cpu_count()}")
    logger.info("=" * 80)

    train_path, val_path = resolve_data_path()

    # 1. Nạp dữ liệu
    logger.info("[1/4] Nạp dữ liệu Parquet...")
    t0 = time.time()
    df_train = ds.dataset(train_path, format="parquet").to_table().to_pandas()
    df_val = ds.dataset(val_path, format="parquet").to_table().to_pandas()
    n_train, n_val = len(df_train), len(df_val)
    logger.info(f"Nạp hoàn tất: Train = {n_train:,} dòng | Val = {n_val:,} dòng trong {time.time()-t0:.2f}s")

    # 2. Xây dựng đồ thị
    logger.info("[2/4] Lập chỉ số đỉnh đồ thị...")
    t_idx = time.time()
    all_nodes_series = pd.concat([df_train["src_node"], df_train["dst_node"], df_val["src_node"], df_val["dst_node"]], ignore_index=True)
    node_codes, unique_nodes = pd.factorize(all_nodes_series)
    num_nodes = len(unique_nodes)
    logger.info(f"Tổng số đỉnh duy nhất: {num_nodes:,} (xong trong {time.time()-t_idx:.2f}s)")

    train_src_ids = torch.tensor(node_codes[:n_train], dtype=torch.long, device=device)
    train_dst_ids = torch.tensor(node_codes[n_train:2*n_train], dtype=torch.long, device=device)
    val_offset = 2 * n_train
    val_src_ids = torch.tensor(node_codes[val_offset:val_offset+n_val], dtype=torch.long, device=device)
    val_dst_ids = torch.tensor(node_codes[val_offset+n_val:val_offset+2*n_val], dtype=torch.long, device=device)
    del all_nodes_series, node_codes, unique_nodes
    import gc; gc.collect()

    # 3. Chuẩn hóa đặc trưng trên GPU
    logger.info("[3/4] Chuẩn hóa đặc trưng GPU (FP16)...")
    for c in ["SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES"]:
        if c in df_train.columns:
            df_train[c] = df_train[c].clip(0, 1e9)
            df_val[c] = df_val[c].clip(0, 1e9)

    X_train_t = torch.from_numpy(df_train[FEATURE_COLS].to_numpy(dtype=np.float32)).to(device)
    X_train_t = torch.nan_to_num(X_train_t, nan=0.0, posinf=1e9, neginf=-1e9)
    mean = X_train_t.mean(dim=0, keepdim=True)
    std = X_train_t.std(dim=0, keepdim=True).clamp(min=1e-5)
    train_edge_feat = ((X_train_t - mean) / std).half()
    del X_train_t
    torch.cuda.empty_cache()

    X_val_t = torch.from_numpy(df_val[FEATURE_COLS].to_numpy(dtype=np.float32)).to(device)
    X_val_t = torch.nan_to_num(X_val_t, nan=0.0, posinf=1e9, neginf=-1e9)
    val_edge_feat = ((X_val_t - mean) / std).half()
    del X_val_t
    torch.cuda.empty_cache()

    y_train_bin = torch.tensor(df_train["Label"].values, dtype=torch.long, device=device)
    y_val_bin = torch.tensor(df_val["Label"].values, dtype=torch.long, device=device)
    y_val_bin_np = y_val_bin.cpu().numpy()

    hidden_dim = 64
    x_nodes = torch.ones((num_nodes, hidden_dim), dtype=torch.float16, device=device)
    del df_train, df_val
    gc.collect()

    # 4. Huấn luyện tự giám sát Anomal-E (DGI)
    logger.info("[4/4] Bắt đầu huấn luyện tự giám sát Anomal-E (DGI Contrastive Learning)...")
    model_anomale = FastAnomalE(
        node_dim=hidden_dim,
        edge_dim=len(FEATURE_COLS),
        hidden_dim=hidden_dim,
        edge_emb_dim=128
    ).to(device)

    optimizer = torch.optim.Adam(model_anomale.parameters(), lr=0.005, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    epochs = 8
    last_w_summary = None
    dgi_history = []

    for ep in range(1, epochs + 1):
        t_ep = time.time()
        model_anomale.train()
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.float16):
            loss, w_summary, _ = model_anomale(x_nodes, train_edge_feat, train_src_ids, train_dst_ids, num_nodes)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        last_w_summary = w_summary.detach()

        ep_duration = time.time() - t_ep
        vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        dgi_history.append({"epoch": ep, "dgi_loss": round(float(loss.item()), 4), "time_sec": round(ep_duration, 2), "vram_mb": round(vram_mb, 1)})
        logger.info(f"Epoch {ep:02d}/{epochs:02d} | Contrastive Loss: {loss.item():.4f} | {ep_duration:.2f}s | VRAM: {vram_mb:.0f} MB")

        del loss, w_summary
        torch.cuda.empty_cache()

    torch.save(model_anomale.state_dict(), os.path.join(OUTPUT_DIR, "anomale_best.pt"))
    pd.DataFrame(dgi_history).to_csv(os.path.join(OUTPUT_DIR, "anomale_training_history.csv"), index=False)

    # 5. Đánh giá Anomaly Scoring không giám sát
    logger.info("\n=== ĐÁNH GIÁ ANOMAL-E KHÔNG GIÁM SÁT TRÊN 1.69 TRIỆU DÒNG VAL ===")
    model_anomale.eval()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        val_edge_embs, _ = model_anomale.get_edge_embeddings(x_nodes, val_edge_feat, val_src_ids, val_dst_ids, num_nodes)
        scores = torch.matmul(val_edge_embs, last_w_summary).cpu().numpy()
        anomaly_scores = -scores

    # Tính ROC-AUC
    roc_auc = roc_auc_score(y_val_bin_np, anomaly_scores)
    logger.info(f"Anomal-E ROC-AUC: {roc_auc:.4f}")

    # Ngưỡng phân tách 95% Benign
    benign_scores = anomaly_scores[y_val_bin_np == 0]
    threshold = np.percentile(benign_scores, 95)
    anomale_preds = (anomaly_scores > threshold).astype(int)

    acc = accuracy_score(y_val_bin_np, anomale_preds)
    f1 = f1_score(y_val_bin_np, anomale_preds, average="binary")
    macro_f1 = f1_score(y_val_bin_np, anomale_preds, average="macro")

    logger.info(f"Ngưỡng Anomaly Score (95% Benign Recall): {threshold:.4f}")
    logger.info(f"Độ chính xác (Accuracy): {acc*100:.2f}% | Binary F1: {f1:.4f} | Macro F1: {macro_f1:.4f}")
    rep = classification_report(y_val_bin_np, anomale_preds, target_names=["Benign", "Attack"], digits=4)
    logger.info(f"\n{rep}")

    # Lưu Ma trận nhầm lẫn nhị phân
    cm = confusion_matrix(y_val_bin_np, anomale_preds)
    pd.DataFrame(cm, index=["Benign", "Attack"], columns=["Benign", "Attack"]).to_csv(os.path.join(OUTPUT_DIR, "confusion_matrix_anomale.csv"))

    # Vẽ ma trận nhầm lẫn
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=["Pred Benign", "Pred Attack"], yticklabels=["True Benign", "True Attack"])
    plt.title(f"Anomal-E Anomaly Confusion Matrix (ROC-AUC = {roc_auc:.4f})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plots", "fig4_anomale_cm.png"), dpi=300)
    plt.close()

    # Hạ nguồn: Isolation Forest trên CPU
    logger.info("\n=== HUẤN LUYỆN HẠ NGUỒN ISOLATION FOREST TRÊN 32 vCPUs ===")
    t_if = time.time()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        sample_benign_idx = torch.where(y_train_bin == 0)[0][:200000]
        train_benign_embs, _ = model_anomale.get_edge_embeddings(
            x_nodes, train_edge_feat[sample_benign_idx], train_src_ids[sample_benign_idx], train_dst_ids[sample_benign_idx], num_nodes
        )
        train_benign_np = train_benign_embs.cpu().numpy()

    iso_forest = IsolationForest(n_estimators=100, contamination=0.1, n_jobs=-1, random_state=42)
    iso_forest.fit(train_benign_np)
    logger.info(f"Isolation Forest fit xong trong {time.time()-t_if:.2f}s!")

    val_sample_np = val_edge_embs[:200000].cpu().numpy()
    val_sample_labels = y_val_bin_np[:200000]
    if_preds = iso_forest.predict(val_sample_np)
    if_preds_bin = np.where(if_preds == -1, 1, 0)
    if_acc = accuracy_score(val_sample_labels, if_preds_bin)
    if_f1 = f1_score(val_sample_labels, if_preds_bin, average="binary")
    logger.info(f"Isolation Forest (200k mẫu): Accuracy = {if_acc*100:.2f}% | Binary F1 = {if_f1:.4f}")

    logger.info(f"\nHoàn tất toàn bộ thực nghiệm Anomal-E! Kết quả lưu tại: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
