import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
import json
import logging
import traceback
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from sklearn.metrics import classification_report, f1_score, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import IsolationForest

# -------------------------------------------------------------
# Logging Setup with Immediate Auto-Flush
# -------------------------------------------------------------
class FlushingFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

LOG_FILE = "/workspace/reseach1/output/train_full.log"
os.makedirs("/workspace/reseach1/output", exist_ok=True)

logger = logging.getLogger("train_full")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

fh = FlushingFileHandler(LOG_FILE, mode="w", encoding="utf-8")
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

# -------------------------------------------------------------
# 1. Native High-Performance PyTorch E-GraphSAGE Layer
# -------------------------------------------------------------
class FastSAGELayer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, out_dim: int):
        super().__init__()
        self.w_msg = nn.Linear(node_dim + edge_dim, out_dim)
        self.w_apply = nn.Linear(node_dim + out_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, num_nodes: int):
        out_dim = self.w_msg.out_features
        # Compute messages
        msg = F.relu(self.w_msg(torch.cat([x[src], edge_attr], dim=-1)))

        # Native CUDA Scatter Add Aggregation
        aggr = torch.zeros(num_nodes, out_dim, device=x.device, dtype=x.dtype)
        aggr.scatter_add_(0, dst.unsqueeze(1).expand(-1, out_dim), msg)

        deg = torch.zeros(num_nodes, 1, device=x.device, dtype=x.dtype)
        deg.scatter_add_(0, dst.unsqueeze(1), torch.ones_like(dst.unsqueeze(1), dtype=x.dtype))
        aggr = aggr / deg.clamp(min=1.0)

        # Update node representation
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
        if self.training:
            h = checkpoint(self.layer1, x, edge_attr, src, dst, num_nodes, use_reentrant=False)
            h = self.dropout(h)
            h = checkpoint(self.layer2, h, edge_attr, src, dst, num_nodes, use_reentrant=False)
        else:
            h = self.layer1(x, edge_attr, src, dst, num_nodes)
            h = self.dropout(h)
            h = self.layer2(h, edge_attr, src, dst, num_nodes)
        edge_repr = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        edge_pred = self.edge_mlp(edge_repr)
        return edge_pred, h


# -------------------------------------------------------------
# 2. Native High-Performance Anomal-E (DGI + Contrastive)
# -------------------------------------------------------------
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

        # Corrupt edge features (negative edges)
        e_perm = torch.randperm(edge_attr.size(0), device=x.device)
        neg_attr = edge_attr[e_perm]
        neg_edge_emb = self.w_edge(torch.cat([h_pos[src], h_pos[dst], neg_attr], dim=-1))

        # Global graph summary
        summary = torch.sigmoid(pos_edge_emb.mean(dim=0))

        # Bilinear discriminator scores
        w_summary = torch.matmul(self.disc_weight, summary)
        pos_scores = torch.matmul(pos_edge_emb, w_summary)
        neg_scores = torch.matmul(neg_edge_emb, w_summary)

        loss_pos = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
        loss_neg = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
        return loss_pos + loss_neg, w_summary, pos_edge_emb


# -------------------------------------------------------------
# Main Execution Pipeline
# -------------------------------------------------------------
def main():
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(os.cpu_count() or 16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
    
    logger.info("=" * 75)
    logger.info("=== BẮT ĐẦU HUẤN LUYỆN TOÀN BỘ 13.55 TRIỆU DÒNG (NF-ToN-IoT-v2) ===")
    logger.info(f"Phần cứng: {gpu_name} ({total_vram_gb:.1f} GB VRAM) | CPU Cores: {os.cpu_count()}")
    logger.info("=" * 75)

    train_path = "/workspace/reseach1/data/dl_processed/train"
    val_path   = "/workspace/reseach1/data/dl_processed/val"

    # 1. Load Data with PyArrow Multi-threaded
    logger.info("[1/5] Đang tải tập Train và Val qua PyArrow...")
    t0 = time.time()
    ds_train = ds.dataset(train_path, format="parquet")
    table_train = ds_train.to_table()
    df_train = table_train.to_pandas()

    ds_val = ds.dataset(val_path, format="parquet")
    table_val = ds_val.to_table()
    df_val = table_val.to_pandas()
    logger.info(f"Tải hoàn tất: Train = {len(df_train):,} dòng | Val = {len(df_val):,} dòng trong {time.time()-t0:.2f}s")

    # 2. Build Unified Node ID Index
    logger.info("[2/5] Xây dựng đồ thị (Graph Indexing) cho đỉnh nguồn/đích...")
    t_idx = time.time()
    all_nodes_series = pd.concat([df_train["src_node"], df_train["dst_node"], df_val["src_node"], df_val["dst_node"]], ignore_index=True)
    node_codes, unique_nodes = pd.factorize(all_nodes_series)
    num_nodes = len(unique_nodes)
    logger.info(f"Đồ thị có tổng cộng: {num_nodes:,} đỉnh duy nhất (IP:Port pairs). Xử lý xong trong {time.time()-t_idx:.2f}s")

    n_train = len(df_train)
    train_src_ids = torch.tensor(node_codes[:n_train], dtype=torch.long, device=device)
    train_dst_ids = torch.tensor(node_codes[n_train:2*n_train], dtype=torch.long, device=device)
    
    val_offset = 2 * n_train
    n_val = len(df_val)
    val_src_ids = torch.tensor(node_codes[val_offset:val_offset+n_val], dtype=torch.long, device=device)
    val_dst_ids = torch.tensor(node_codes[val_offset+n_val:val_offset+2*n_val], dtype=torch.long, device=device)

    # Free memory
    del all_nodes_series, node_codes, unique_nodes
    import gc; gc.collect()

    # 3. Standardize Edge Features directly on GPU (Ultra-fast, zero CPU RAM overhead)
    logger.info("[3/5] Chuẩn hóa 39 đặc trưng NetFlow trực tiếp trên GPU CUDA...")
    t_scale = time.time()
    
    # Pre-clamp extreme float outliers in pandas before converting to float32
    for c in ["SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES"]:
        if c in df_train.columns:
            df_train[c] = df_train[c].clip(0, 1e9)
            df_val[c] = df_val[c].clip(0, 1e9)

    # Directly create float32 tensors on GPU
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
    logger.info(f"Chuẩn hóa GPU hoàn tất trong {time.time()-t_scale:.2f}s!")

    # 4. Fast Label Factorization (Hash-based, avoids O(N log N) string sort)
    logger.info("[4/5] Mã hóa nhãn (pd.factorize)...")
    t_lbl = time.time()
    y_train_multi_np, class_names_arr = pd.factorize(df_train["Attack"], sort=True)
    class_names = [str(c) for c in class_names_arr]
    num_classes = len(class_names)
    class_to_code = {c: i for i, c in enumerate(class_names)}
    y_val_multi_np = df_val["Attack"].map(lambda x: class_to_code.get(x, 0)).to_numpy(dtype=np.int64)
    logger.info(f"Mã hóa nhãn {num_classes} lớp hoàn tất trong {time.time()-t_lbl:.2f}s!")
    logger.info(f"Danh sách {num_classes} lớp tấn công: {class_names}")

    y_train_multi = torch.tensor(y_train_multi_np, dtype=torch.long, device=device)
    y_val_multi   = torch.tensor(y_val_multi_np, dtype=torch.long, device=device)

    y_train_bin = torch.tensor(df_train["Label"].values, dtype=torch.long, device=device)
    y_val_bin   = torch.tensor(df_val["Label"].values, dtype=torch.long, device=device)

    # Compute balanced class weights with square-root damping
    class_counts = np.bincount(y_train_multi_np, minlength=num_classes)
    weights = np.sqrt(len(y_train_multi_np) / (num_classes * np.maximum(class_counts, 1.0)))
    weights = torch.tensor(weights, dtype=torch.float32, device=device)

    # Initial Node representation vector (ones)
    hidden_dim = 64
    x_nodes = torch.ones((num_nodes, hidden_dim), dtype=torch.float16, device=device)

    # Free raw DataFrames completely
    del df_train, df_val
    import gc; gc.collect()

    # -------------------------------------------------------------
    # MODEL 1: E-GraphSAGE (Multiclass Edge Classification)
    # -------------------------------------------------------------
    logger.info("=" * 75)
    logger.info("=== BƯỚC 1: HUẤN LUYỆN MÔ HÌNH E-GRAPHSAGE (10 LỚP TẤN CÔNG) ===")
    logger.info("=" * 75)

    model_egraph = FastEGraphSAGE(
        node_dim=hidden_dim,
        edge_dim=len(FEATURE_COLS),
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=0.2
    ).to(device)

    optimizer = torch.optim.AdamW(model_egraph.parameters(), lr=0.005, weight_decay=1e-4)
    scaler_amp = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss(weight=weights)

    epochs = 10
    best_val_macro_f1 = 0.0
    best_report = ""
    best_val_acc = 0.0
    best_val_weighted_f1 = 0.0

    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model_egraph.train()
        optimizer.zero_grad()

        # Full-Graph Forward Pass on 11.85M edges via CUDA FP16
        with torch.amp.autocast('cuda', dtype=torch.float16):
            train_logits, _ = model_egraph(x_nodes, train_edge_feat, train_src_ids, train_dst_ids, num_nodes)
            loss = criterion(train_logits, y_train_multi)

        scaler_amp.scale(loss).backward()
        scaler_amp.step(optimizer)
        scaler_amp.update()

        # Validation on full 1.69M validation set
        model_egraph.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            val_logits, _ = model_egraph(x_nodes, val_edge_feat, val_src_ids, val_dst_ids, num_nodes)
            val_preds = val_logits.argmax(dim=-1).cpu().numpy()
            val_acc = accuracy_score(y_val_multi_np, val_preds)
            val_macro_f1 = f1_score(y_val_multi_np, val_preds, average="macro", zero_division=0)
            val_weighted_f1 = f1_score(y_val_multi_np, val_preds, average="weighted", zero_division=0)

        ep_duration = time.time() - t_ep
        vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        logger.info(f"E-GraphSAGE Epoch {epoch:02d}/{epochs:02d} | Train Loss: {loss.item():.4f} | Val Acc: {val_acc*100:.2f}% | Macro F1: {val_macro_f1:.4f} | Weighted F1: {val_weighted_f1:.4f} | Thời gian: {ep_duration:.2f}s | VRAM: {vram_mb:.0f} MB")

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_val_acc = val_acc
            best_val_weighted_f1 = val_weighted_f1
            torch.save(model_egraph.state_dict(), "/workspace/reseach1/output/egraphsage_best.pt")
            best_report = classification_report(y_val_multi_np, val_preds, target_names=class_names, digits=4, zero_division=0)
            logger.info("--> [CHECKPOINT] Đã lưu model E-GraphSAGE tốt nhất!")

        # Free per-epoch tensors to prevent memory accumulation
        del train_logits, val_logits, val_preds, loss
        torch.cuda.empty_cache()

    logger.info("\n=== BÁO CÁO PHÂN LỚP CHI TIẾT E-GRAPHSAGE (TẬP VALIDATION 1.69M DÒNG) ===")
    logger.info(f"\n{best_report}")

    del model_egraph, optimizer, scaler_amp, criterion
    torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # MODEL 2: Anomal-E (Unsupervised / DGI Contrastive)
    # -------------------------------------------------------------
    logger.info("=" * 75)
    logger.info("=== BƯỚC 2: HUẤN LUYỆN MÔ HÌNH ANOMAL-E (SELF-SUPERVISED DGI) ===")
    logger.info("=" * 75)

    model_anomale = FastAnomalE(
        node_dim=hidden_dim,
        edge_dim=len(FEATURE_COLS),
        hidden_dim=hidden_dim,
        edge_emb_dim=128
    ).to(device)

    dgi_optimizer = torch.optim.Adam(model_anomale.parameters(), lr=0.005, weight_decay=1e-4)
    dgi_scaler = torch.amp.GradScaler('cuda')
    dgi_epochs = 8
    last_w_summary = None

    for ep in range(1, dgi_epochs + 1):
        t_dgi = time.time()
        model_anomale.train()
        dgi_optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.float16):
            dgi_loss, w_summary, _ = model_anomale(x_nodes, train_edge_feat, train_src_ids, train_dst_ids, num_nodes)

        dgi_scaler.scale(dgi_loss).backward()
        dgi_scaler.step(dgi_optimizer)
        dgi_scaler.update()
        last_w_summary = w_summary.detach()

        ep_duration = time.time() - t_dgi
        vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        logger.info(f"Anomal-E DGI Epoch {ep:02d}/{dgi_epochs:02d} | Contrastive Loss: {dgi_loss.item():.4f} | Thời gian: {ep_duration:.2f}s | VRAM: {vram_mb:.0f} MB")

        del dgi_loss, w_summary
        torch.cuda.empty_cache()

    torch.save(model_anomale.state_dict(), "/workspace/reseach1/output/anomale_best.pt")
    logger.info("--> [CHECKPOINT] Đã lưu model Anomal-E thành công!")

    # Evaluate Anomal-E via Discriminator Anomaly Scoring on Val Set
    logger.info("[Đánh giá Anomal-E] Trích xuất Edge Embeddings và tính Anomaly Score...")
    model_anomale.eval()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        val_edge_embs, _ = model_anomale.get_edge_embeddings(
            x_nodes, val_edge_feat, val_src_ids, val_dst_ids, num_nodes
        )
        scores = torch.matmul(val_edge_embs, last_w_summary).cpu().numpy()
        anomaly_scores = -scores

    y_val_bin_np = y_val_bin.cpu().numpy()
    roc_auc = roc_auc_score(y_val_bin_np, anomaly_scores)
    logger.info(f"Anomal-E ROC-AUC (Không giám sát trên 1.69 triệu mẫu Val): {roc_auc:.4f}")

    benign_scores = anomaly_scores[y_val_bin_np == 0]
    threshold = np.percentile(benign_scores, 95)
    anomale_preds = (anomaly_scores > threshold).astype(int)

    anomale_acc = accuracy_score(y_val_bin_np, anomale_preds)
    anomale_f1 = f1_score(y_val_bin_np, anomale_preds, average="binary")
    anomale_macro_f1 = f1_score(y_val_bin_np, anomale_preds, average="macro")

    logger.info(f"Anomal-E Accuracy: {anomale_acc*100:.2f}% | Binary F1: {anomale_f1:.4f} | Macro F1: {anomale_macro_f1:.4f}")
    anomale_report = classification_report(y_val_bin_np, anomale_preds, target_names=["Benign", "Attack"], digits=4)
    logger.info(f"\n{anomale_report}")

    # Downstream: Multi-threaded Isolation Forest on CPU (32 vCPUs)
    logger.info("[Đánh giá Anomal-E + IsolationForest] Huấn luyện IsolationForest trên 32 vCPUs...")
    t_if = time.time()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        sample_benign_idx = torch.where(y_train_bin == 0)[0][:200000]
        train_benign_embs, _ = model_anomale.get_edge_embeddings(
            x_nodes, train_edge_feat[sample_benign_idx], train_src_ids[sample_benign_idx], train_dst_ids[sample_benign_idx], num_nodes
        )
        train_benign_np = train_benign_embs.cpu().numpy()

    iso_forest = IsolationForest(n_estimators=100, contamination=0.1, n_jobs=-1, random_state=42)
    iso_forest.fit(train_benign_np)
    logger.info(f"Isolation Forest huấn luyện xong trong {time.time()-t_if:.2f}s!")

    val_sample_np = val_edge_embs[:200000].cpu().numpy()
    val_sample_labels = y_val_bin_np[:200000]
    if_preds = iso_forest.predict(val_sample_np)
    if_preds_bin = np.where(if_preds == -1, 1, 0)

    if_acc = accuracy_score(val_sample_labels, if_preds_bin)
    if_f1 = f1_score(val_sample_labels, if_preds_bin, average="binary")
    logger.info(f"Isolation Forest Accuracy: {if_acc*100:.2f}% | Binary F1: {if_f1:.4f}")

    # -------------------------------------------------------------
    # Save Final Experiment Results
    # -------------------------------------------------------------
    results = {
        "dataset": "NF-ToN-IoT-v2 Full Dataset",
        "total_flows": n_train + n_val,
        "train_flows": n_train,
        "val_flows": n_val,
        "unique_graph_nodes": num_nodes,
        "hardware": {
            "gpu": gpu_name,
            "vram_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
            "cpu_cores": 32,
        },
        "egraphsage_multiclass": {
            "num_classes": num_classes,
            "classes": class_names,
            "best_val_accuracy": round(float(best_val_acc), 4),
            "best_val_macro_f1": round(float(best_val_macro_f1), 4),
            "best_val_weighted_f1": round(float(best_val_weighted_f1), 4),
        },
        "anomal_e_unsupervised": {
            "val_roc_auc": round(float(roc_auc), 4),
            "val_accuracy": round(float(anomale_acc), 4),
            "val_binary_f1": round(float(anomale_f1), 4),
            "val_macro_f1": round(float(anomale_macro_f1), 4),
            "isolation_forest_f1": round(float(if_f1), 4)
        }
    }
    with open("/workspace/reseach1/output/results_full_13m.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 75)
    logger.info("=== TOÀN BỘ QUÁ TRÌNH HUẤN LUYỆN ĐÃ HOÀN TẤT THÀNH CÔNG ===")
    logger.info(f"Kết quả lưu tại: /workspace/reseach1/output/results_full_13m.json")
    logger.info("=" * 75)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Đã xảy ra ngoại lệ: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
