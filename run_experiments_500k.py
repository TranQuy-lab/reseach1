import time
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.ensemble import IsolationForest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("exp_500k")

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
        # x: (N, D_node), edge_attr: (E, D_edge), src: (E,), dst: (E,)
        # Step 1: Compute messages from source node and edge feature
        msg = self.w_msg(torch.cat([x[src], edge_attr], dim=-1))  # (E, out_dim)

        # Step 2: Mean aggregation to destination nodes via CUDA scatter_add
        out_dim = msg.size(-1)
        aggr = torch.zeros(num_nodes, out_dim, device=x.device, dtype=msg.dtype)
        aggr.scatter_add_(0, dst.unsqueeze(1).expand(-1, out_dim), msg)

        deg = torch.zeros(num_nodes, 1, device=x.device, dtype=msg.dtype)
        deg.scatter_add_(0, dst.unsqueeze(1), torch.ones_like(dst.unsqueeze(1), dtype=msg.dtype))
        aggr = aggr / deg.clamp(min=1.0)

        # Step 3: Update node representation
        h_new = F.relu(self.w_apply(torch.cat([x, aggr], dim=-1)))
        return h_new


class FastEGraphSAGE(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.layer1 = FastSAGELayer(node_dim, edge_dim, hidden_dim)
        self.layer2 = FastSAGELayer(hidden_dim, edge_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # Edge classifier: maps concatenated endpoints (u, v) to edge label classes
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, num_nodes: int):
        h = self.layer1(x, edge_attr, src, dst, num_nodes)
        h = self.dropout(h)
        h = self.layer2(h, edge_attr, src, dst, num_nodes)
        # Edge score
        edge_pred = self.edge_mlp(torch.cat([h[src], h[dst]], dim=-1))
        return edge_pred, h


# -------------------------------------------------------------
# 2. Native High-Performance Anomal-E (DGI + Edge Embeddings)
# -------------------------------------------------------------
class FastAnomalE(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 128, edge_emb_dim: int = 256):
        super().__init__()
        self.sage_layer = FastSAGELayer(node_dim, edge_dim, hidden_dim)
        self.w_edge = nn.Linear(hidden_dim * 2, edge_emb_dim)
        self.disc_weight = nn.Parameter(torch.Tensor(edge_emb_dim, edge_emb_dim))
        nn.init.xavier_uniform_(self.disc_weight)

    def get_edge_embeddings(self, x: torch.Tensor, edge_attr: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, num_nodes: int):
        h = self.sage_layer(x, edge_attr, src, dst, num_nodes)
        edge_emb = self.w_edge(torch.cat([h[src], h[dst]], dim=-1))
        return edge_emb

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, num_nodes: int):
        # Positive embeddings
        h_pos = self.sage_layer(x, edge_attr, src, dst, num_nodes)
        pos_edge_emb = self.w_edge(torch.cat([h_pos[src], h_pos[dst]], dim=-1))

        # Negative embeddings (corrupted edge features)
        e_perm = torch.randperm(edge_attr.size(0), device=edge_attr.device)
        edge_attr_neg = edge_attr[e_perm]
        h_neg = self.sage_layer(x, edge_attr_neg, src, dst, num_nodes)
        neg_edge_emb = self.w_edge(torch.cat([h_neg[src], h_neg[dst]], dim=-1))

        # Summary vector
        summary = torch.sigmoid(pos_edge_emb.mean(dim=0))  # (edge_emb_dim,)

        # Discriminator bilinear score
        # pos_score = pos_edge_emb @ (disc_weight @ summary)
        w_summary = torch.matmul(self.disc_weight, summary)  # (edge_emb_dim,)
        pos_scores = torch.matmul(pos_edge_emb, w_summary)
        neg_scores = torch.matmul(neg_edge_emb, w_summary)

        loss_pos = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
        loss_neg = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
        return loss_pos + loss_neg, pos_edge_emb


# -------------------------------------------------------------
# 3. Main Experiment Execution
# -------------------------------------------------------------
def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load 500k sample data from Parquet
    sample_dir = "/workspace/gnn_data/sample_500k"
    logger.info("Loading Train, Val, Test Parquet files...")
    t_load = time.time()
    df_train = pd.read_parquet(f"{sample_dir}/train.parquet")
    df_val   = pd.read_parquet(f"{sample_dir}/val.parquet")
    df_test  = pd.read_parquet(f"{sample_dir}/test.parquet")
    logger.info(f"Loaded: Train={len(df_train):,}, Val={len(df_val):,}, Test={len(df_test):,} in {time.time() - t_load:.2f}s")

    # Build Global Node Mapping (IP:Port -> integer ID 0..N-1)
    all_nodes = np.unique(np.concatenate([
        df_train["src_node"].values, df_train["dst_node"].values,
        df_val["src_node"].values, df_val["dst_node"].values,
        df_test["src_node"].values, df_test["dst_node"].values,
    ]))
    node_to_id = {node: i for i, node in enumerate(all_nodes)}
    num_nodes = len(node_to_id)
    logger.info(f"Total Unique Nodes (IP:Port pairs) in Graph: {num_nodes:,}")

    # Standardize Edge Features using Train Scaler
    scaler = StandardScaler()
    X_train_raw = df_train[FEATURE_COLS].values.astype(np.float32)
    X_val_raw   = df_val[FEATURE_COLS].values.astype(np.float32)
    X_test_raw  = df_test[FEATURE_COLS].values.astype(np.float32)

    X_train_feat = torch.from_numpy(scaler.fit_transform(X_train_raw)).to(device)
    X_val_feat   = torch.from_numpy(scaler.transform(X_val_raw)).to(device)
    X_test_feat  = torch.from_numpy(scaler.transform(X_test_raw)).to(device)

    # Encode Labels
    # 1. Binary: Label (0=Benign, 1=Attack)
    # 2. Multiclass: Attack
    le = LabelEncoder()
    y_train_multi = torch.tensor(le.fit_transform(df_train["Attack"].values), dtype=torch.long, device=device)
    y_val_multi   = torch.tensor(le.transform(df_val["Attack"].values), dtype=torch.long, device=device)
    y_test_multi  = torch.tensor(le.transform(df_test["Attack"].values), dtype=torch.long, device=device)
    class_names = list(le.classes_)
    num_classes = len(class_names)
    logger.info(f"Classes ({num_classes}): {class_names}")

    y_train_bin = torch.tensor(df_train["Label"].values, dtype=torch.long, device=device)
    y_val_bin   = torch.tensor(df_val["Label"].values, dtype=torch.long, device=device)
    y_test_bin  = torch.tensor(df_test["Label"].values, dtype=torch.long, device=device)

    # Convert Edges to Tensor IDs
    train_src = torch.tensor(df_train["src_node"].map(node_to_id).values, dtype=torch.long, device=device)
    train_dst = torch.tensor(df_train["dst_node"].map(node_to_id).values, dtype=torch.long, device=device)
    val_src   = torch.tensor(df_val["src_node"].map(node_to_id).values, dtype=torch.long, device=device)
    val_dst   = torch.tensor(df_val["dst_node"].map(node_to_id).values, dtype=torch.long, device=device)
    test_src  = torch.tensor(df_test["src_node"].map(node_to_id).values, dtype=torch.long, device=device)
    test_dst  = torch.tensor(df_test["dst_node"].map(node_to_id).values, dtype=torch.long, device=device)

    # Initial Node Features: ones vector as in E-GraphSAGE paper
    x_nodes = torch.ones(num_nodes, len(FEATURE_COLS), device=device, dtype=torch.float32)

    # Compute Class Weights for balanced loss
    class_counts = np.bincount(y_train_multi.cpu().numpy(), minlength=num_classes)
    total_samples = len(y_train_multi)
    weights = total_samples / (num_classes * np.maximum(class_counts, 1).astype(np.float32))
    weights = torch.tensor(weights, dtype=torch.float32, device=device)

    # ---------------------------------------------------------
    # EXPERIMENT 1: E-GraphSAGE Multi-Class Training
    # ---------------------------------------------------------
    logger.info("=" * 60)
    logger.info("=== THỬ NGHIỆM 1: E-GRAPHSAGE (MULTICLASS EDGE CLASSIFICATION) ===")
    logger.info("=" * 60)

    model = FastEGraphSAGE(
        node_dim=len(FEATURE_COLS),
        edge_dim=len(FEATURE_COLS),
        hidden_dim=128,
        num_classes=num_classes,
        dropout=0.2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)
    scaler_amp = torch.amp.GradScaler('cuda')

    best_val_f1 = 0.0
    best_test_report = None
    epochs = 15

    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        optimizer.zero_grad()

        # Mixed Precision Forward Pass (Tensor Cores on RTX 3090)
        with torch.amp.autocast('cuda'):
            train_logits, _ = model(x_nodes, X_train_feat, train_src, train_dst, num_nodes)
            loss = criterion(train_logits, y_train_multi)

        scaler_amp.scale(loss).backward()
        scaler_amp.step(optimizer)
        scaler_amp.update()

        # Validation
        model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda'):
            val_logits, _ = model(x_nodes, X_val_feat, val_src, val_dst, num_nodes)
            val_preds = val_logits.argmax(dim=-1).cpu().numpy()
            y_val_np = y_val_multi.cpu().numpy()
            val_acc = accuracy_score(y_val_np, val_preds)
            val_f1 = f1_score(y_val_np, val_preds, average="macro", zero_division=0)

        ep_time = time.time() - t_ep
        vram_used = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

        logger.info(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {loss.item():.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_f1:.4f} | Time: {ep_time:.2f}s | VRAM: {vram_used:.0f} MB")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            with torch.no_grad(), torch.amp.autocast('cuda'):
                test_logits, _ = model(x_nodes, X_test_feat, test_src, test_dst, num_nodes)
                test_preds = test_logits.argmax(dim=-1).cpu().numpy()
                y_test_np = y_test_multi.cpu().numpy()
                best_test_report = classification_report(y_test_np, test_preds, target_names=class_names, digits=4, zero_division=0)
                best_test_acc = accuracy_score(y_test_np, test_preds)
                best_test_f1 = f1_score(y_test_np, test_preds, average="macro", zero_division=0)

    logger.info("\n=== KẾT QUẢ KIỂM THỬ E-GRAPHSAGE TRÊN TEST SET (100.000 MẪU) ===")
    logger.info(f"Accuracy: {best_test_acc*100:.2f}% | Macro F1-Score: {best_test_f1:.4f}")
    logger.info(f"\n{best_test_report}")

    # ---------------------------------------------------------
    # EXPERIMENT 2: Anomal-E (Self-Supervised Anomaly Detection)
    # ---------------------------------------------------------
    logger.info("=" * 60)
    logger.info("=== THỬ NGHIỆM 2: ANOMAL-E (UNSUPERVISED / SELF-SUPERVISED DGI) ===")
    logger.info("=" * 60)

    anomal_model = FastAnomalE(
        node_dim=len(FEATURE_COLS),
        edge_dim=len(FEATURE_COLS),
        hidden_dim=128,
        edge_emb_dim=256
    ).to(device)

    dgi_opt = torch.optim.Adam(anomal_model.parameters(), lr=0.002, weight_decay=1e-4)
    dgi_epochs = 10

    for ep in range(1, dgi_epochs + 1):
        t_dgi = time.time()
        anomal_model.train()
        dgi_opt.zero_grad()

        with torch.amp.autocast('cuda'):
            loss_dgi, _ = anomal_model(x_nodes, X_train_feat, train_src, train_dst, num_nodes)

        scaler_amp.scale(loss_dgi).backward()
        scaler_amp.step(dgi_opt)
        scaler_amp.update()

        ep_time = time.time() - t_dgi
        logger.info(f"DGI Epoch {ep:02d}/{dgi_epochs:02d} | Loss: {loss_dgi.item():.4f} | Time: {ep_time:.2f}s")

    # Extract 256-D Edge Embeddings
    logger.info("Trích xuất Embeddings 256 chiều cho cạnh trên GPU...")
    anomal_model.eval()
    with torch.no_grad(), torch.amp.autocast('cuda'):
        train_edge_embs = anomal_model.get_edge_embeddings(x_nodes, X_train_feat, train_src, train_dst, num_nodes).cpu().numpy()
        test_edge_embs  = anomal_model.get_edge_embeddings(x_nodes, X_test_feat, test_src, test_dst, num_nodes).cpu().numpy()

    logger.info(f"Train embeddings shape: {train_edge_embs.shape}, Test embeddings shape: {test_edge_embs.shape}")

    # Train IsolationForest on CPU using all 32 vCPUs
    logger.info("Huấn luyện IsolationForest đa luồng (32 vCPUs, n_jobs=-1) trên Embeddings...")
    t_if = time.time()
    # Train on Benign samples (Unsupervised Anomaly Detection baseline)
    benign_mask = (y_train_bin.cpu().numpy() == 0)
    train_benign_embs = train_edge_embs[benign_mask]

    clf = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(train_benign_embs)
    logger.info(f"IsolationForest Fit hoàn tất sau: {time.time() - t_if:.2f}s")

    # Predict Anomaly on Test Set (1 = Normal, -1 = Anomaly -> map to 0=Benign, 1=Attack)
    raw_preds = clf.predict(test_edge_embs)
    test_preds_bin = np.where(raw_preds == -1, 1, 0)
    y_test_bin_np = y_test_bin.cpu().numpy()

    anomal_acc = accuracy_score(y_test_bin_np, test_preds_bin)
    anomal_f1_macro = f1_score(y_test_bin_np, test_preds_bin, average="macro")
    anomal_f1_binary = f1_score(y_test_bin_np, test_preds_bin, average="binary")

    logger.info("\n=== KẾT QUẢ ANOMAL-E (PHÁT HIỆN BẤT THƯỜNG TRÊN 100.000 MẪU TEST) ===")
    logger.info(f"Accuracy: {anomal_acc*100:.2f}%")
    logger.info(f"Binary F1-Score: {anomal_f1_binary:.4f}")
    logger.info(f"Macro F1-Score: {anomal_f1_macro:.4f}")
    logger.info(f"\n{classification_report(y_test_bin_np, test_preds_bin, target_names=['Benign', 'Attack'], digits=4)}")

    # Save summary
    results = {
        "dataset": "NF-BoT-IoT-v2 (500k Sample)",
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "vram_peak_mb": round(vram_used, 2),
            "cpu_cores": 32,
        },
        "e_graphsage": {
            "best_test_acc": round(float(best_test_acc), 4),
            "best_test_f1_macro": round(float(best_test_f1), 4),
            "training_time_per_epoch_s": round(ep_time, 2)
        },
        "anomal_e": {
            "test_acc": round(float(anomal_acc), 4),
            "test_f1_binary": round(float(anomal_f1_binary), 4),
            "test_f1_macro": round(float(anomal_f1_macro), 4),
        }
    }
    with open("/workspace/gnn_data/experiment_results_500k.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results to /workspace/gnn_data/experiment_results_500k.json")

if __name__ == "__main__":
    run()
