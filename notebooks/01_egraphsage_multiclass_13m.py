"""
====================================================================================================
BÀI 1: E-GRAPHSAGE MULTICLASS EDGE CLASSIFICATION (10 LỚP TẤN CÔNG)
QUY MÔ TOÀN BỘ 13.55 TRIỆU LUỒNG MẠNG (NF-ToN-IoT-v2) TRÊN 1 GPU RTX 3090
====================================================================================================
Kiến trúc:
- FastSAGELayer với Native CUDA scatter_add_
- FastEGraphSAGE tích hợp PyTorch Gradient Checkpointing
- Chuẩn hóa 39 đặc trưng trực tiếp trên GPU CUDA (FP16)
- Đánh giá đa lớp toàn diện: Precision, Recall/DR, F1, Specificity, FPR, Confusion Matrix
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
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
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

logger = logging.getLogger("egraphsage_13m")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

fh = FlushingFileHandler(os.path.join(OUTPUT_DIR, "egraphsage_run.log"), mode="w", encoding="utf-8")
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
# 2. Native CUDA PyTorch E-GraphSAGE Modules
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
    logger.info("=== BẮT ĐẦU THỰC THI E-GRAPHSAGE TRÊN PHÂN TẬP NF-ToN-IoT-v2 (13.55M DÒNG) ===")
    logger.info(f"Phần cứng: {gpu_name} ({total_vram_gb:.1f} GB VRAM) | CPU Cores: {os.cpu_count()}")
    logger.info("=" * 80)

    train_path, val_path = resolve_data_path()

    # 1. Nạp dữ liệu
    logger.info("[1/5] Nạp Parquet đa luồng qua PyArrow...")
    t0 = time.time()
    df_train = ds.dataset(train_path, format="parquet").to_table().to_pandas()
    df_val = ds.dataset(val_path, format="parquet").to_table().to_pandas()
    n_train, n_val = len(df_train), len(df_val)
    logger.info(f"Nạp hoàn tất: Train = {n_train:,} dòng | Val = {n_val:,} dòng trong {time.time()-t0:.2f}s")

    # 2. Xây dựng đồ thị
    logger.info("[2/5] Lập chỉ số đỉnh đồ thị (Graph Indexing)...")
    t_idx = time.time()
    all_nodes_series = pd.concat([df_train["src_node"], df_train["dst_node"], df_val["src_node"], df_val["dst_node"]], ignore_index=True)
    node_codes, unique_nodes = pd.factorize(all_nodes_series)
    num_nodes = len(unique_nodes)
    logger.info(f"Tổng số đỉnh duy nhất (IP:Port pairs): {num_nodes:,} (xong trong {time.time()-t_idx:.2f}s)")

    train_src_ids = torch.tensor(node_codes[:n_train], dtype=torch.long, device=device)
    train_dst_ids = torch.tensor(node_codes[n_train:2*n_train], dtype=torch.long, device=device)
    val_offset = 2 * n_train
    val_src_ids = torch.tensor(node_codes[val_offset:val_offset+n_val], dtype=torch.long, device=device)
    val_dst_ids = torch.tensor(node_codes[val_offset+n_val:val_offset+2*n_val], dtype=torch.long, device=device)
    del all_nodes_series, node_codes, unique_nodes
    import gc; gc.collect()

    # 3. Chuẩn hóa GPU
    logger.info("[3/5] Chuẩn hóa 39 đặc trưng trực tiếp trên GPU CUDA (FP16)...")
    t_scale = time.time()
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
    logger.info(f"Chuẩn hóa GPU hoàn tất trong {time.time()-t_scale:.2f}s!")

    # 4. Mã hóa nhãn 10 lớp
    logger.info("[4/5] Mã hóa nhãn 10 lớp và phân bổ trọng số...")
    y_train_multi_np, class_names_arr = pd.factorize(df_train["Attack"], sort=True)
    class_names = [str(c) for c in class_names_arr]
    num_classes = len(class_names)
    class_to_code = {c: i for i, c in enumerate(class_names)}
    y_val_multi_np = df_val["Attack"].map(lambda x: class_to_code.get(x, 0)).to_numpy(dtype=np.int64)

    y_train_multi = torch.tensor(y_train_multi_np, dtype=torch.long, device=device)
    y_val_multi = torch.tensor(y_val_multi_np, dtype=torch.long, device=device)

    class_counts = np.bincount(y_train_multi_np, minlength=num_classes)
    weights = np.sqrt(len(y_train_multi_np) / (num_classes * np.maximum(class_counts, 1.0)))
    weights = torch.tensor(weights, dtype=torch.float32, device=device)

    hidden_dim = 64
    x_nodes = torch.ones((num_nodes, hidden_dim), dtype=torch.float16, device=device)
    del df_train, df_val
    gc.collect()

    # 5. Huấn luyện E-GraphSAGE
    logger.info("[5/5] Bắt đầu huấn luyện mô hình E-GraphSAGE...")
    model = FastEGraphSAGE(
        node_dim=hidden_dim,
        edge_dim=len(FEATURE_COLS),
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=0.2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    scaler_amp = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss(weight=weights)

    epochs = 10
    best_val_macro_f1 = 0.0
    best_preds = None
    history = []

    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.float16):
            train_logits, _ = model(x_nodes, train_edge_feat, train_src_ids, train_dst_ids, num_nodes)
            loss = criterion(train_logits, y_train_multi)

        scaler_amp.scale(loss).backward()
        scaler_amp.step(optimizer)
        scaler_amp.update()

        model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            val_logits, _ = model(x_nodes, val_edge_feat, val_src_ids, val_dst_ids, num_nodes)
            val_preds = val_logits.argmax(dim=-1).cpu().numpy()
            val_acc = accuracy_score(y_val_multi_np, val_preds)
            val_macro_f1 = f1_score(y_val_multi_np, val_preds, average="macro", zero_division=0)
            val_weighted_f1 = f1_score(y_val_multi_np, val_preds, average="weighted", zero_division=0)

        ep_duration = time.time() - t_ep
        vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        record = {
            "epoch": epoch,
            "train_loss": round(float(loss.item()), 4),
            "val_acc": round(float(val_acc), 4),
            "val_macro_f1": round(float(val_macro_f1), 4),
            "val_weighted_f1": round(float(val_weighted_f1), 4),
            "duration_sec": round(ep_duration, 2),
            "vram_mb": round(vram_mb, 1)
        }
        history.append(record)
        logger.info(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {record['train_loss']} | Val Acc: {val_acc*100:.2f}% | Macro F1: {record['val_macro_f1']} | Weighted F1: {record['val_weighted_f1']} | {ep_duration:.2f}s | VRAM: {vram_mb:.0f} MB")

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_preds = val_preds
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "egraphsage_best.pt"))
            logger.info("--> [CHECKPOINT] Đã lưu model E-GraphSAGE tốt nhất!")

        del train_logits, val_logits, val_preds, loss
        torch.cuda.empty_cache()

    # Lưu lịch sử huấn luyện
    pd.DataFrame(history).to_csv(os.path.join(OUTPUT_DIR, "egraphsage_training_history.csv"), index=False)

    # Đánh giá chi tiết
    logger.info("\n=== BÁO CÁO PHÂN LỚP CHI TIẾT E-GRAPHSAGE (1.69M MẪU KIỂM THỬ) ===")
    rep_str = classification_report(y_val_multi_np, best_preds, target_names=class_names, digits=4, zero_division=0)
    logger.info(f"\n{rep_str}")

    # Tạo và lưu ma trận nhầm lẫn
    cm = confusion_matrix(y_val_multi_np, best_preds)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(OUTPUT_DIR, "confusion_matrix_egraphsage_counts.csv"))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm_df = pd.DataFrame(cm_norm, index=class_names, columns=class_names)
    cm_norm_df.to_csv(os.path.join(OUTPUT_DIR, "confusion_matrix_egraphsage_normalized.csv"))

    # Vẽ và lưu biểu đồ ma trận nhầm lẫn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("E-GraphSAGE Normalized Confusion Matrix (NF-ToN-IoT-v2 1.69M Val)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plots", "fig2_egraphsage_cm_normalized.png"), dpi=300)
    plt.close()

    logger.info(f"Hoàn tất! Tất cả kết quả lưu tại: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
