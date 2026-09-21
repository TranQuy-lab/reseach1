import time
from pathlib import Path
import duckdb

sample_dir = Path('/workspace/gnn_data/sample_500k')
sample_dir.mkdir(parents=True, exist_ok=True)
con = duckdb.connect()

def make_stratified_split(source_file, target_file, target_total):
    t0 = time.time()
    n_ddos = int(target_total * 0.485)
    n_dos  = int(target_total * 0.441)
    n_rec  = int(target_total * 0.069)
    n_ben  = max(int(target_total * 0.004), 100)
    n_theft = max(int(target_total * 0.001), 50)
    
    query = f"""
    COPY (
        (SELECT * FROM '{source_file}' WHERE Attack = 'DDoS' LIMIT {n_ddos})
        UNION ALL
        (SELECT * FROM '{source_file}' WHERE Attack = 'DoS' LIMIT {n_dos})
        UNION ALL
        (SELECT * FROM '{source_file}' WHERE Attack = 'Reconnaissance' LIMIT {n_rec})
        UNION ALL
        (SELECT * FROM '{source_file}' WHERE Attack = 'Benign' LIMIT {n_ben})
        UNION ALL
        (SELECT * FROM '{source_file}' WHERE Attack = 'Theft' LIMIT {n_theft})
    ) TO '{target_file}' (FORMAT PARQUET, COMPRESSION 'SNAPPY');
    """
    con.execute(query)
    print(f"Tạo {target_file.name} ({target_total:,} mẫu) trong {time.time() - t0:.2f}s")

print("=== TRÍCH XUẤT CHUẨN PHÂN TẦNG 500K MẪU CHO CẢ 5 LỚP ===")
make_stratified_split('/workspace/gnn_data/e_graphsage/splits/bot_iot_train.parquet', sample_dir / 'train.parquet', 350000)
make_stratified_split('/workspace/gnn_data/e_graphsage/splits/bot_iot_val.parquet', sample_dir / 'val.parquet', 50000)
make_stratified_split('/workspace/gnn_data/e_graphsage/splits/bot_iot_test.parquet', sample_dir / 'test.parquet', 100000)

print("\nPhân bố nhãn thực tế sau khi trích xuất:")
for name in ['train', 'val', 'test']:
    target_path = str(sample_dir / f"{name}.parquet")
    print(f"-- Tập {name.upper()}:")
    rows = con.execute(f"SELECT Attack, Label, count(*) FROM '{target_path}' GROUP BY Attack, Label ORDER BY count(*) DESC").fetchall()
    for r in rows:
        print(f"   {r[0]:<15} (Label={r[1]}): {r[2]:,}")
