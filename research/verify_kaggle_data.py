"""Read-only, bounded-memory inventory of the downloaded merged CSV."""
import json
from pathlib import Path
import duckdb
root=Path(__file__).resolve().parents[1]
p=root/'data/raw/NF-UQ-NIDS-v2/NF-UQ-NIDS-v2.csv'
manifest=root/'research/results/kaggle_download_manifest.json'
assert p.exists() and manifest.exists(), 'Download and extraction must finish first'
con=duckdb.connect()
con.execute("SET memory_limit='512MB'")
con.execute('SET threads=2')
rows=con.execute('SELECT Dataset, count(*) FROM read_csv(?, header=true, all_varchar=true) GROUP BY Dataset ORDER BY Dataset',[str(p)]).fetchall()
counts=dict(rows)
expected={'NF-UNSW-NB15-v2':2390275,'NF-BoT-IoT-v2':37763497,'NF-ToN-IoT-v2':16940496,'NF-CSE-CIC-IDS2018-v2':18893708}
result={'counts':counts,'total_rows':sum(counts.values()),'expected_source':'https://www.cyber.uq.edu.au/node/824','expected_counts':expected,'matches_expected':counts==expected,'scope':'CSV parsing and dataset inventory only; not preprocessing or label-quality audit'}
(root/'research/results/kaggle_data_inventory.json').write_text(json.dumps(result,indent=2)+'\n')
data=json.loads(manifest.read_text()); data['dataset_counts_status']='verified' if counts==expected else 'mismatch'; data['inventory']='kaggle_data_inventory.json'; manifest.write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(result,indent=2))
if counts!=expected: raise SystemExit('Dataset counts differ; inspect before preprocessing')
