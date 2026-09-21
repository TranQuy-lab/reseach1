"""Independent on-disk audit, using DuckDB with bounded engine memory."""
from pathlib import Path
import json,time
import duckdb
root=Path(__file__).resolve().parents[1]
out=root/'data/processed_four'
report=json.loads((root/'research/results/preprocessing_four.json').read_text())
con=duckdb.connect()
con.execute("SET memory_limit='1GB'");con.execute('SET threads=2')
temp=out/'audit_tmp';temp.mkdir(exist_ok=True)
con.execute("SET temp_directory='"+str(temp).replace("'","''")+"'")
con.execute("SET max_temp_directory_size='12GB'")
results={}
for d,entry in report['outputs'].items():
 t=time.monotonic();p=out/(d+'.parquet')
 con.from_parquet(str(p)).create_view('data',replace=True)
 row=con.execute('''SELECT count(*), count(DISTINCT Dataset), min(Dataset), min(L4_SRC_PORT),max(L4_SRC_PORT),min(L4_DST_PORT),max(L4_DST_PORT),count(*) FILTER (WHERE L4_SRC_PORT<0 OR L4_SRC_PORT>65535 OR L4_DST_PORT<0 OR L4_DST_PORT>65535),count(*) FILTER (WHERE Label NOT IN (0,1) OR Label != CAST(Attack!='Benign' AS INTEGER)),count(*) FILTER (WHERE IPV4_SRC_ADDR IS NULL OR IPV4_DST_ADDR IS NULL),count(DISTINCT source_row_id) FROM data''').fetchone()
 assert row[0]==entry['rows'] and row[1]==1 and row[2]==d and row[7:10]==(0,0,0) and row[10]==row[0]
 grouped=con.execute('''SELECT count(*),coalesce(sum(n-1),0),coalesce(sum(CASE WHEN classes>1 THEN 1 ELSE 0 END),0),coalesce(sum(CASE WHEN classes>1 THEN n ELSE 0 END),0) FROM (SELECT flow_group_id,count(*) n,CASE WHEN min(Attack)!=max(Attack) THEN 2 ELSE 1 END classes FROM data GROUP BY flow_group_id HAVING count(*)>1)''').fetchone()
 classes=dict(con.execute('SELECT Attack,count(*) FROM data GROUP BY Attack').fetchall())
 assert classes==report['valid_classes'][d]
 results[d]={'rows':row[0],'unique_source_row_ids':row[10],'src_port_range':[row[3],row[4]],'dst_port_range':[row[5],row[6]],'label_errors':row[8],'repeated_predictor_groups':grouped[0],'excess_rows_in_repeated_predictor_groups':grouped[1],'conflicting_label_predictor_groups':grouped[2],'rows_in_conflicting_label_predictor_groups':grouped[3],'classes':classes,'elapsed_s':round(time.monotonic()-t,2)}
 print(json.dumps({d:results[d]}),flush=True)
assert report['raw_rows']==report['valid_rows']+report['rejected_rows']
assert sum(v['rows'] for v in results.values())==report['valid_rows']
(root/'research/results/preprocessing_four_verification.json').write_text(json.dumps({'passed':True,'datasets':results,'note':'Repeated predictor grouping uses 128-bit fingerprints, not an exact all-column deduplication proof.'},indent=2)+'\n')
con.close()
if temp.exists() and not any(temp.iterdir()):temp.rmdir()
