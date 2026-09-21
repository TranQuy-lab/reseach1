"""Streaming raw CSV validation and loss-conscious four-dataset export.
Retain repeated valid observations; group fingerprints exclude labels and IDs.
"""
from pathlib import Path
from collections import Counter, defaultdict
from functools import lru_cache
import argparse, hashlib, ipaddress, json, time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

DATASETS=['NF-UNSW-NB15-v2','NF-BoT-IoT-v2','NF-ToN-IoT-v2','NF-CSE-CIC-IDS2018-v2']
TEXT=['IPV4_SRC_ADDR','IPV4_DST_ADDR','Attack','Dataset']
FLOAT=['L7_PROTO','SRC_TO_DST_SECOND_BYTES','DST_TO_SRC_SECOND_BYTES','FTP_COMMAND_RET_CODE']
PORT=['L4_SRC_PORT','L4_DST_PORT']
@lru_cache(maxsize=131072)
def valid_ip(s):
 try: return isinstance(ipaddress.ip_address(s),ipaddress.IPv4Address)
 except (ValueError,TypeError): return False

def transform(frame, start):
 df=frame.copy()
 reasons={}
 numeric=[c for c in df if c not in TEXT]
 reasons['missing']=df.isna().any(axis=1).to_numpy()
 reasons['nonfinite']=~np.isfinite(df[numeric].to_numpy(dtype=np.float64,na_value=np.nan)).all(axis=1)
 reasons['port_invalid']=np.logical_or.reduce([(~df[c].between(0,65535) | (df[c]%1!=0)).fillna(True).to_numpy(dtype=bool) for c in PORT])
 reasons['ip_invalid']=np.logical_or.reduce([~df[c].map(valid_ip).to_numpy(dtype=bool) for c in TEXT[:2]])
 reasons['dataset_invalid']=(~df.Dataset.isin(DATASETS)).to_numpy()
 reasons['attack_empty']=(df.Attack.fillna('').str.strip()=='').to_numpy()
 reasons['label_invalid']=(~df.Label.isin([0,1]) | (df.Label != (df.Attack!='Benign').astype(int))).fillna(True).to_numpy(dtype=bool)
 bad=np.logical_or.reduce(list(reasons.values()))
 ids=np.arange(start,start+len(df),dtype=np.int64)
 df['source_row_id']=ids
 clean=df.loc[~bad].copy()
 for c in numeric:
  clean[c]=clean[c].astype('float64' if c in FLOAT else 'int64')
 for c in TEXT: clean[c]=clean[c].astype('str')
 # Two separately keyed 64-bit fingerprints; practical grouping, not proof of no collisions.
 keycols=[c for c in frame.columns if c not in ['Label','Attack']]
 h1=pd.util.hash_pandas_object(clean[keycols],index=False,hash_key='nids-group-key01').to_numpy(dtype='uint64')
 h2=pd.util.hash_pandas_object(clean[keycols],index=False,hash_key='nids-group-key02').to_numpy(dtype='uint64')
 clean['flow_group_id']=[f'{a:016x}{b:016x}' for a,b in zip(h1,h2)]
 rejected=df.loc[bad].copy()
 rejected['rejection_reasons']=['|'.join(k for k,v in reasons.items() if v[i]) for i in np.flatnonzero(bad)]
 return clean,rejected,{k:int(v.sum()) for k,v in reasons.items()}

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024**2):h.update(b)
 return h.hexdigest()

def run(source,out,report):
 if out.exists():raise ValueError('Output exists; refusing overwrite')
 out.mkdir(parents=True)
 header=source.open().readline().strip().split(',')
 if len(header)!=46 or not set(TEXT+FLOAT+PORT+['Label'])<=set(header):raise ValueError('Unexpected CSV schema')
 types={c:pa.string() if c in TEXT else pa.float64() if c in FLOAT else pa.int64() for c in header}
 reader=pcsv.open_csv(source,read_options=pcsv.ReadOptions(block_size=16*1024**2,use_threads=False),convert_options=pcsv.ConvertOptions(column_types=types,strings_can_be_null=True))
 schema=pa.schema([(c,types[c]) for c in header]+[('source_row_id',pa.int64()),('flow_group_id',pa.string())])
 writers={d:pq.ParquetWriter(out/(d+'.parquet.partial'),schema,compression='zstd') for d in DATASETS}
 reject_writer=None
 stats={'policy':'retain valid repeated flows; no imputation, scaling, balancing or feature clipping; keep IP and ports','grouping':'two keyed pandas uint64 fingerprints of all source predictors and Dataset, excluding Label/Attack/source_row_id; theoretical collisions possible','raw_rows':0,'valid_rows':0,'rejected_rows':0,'reasons_overlapping':Counter(),'raw_by_dataset':Counter(),'valid_by_dataset':Counter(),'raw_classes':defaultdict(Counter),'valid_classes':defaultdict(Counter),'outputs':{}}
 t=time.monotonic(); last=t
 try:
  for batch in reader:
   frame=batch.to_pandas(types_mapper=lambda t: pd.Int64Dtype() if pa.types.is_int64(t) else None); start=stats['raw_rows']+1
   for (d,a),n in frame.groupby(['Dataset','Attack'],dropna=False).size().items():stats['raw_classes'][str(d)][str(a)]+=int(n)
   stats['raw_by_dataset'].update(frame.Dataset.value_counts().to_dict())
   clean,rejected,reasons=transform(frame,start)
   stats['raw_rows']+=len(frame);stats['valid_rows']+=len(clean);stats['rejected_rows']+=len(rejected);stats['reasons_overlapping'].update(reasons)
   for d,sub in clean.groupby('Dataset',sort=False):
    writers[d].write_table(pa.Table.from_pandas(sub,schema=schema,preserve_index=False));stats['valid_by_dataset'][d]+=len(sub)
    stats['valid_classes'][d].update(sub.Attack.value_counts().to_dict())
   if len(rejected):
    tab=pa.Table.from_pandas(rejected,preserve_index=False)
    if reject_writer is None:reject_writer=pq.ParquetWriter(out/'rejected_rows.parquet.partial',tab.schema,compression='zstd')
    reject_writer.write_table(tab)
   if time.monotonic()-last>25:
    print(json.dumps({'rows':stats['raw_rows'],'rejected':stats['rejected_rows'],'elapsed_s':round(time.monotonic()-t)}),flush=True);last=time.monotonic()
 finally:
  for w in writers.values():w.close()
  if reject_writer:reject_writer.close()
 for d in DATASETS:
  part=out/(d+'.parquet.partial');final=out/(d+'.parquet');part.rename(final)
  stats['outputs'][d]={'rows':pq.ParquetFile(final).metadata.num_rows,'bytes':final.stat().st_size,'sha256':digest(final)}
 if reject_writer:(out/'rejected_rows.parquet.partial').rename(out/'rejected_rows.parquet')
 stats['elapsed_s']=round(time.monotonic()-t,2)
 stats['source_sha256']=digest(source);stats['script_sha256']=digest(Path(__file__))
 stats['versions']={'pandas':pd.__version__,'pyarrow':pa.__version__,'numpy':np.__version__}
 report.write_text(json.dumps(stats,indent=2)+'\n'); print(json.dumps({'complete':True,'raw':stats['raw_rows'],'valid':stats['valid_rows'],'rejected':stats['rejected_rows']}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args();run(a.source,a.output,a.report)
