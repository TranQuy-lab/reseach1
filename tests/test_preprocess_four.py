import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
spec=importlib.util.spec_from_file_location('four',Path(__file__).parents[1]/'research/preprocess_four.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def sample():
 row={'IPV4_SRC_ADDR':'192.0.2.1','IPV4_DST_ADDR':'198.51.100.2','L4_SRC_PORT':65535,'L4_DST_PORT':443,'IN_BYTES':2**53+1,'Label':0,'Attack':'Benign','Dataset':m.DATASETS[0]}
 for f in m.FLOAT:row[f]=0.25
 return pd.DataFrame([row,row])
def test_preserve_port_integer_and_duplicate_group_across_batches():
 df=sample();a,_,_=m.transform(df.iloc[:1],1);b,_,_=m.transform(df.iloc[1:],2)
 assert a.L4_SRC_PORT.iloc[0]==65535 and a.IN_BYTES.iloc[0]==2**53+1
 assert a.flow_group_id.iloc[0]==b.flow_group_id.iloc[0]
 assert a.source_row_id.iloc[0]!=b.source_row_id.iloc[0]
 df.loc[1,['Label','Attack']]=[1,'DoS'];c,_,_=m.transform(df,1)
 assert c.flow_group_id.nunique()==1
@pytest.mark.parametrize('column,value,reason',[('L4_SRC_PORT',65536,'port_invalid'),('IPV4_SRC_ADDR','bad','ip_invalid'),('Label',1,'label_invalid'),('L7_PROTO',np.inf,'nonfinite'),('Attack',' ','attack_empty'),('Dataset','wrong','dataset_invalid')])
def test_quarantine(column,value,reason):
 df=sample();df.loc[0,column]=value
 clean,rejected,counts=m.transform(df,1)
 assert len(clean)==len(rejected)==1 and counts[reason]==1
 assert reason in rejected.rejection_reasons.iloc[0]
def test_nullable_int_no_precision_loss():
 df=sample();df['IN_BYTES']=pd.array([pd.NA,2**53+1],dtype='Int64')
 clean,rejected,_=m.transform(df,1)
 assert len(rejected)==1 and clean.IN_BYTES.iloc[0]==2**53+1
def test_end_to_end_exports_four_and_quarantines(tmp_path):
 from nids_research.schema import FEATURES
 import json
 cols=['IPV4_SRC_ADDR','L4_SRC_PORT','IPV4_DST_ADDR','L4_DST_PORT']+FEATURES+['Label','Attack','Dataset']
 rows=[]
 for d in m.DATASETS:
  row={c:0 for c in cols};row.update(sample().iloc[0].to_dict());row['Dataset']=d;rows.append(row)
 rows.append(dict(rows[0],IPV4_SRC_ADDR='invalid'))
 source=tmp_path/'input.csv';pd.DataFrame(rows)[cols].to_csv(source,index=False)
 out=tmp_path/'out';report=tmp_path/'report.json';m.run(source,out,report)
 stats=json.loads(report.read_text())
 assert stats['raw_rows']==5 and stats['valid_rows']==4 and stats['rejected_rows']==1
 for d in m.DATASETS:
  df=pq.read_table(out/(d+'.parquet')).to_pandas()
  assert len(df)==1 and df.L4_SRC_PORT.iloc[0]==65535
  assert df.IN_BYTES.iloc[0]==2**53+1
 assert pq.read_table(out/'rejected_rows.parquet').num_rows==1
 with pytest.raises(ValueError,match='refusing overwrite'):m.run(source,out,report)
