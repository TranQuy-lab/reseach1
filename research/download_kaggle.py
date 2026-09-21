"""Download the public Kaggle archive in resumable, verified byte ranges."""
import concurrent.futures as cf
import hashlib,json,os,shutil,time,urllib.request,zipfile,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/raw/NF-UQ-NIDS-v2'
URL='https://www.kaggle.com/api/v1/datasets/download/aryashah2k/nfuqnidsv2-network-intrusion-detection-dataset'
OUT.mkdir(parents=True,exist_ok=True)
PARTS=OUT/'download_parts'; PARTS.mkdir(exist_ok=True)
STATE=OUT/'download_status.json'
def status(**kw):
 kw['updated_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
 tmp=STATE.with_suffix('.tmp'); tmp.write_text(json.dumps(kw,indent=2)); tmp.replace(STATE)
 print(json.dumps(kw),flush=True)
def run():
 with urllib.request.urlopen(URL,timeout=60) as r:
  target=r.url; size=int(r.headers['Content-Length'])
 chunk=1024*1024
 ranges=[(i,a,min(a+chunk,size)-1) for i,a in enumerate(range(0,size,chunk))]
 def get(part):
  i,a,b=part; p=PARTS/f'{i:06}.part'; n=b-a+1
  if p.exists() and p.stat().st_size==n: return n
  for attempt in range(6):
   try:
    req=urllib.request.Request(target,headers={'Range':f'bytes={a}-{b}'})
    with urllib.request.urlopen(req,timeout=60) as r:
     if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {a}-{b}/{size}': raise ValueError('Incorrect range response')
     data=r.read(n+1)
    if len(data)!=n: raise ValueError('Incomplete range')
    p.write_bytes(data); return n
   except Exception:
    if attempt==5: raise RuntimeError(f'Download failed at part {i}; rerun to resume') from None
    time.sleep(min(2**attempt,16))
 status(stage='downloading',total_bytes=size)
 completed=0; last=time.monotonic()
 with cf.ThreadPoolExecutor(max_workers=24) as pool:
  for future in cf.as_completed([pool.submit(get,part) for part in ranges]):
   completed+=future.result()
   if time.monotonic()-last>30:
    status(stage='downloading',downloaded_bytes=completed,total_bytes=size); last=time.monotonic()
 archive=OUT/'kaggle.zip'
 if not archive.exists():
  with archive.with_suffix('.assembling').open('wb') as w:
   for i,_,_ in ranges:
    with (PARTS/f'{i:06}.part').open('rb') as r: shutil.copyfileobj(r,w)
  archive.with_suffix('.assembling').rename(archive)
 if archive.stat().st_size!=size: raise ValueError('Archive size mismatch')
 # Parts are redundant once the archive is assembled; remove only these download artifacts.
 for i,_,_ in ranges: (PARTS/f'{i:06}.part').unlink()
 PARTS.rmdir()
 status(stage='extracting',archive_bytes=size)
 with zipfile.ZipFile(archive) as z:
  entries=z.infolist()
  if len(entries)!=1 or entries[0].filename!='NF-UQ-NIDS-v2.csv': raise ValueError('Unexpected archive members')
  entry=entries[0]; dest=OUT/entry.filename
  if dest.exists(): raise ValueError('CSV already exists; refusing overwrite')
  if shutil.disk_usage(OUT).free<entry.file_size+5*1024**3: raise ValueError('Insufficient free disk')
  h=hashlib.sha256()
  with z.open(entry) as r,dest.with_suffix('.csv.part').open('wb') as w:
   while data:=r.read(8*1024**2): w.write(data); h.update(data)
  dest.with_suffix('.csv.part').rename(dest)
 with dest.open(newline='') as f:
  header=next(csv.reader(f))
 if not {'Dataset','Attack','Label','IPV4_SRC_ADDR','IPV4_DST_ADDR'}<=set(header): raise ValueError('Missing required columns')
 manifest={'source':URL,'retrieved_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'file':dest.name,'bytes':dest.stat().st_size,'sha256':h.hexdigest(),'columns':header,'archive_crc_verified':True,'credentials_used':False,'dataset_counts_status':'pending streaming scan'}
 (ROOT/'research/results/kaggle_download_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 status(stage='complete',csv_bytes=dest.stat().st_size,sha256=h.hexdigest())
if __name__=='__main__':
 try: run()
 except Exception as e:
  status(stage='failed',error=type(e).__name__+': '+str(e)); raise SystemExit(1)
