"""Stream a full Parquet audit, then sample unique groups from legacy train.

All writes go to a NEW output directory. A 128-bit pair of independent pandas
hashes is a practical duplicate screen, not a cryptographic proof of equality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

from .schema import FEATURES, ENDPOINTS, REQUIRED


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def fingerprints(df):
    values = df[FEATURES + ENDPOINTS]
    return tuple(pd.util.hash_pandas_object(values, index=False, hash_key=key).to_numpy()
                 for key in ("0123456789abcdef", "fedcba9876543210"))


def priority_hash(h, seed):
    # SplitMix64: deterministic permutation to randomize the sampling priority.
    with np.errstate(over="ignore"):
        z = h.astype(np.uint64) + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def split_unique(df, seed):
    if df['_flow_id'].duplicated().any():
        raise ValueError('Duplicate flow groups must be resolved before splitting')
    counts = df.Attack.value_counts()
    if counts.min() < 10:
        raise ValueError(f'Insufficient unique samples for a three-way split: {counts.to_dict()}')
    train_val, test = train_test_split(df, test_size=.2, stratify=df.Attack, random_state=seed)
    train, val = train_test_split(train_val, test_size=.125, stratify=train_val.Attack, random_state=seed)
    splits = dict(train=train, val=val, test=test)
    classes = set(df.Attack)
    for name, frame in splits.items():
        if set(frame.Attack) != classes:
            raise ValueError(f'{name} lacks at least one class; increase sample size')
    sets = [set(x._flow_id) for x in splits.values()]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise AssertionError('Flow group overlap')
    return splits


def prepare(root, output, sample_size=50000, seed=20260919):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    files = [(p, 'train') for p in sorted((root / 'train').glob('*.parquet'))]
    files += [(p, 'legacy_val') for p in sorted((root / 'val').glob('*.parquet'))]
    if not files or not any(tag == 'train' for _, tag in files):
        raise ValueError('No training Parquet files')
    manifest, counts, invalid = [], {}, {}
    feature_min = np.full(len(FEATURES), np.inf)
    feature_max = np.full(len(FEATURES), -np.inf)
    nonfinite = np.zeros(len(FEATURES), dtype=np.int64)
    global_row = 0
    index_path = output / 'fingerprints.parquet'
    writer = None
    try:
        for file_num, (path, tag) in enumerate(files):
            pf = pq.ParquetFile(path)
            if set(REQUIRED) - set(pf.schema_arrow.names):
                raise ValueError(f'Missing required columns in source file {file_num}')
            manifest.append(dict(file=path.relative_to(root).as_posix(), tag=tag,
                                 sha256=sha256_file(path), rows=pf.metadata.num_rows,
                                 global_start=global_row))
            counts.setdefault(tag, Counter())
            invalid.setdefault(tag, Counter())
            local_row = 0
            for batch in pf.iter_batches(batch_size=65536, columns=REQUIRED):
                df = batch.to_pandas()
                if set(df.Dataset.dropna().unique()) != {'NF-ToN-IoT-v2'} or df.Dataset.isna().any():
                    raise ValueError('This protocol accepts NF-ToN-IoT-v2 only')
                X = df[FEATURES].to_numpy(dtype=np.float64)
                finite = np.isfinite(X)
                nonfinite += (~finite).sum(axis=0)
                feature_min = np.minimum(feature_min, np.where(finite, X, np.inf).min(axis=0))
                feature_max = np.maximum(feature_max, np.where(finite, X, -np.inf).max(axis=0))
                bad_nodes = df[ENDPOINTS].isna().any(axis=1)
                for col in ENDPOINTS:
                    ports = pd.to_numeric(df[col].str.rsplit(':', n=1).str[-1], errors='coerce')
                    bad_nodes |= ~ports.between(0, 65535)
                bad_labels = df.Attack.isna() | ~df.Label.isin([0, 1])
                bad_labels |= (df.Label != (df.Attack != 'Benign').astype(int))
                valid = finite.all(axis=1) & ~bad_nodes.to_numpy() & ~bad_labels.to_numpy()
                invalid[tag].update(dict(nonfinite_rows=int((~finite.all(axis=1)).sum()),
                                         bad_endpoint_rows=int(bad_nodes.sum()),
                                         bad_label_rows=int(bad_labels.sum()),
                                         excluded_rows=int((~valid).sum())))
                counts[tag].update(df.Attack.fillna('<missing>').value_counts().to_dict())
                h1, h2 = fingerprints(df)
                index = pa.table(dict(h1=h1, h2=h2, priority=priority_hash(h1, seed),
                                      attack=df.Attack.fillna('<missing>').astype(str),
                                      row_id=np.arange(global_row, global_row+len(df), dtype=np.int64),
                                      source=np.repeat(tag, len(df)), valid=valid))
                if writer is None:
                    writer = pq.ParquetWriter(index_path, index.schema, compression='zstd')
                writer.write_table(index)
                global_row += len(df)
                local_row += len(df)
            print(f'Audited {tag} file {file_num+1}/{len(files)}: {local_row:,} rows', flush=True)
    finally:
        if writer:
            writer.close()

    con = duckdb.connect(str(output / 'audit.duckdb'))
    con.execute("SET memory_limit='600MB'")
    con.execute('SET threads=2')
    con.execute('SET preserve_insertion_order=false')
    con.read_parquet(str(index_path)).create_view('idx')
    con.execute('''CREATE TABLE groups AS
        SELECT h1,h2,min(priority) priority, count(*) n,
        count(*) FILTER (WHERE source='train') nt,
        count(*) FILTER (WHERE source='legacy_val') nv,
        count(DISTINCT attack) classes,
        count(DISTINCT attack) FILTER (WHERE source='train') train_classes,
        min(row_id) FILTER (WHERE source='train') first_train_row
        FROM idx WHERE valid GROUP BY h1,h2''')
    row = con.execute('''SELECT count(*), sum(n-1),
        count(*) FILTER (WHERE nt>0 AND nv>0),
        coalesce(sum(nv) FILTER (WHERE nt>0),0),
        count(*) FILTER (WHERE classes>1),
        count(*) FILTER (WHERE nt>0),
        coalesce(sum(nt-1) FILTER (WHERE nt>0),0),
        count(*) FILTER (WHERE train_classes>1)
        FROM groups''').fetchone()
    duplicate_stats = dict(zip(['unique_valid_groups','extra_duplicate_rows',
        'groups_shared_by_legacy_train_val','legacy_val_rows_matching_train',
        'conflicting_label_groups_all','unique_train_groups','extra_train_duplicate_rows',
        'conflicting_train_groups'], map(int, row)))
    selected = con.execute('''SELECT h1,h2,first_train_row AS row_id FROM groups
        WHERE nt>0 AND train_classes=1 ORDER BY priority,h1,h2 LIMIT ?''', [sample_size]).df()
    con.close()
    selected['_flow_id'] = [f'{a:016x}{b:016x}' for a,b in zip(selected.h1,selected.h2)]
    selected = selected.set_index('row_id').sort_index()
    parts = []
    for entry, (path, tag) in zip(manifest, files):
        if tag != 'train':
            continue
        offset = entry['global_start']
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=REQUIRED):
            stop = offset + len(batch)
            chosen = selected.loc[(selected.index >= offset) & (selected.index < stop)]
            if len(chosen):
                frame = batch.take(pa.array((chosen.index-offset).to_numpy())).to_pandas()
                frame['_flow_id'] = chosen._flow_id.to_numpy()
                frame['_source_file'] = entry['file']
                frame['_source_row'] = chosen.index.to_numpy() - entry['global_start']
                parts.append(frame)
            offset = stop
    sample = pd.concat(parts, ignore_index=True).sort_values('_flow_id').reset_index(drop=True)
    # Verify each chosen fingerprint against the retained source row.
    a, b = fingerprints(sample)
    assert sample._flow_id.tolist() == [f'{x:016x}{y:016x}' for x,y in zip(a,b)]
    audit = dict(total_rows=global_row, source_files=manifest, class_counts=counts,
                 invalid_rows=invalid, duplicates=duplicate_stats,
                 feature_ranges={c: dict(min=float(lo) if np.isfinite(lo) else None,
                                         max=float(hi) if np.isfinite(hi) else None,
                                         nonfinite=int(n))
                                 for c,lo,hi,n in zip(FEATURES,feature_min,feature_max,nonfinite)},
                 fingerprint='two independent pandas uint64 hashes; practical collision risk is nonzero',
                 sampling='bottom-k SplitMix64 priority on unique train-only fingerprints; no label balancing',
                 seed=seed, requested_sample_size=sample_size, sampled_rows=len(sample),
                 sampled_class_counts=sample.Attack.value_counts().to_dict())
    write_json(output/'audit.json', audit)
    splits = split_unique(sample, seed)
    split_manifest = {}
    for name, frame in splits.items():
        dest = output/f'{name}.parquet'
        frame.reset_index(drop=True).to_parquet(dest, index=False)
        split_manifest[name] = dict(rows=len(frame), sha256=sha256_file(dest),
                                    classes=frame.Attack.value_counts().to_dict())
    write_json(output/'splits.json', dict(seed=seed, splits=split_manifest, features=FEATURES,
                                         scope='pilot within legacy train; not external test',
                                         group_overlap=0))
    print(json.dumps(dict(duplicates=duplicate_stats, sample=split_manifest), indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/dl_processed')
    p.add_argument('--output', required=True)
    p.add_argument('--sample-size', type=int, default=50000)
    p.add_argument('--seed', type=int, default=20260919)
    args = p.parse_args()
    prepare(args.data, args.output, args.sample_size, args.seed)


if __name__ == '__main__':
    main()
