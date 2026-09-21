"""Exploratory structural diagnostics; never used to select pilot models."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    frames = {s:pd.read_parquet(Path(a.data)/f'{s}.parquet') for s in ('train','val','test')}
    known = set(frames['train'].src_node) | set(frames['train'].dst_node)
    report = {'scope':'post-hoc descriptive graph audit; not a model selection criterion'}
    for split, df in frames.items():
        nodes = pd.concat([df.src_node, df.dst_node], ignore_index=True)
        degree = nodes.value_counts()
        # Each ordered endpoint pair receives one deterministic prediction in
        # an endpoint-only decoder on a fixed graph, regardless of edge count.
        pair_class = df.groupby(['src_node','dst_node','Attack'], observed=True).size()
        pairs = pair_class.groupby(level=[0,1]).sum()
        maxima = pair_class.groupby(level=[0,1]).max()
        classes = pair_class.groupby(level=[0,1]).size()
        report[split] = dict(flows=len(df), nodes=int(degree.size),
            endpoint_degree=dict(min=int(degree.min()), median=float(degree.median()),
                                 p95=float(degree.quantile(.95)), max=int(degree.max())),
            degree_one_node_fraction=float((degree==1).mean()),
            flows_with_any_unseen_endpoint=float((~df.src_node.isin(known) | ~df.dst_node.isin(known)).mean()),
            flows_with_both_unseen_endpoints=float((~df.src_node.isin(known) & ~df.dst_node.isin(known)).mean()),
            ordered_endpoint_pairs=int(len(pairs)), multi_flow_pairs=int((pairs>1).sum()),
            conflicting_label_endpoint_pairs=int((classes>1).sum()),
            endpoint_only_empirical_accuracy_ceiling=float(maxima.sum()/len(df)),
            note='Ceiling is a label-based descriptive upper bound for endpoint-only decoding, not achieved performance.')
    Path(a.output).write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
