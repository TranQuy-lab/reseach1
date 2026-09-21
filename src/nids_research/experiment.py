"""Run the prespecified CPU pilot and persist reproducible evidence."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .forest import save_forest, predict_forest
from .models import build_model, make_graph
from .prepare import write_json, sha256_file
from .preprocessing import Preprocessor


def metrics(y, pred, classes):
    labels = list(range(len(classes)))
    return dict(accuracy=float(accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, labels=labels, average='macro', zero_division=0)),
                weighted_f1=float(f1_score(y, pred, labels=labels, average='weighted', zero_division=0)),
                per_class=classification_report(y, pred, labels=labels, target_names=classes,
                                                output_dict=True, zero_division=0),
                confusion_matrix=confusion_matrix(y, pred, labels=labels).tolist())


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def fit_neural(name, graphs, ys, seed, epochs, patience, config):
    seed_everything(seed)
    model = build_model(name, config['n_features'], config['n_classes'], config['hidden'], config['dropout'])
    count = np.bincount(ys['train'], minlength=config['n_classes'])
    weights = torch.tensor(len(ys['train']) / (len(count)*count), dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    target = torch.from_numpy(ys['train'])
    history, best_state, best_score, best_epoch, stale = [], None, -1, 0, 0
    for epoch in range(1, epochs+1):
        start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(graphs['train'])
        loss = loss_fn(logits, target)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite gradient')
        optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model(graphs['val']).argmax(1).numpy()
        score = f1_score(ys['val'], pred, labels=list(range(config['n_classes'])), average='macro', zero_division=0)
        history.append(dict(epoch=epoch, loss=float(loss.detach()), val_macro_f1=float(score),
                            seconds=time.perf_counter()-start))
        if score > best_score:
            best_state, best_score, best_epoch, stale = copy.deepcopy(model.state_dict()), float(score), epoch, 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f'{name} seed={seed} epoch={epoch} loss={loss.item():.4f} val_macro_f1={score:.4f}', flush=True)
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probabilities = {s: torch.softmax(model(graphs[s]), 1).numpy() for s in ('val', 'test')}
    return model, history, probabilities, best_epoch


def predict_artifact(run_dir, frame):
    """Infer unlabeled flows. The entire supplied frame is the graph context."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir/'config.json').read_text())
    pre = Preprocessor.from_dict(json.loads((run_dir/'preprocessor.json').read_text()))
    x = pre.transform(frame)
    if config['model'] == 'random_forest':
        return predict_forest(run_dir/'forest.npz', x), pre.classes
    graph = make_graph(frame, x, bidirectional=config['bidirectional'])
    model = build_model(config['model'], len(pre.features), len(pre.classes), config['hidden'], config['dropout'])
    model.load_state_dict(torch.load(run_dir/'model.pt', map_location='cpu', weights_only=True))
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(graph), 1).numpy(), pre.classes


def run(data, output, seeds=(11,22,33), epochs=120, patience=20, threads=2):
    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    frames = {s: pd.read_parquet(data/f'{s}.parquet') for s in ('train','val','test')}
    manifest = json.loads((data/'splits.json').read_text())
    for s in frames:
        if sha256_file(data/f'{s}.parquet') != manifest['splits'][s]['sha256']:
            raise ValueError('Input checksum does not match split manifest')
    ids = [set(x._flow_id) for x in frames.values()]
    if any(ids[i] & ids[j] for i in range(3) for j in range(i)):
        raise ValueError('Input split overlap')
    pre = Preprocessor.fit(frames['train'])
    xs = {s: pre.transform(f) for s,f in frames.items()}
    ys = {s: pre.labels(f) for s,f in frames.items()}
    graphs = {s: make_graph(frames[s], xs[s]) for s in frames}
    config_base = dict(n_features=len(pre.features), n_classes=len(pre.classes), hidden=128,
                       dropout=.2, lr=.001, epochs=epochs, patience=patience, threads=threads,
                       bidirectional=True, loss='balanced weighted CE from train',
                       evaluation='one prediction per original flow; offline separate split graphs')
    environment = dict(python=platform.python_version(), platform=platform.platform(),
                       packages={p:importlib.metadata.version(p) for p in
                                 ('numpy','pandas','pyarrow','scikit-learn','torch','duckdb')},
                       torch_threads=torch.get_num_threads(), cuda_available=torch.cuda.is_available())
    source_root = Path(__file__).parent
    source_digests = {p.name:sha256_file(p) for p in sorted(source_root.glob('*.py'))}
    provenance = dict(environment=environment, source_sha256=source_digests,
                      protocol_sha256=sha256_file('research/PROTOCOL.md'),
                      split_manifest=manifest, seeds=list(seeds), config=config_base,
                      graphs={s:dict(nodes=g.num_nodes, original_flows=len(g.src),
                                     message_edges=len(g.message_src)) for s,g in graphs.items()})
    write_json(output/'provenance.json', provenance)
    write_json(output/'preprocessor.json', pre.as_dict())
    rows = []
    for name in ('edge_mlp','sage','sage_edge','random_forest'):
        for seed in seeds:
            dest = output/f'{name}_seed{seed}'
            dest.mkdir()
            config = dict(config_base, model=name, seed=seed)
            if name == 'random_forest':
                config.update(n_estimators=100, min_samples_leaf=2, class_weight='balanced')
            write_json(dest/'config.json', config)
            write_json(dest/'preprocessor.json', pre.as_dict())
            start = time.perf_counter()
            print(f'Start {name} seed={seed}', flush=True)
            if name == 'random_forest':
                model = RandomForestClassifier(n_estimators=100, min_samples_leaf=2,
                    class_weight='balanced', random_state=seed, n_jobs=threads)
                model.fit(xs['train'], ys['train'])
                probabilities = {s:model.predict_proba(xs[s]) for s in ('val','test')}
                save_forest(model, dest/'forest.npz')
                history, best_epoch, parameter_count = [], None, None
                forest_nodes = sum(t.tree_.node_count for t in model.estimators_)
            else:
                model, history, probabilities, best_epoch = fit_neural(name, graphs, ys, seed, epochs, patience, config)
                torch.save(model.state_dict(), dest/'model.pt')
                parameter_count = sum(p.numel() for p in model.parameters())
                forest_nodes = None
            elapsed = time.perf_counter()-start
            write_json(dest/'history.json', history)
            # Reload with NO label columns: demonstrate real usable inference.
            unlabeled = frames['test'].drop(columns=['Attack','Label','Dataset'])
            replay, replay_classes = predict_artifact(dest, unlabeled)
            if replay_classes != pre.classes or not np.allclose(replay, probabilities['test'], rtol=1e-6, atol=1e-7):
                raise AssertionError('Checkpoint replay differs from original inference')
            result = dict(model=name, seed=seed, seconds_fit_and_evaluate=elapsed,
                          best_epoch=best_epoch, epochs_ran=len(history), parameters=parameter_count,
                          forest_nodes=forest_nodes, replay_max_abs_error=float(np.max(np.abs(replay-probabilities['test']))),
                          val=metrics(ys['val'], probabilities['val'].argmax(1), pre.classes),
                          test=metrics(ys['test'], probabilities['test'].argmax(1), pre.classes))
            write_json(dest/'metrics.json', result)
            pred_frame = pd.DataFrame(dict(flow_id=frames['test']._flow_id, y_true=ys['test'],
                                           y_pred=probabilities['test'].argmax(1)))
            for i,c in enumerate(pre.classes):
                pred_frame[f'p_{c}'] = probabilities['test'][:,i]
            pred_frame.to_parquet(dest/'test_predictions.parquet', index=False)
            row = {k:result[k] for k in ('model','seed','seconds_fit_and_evaluate','best_epoch','epochs_ran','parameters')}
            row.update({f'{s}_{m}':result[s][m] for s in ('val','test') for m in ('macro_f1','weighted_f1','accuracy')})
            rows.append(row)
            pd.DataFrame(rows).to_csv(output/'runs.csv', index=False)
            print(f'Completed {name} seed={seed}; test macro-F1={result["test"]["macro_f1"]:.4f}; {elapsed:.1f}s', flush=True)
    runs = pd.DataFrame(rows)
    summary = runs.groupby('model')[['test_macro_f1','test_weighted_f1','test_accuracy','seconds_fit_and_evaluate']].agg(['mean','std'])
    summary.to_csv(output/'summary.csv')
    print(summary.to_string(), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--seeds', type=int, nargs='+', default=[11,22,33])
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    if min(a.epochs, a.patience, a.threads) < 1 or not a.seeds:
        p.error('Positive epochs/patience/threads and at least one seed required')
    run(a.data, a.output, a.seeds, a.epochs, a.patience, a.threads)


if __name__ == '__main__':
    main()
