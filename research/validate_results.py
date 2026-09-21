"""Independently recompute saved metrics and check provenance/artifact contracts."""
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from nids_research.prepare import sha256_file, write_json
from nids_research.preprocessing import Preprocessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',default='research/artifacts/pilot_data')
    parser.add_argument('--runs',default='research/artifacts/pilot_runs')
    parser.add_argument('--cli',default='research/artifacts/cli_predictions.parquet')
    parser.add_argument('--output',default='research/results/verification.json')
    args = parser.parse_args()
    data, runs = Path(args.data), Path(args.runs)
    frames = {s:pd.read_parquet(data/f'{s}.parquet') for s in ('train','val','test')}
    pre = Preprocessor.from_dict(json.loads((runs/'preprocessor.json').read_text()))
    expected_pre = Preprocessor.fit(frames['train'])
    np.testing.assert_array_equal(expected_pre.mean,pre.mean)
    np.testing.assert_array_equal(expected_pre.scale,pre.scale)
    manifest = json.loads((runs/'provenance.json').read_text())
    for name,digest in manifest['source_sha256'].items():
        assert sha256_file(Path('src/nids_research')/name) == digest, name
    assert sha256_file('research/PROTOCOL.md') == manifest['protocol_sha256']
    for s in frames:
        assert sha256_file(data/f'{s}.parquet') == manifest['split_manifest']['splits'][s]['sha256']
    classes = pre.classes
    records = []
    for path in sorted(runs.glob('*_seed*')):
        rec = json.loads((path/'metrics.json').read_text())
        pred = pd.read_parquet(path/'test_predictions.parquet')
        np.testing.assert_array_equal(pred.flow_id.to_numpy(),frames['test']._flow_id.to_numpy())
        np.testing.assert_array_equal(pred.y_true.to_numpy(),pre.labels(frames['test']))
        prob = pred[[f'p_{c}' for c in classes]].to_numpy()
        assert np.isfinite(prob).all() and (prob >= 0).all()
        np.testing.assert_allclose(prob.sum(axis=1),1,rtol=1e-6,atol=1e-6)
        np.testing.assert_array_equal(prob.argmax(axis=1),pred.y_pred)
        for average in ('macro','weighted'):
            recomputed = f1_score(pred.y_true,pred.y_pred,labels=list(range(len(classes))),
                                   average=average,zero_division=0)
            assert abs(recomputed-rec['test'][f'{average}_f1']) < 1e-12
        assert abs(accuracy_score(pred.y_true,pred.y_pred)-rec['test']['accuracy']) < 1e-12
        np.testing.assert_array_equal(confusion_matrix(pred.y_true,pred.y_pred,labels=list(range(len(classes)))),
                                      rec['test']['confusion_matrix'])
        history = json.loads((path/'history.json').read_text())
        if history:
            best = max(history,key=lambda r:r['val_macro_f1'])
            assert best['epoch'] == rec['best_epoch']
            assert abs(best['val_macro_f1'] - rec['val']['macro_f1']) < 1e-12
        assert rec['replay_max_abs_error'] < 1e-6
        records.append(dict(run=path.name,rows=len(pred),metrics_recomputed=True,
                            replay_max_abs_error=rec['replay_max_abs_error']))
    assert len(records) == 12
    cli = pd.read_parquet(args.cli)
    reference = pd.read_parquet(runs/'sage_edge_seed11'/'test_predictions.parquet')
    np.testing.assert_allclose(cli[[f'p_{c}' for c in classes]].to_numpy(),
                               reference[[f'p_{c}' for c in classes]].to_numpy(),atol=1e-6,rtol=1e-5)
    result = dict(status='passed',scope='saved artifacts, metrics, provenance and CLI replay',
                  verified_runs=records, protocol_hash_matches=True,source_hashes_match=True,
                  split_hashes_match=True,preprocessing_matches_training_only=True,
                  pytest=dict(result='45 passed',command='PYTHONPATH=src .venv-research/bin/python -m pytest tests -q',
                              recorded_from='successful test execution before the pilot'),
                  limitations=['No raw CSV validation','No GPU validation','No DGL runtime parity test',
                               'No full-scale training or external evaluation'])
    write_json(args.output,result)
    print('PASS: all 12 runs, saved metrics, source/split/protocol hashes, preprocessing and CLI replay')


if __name__ == '__main__':
    main()
