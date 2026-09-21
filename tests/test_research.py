"""Scientific invariants of the new pilot; all fixtures are synthetic."""
import json

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.ensemble import RandomForestClassifier

from nids_research.schema import FEATURES
from nids_research.prepare import fingerprints, prepare, split_unique, write_json
from nids_research.preprocessing import Preprocessor
from nids_research.models import SAGELayer, build_model, make_graph
from nids_research.experiment import predict_artifact
from nids_research.forest import predict_forest, save_forest


def frame(n=100):
    rng = np.random.default_rng(17)
    f = pd.DataFrame(rng.uniform(0, 10, (n,len(FEATURES))), columns=FEATURES)
    f['src_node'] = [f'10.0.0.1:{i+1000}' for i in range(n)]
    f['dst_node'] = '10.0.0.2:80'
    f['Attack'] = np.where(np.arange(n)%2, 'DDoS', 'Benign')
    f['Label'] = (f.Attack != 'Benign').astype(int)
    f['Dataset'] = 'NF-ToN-IoT-v2'
    f['_flow_id'] = [f'flow{i}' for i in range(n)]
    return f


def test_scaler_train_only_and_order():
    train, val = frame(), frame(20)
    pre = Preprocessor.fit(train)
    expected_mean = train[FEATURES].mean().to_numpy()
    val[FEATURES] += 100000
    before = pre.mean.copy()
    pre.transform(val)
    np.testing.assert_allclose(pre.mean, expected_mean)
    np.testing.assert_array_equal(before, pre.mean)
    shuffled = val[val.columns[::-1]]
    np.testing.assert_array_equal(pre.transform(shuffled), pre.transform(val))


def test_scaler_serialization_constant_nonfinite_unknown():
    f = frame()
    f[FEATURES[0]] = 5.
    pre = Preprocessor.fit(f)
    assert pre.scale[0] == 1
    restored = Preprocessor.from_dict(json.loads(json.dumps(pre.as_dict())))
    np.testing.assert_array_equal(pre.transform(f), restored.transform(f))
    with pytest.raises(ValueError, match='Unknown'):
        pre.labels(f.assign(Attack='unseen'))
    f.loc[0,FEATURES[0]] = np.inf
    with pytest.raises(ValueError, match='Nonfinite'):
        pre.transform(f)


def test_fingerprint_ignores_labels_but_includes_features():
    f = frame(5)
    a,b = fingerprints(f)
    x,y = fingerprints(f.assign(Attack='Other', Label=1))
    np.testing.assert_array_equal(a,x)
    np.testing.assert_array_equal(b,y)
    f.loc[0,FEATURES[0]] += 1
    assert fingerprints(f)[0][0] != a[0]


def test_split_disjoint_deterministic_and_reject_duplicate():
    f = frame(200)
    splits = split_unique(f,42)
    again = split_unique(f,42)
    assert [len(splits[s]) for s in ('train','val','test')] == [140,20,40]
    for s in splits:
        assert splits[s]._flow_id.tolist() == again[s]._flow_id.tolist()
    assert not set(splits['train']._flow_id) & set(splits['test']._flow_id)
    with pytest.raises(ValueError, match='Duplicate'):
        split_unique(pd.concat([f,f.iloc[:1]]),42)


def test_reverse_edges_keep_parallel_and_self_loop_once():
    f = pd.DataFrame(dict(src_node=['a:1','a:1','b:2'], dst_node=['b:2','b:2','b:2']))
    g = make_graph(f, np.ones((3,2), dtype=np.float32))
    assert len(g.message_src) == 5
    assert len(g.src) == 3
    assert g.message_flow.tolist() == [0,1,2,0,1]
    assert make_graph(f, np.ones((3,2)), False).message_flow.tolist() == [0,1,2]


def test_sage_aggregation_matches_manual_equations():
    torch.manual_seed(3)
    f = pd.DataFrame(dict(src_node=['a:1','b:1','a:1'], dst_node=['c:1','c:1','c:1']))
    e = np.array([[1.,2.],[3.,4.],[5.,6.]], dtype=np.float32)
    g = make_graph(f,e,False)
    x = torch.tensor([[1.,2.],[3.,4.],[5.,6.]])
    layer = SAGELayer(2,2,3)
    actual = layer(x,g).detach().numpy()
    wm, bm = layer.w_msg.weight.detach().numpy(), layer.w_msg.bias.detach().numpy()
    wa, ba = layer.w_apply.weight.detach().numpy(), layer.w_apply.bias.detach().numpy()
    expected = []
    for v in range(3):
        messages = [np.concatenate([x[u].numpy(), e[i]]) @ wm.T + bm
                    for i,(u,d) in enumerate(zip(g.src.tolist(),g.dst.tolist())) if d == v]
        agg = np.mean(messages,axis=0) if messages else np.zeros(3)
        expected.append(np.maximum(np.concatenate([x[v].numpy(),agg]) @ wa.T + ba,0))
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)


@pytest.mark.parametrize('name',['edge_mlp','sage','sage_edge'])
def test_finite_gradients_and_one_prediction_per_flow(name):
    torch.manual_seed(4)
    f = frame(20)
    pre = Preprocessor.fit(f)
    g = make_graph(f,pre.transform(f))
    model = build_model(name,len(FEATURES),2,hidden=8)
    logits = model(g)
    assert logits.shape == (20,2)
    loss = torch.nn.functional.cross_entropy(logits,torch.from_numpy(pre.labels(f)))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(float(p.grad.abs().sum()) for p in model.parameters()) > 0


def test_same_endpoints_limitation_and_direct_edge_distinction():
    torch.manual_seed(12)
    f = pd.DataFrame(dict(src_node=['a:1','a:1'],dst_node=['b:2','b:2']))
    g = make_graph(f,np.array([[0.,0.],[3.,4.]],dtype=np.float32))
    model = build_model('sage',2,2,hidden=4).eval()
    assert torch.equal(model(g)[0],model(g)[1])
    direct = build_model('sage_edge',2,2,hidden=4).eval()
    assert not torch.equal(direct(g)[0],direct(g)[1])


def test_checkpoint_replay_unlabeled_and_node_rename(tmp_path):
    torch.set_num_threads(1)
    f = frame(20)
    pre = Preprocessor.fit(f)
    model = build_model('sage_edge',len(FEATURES),2,hidden=8).eval()
    torch.save(model.state_dict(),tmp_path/'model.pt')
    write_json(tmp_path/'config.json',dict(model='sage_edge',hidden=8,dropout=.2,bidirectional=True))
    write_json(tmp_path/'preprocessor.json',pre.as_dict())
    with torch.no_grad():
        original = torch.softmax(model(make_graph(f,pre.transform(f))),1).numpy()
    unlabeled = f.drop(columns=['Attack','Label','Dataset'])
    p,_ = predict_artifact(tmp_path,unlabeled)
    np.testing.assert_array_equal(original,p)
    unlabeled['src_node'] = [f'z:{i}' for i in range(len(f))]
    unlabeled['dst_node'] = 'a:80'
    renamed,_ = predict_artifact(tmp_path,unlabeled)
    np.testing.assert_allclose(p,renamed,rtol=1e-5,atol=1e-6)


def test_forest_export_replays_without_pickle(tmp_path):
    f = frame(30)
    pre = Preprocessor.fit(f)
    x,y = pre.transform(f),pre.labels(f)
    model = RandomForestClassifier(n_estimators=5,random_state=13).fit(x,y)
    save_forest(model,tmp_path/'forest.npz')
    np.testing.assert_allclose(model.predict_proba(x),predict_forest(tmp_path/'forest.npz',x),atol=1e-12)


def test_streaming_prepare_detects_overlap_conflicts_and_excludes(tmp_path):
    root = tmp_path/'input'
    (root/'train').mkdir(parents=True)
    (root/'val').mkdir()
    f = frame(200).drop(columns='_flow_id')
    duplicate = f.iloc[[1]].copy()
    conflict = f.iloc[[0]].copy().assign(Attack='DDoS',Label=1)
    pd.concat([f,duplicate,conflict],ignore_index=True).to_parquet(root/'train'/'a.parquet',index=False)
    f.iloc[:20].to_parquet(root/'val'/'a.parquet',index=False)
    out = tmp_path/'prepared'
    prepare(root,out,sample_size=100,seed=12)
    audit = json.loads((out/'audit.json').read_text())
    assert audit['duplicates']['groups_shared_by_legacy_train_val'] == 20
    assert audit['duplicates']['extra_train_duplicate_rows'] == 2
    assert audit['duplicates']['conflicting_train_groups'] == 1
    combined = pd.concat([pd.read_parquet(out/f'{s}.parquet') for s in ('train','val','test')])
    assert len(combined) == 100
    assert combined._flow_id.nunique() == 100
    assert not ((combined.src_node == f.iloc[0].src_node) & (combined.dst_node == f.iloc[0].dst_node)).any()
