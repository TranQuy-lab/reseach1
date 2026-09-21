import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.loader import LinkNeighborLoader

from nids_minibatch.data import Preprocessor, make_graph
from nids_minibatch.models import build_model
from nids_minibatch.schema import FEATURES
from nids_minibatch.training import fit_model, full_probabilities, load_frames, run, save_predictions


def frame(n=80):
    rng = np.random.default_rng(902)
    result = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    result["IPV4_SRC_ADDR"] = [f"10.0.0.{i % 11 + 1}" for i in range(n)]
    result["L4_SRC_PORT"] = 1000 + np.arange(n) % 7
    result["IPV4_DST_ADDR"] = [f"192.0.2.{i % 9 + 1}" for i in range(n)]
    result["L4_DST_PORT"] = 80 + np.arange(n) % 3
    result["Dataset"] = "synthetic"
    result["Attack"] = np.where(np.arange(n) % 2, "DDoS", "Benign")
    result["Label"] = (result.Attack != "Benign").astype("int64")
    result["source_row_id"] = np.arange(1, n + 1)
    result["flow_group_id"] = [f"{i:032x}" for i in range(n)]
    return result


@pytest.mark.parametrize("task,classes", [("binary", ["Benign", "Attack"]),
                                           ("multiclass", ["Benign", "DDoS"])])
def test_preprocessor_train_only_and_tasks(task, classes):
    train, val = frame(), frame(20)
    pre = Preprocessor.fit(train, task)
    before = pre.mean.copy()
    val[FEATURES] += 1000
    transformed = pre.transform(val)
    np.testing.assert_array_equal(pre.mean, before)
    assert transformed.dtype == np.float32
    assert pre.classes == classes
    assert set(pre.labels(train)) == {0, 1}


def test_preprocessor_stabilizes_extreme_finite_rates():
    sample = frame(12)
    sample.loc[0, "SRC_TO_DST_SECOND_BYTES"] = 1e200
    sample.loc[1, "DST_TO_SRC_SECOND_BYTES"] = 1e100
    pre = Preprocessor.fit(sample, "binary")
    transformed = pre.transform(sample)
    assert set(pre.log_features) == {"SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES"}
    assert np.isfinite(transformed).all()
    assert Preprocessor.from_dict(pre.as_dict()).as_dict() == pre.as_dict()


def test_ip_and_port_define_nodes_and_parallel_edges_survive():
    f = frame(4)
    f["IPV4_SRC_ADDR"] = "10.0.0.1"
    f["L4_SRC_PORT"] = [1, 2, 2, 2]
    f["IPV4_DST_ADDR"] = "10.0.0.2"
    f["L4_DST_PORT"] = 443
    g = make_graph(f, f[FEATURES].to_numpy(np.float32), f.Label.to_numpy())
    assert g.data.num_edges == 8
    assert g.data.num_nodes == 3
    assert g.label_edge_index.shape == (2, 4)


def test_half_storage_preserves_graph_identity_and_reduces_feature_bytes():
    f = frame(12)
    values = f[FEATURES].to_numpy(np.float32)
    g32 = make_graph(f, values, f.Label.to_numpy(), torch.float32)
    g16 = make_graph(f, values, f.Label.to_numpy(), torch.float16)
    assert torch.equal(g32.label_edge_index, g16.label_edge_index)
    assert g16.label_edge_attr.element_size() * 2 == g32.label_edge_attr.element_size()


@pytest.mark.parametrize("name", ["sage", "sage_edge"])
def test_all_neighbor_minibatch_matches_full_batch(name):
    torch.manual_seed(7)
    f = frame(40)
    pre = Preprocessor.fit(f, "binary")
    g = make_graph(f, pre.transform(f), pre.labels(f))
    model = build_model(name, len(FEATURES), 2, hidden=8, dropout=0).eval()
    full = full_probabilities(model, name, g)
    loader = LinkNeighborLoader(
        g.data, num_neighbors=[-1, -1], edge_label_index=g.label_edge_index,
        edge_label=g.labels, batch_size=len(f), shuffle=False, neg_sampling=None,
        subgraph_type="directional", num_workers=0,
    )
    batch = next(iter(loader))
    with torch.no_grad():
        logits = model(batch.edge_index, batch.edge_attr, batch.edge_label_index,
                       g.label_edge_attr[batch.input_id], batch.num_nodes)
        sampled = torch.softmax(logits, dim=1).numpy()
    np.testing.assert_allclose(sampled, full[batch.input_id], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", ["edge_mlp", "sage", "sage_edge"])
def test_one_epoch_smoke_has_finite_artifact(name):
    frames = {s: frame(n) for s, n in {"train": 80, "val": 30, "test": 30}.items()}
    pre = Preprocessor.fit(frames["train"], "binary")
    graphs = {s: make_graph(f, pre.transform(f), pre.labels(f)) for s, f in frames.items()}
    model, history, best = fit_model(name, graphs, 2, seed=11, epochs=1, patience=1,
                                     batch_size=16, fanout=(5, 3), hidden=8)
    assert best == 1 and len(history) == 1
    assert np.isfinite(history[0]["loss"])
    assert history[0]["train_edges"] == 80
    assert history[0]["train_batches"] == 5
    probability = full_probabilities(model, name, graphs["test"])
    np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-6)


def test_benchmark_batch_cap_does_not_claim_full_epoch():
    frames = {s: frame(n) for s, n in {"train": 80, "val": 30, "test": 30}.items()}
    pre = Preprocessor.fit(frames["train"], "binary")
    graphs = {s: make_graph(f, pre.transform(f), pre.labels(f)) for s, f in frames.items()}
    _, history, _ = fit_model(
        "edge_mlp", graphs, 2, seed=11, epochs=1, patience=1,
        batch_size=16, fanout=(5, 3), hidden=8, max_train_batches=2,
    )
    assert history[0]["train_edges"] == 32
    assert history[0]["train_batches"] == 2


def test_full_scope_reads_partition_files_and_caps_audit_predictions(tmp_path):
    root = tmp_path / "splits"
    dataset = "NF-UNSW-NB15-v2"
    frames = {}
    for i, split in enumerate(("train", "val", "test")):
        value = frame(24)
        value["Dataset"] = dataset
        value["source_row_id"] += i * 100
        value["flow_group_id"] = [f"{i + 1:02x}{j:030x}" for j in range(len(value))]
        folder = root / dataset / f"split={split}"
        folder.mkdir(parents=True)
        value.to_parquet(folder / "part.parquet", index=False)
        frames[split] = value
    loaded = load_frames(root, dataset, "full")
    assert {name: len(value) for name, value in loaded.items()} == {"train": 24, "val": 24, "test": 24}
    pre = Preprocessor.fit(loaded["train"], "binary")
    graph = make_graph(loaded["test"], pre.transform(loaded["test"]), pre.labels(loaded["test"]))
    probability = np.full((24, 2), 0.5, dtype=np.float32)
    path = tmp_path / "audit.parquet"
    assert save_predictions(path, graph, probability, pre.classes, max_rows=7) == 7
    stored = pd.read_parquet(path)
    assert len(stored) == 7
    assert stored.row_offset.is_monotonic_increasing


def test_full_scope_cli_core_completes_one_reproducible_run(tmp_path):
    root = tmp_path / "splits"
    dataset = "NF-UNSW-NB15-v2"
    for i, split in enumerate(("train", "val", "test")):
        value = frame(32)
        value["Dataset"] = dataset
        value["source_row_id"] += i * 100
        value["flow_group_id"] = [f"{i + 1:02x}{j:030x}" for j in range(len(value))]
        folder = root / dataset / f"split={split}"
        folder.mkdir(parents=True)
        value.to_parquet(folder / "part.parquet", index=False)
    output = tmp_path / "runs"
    run(
        root, output, [dataset], ["binary"], ["edge_mlp"], [11],
        epochs=1, patience=1, batch_size=16, fanout=(5, 3), threads=1,
        device="cpu", scope="full", prediction_cap=7,
    )
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["scope"] == "full"
    table = pd.read_csv(output / "runs.csv")
    assert len(table) == 1
    audit = pd.read_parquet(output / f"{dataset}__binary__edge_mlp__seed11" / "test_predictions.parquet")
    assert len(audit) == 7
