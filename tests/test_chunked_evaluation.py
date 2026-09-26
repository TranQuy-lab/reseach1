"""Layer-wise bounded evaluation must equal the single-pass reference."""

import numpy as np
import pandas as pd
import pytest
import torch

from nids_minibatch.data import make_graph
from nids_minibatch.models import build_model
from nids_minibatch.schema import FEATURES
from nids_minibatch.training import (
    EVALUATION_CHUNK_EDGES,
    chunked_logits,
    full_logits,
)


def synthetic_graph(rows: int = 600, hubs: int = 12, seed: int = 7):
    """Directed flows with a few high-degree endpoints, like real NetFlow."""
    rng = np.random.default_rng(seed)
    share = [f"10.0.0.{index}" for index in range(hubs)]
    source = rng.choice(share + [f"192.168.1.{i}" for i in range(40)], size=rows)
    dest = rng.choice(share + [f"172.16.0.{i}" for i in range(60)], size=rows)
    frame = pd.DataFrame({
        "IPV4_SRC_ADDR": source,
        "IPV4_DST_ADDR": dest,
        "L4_SRC_PORT": rng.integers(1024, 65535, size=rows),
        "L4_DST_PORT": rng.choice([80, 443, 22, 8080, 53], size=rows),
        "source_row_id": np.arange(rows),
        "flow_group_id": np.arange(rows),
        "Dataset": "SYNTH",
    })
    features = rng.normal(size=(rows, len(FEATURES))).astype(np.float32)
    labels = rng.integers(0, 3, size=rows)
    return make_graph(frame, features, labels, torch.float32)


@pytest.mark.parametrize("model_name", ["edge_mlp", "sage", "sage_edge"])
@pytest.mark.parametrize("chunk_edges", [16, 97, EVALUATION_CHUNK_EDGES])
def test_chunked_evaluation_matches_full_graph(model_name, chunk_edges):
    graph = synthetic_graph()
    torch.manual_seed(11)
    model = build_model(model_name, len(FEATURES), 3, 32, 0.2)
    reference = full_logits(model, model_name, graph).float()
    chunked = chunked_logits(model, model_name, graph, chunk_edges=chunk_edges)
    assert chunked.shape == reference.shape
    difference = float((chunked - reference).abs().max())
    assert difference < 1e-4, f"{model_name} chunk={chunk_edges} diff={difference}"


def test_chunked_evaluation_rejects_empty_graph_and_chunk():
    graph = synthetic_graph()
    torch.manual_seed(11)
    model = build_model("sage", len(FEATURES), 3, 32, 0.2)
    with pytest.raises(ValueError):
        chunked_logits(model, "sage", graph, chunk_edges=0)
    graph.labels = graph.labels[:0]
    graph.label_edge_attr = graph.label_edge_attr[:0]
    graph.label_edge_index = graph.label_edge_index[:, :0]
    with pytest.raises(ValueError):
        chunked_logits(model, "sage", graph)


def test_validation_loss_uses_host_logits_with_device_independent_weights():
    from nids_minibatch.training import validation_metrics

    graph = synthetic_graph()
    torch.manual_seed(11)
    model = build_model("sage", len(FEATURES), 3, 32, 0.2)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0]))
    probability, loss = validation_metrics(model, "sage", graph, loss_fn)
    assert probability.shape == (len(graph.labels), 3)
    assert np.isfinite(loss)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("model_name", ["sage", "edge_mlp"])
def test_chunked_evaluation_matches_on_cuda(model_name):
    """CUDA chunk accumulation and host output must match the reference."""
    graph = synthetic_graph(rows=400, hubs=8)
    device = torch.device("cuda")
    torch.manual_seed(11)
    model = build_model(model_name, len(FEATURES), 3, 32, 0.2).to(device)
    # `full_logits` returns on the model device; `chunked_logits` returns on the
    # host so the caller can build probabilities without a device transfer.
    reference = full_logits(model, model_name, graph).float().cpu()
    chunked = chunked_logits(model, model_name, graph, chunk_edges=64)
    assert chunked.shape == reference.shape
    assert float((chunked - reference).abs().max()) < 1e-4
