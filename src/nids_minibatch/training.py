"""Training, deterministic evaluation, artifact storage and experiment orchestration."""

from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import resource
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import LinkNeighborLoader

from .data import EdgeGraph, Preprocessor, make_graph
from .models import EGraphSAGE, build_model
from .schema import DATASETS, FEATURES


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but torch.cuda.is_available() is false")
    return device


def metrics(y: np.ndarray, probabilities: np.ndarray, classes: list[str]) -> dict:
    pred = probabilities.argmax(axis=1)
    labels = list(range(len(classes)))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": classification_report(
            y, pred, labels=labels, target_names=classes, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
    }


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def full_probabilities(model, name: str, graph: EdgeGraph, amp: bool = False) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad(), _autocast(device, amp):
        if name == "edge_mlp":
            logits = model(graph.label_edge_attr.to(device))
        else:
            logits = model(
                graph.data.edge_index.to(device),
                graph.data.edge_attr.to(device),
                graph.label_edge_index.to(device),
                graph.label_edge_attr.to(device),
                graph.data.num_nodes,
            )
        return torch.softmax(logits.float(), dim=1).cpu().numpy()


def _class_weights(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    count = torch.bincount(labels, minlength=n_classes).float()
    if (count == 0).any():
        raise ValueError("Training split lacks a class")
    return len(labels) / (n_classes * count)


def _timing_profile(batch_seconds: list[float], warmup: int = 50,
                    window: int = 50) -> dict:
    """Summarize bounded benchmark batches without treating warm-up as steady state."""
    usable = batch_seconds[warmup:]
    windows = [sum(usable[i:i + window]) / len(usable[i:i + window])
               for i in range(0, len(usable), window)
               if len(usable[i:i + window]) == window]
    median = float(np.median(windows)) if windows else None
    p90 = float(np.quantile(windows, 0.9)) if windows else None
    mad = float(np.median(np.abs(np.asarray(windows) - median))) if windows else None
    return {
        "warmup_batches": min(warmup, len(batch_seconds)),
        "window_batches": window,
        "measured_windows": len(windows),
        "seconds_per_batch_windows": windows,
        "median_seconds_per_batch": median,
        "p90_seconds_per_batch": p90,
        "mad_seconds_per_batch": mad,
    }


def _mlp_epoch(model, graph: EdgeGraph, optimizer, loss_fn, batch_size: int,
               seed: int, amp: bool = False, max_batches: int = 0,
               profile_batches: bool = False) -> tuple[float, int, int, dict | None]:
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(graph.label_edge_attr, graph.labels), batch_size=batch_size,
        shuffle=True, generator=generator, num_workers=0,
    )
    total, seen = 0.0, 0
    device = next(model.parameters()).device
    model.train()
    batches = 0
    batch_seconds = []
    for x, y in loader:
        batch_started = time.perf_counter()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            loss = loss_fn(model(x), y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss")
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * len(y)
        seen += len(y)
        batches += 1
        if profile_batches:
            batch_seconds.append(time.perf_counter() - batch_started)
        if max_batches and batches >= max_batches:
            break
    profile = _timing_profile(batch_seconds) if profile_batches else None
    return total / seen, seen, batches, profile


def _sage_epoch(model: EGraphSAGE, graph: EdgeGraph, optimizer, loss_fn,
                batch_size: int, fanout: tuple[int, int], seed: int,
                amp: bool = False, max_batches: int = 0,
                profile_batches: bool = False) -> tuple[float, int, int, dict | None]:
    generator = torch.Generator().manual_seed(seed)
    loader = LinkNeighborLoader(
        graph.data,
        num_neighbors=list(fanout),
        edge_label_index=graph.label_edge_index,
        edge_label=graph.labels,
        batch_size=batch_size,
        shuffle=True,
        neg_sampling=None,
        subgraph_type="directional",
        num_workers=0,
        generator=generator,
    )
    total, seen = 0.0, 0
    device = next(model.parameters()).device
    model.train()
    batches = 0
    batch_seconds = []
    for batch in loader:
        batch_started = time.perf_counter()
        seed_attr = graph.label_edge_attr[batch.input_id].to(device)
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            logits = model(batch.edge_index, batch.edge_attr, batch.edge_label_index,
                           seed_attr, batch.num_nodes)
            loss = loss_fn(logits, batch.edge_label.long())
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss")
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite gradient")
        optimizer.step()
        total += float(loss.detach()) * batch.edge_label.numel()
        seen += batch.edge_label.numel()
        batches += 1
        if profile_batches:
            batch_seconds.append(time.perf_counter() - batch_started)
        if max_batches and batches >= max_batches:
            break
    profile = _timing_profile(batch_seconds) if profile_batches else None
    return total / seen, seen, batches, profile


def fit_model(name: str, graphs: dict[str, EdgeGraph], n_classes: int, seed: int,
              epochs: int, patience: int, batch_size: int, fanout: tuple[int, int],
              hidden: int = 128, dropout: float = 0.2, lr: float = 0.001,
              device: str | torch.device = "cpu", eval_every: int = 1,
              amp: bool = False, max_train_batches: int = 0):
    seed_everything(seed)
    device = resolve_device(str(device))
    model = build_model(name, len(FEATURES), n_classes, hidden, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    weights = _class_weights(graphs["train"].labels, n_classes).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    best_state, best_score, best_epoch, stale = None, -1.0, 0, 0
    history = []
    profile_batches = max_train_batches > 0
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        if name == "edge_mlp":
            loss, train_edges, train_batches, timing = _mlp_epoch(
                model, graphs["train"], optimizer, loss_fn, batch_size,
                seed + epoch, amp, max_train_batches, profile_batches,
            )
        else:
            loss, train_edges, train_batches, timing = _sage_epoch(
                model, graphs["train"], optimizer, loss_fn,
                batch_size, fanout, seed + epoch, amp, max_train_batches,
                profile_batches,
            )
        train_seconds = time.perf_counter() - started
        should_evaluate = epoch == 1 or epoch % eval_every == 0 or epoch == epochs
        if not should_evaluate:
            history.append({"epoch": epoch, "loss": loss, "val_macro_f1": None,
                            "train_edges": train_edges, "train_batches": train_batches,
                            "train_seconds": train_seconds, "validation_seconds": 0.0,
                            "batch_timing": timing,
                            "seconds": time.perf_counter() - started})
            continue
        validation_started = time.perf_counter()
        val_prob = full_probabilities(model, name, graphs["val"], amp)
        val_score = f1_score(
            graphs["val"].labels.numpy(), val_prob.argmax(1), labels=list(range(n_classes)),
            average="macro", zero_division=0,
        )
        history.append({"epoch": epoch, "loss": loss, "val_macro_f1": float(val_score),
                        "train_edges": train_edges, "train_batches": train_batches,
                        "train_seconds": train_seconds,
                        "batch_timing": timing,
                        "validation_seconds": time.perf_counter() - validation_started,
                        "seconds": time.perf_counter() - started})
        if val_score > best_score:
            best_state, best_score, best_epoch, stale = copy.deepcopy(model.state_dict()), float(val_score), epoch, 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f"{name} seed={seed} epoch={epoch} loss={loss:.4f} val_macro_f1={val_score:.4f}", flush=True)
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return model, history, best_epoch


def fit_model_steps(name: str, graphs: dict[str, EdgeGraph], n_classes: int, seed: int,
                    max_steps: int, eval_every_steps: int, patience: int,
                    batch_size: int, fanout: tuple[int, int], hidden: int = 128,
                    dropout: float = 0.2, lr: float = 0.001,
                    device: str | torch.device = "cpu", amp: bool = False):
    """Train to a fixed optimizer-step budget after one complete data pass.

    A checkpoint is eligible only after every train edge has been presented at
    least once. Validation then occurs at the first completed pass and at fixed
    step intervals. This keeps the full-data claim while making compute budgets
    comparable across datasets with very different edge counts.
    """
    seed_everything(seed)
    device = resolve_device(str(device))
    model = build_model(name, len(FEATURES), n_classes, hidden, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    weights = _class_weights(graphs["train"].labels, n_classes).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    steps_per_pass = math.ceil(len(graphs["train"].labels) / batch_size)
    if max_steps < steps_per_pass:
        raise ValueError(
            f"max_train_steps={max_steps} is below one full pass ({steps_per_pass})"
        )
    best_state, best_score, best_step, stale = None, -1.0, 0, 0
    history: list[dict] = []
    total_steps = 0
    total_edges = 0
    epoch = 0
    last_eval_step = 0
    stop = False

    def evaluate(loss: float, interval_edges: int, interval_batches: int,
                 interval_train_seconds: float, current_epoch: int) -> bool:
        nonlocal best_state, best_score, best_step, stale, last_eval_step
        validation_started = time.perf_counter()
        val_prob = full_probabilities(model, name, graphs["val"], amp)
        val_score = f1_score(
            graphs["val"].labels.numpy(), val_prob.argmax(1),
            labels=list(range(n_classes)), average="macro", zero_division=0,
        )
        validation_seconds = time.perf_counter() - validation_started
        history.append({
            "epoch": current_epoch, "step": total_steps, "loss": loss,
            "val_macro_f1": float(val_score), "train_edges": interval_edges,
            "train_batches": interval_batches,
            "train_seconds": interval_train_seconds,
            "validation_seconds": validation_seconds,
            "seconds": interval_train_seconds + validation_seconds,
            "eligible_after_full_pass": True,
        })
        if val_score > best_score:
            best_state = copy.deepcopy(model.state_dict())
            best_score, best_step, stale = float(val_score), total_steps, 0
        else:
            stale += 1
        last_eval_step = total_steps
        print(f"{name} seed={seed} step={total_steps} epoch={current_epoch} "
              f"loss={loss:.4f} val_macro_f1={val_score:.4f}", flush=True)
        return stale >= patience

    interval_loss_sum = 0.0
    interval_edges = 0
    interval_batches = 0
    interval_started = time.perf_counter()
    while total_steps < max_steps and not stop:
        epoch += 1
        if name == "edge_mlp":
            generator = torch.Generator().manual_seed(seed + epoch)
            loader = DataLoader(
                TensorDataset(graphs["train"].label_edge_attr, graphs["train"].labels),
                batch_size=batch_size, shuffle=True, generator=generator, num_workers=0,
            )
        else:
            generator = torch.Generator().manual_seed(seed + epoch)
            loader = LinkNeighborLoader(
                graphs["train"].data, num_neighbors=list(fanout),
                edge_label_index=graphs["train"].label_edge_index,
                edge_label=graphs["train"].labels, batch_size=batch_size,
                shuffle=True, neg_sampling=None, subgraph_type="directional",
                num_workers=0, generator=generator,
            )
        model.train()
        for value in loader:
            if name == "edge_mlp":
                x, y = value
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device, amp):
                    loss = loss_fn(model(x), y)
                edge_count = len(y)
            else:
                batch = value
                seed_attr = graphs["train"].label_edge_attr[batch.input_id].to(device)
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device, amp):
                    logits = model(batch.edge_index, batch.edge_attr,
                                   batch.edge_label_index, seed_attr, batch.num_nodes)
                    loss = loss_fn(logits, batch.edge_label.long())
                edge_count = batch.edge_label.numel()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all()
                       for p in model.parameters()):
                raise FloatingPointError("Nonfinite gradient")
            optimizer.step()
            loss_value = float(loss.detach())
            total_steps += 1
            total_edges += edge_count
            interval_loss_sum += loss_value * edge_count
            interval_edges += edge_count
            interval_batches += 1

            full_pass_reached = total_edges >= len(graphs["train"].labels)
            interval_due = total_steps - last_eval_step >= eval_every_steps
            budget_reached = total_steps >= max_steps
            if full_pass_reached and (interval_due or budget_reached):
                train_seconds = time.perf_counter() - interval_started
                stop = evaluate(
                    interval_loss_sum / interval_edges, interval_edges,
                    interval_batches, train_seconds, epoch,
                )
                interval_loss_sum = 0.0
                interval_edges = 0
                interval_batches = 0
                interval_started = time.perf_counter()
                model.train()
            if stop or total_steps >= max_steps:
                break
        # Evaluate exactly at the first completed pass even if the fixed interval
        # boundary does not coincide with the end of the loader.
        if (not stop and best_state is None and total_edges >= len(graphs["train"].labels)
                and interval_batches):
            train_seconds = time.perf_counter() - interval_started
            stop = evaluate(
                interval_loss_sum / interval_edges, interval_edges,
                interval_batches, train_seconds, epoch,
            )
            interval_loss_sum = 0.0
            interval_edges = 0
            interval_batches = 0
            interval_started = time.perf_counter()
    if best_state is None:
        raise RuntimeError("No eligible checkpoint after a complete train pass")
    model.load_state_dict(best_state)
    return model, history, best_step


def load_frames(root: Path, dataset: str, scope: str = "pilot") -> dict[str, pd.DataFrame]:
    if scope not in {"pilot", "full"}:
        raise ValueError("scope must be pilot or full")
    if scope == "pilot":
        paths = {s: root / dataset / f"pilot_{s}.parquet" for s in ("train", "val", "test")}
    else:
        paths = {
            s: sorted((root / dataset / f"split={s}").glob("*.parquet"))
            for s in ("train", "val", "test")
        }
        if any(not value for value in paths.values()):
            raise FileNotFoundError(f"Missing full-data partition for {dataset}")
    frames = {s: pd.read_parquet(path) for s, path in paths.items()}
    if scope == "pilot":
        ids = {s: set(f.flow_group_id) for s, f in frames.items()}
        if ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"]:
            raise ValueError("Flow group leakage across pilot splits")
    if any(set(f.Dataset) != {dataset} for f in frames.values()):
        raise ValueError("Dataset column mismatch")
    return frames


def prepare_graphs(frames: dict[str, pd.DataFrame], task: str,
                   storage_dtype: torch.dtype = torch.float32):
    pre = Preprocessor.fit(frames["train"], task)
    graphs = {s: make_graph(f, pre.transform(f), pre.labels(f), storage_dtype)
              for s, f in frames.items()}
    return pre, graphs


def save_predictions(path: Path, graph: EdgeGraph, probabilities: np.ndarray,
                     classes: list[str], max_rows: int | None = None) -> int:
    row_offset = np.arange(len(graph.labels))
    if max_rows and len(row_offset) > max_rows:
        row_offset = np.linspace(0, len(row_offset) - 1, max_rows, dtype=np.int64)
    frame = pd.DataFrame({
        "row_offset": row_offset,
        "source_row_id": graph.source_row_id[row_offset],
        "flow_group_id": graph.flow_group_id[row_offset],
        "y_true": graph.labels.numpy()[row_offset],
        "y_pred": probabilities.argmax(1)[row_offset],
    })
    for i, name in enumerate(classes):
        frame[f"p_{name}"] = probabilities[row_offset, i]
    frame.to_parquet(path, index=False)
    return len(frame)


def environment() -> dict:
    packages = ["numpy", "pandas", "pyarrow", "scikit-learn", "torch", "torch-geometric", "pyg-lib"]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "cuda_available": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(),
    }


def run(data: Path, output: Path, datasets: list[str], tasks: list[str], models: list[str],
        seeds: list[int], epochs: int, patience: int, batch_size: int,
        fanout: tuple[int, int], threads: int, resume: bool = False,
        device: str = "auto", scope: str = "pilot", eval_every: int = 1,
        amp: bool = False, prediction_cap: int = 0,
        max_train_batches: int = 0, max_train_steps: int = 0,
        eval_every_steps: int = 0) -> None:
    data, output = Path(data), Path(output)
    if output.exists() and not resume:
        raise ValueError("Output exists; refusing overwrite")
    output.mkdir(parents=True, exist_ok=resume)
    torch.set_num_threads(threads)
    effective_device = resolve_device(device)
    storage_dtype = torch.float16 if scope == "full" and effective_device.type == "cuda" else torch.float32
    protocol = Path("research/PROTOCOL_FULL_DATA_VI.md" if scope == "full"
                    else "research/PROTOCOL_MINIBATCH_VI.md")
    expected_provenance = {
        "environment": environment(),
        "protocol_sha256": sha256_file(protocol),
        "datasets": datasets, "tasks": tasks, "models": models, "seeds": seeds,
        "epochs": epochs, "patience": patience, "batch_size": batch_size,
        "fanout": list(fanout), "threads": threads,
        "scope": scope, "eval_every": eval_every, "amp": amp,
        "prediction_cap": prediction_cap,
        "max_train_batches": max_train_batches,
        "max_train_steps": max_train_steps,
        "eval_every_steps": eval_every_steps,
        "training_budget_mode": "steps" if max_train_steps else "epochs",
        "edge_storage_dtype": str(storage_dtype),
        "requested_device": device, "effective_device": str(effective_device),
    }
    if resume:
        stored = json.loads((output / "provenance.json").read_text())
        comparable = {k: v for k, v in stored.items() if k != "environment"}
        expected_comparable = {k: v for k, v in expected_provenance.items() if k != "environment"}
        if comparable != expected_comparable:
            raise ValueError("Resume configuration/protocol differs from existing run")
        runs_file = output / "runs.csv"
        rows = pd.read_csv(runs_file).to_dict("records") if runs_file.is_file() else []
    else:
        rows = []
        write_json(output / "provenance.json", expected_provenance)
    completed = {(r["dataset"], r["task"], r["model"], int(r["seed"])) for r in rows}
    for dataset in datasets:
        load_started = time.perf_counter()
        frames = load_frames(data, dataset, scope)
        dataset_load_seconds = time.perf_counter() - load_started
        for task in tasks:
            graph_started = time.perf_counter()
            pre, graphs = prepare_graphs(frames, task, storage_dtype)
            graph_prepare_seconds = time.perf_counter() - graph_started
            for name in models:
                for seed in seeds:
                    run_id = f"{dataset}__{task}__{name}__seed{seed}"
                    dest = output / run_id
                    key = (dataset, task, name, seed)
                    if key in completed:
                        required = [dest / filename for filename in (
                            "config.json", "preprocessor.json", "model.pt", "history.json",
                            "metrics.json", "test_predictions.parquet",
                        )]
                        if not all(path.is_file() for path in required):
                            raise ValueError(f"Completed run has missing artifacts: {run_id}")
                        print(f"SKIP {run_id}", flush=True)
                        continue
                    if dest.exists():
                        shutil.rmtree(dest)
                    dest.mkdir()
                    config = {
                        "dataset": dataset, "task": task, "model": name, "seed": seed,
                        "epochs": epochs, "patience": patience, "batch_size": batch_size,
                        "fanout": list(fanout), "hidden": 128, "dropout": 0.2,
                        "learning_rate": 0.001, "bidirectional_messages": True,
                        "scope": scope, "eval_every": eval_every, "amp": amp,
                        "prediction_cap": prediction_cap,
                        "max_train_batches": max_train_batches,
                        "max_train_steps": max_train_steps,
                        "eval_every_steps": eval_every_steps,
                        "training_budget_mode": "steps" if max_train_steps else "epochs",
                        "edge_storage_dtype": str(storage_dtype),
                        "train_sampling": "LinkNeighborLoader directional; loss on seed edges",
                        "validation_test": "full split graph",
                    }
                    write_json(dest / "config.json", config)
                    write_json(dest / "preprocessor.json", pre.as_dict())
                    started = time.perf_counter()
                    if effective_device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(effective_device)
                    print(f"START {run_id}", flush=True)
                    if max_train_steps:
                        model, history, best_step = fit_model_steps(
                            name, graphs, len(pre.classes), seed, max_train_steps,
                            eval_every_steps, patience, batch_size, fanout,
                            device=effective_device, amp=amp,
                        )
                        best_epoch = next(
                            item["epoch"] for item in history if item["step"] == best_step
                        )
                    else:
                        model, history, best_epoch = fit_model(
                            name, graphs, len(pre.classes), seed, epochs, patience,
                            batch_size, fanout,
                            device=effective_device, eval_every=eval_every, amp=amp,
                            max_train_batches=max_train_batches,
                        )
                        best_step = sum(item["train_batches"] for item in history[:best_epoch])
                    fit_seconds = time.perf_counter() - started
                    torch.save(model.state_dict(), dest / "model.pt")
                    write_json(dest / "history.json", history)
                    evaluation_started = time.perf_counter()
                    val_prob = full_probabilities(model, name, graphs["val"], amp)
                    test_prob = full_probabilities(model, name, graphs["test"], amp)
                    replay_model = build_model(name, len(FEATURES), len(pre.classes), 128, 0.2).to(effective_device)
                    replay_model.load_state_dict(torch.load(dest / "model.pt", map_location="cpu", weights_only=True))
                    replay_prob = full_probabilities(replay_model, name, graphs["test"], amp)
                    replay_error = float(np.max(np.abs(replay_prob - test_prob)))
                    if replay_error > 1e-7:
                        raise AssertionError(f"Checkpoint replay differs: {replay_error}")
                    final_evaluation_seconds = time.perf_counter() - evaluation_started
                    elapsed = fit_seconds + final_evaluation_seconds
                    result = {
                        "dataset": dataset, "task": task, "model": name, "seed": seed,
                        "seconds_fit_and_evaluate": elapsed,
                        "seconds_fit": fit_seconds,
                        "seconds_final_evaluation": final_evaluation_seconds,
                        "seconds_dataset_load": dataset_load_seconds,
                        "seconds_graph_prepare": graph_prepare_seconds,
                        "best_epoch": best_epoch, "best_step": best_step,
                        "epochs_ran": max(item["epoch"] for item in history),
                        "steps_ran": max(item.get("step", 0) for item in history)
                                     or sum(item["train_batches"] for item in history),
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "replay_max_abs_error": replay_error,
                        "peak_rss_kib_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        "peak_cuda_bytes": (torch.cuda.max_memory_allocated(effective_device)
                                            if effective_device.type == "cuda" else 0),
                        "val": metrics(graphs["val"].labels.numpy(), val_prob, pre.classes),
                        "test": metrics(graphs["test"].labels.numpy(), test_prob, pre.classes),
                    }
                    result["test_prediction_rows_stored"] = save_predictions(
                        dest / "test_predictions.parquet", graphs["test"], test_prob,
                        pre.classes, prediction_cap or None,
                    )
                    result["test_prediction_rows_total"] = len(graphs["test"].labels)
                    write_json(dest / "metrics.json", result)
                    row = {k: result[k] for k in (
                        "dataset", "task", "model", "seed", "seconds_fit_and_evaluate",
                        "seconds_fit", "seconds_final_evaluation", "seconds_dataset_load",
                        "seconds_graph_prepare", "best_epoch", "epochs_ran", "parameters",
                        "best_step", "steps_ran", "peak_rss_kib_process", "peak_cuda_bytes",
                    )}
                    row.update({f"{split}_{metric}": result[split][metric]
                                for split in ("val", "test")
                                for metric in ("macro_f1", "weighted_f1", "accuracy")})
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(output / "runs.csv", index=False)
                    print(f"DONE {run_id} test_macro_f1={result['test']['macro_f1']:.4f} seconds={elapsed:.1f}", flush=True)
            del graphs
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["dataset", "task", "model"])[
        ["test_macro_f1", "test_weighted_f1", "test_accuracy", "seconds_fit_and_evaluate"]
    ].agg(["mean", "std"])
    summary.to_csv(output / "summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--tasks", nargs="+", choices=["multiclass", "binary"], default=["multiclass", "binary"])
    parser.add_argument("--models", nargs="+", choices=["edge_mlp", "sage", "sage_edge"], default=["edge_mlp", "sage", "sage_edge"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--fanout", nargs=2, type=int, default=[15, 10])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--scope", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prediction-cap", type=int, default=0,
                        help="Store at most this many deterministic test rows; 0 stores all")
    parser.add_argument("--max-train-batches", type=int, default=0,
                        help="Benchmark gate only; 0 processes every training edge")
    parser.add_argument("--max-train-steps", type=int, default=0,
                        help="Full-data total optimizer-step budget; 0 uses epoch mode")
    parser.add_argument("--eval-every-steps", type=int, default=0,
                        help="Validate after this many steps once a full pass is complete")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.threads, args.eval_every) < 1:
        parser.error("epochs, patience, batch-size, threads and eval-every must be positive")
    if min(args.prediction_cap, args.max_train_batches, args.max_train_steps,
           args.eval_every_steps) < 0:
        parser.error("prediction and training limits must be nonnegative")
    if args.max_train_batches and args.max_train_steps:
        parser.error("max-train-batches and max-train-steps are mutually exclusive")
    if bool(args.max_train_steps) != bool(args.eval_every_steps):
        parser.error("max-train-steps and eval-every-steps must be used together")
    run(args.data, args.output, args.datasets, args.tasks, args.models, args.seeds,
        args.epochs, args.patience, args.batch_size, tuple(args.fanout), args.threads,
        args.resume, args.device, args.scope, args.eval_every, args.amp,
        args.prediction_cap, args.max_train_batches, args.max_train_steps,
        args.eval_every_steps)


if __name__ == "__main__":
    main()
