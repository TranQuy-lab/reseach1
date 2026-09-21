"""In-memory pilot graph construction with disk-prepared, traceable inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from .schema import FEATURES

LOG_THRESHOLD = 1e20


@dataclass
class Preprocessor:
    features: list[str]
    classes: list[str]
    mean: np.ndarray
    scale: np.ndarray
    task: str
    log_features: list[str]

    @staticmethod
    def _signed_log1p(x: np.ndarray) -> np.ndarray:
        return np.sign(x) * np.log1p(np.abs(x))

    @classmethod
    def _stabilize(cls, x: np.ndarray, features: list[str], log_features: list[str]) -> np.ndarray:
        result = x.copy()
        selected = [features.index(name) for name in log_features]
        if selected:
            result[:, selected] = cls._signed_log1p(result[:, selected])
        return result

    @classmethod
    def fit(cls, frame: pd.DataFrame, task: str):
        if task not in {"multiclass", "binary"}:
            raise ValueError(f"Unknown task: {task}")
        x = frame[FEATURES].to_numpy(dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite training feature")
        # Some ToN-IoT rate fields contain finite values up to ~1e261. Squaring
        # those values overflows even float64. Choose this transform using only
        # training data and persist the selected columns for val/test replay.
        log_features = [FEATURES[i] for i, value in enumerate(np.max(np.abs(x), axis=0))
                        if value > LOG_THRESHOLD]
        x = cls._stabilize(x, list(FEATURES), log_features)
        scaler = StandardScaler().fit(x)
        scale = scaler.scale_.copy()
        scale[scale == 0] = 1.0
        classes = sorted(frame.Attack.unique().tolist()) if task == "multiclass" else ["Benign", "Attack"]
        return cls(list(FEATURES), classes, scaler.mean_.copy(), scale, task, log_features)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        missing = set(self.features) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing features: {sorted(missing)}")
        x = frame[self.features].to_numpy(dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite feature")
        x = self._stabilize(x, self.features, self.log_features)
        result = ((x - self.mean) / self.scale).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite standardized feature")
        return result

    def labels(self, frame: pd.DataFrame) -> np.ndarray:
        if self.task == "binary":
            y = frame.Label.to_numpy(dtype=np.int64)
            if not np.isin(y, [0, 1]).all():
                raise ValueError("Invalid binary label")
            return y
        lookup = {name: i for i, name in enumerate(self.classes)}
        mapped = frame.Attack.map(lookup)
        if mapped.isna().any():
            unknown = sorted(frame.loc[mapped.isna(), "Attack"].unique().tolist())
            raise ValueError(f"Unknown attacks: {unknown}")
        return mapped.to_numpy(dtype=np.int64)

    def as_dict(self):
        return {
            "features": self.features,
            "classes": self.classes,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "task": self.task,
            "log_features": self.log_features,
        }

    @classmethod
    def from_dict(cls, value):
        features = list(value["features"])
        classes = list(value["classes"])
        mean = np.asarray(value["mean"], dtype=np.float64)
        scale = np.asarray(value["scale"], dtype=np.float64)
        task = value["task"]
        if features != list(FEATURES) or len(mean) != len(features) or len(scale) != len(features):
            raise ValueError("Invalid preprocessor dimensions or feature order")
        if task not in {"binary", "multiclass"} or not classes:
            raise ValueError("Invalid preprocessor task/classes")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Invalid preprocessor statistics")
        log_features = list(value.get("log_features", []))
        if not set(log_features).issubset(features):
            raise ValueError("Unknown stabilized feature")
        return cls(features, classes, mean, scale, task, log_features)


@dataclass
class EdgeGraph:
    data: Data
    label_edge_index: torch.Tensor
    label_edge_attr: torch.Tensor
    labels: torch.Tensor
    source_row_id: np.ndarray
    flow_group_id: np.ndarray
    dataset: str


def endpoint_strings(frame: pd.DataFrame, side: str) -> pd.Series:
    if side not in {"SRC", "DST"}:
        raise ValueError("side must be SRC or DST")
    ip = frame[f"IPV4_{side}_ADDR"].astype(str)
    port = frame[f"L4_{side}_PORT"].astype("int64").astype(str)
    return frame.Dataset.astype(str) + "|" + ip + ":" + port


def make_graph(frame: pd.DataFrame, features: np.ndarray, labels: np.ndarray,
               storage_dtype: torch.dtype = torch.float32) -> EdgeGraph:
    if not len(frame) or features.shape != (len(frame), len(FEATURES)) or len(labels) != len(frame):
        raise ValueError("Invalid graph inputs")
    if storage_dtype not in {torch.float16, torch.float32}:
        raise ValueError("storage_dtype must be float16 or float32")
    # Factorize the exact (IP, port) tuple without materializing tens of millions
    # of temporary "IP:port" Python strings on full datasets.
    ip = pd.concat([frame["IPV4_SRC_ADDR"], frame["IPV4_DST_ADDR"]], ignore_index=True)
    port = np.concatenate([
        frame["L4_SRC_PORT"].to_numpy(dtype=np.int64, copy=False),
        frame["L4_DST_PORT"].to_numpy(dtype=np.int64, copy=False),
    ])
    endpoints = pd.MultiIndex.from_arrays([ip, port], names=["ip", "port"])
    codes, nodes = pd.factorize(endpoints, sort=True)
    src = torch.from_numpy(codes[: len(frame)].copy()).long()
    dst = torch.from_numpy(codes[len(frame) :].copy()).long()
    original = torch.stack([src, dst])
    reverse_mask = src != dst
    message_index = torch.cat([original, torch.stack([dst[reverse_mask], src[reverse_mask]])], dim=1)
    edge_attr = torch.from_numpy(features).to(storage_dtype)
    message_attr = torch.cat([edge_attr, edge_attr[reverse_mask]])
    data = Data(edge_index=message_index, edge_attr=message_attr, num_nodes=len(nodes))
    return EdgeGraph(
        data=data,
        label_edge_index=original,
        label_edge_attr=edge_attr,
        labels=torch.from_numpy(labels.copy()).long(),
        source_row_id=frame.source_row_id.to_numpy(copy=True),
        flow_group_id=frame.flow_group_id.to_numpy(copy=True),
        dataset=str(frame.Dataset.iloc[0]),
    )
