"""Training-only standardization and JSON persistence without pickle."""
from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import StandardScaler

from .schema import FEATURES


@dataclass
class Preprocessor:
    features: list
    classes: list
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, train):
        x = train[FEATURES].to_numpy(dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite training feature')
        if train.Attack.isna().any():
            raise ValueError('Missing training label')
        scaler = StandardScaler().fit(x)
        return cls(list(FEATURES), sorted(train.Attack.unique().tolist()), scaler.mean_, scaler.scale_)

    def transform(self, frame):
        x = frame[self.features].to_numpy(dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite inference feature')
        x = ((x-self.mean)/self.scale).astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError('Overflow after standardization')
        return x

    def labels(self, frame):
        lookup = {c:i for i,c in enumerate(self.classes)}
        codes = frame.Attack.map(lookup)
        if codes.isna().any():
            raise ValueError('Unknown or missing label; refusing silent remap to class zero')
        return codes.to_numpy(dtype=np.int64, copy=True)

    def as_dict(self):
        return dict(features=self.features, classes=self.classes, mean=self.mean.tolist(), scale=self.scale.tolist())

    @classmethod
    def from_dict(cls, d):
        if len(d['features']) != len(d['mean']) or len(d['mean']) != len(d['scale']):
            raise ValueError('Invalid preprocessor dimensions')
        mean, scale = np.asarray(d['mean']), np.asarray(d['scale'])
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('Invalid preprocessor statistics')
        return cls(d['features'], d['classes'], mean, scale)
