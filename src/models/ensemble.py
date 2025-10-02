"""Simple ensemble helpers."""
from __future__ import annotations

from typing import Iterable, List

import numpy as np


def average_ensemble(predictions: Iterable[np.ndarray]) -> np.ndarray:
    preds: List[np.ndarray] = [np.asarray(p) for p in predictions if p is not None]
    if not preds:
        raise ValueError("No predictions provided for ensembling.")
    min_len = min(p.shape[0] for p in preds)
    aligned = [p[:min_len] for p in preds]
    return np.mean(np.vstack(aligned), axis=0)


__all__ = ["average_ensemble"]
