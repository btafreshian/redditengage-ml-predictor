"""Embedding based regression models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
from lightgbm import LGBMRegressor
from pandas import DataFrame
from sklearn.linear_model import ElasticNet


@dataclass
class EmbeddingResult:
    model: object
    y_true: np.ndarray
    y_pred_log: np.ndarray
    y_pred: np.ndarray

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "model.joblib")


def _stack_features(embeddings: np.ndarray, meta: Optional[DataFrame] = None) -> np.ndarray:
    matrix = embeddings
    if meta is not None and not meta.empty:
        matrix = np.hstack([matrix, meta.to_numpy(dtype=np.float32)])
    return matrix.astype(np.float32)


def _build_model(config: Dict) -> object:
    model_type = config.get("type", "elasticnet").lower()
    params = {k: v for k, v in config.items() if k not in {"type"}}
    if model_type == "elasticnet":
        return ElasticNet(**params)
    if model_type == "lightgbm":
        return LGBMRegressor(**params)
    raise ValueError(f"Unsupported embedding regressor: {model_type}")


def train_embedding_regressor(
    train_embeddings: np.ndarray,
    train_targets: np.ndarray,
    test_embeddings: np.ndarray,
    test_targets: np.ndarray,
    config: Dict,
    log1p_target: bool = True,
    train_meta: Optional[DataFrame] = None,
    test_meta: Optional[DataFrame] = None,
) -> EmbeddingResult:
    X_train = _stack_features(train_embeddings, train_meta)
    X_test = _stack_features(test_embeddings, test_meta)

    if log1p_target:
        y_train = np.log1p(train_targets)
    else:
        y_train = train_targets

    model = _build_model(config)
    model.fit(X_train, y_train)

    pred_log = model.predict(X_test)
    if log1p_target:
        preds = np.expm1(pred_log)
    else:
        preds = pred_log

    return EmbeddingResult(
        model=model,
        y_true=test_targets,
        y_pred_log=pred_log,
        y_pred=preds,
    )


__all__ = ["train_embedding_regressor", "EmbeddingResult"]
