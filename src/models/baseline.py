"""Baseline TF-IDF + linear models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from pandas import Series
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MaxAbsScaler

from ..features_tfidf import TfidfBundle, build_tfidf_bundle


@dataclass
class BaselineResult:
    model: Pipeline
    vectorizer: TfidfBundle
    predictions: np.ndarray
    predictions_log: np.ndarray
    y_true: np.ndarray

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        import joblib

        joblib.dump(self.model, path / "model.joblib")
        self.vectorizer.save(path / "tfidf.joblib")


MODEL_MAP = {
    "ridge": Ridge,
    "lasso": Lasso,
    "elasticnet": ElasticNet,
}


def _instantiate_model(config: Dict) -> Pipeline:
    model_type = config.get("type", "ridge").lower()
    model_cls = MODEL_MAP.get(model_type)
    if model_cls is None:
        raise ValueError(f"Unsupported baseline model type: {model_type}")
    params = {k: v for k, v in config.items() if k not in {"type"}}
    base_model = model_cls(**params)
    return make_pipeline(MaxAbsScaler(), base_model)


def train_baseline(
    train_texts: Series,
    train_targets: Series,
    test_texts: Series,
    test_targets: Series,
    vectorizer_cfg: Dict,
    model_cfg: Dict,
    log1p_target: bool = True,
) -> BaselineResult:
    tfidf = build_tfidf_bundle(vectorizer_cfg)
    X_train = tfidf.fit_transform(train_texts)
    X_test = tfidf.transform(test_texts)

    y_train = train_targets.to_numpy(dtype=float)
    y_test = test_targets.to_numpy(dtype=float)

    if log1p_target:
        y_train_transformed = np.log1p(y_train)
    else:
        y_train_transformed = y_train

    model = _instantiate_model(model_cfg)
    model.fit(X_train, y_train_transformed)

    pred_log = model.predict(X_test)
    if log1p_target:
        preds = np.expm1(pred_log)
    else:
        preds = pred_log

    return BaselineResult(
        model=model,
        vectorizer=tfidf,
        predictions=preds,
        predictions_log=pred_log,
        y_true=y_test,
    )


__all__ = ["train_baseline", "BaselineResult"]
