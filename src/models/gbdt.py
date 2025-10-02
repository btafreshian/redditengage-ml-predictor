"""LightGBM based regressors."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
from lightgbm import LGBMRegressor


@dataclass
class GBDTResult:
    model: LGBMRegressor
    y_true: np.ndarray
    y_pred_log: np.ndarray
    y_pred: np.ndarray
    quantile_predictions: Dict[str, np.ndarray] = field(default_factory=dict)
    quantile_models: Dict[str, LGBMRegressor] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "gbdt_model.joblib")
        for name, model in self.quantile_models.items():
            joblib.dump(model, path / f"gbdt_quantile_{name}.joblib")


def _prepare_targets(targets: np.ndarray, log1p: bool) -> np.ndarray:
    return np.log1p(targets) if log1p else targets


def _inverse_targets(predictions: np.ndarray, log1p: bool) -> np.ndarray:
    return np.expm1(predictions) if log1p else predictions


def train_gbdt(
    train_matrix: np.ndarray,
    train_targets: np.ndarray,
    test_matrix: np.ndarray,
    test_targets: np.ndarray,
    config: Dict,
    log1p_target: bool = True,
    quantiles: Optional[List[float]] = None,
) -> GBDTResult:
    params = dict(config.get("lightgbm", {}))
    model = LGBMRegressor(**params)

    y_train = _prepare_targets(train_targets, log1p_target)
    model.fit(train_matrix, y_train)

    pred_log = model.predict(test_matrix)
    preds = _inverse_targets(pred_log, log1p_target)

    quantile_preds: Dict[str, np.ndarray] = {}
    quantile_models: Dict[str, LGBMRegressor] = {}
    quantile_cfg = config.get("quantile", {})
    quantile_list = quantiles if quantiles is not None else quantile_cfg.get("alpha_values", [])
    if quantile_cfg.get("enabled", False) and quantile_list:
        for alpha in quantile_list:
            q_params = dict(params)
            q_params.update({
                "objective": "quantile",
                "alpha": alpha,
            })
            if "metric" in q_params:
                q_params.pop("metric")
            q_params.update({
                "n_estimators": quantile_cfg.get("n_estimators", params.get("n_estimators", 200)),
            })
            q_model = LGBMRegressor(**q_params)
            q_model.fit(train_matrix, y_train)
            q_pred = _inverse_targets(q_model.predict(test_matrix), log1p_target)
            name = f"alpha_{alpha:.2f}".replace(".", "p")
            quantile_models[name] = q_model
            quantile_preds[name] = q_pred

    return GBDTResult(
        model=model,
        y_true=test_targets,
        y_pred_log=pred_log,
        y_pred=preds,
        quantile_predictions=quantile_preds,
        quantile_models=quantile_models,
    )


__all__ = ["train_gbdt", "GBDTResult"]
