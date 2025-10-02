"""Evaluation utilities for models."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics as sk_metrics

from .utils import ensure_dir

sns.set_theme(style="whitegrid")


def compute_regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    return {
        "rmse": float(math.sqrt(sk_metrics.mean_squared_error(y_true, y_pred))),
        "mae": float(sk_metrics.mean_absolute_error(y_true, y_pred)),
        "median_ae": float(np.median(np.abs(y_true - y_pred))),
        "r2": float(sk_metrics.r2_score(y_true, y_pred)),
    }


def _save_fig(path: Path) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _pred_vs_actual(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=df, x="ups", y="prediction", hue="subreddit", alpha=0.6, edgecolor=None)
    max_val = max(df["ups"].max(), df["prediction"].max())
    plt.plot([0, max_val], [0, max_val], color="black", linestyle="--", linewidth=1)
    plt.xlabel("Actual upvotes")
    plt.ylabel("Predicted upvotes")
    plt.title("Predicted vs. Actual Upvotes")
    _save_fig(path)


def _residuals_vs_fitted(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x="prediction", y="residual", alpha=0.5)
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Predicted upvotes")
    plt.ylabel("Residual (y - yhat)")
    plt.title("Residuals vs. Fitted")
    _save_fig(path)


def _error_distribution(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.histplot(df["residual"], bins=40, kde=True, color="#2a9d8f")
    plt.xlabel("Residual")
    plt.title("Residual Distribution")
    _save_fig(path)


def _qq_plot(df: pd.DataFrame, path: Path) -> None:
    import scipy.stats as stats

    plt.figure(figsize=(5, 5))
    stats.probplot(df["residual"], dist="norm", plot=plt)
    plt.title("QQ Plot of Residuals")
    _save_fig(path)


def _error_by_subreddit(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(7, 4))
    grouped = df.groupby("subreddit")["abs_error"].mean().sort_values()
    sns.barplot(x=grouped.values, y=grouped.index, palette="viridis")
    plt.xlabel("Mean Absolute Error")
    plt.ylabel("Subreddit")
    plt.title("Error by Subreddit")
    _save_fig(path)


def _error_by_length(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(6, 4))
    bins = pd.cut(df["text_length"], bins=min(10, max(3, df["text_length"].nunique())))
    agg = df.groupby(bins)["abs_error"].median().reset_index()
    agg["mid"] = agg["text_length"].apply(lambda interval: interval.mid if hasattr(interval, "mid") else 0)
    sns.lineplot(data=agg, x="mid", y="abs_error", marker="o")
    plt.xlabel("Body length (characters)")
    plt.ylabel("Median Absolute Error")
    plt.title("Error by Text Length")
    _save_fig(path)


def _interval_coverage(df: pd.DataFrame, path: Path, quantile_cols: Dict[str, str]) -> None:
    plt.figure(figsize=(6, 4))
    coverage_records = []
    for lower, upper in quantile_cols.values():
        lower_vals = df[lower]
        upper_vals = df[upper]
        coverage = np.mean((df["ups"] >= lower_vals) & (df["ups"] <= upper_vals))
        coverage_records.append({"interval": f"{lower}->{upper}", "coverage": coverage})
    coverage_df = pd.DataFrame(coverage_records)
    sns.barplot(data=coverage_df, x="interval", y="coverage", palette="magma")
    plt.ylim(0, 1)
    plt.title("Prediction Interval Coverage")
    plt.ylabel("Coverage")
    plt.xlabel("Interval")
    _save_fig(path)


def _placeholder(path: Path, title: str, message: str) -> None:
    plt.figure(figsize=(5, 3))
    plt.axis("off")
    plt.title(title)
    plt.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    _save_fig(path)


def generate_figures(
    df: pd.DataFrame,
    figures_dir: Path,
    include_importances: Optional[pd.Series] = None,
    ridge_terms: Optional[pd.Series] = None,
    interval_cols: Optional[Dict[str, str]] = None,
) -> None:
    figures_dir = ensure_dir(figures_dir)
    df = df.copy()
    df["residual"] = df["ups"] - df["prediction"]
    df["abs_error"] = df["residual"].abs()
    df["text_length"] = df["body"].str.len()

    _pred_vs_actual(df, figures_dir / "pred_vs_actual.png")
    _residuals_vs_fitted(df, figures_dir / "residuals_vs_fitted.png")
    _error_distribution(df, figures_dir / "error_distribution.png")
    _qq_plot(df, figures_dir / "qq_plot.png")
    _error_by_subreddit(df, figures_dir / "error_by_subreddit.png")
    _error_by_length(df, figures_dir / "error_by_length.png")

    if interval_cols:
        _interval_coverage(df, figures_dir / "interval_coverage.png", interval_cols)
    else:
        _placeholder(figures_dir / "interval_coverage.png", "Prediction Intervals", "Quantile predictions unavailable.")

    if include_importances is not None and not include_importances.empty:
        plt.figure(figsize=(6, 6))
        include_importances.sort_values().tail(20).plot(kind="barh", color="#264653")
        plt.title("Top LightGBM Feature Importance")
        plt.xlabel("Gain")
        _save_fig(figures_dir / "feature_importance_gbdt.png")
    else:
        _placeholder(
            figures_dir / "feature_importance_gbdt.png",
            "Feature Importance",
            "Train a GBDT model to view feature importances.",
        )

    if ridge_terms is not None and not ridge_terms.empty:
        plt.figure(figsize=(6, 6))
        ridge_terms.sort_values().tail(20).plot(kind="barh", color="#e76f51")
        plt.title("Top Ridge Coefficients")
        plt.xlabel("Coefficient")
        _save_fig(figures_dir / "coef_ridge_top_terms.png")
    else:
        _placeholder(
            figures_dir / "coef_ridge_top_terms.png",
            "Ridge Coefficients",
            "Train the baseline model to view coefficients.",
        )


def build_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    df = df.copy()
    df["error"] = df["ups"] - df["prediction"]
    df["abs_error"] = df["error"].abs()

    subreddit_table = df.groupby("subreddit").agg(
        count=("id", "count"),
        rmse=("error", lambda x: math.sqrt(np.mean(np.square(x)))),
        mae=("abs_error", "mean"),
    ).sort_values("rmse")

    created = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
    monthly_table = df.assign(month=created.dt.to_period("M")).groupby("month").agg(
        count=("id", "count"),
        rmse=("error", lambda x: math.sqrt(np.mean(np.square(x)))),
        mae=("abs_error", "mean"),
    )

    return {
        "subreddit": subreddit_table.reset_index(),
        "monthly": monthly_table.reset_index(),
    }


def save_tables(tables: Dict[str, pd.DataFrame], run_dir: Path) -> Dict[str, str]:
    paths = {}
    for name, table in tables.items():
        path = run_dir / f"table_{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = str(path)
    return paths


def save_metrics(metrics: Dict[str, float], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)


__all__ = [
    "compute_regression_metrics",
    "generate_figures",
    "build_tables",
    "save_tables",
    "save_metrics",
]
