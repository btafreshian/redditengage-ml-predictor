"""Command line interface for the project."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import mlflow
import numpy as np
import pandas as pd
import typer
from rich import box
from rich.table import Table
from sklearn.model_selection import train_test_split

from .data import load_or_generate, save_prepared_dataset
from .embeddings import encode_texts, load_embedding_model
from .eval import build_tables, compute_regression_metrics, generate_figures, save_metrics, save_tables
from .features_meta import compute_meta_features
from .models.baseline import train_baseline
from .models.embed_reg import train_embedding_regressor
from .models.gbdt import train_gbdt
from .report import build_html_report
from .text_clean import clean_text
from .utils import (
    CONSOLE,
    config_to_dict,
    configure_logging,
    ensure_dir,
    infer_override_help,
    load_config,
    load_latest_run,
    save_latest_run,
)

app = typer.Typer(help="Reddit upvote prediction toolkit")


def _load_dataframe(data_cfg) -> pd.DataFrame:
    prepared_path = Path(data_cfg.prepared_path)
    if prepared_path.exists():
        return pd.read_parquet(prepared_path)
    df = load_or_generate(data_cfg)
    text_col = data_cfg.text_column
    df[text_col] = df[text_col].apply(clean_text)
    save_prepared_dataset(df, prepared_path)
    return df


def _split_dataset(df: pd.DataFrame, target_col: str, test_size: float, random_state: int):
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col] > df[target_col].median(),
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


@contextmanager
def mlflow_run(train_cfg, run_name: str, params: Dict):
    mlflow.set_tracking_uri(str(train_cfg.mlflow_tracking_uri))
    mlflow.set_experiment(str(train_cfg.experiment_name))
    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params(params)
        yield run


def _summarise_config(data_cfg, train_cfg, model_name: str) -> Dict[str, str]:
    return {
        "data_prepared_path": str(data_cfg.prepared_path),
        "model": model_name,
        "test_size": train_cfg.test_size,
        "log1p_target": bool(train_cfg.log1p_target),
        "timestamp": datetime.utcnow().isoformat(),
    }


def _create_run_dir(train_cfg, model_name: str) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(Path(train_cfg.artifacts_dir) / f"{model_name}_{ts}")
    return run_dir


def _display_metrics(metrics: Dict[str, float]) -> None:
    table = Table(title="Evaluation Metrics", box=box.SIMPLE_HEAVY)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, value in metrics.items():
        table.add_row(key.upper(), f"{value:.4f}")
    CONSOLE.print(table)


@app.command()
def prepare(
    overrides: List[str] = typer.Option([], "--override", "-o", help=infer_override_help()),
) -> None:
    """Generate or load a dataset and persist the cleaned parquet file."""
    configure_logging()
    data_cfg = load_config("data", overrides)
    df = _load_dataframe(data_cfg)
    sample_path = ensure_dir(Path(data_cfg.cache_dir)) / "sample_preview.json"
    df.head(20).to_json(sample_path, orient="records", lines=True)
    CONSOLE.print(f"[green]Prepared dataset available at[/green] {data_cfg.prepared_path}")
    CONSOLE.print(f"[green]Preview saved to[/green] {sample_path}")


def _baseline_training_pipeline(
    df: pd.DataFrame,
    data_cfg,
    train_cfg,
    model_cfg,
    run_name: str,
) -> Dict:
    train_df, test_df = _split_dataset(df, data_cfg.target_column, train_cfg.test_size, train_cfg.random_state)
    result = train_baseline(
        train_df[data_cfg.text_column],
        train_df[data_cfg.target_column],
        test_df[data_cfg.text_column],
        test_df[data_cfg.target_column],
        config_to_dict(model_cfg.vectorizer),
        config_to_dict(model_cfg.model),
        log1p_target=bool(train_cfg.log1p_target),
    )

    run_dir = _create_run_dir(train_cfg, run_name)
    result.save(run_dir)

    predictions_df = test_df[[
        data_cfg.id_column,
        data_cfg.subreddit_column,
        data_cfg.author_column,
        data_cfg.time_column,
        data_cfg.text_column,
        data_cfg.target_column,
    ]].copy()
    predictions_df.rename(columns={data_cfg.target_column: "ups", data_cfg.text_column: "body"}, inplace=True)
    predictions_df["prediction"] = result.predictions
    predictions_df["prediction_log"] = result.predictions_log

    predictions_path = run_dir / "predictions.parquet"
    predictions_df.to_parquet(predictions_path, index=False)

    estimator = list(result.model.named_steps.values())[-1]
    names = result.vectorizer.feature_names()
    coef = getattr(estimator, "coef_", np.zeros(len(names)))
    coef_df = pd.DataFrame({"feature": names, "coefficient": coef})
    coef_path = run_dir / "ridge_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)

    metrics = compute_regression_metrics(predictions_df["ups"], predictions_df["prediction"])
    metrics_path = run_dir / "metrics.json"
    save_metrics(metrics, metrics_path)

    _display_metrics(metrics)

    return {
        "run_dir": str(run_dir),
        "predictions_path": str(predictions_path),
        "metrics": metrics,
        "metrics_path": str(metrics_path),
        "coefficients_path": str(coef_path),
        "model_type": run_name,
    }


@app.command("train-baseline")
def train_baseline_cli(
    overrides: List[str] = typer.Option([], "--override", "-o", help=infer_override_help()),
) -> None:
    """Train the TF-IDF + linear baseline."""
    configure_logging()
    data_cfg = load_config("data", overrides)
    train_cfg = load_config("train", overrides)
    model_cfg = load_config("model_baseline", overrides)

    df = _load_dataframe(data_cfg)
    params = config_to_dict(model_cfg.model)
    run_name = "baseline"

    with mlflow_run(train_cfg, run_name, params) as run:
        metadata = _baseline_training_pipeline(df, data_cfg, train_cfg, model_cfg, run_name)
        mlflow.log_metrics(metadata["metrics"])
        mlflow.log_artifact(metadata["predictions_path"])
        mlflow.log_artifact(metadata["coefficients_path"])
        mlflow.log_artifact(metadata["metrics_path"])
        metadata.update(
            {
                "mlflow_run_id": run.info.run_id,
                "mlflow_tracking_uri": str(train_cfg.mlflow_tracking_uri),
                "mlflow_experiment": str(train_cfg.experiment_name),
            }
        )

    save_latest_run(metadata)
    CONSOLE.print("[green]Baseline training complete.[/green]")


@app.command("train-embed")
def train_embed_cli(
    overrides: List[str] = typer.Option([], "--override", "-o", help=infer_override_help()),
) -> None:
    """Train an embedding based regressor with TF-IDF fallback."""
    configure_logging()
    data_cfg = load_config("data", overrides)
    train_cfg = load_config("train", overrides)
    model_cfg = load_config("model_embed", overrides)

    df = _load_dataframe(data_cfg)
    train_df, test_df = _split_dataset(df, data_cfg.target_column, train_cfg.test_size, train_cfg.random_state)

    meta_train, stats = compute_meta_features(
        train_df,
        data_cfg.time_column,
        data_cfg.text_column,
        target_column=data_cfg.target_column,
    )
    meta_test, _ = compute_meta_features(
        test_df,
        data_cfg.time_column,
        data_cfg.text_column,
        target_column=None,
        stats=stats,
    )

    model = load_embedding_model(model_cfg.model_name, model_cfg.cache_dir)
    train_embeddings = encode_texts(train_df[data_cfg.text_column], model, batch_size=model_cfg.batch_size)
    test_embeddings = encode_texts(test_df[data_cfg.text_column], model, batch_size=model_cfg.batch_size)

    if train_embeddings is None or test_embeddings is None:
        CONSOLE.print("[yellow]Falling back to TF-IDF baseline due to embedding availability.[/yellow]")
        baseline_cfg = load_config("model_baseline", overrides)
        metadata = _baseline_training_pipeline(df, data_cfg, train_cfg, baseline_cfg, "baseline_fallback")
        metadata.update(
            {
                "mlflow_run_id": None,
                "mlflow_tracking_uri": str(train_cfg.mlflow_tracking_uri),
                "mlflow_experiment": str(train_cfg.experiment_name),
            }
        )
        save_latest_run(metadata)
        CONSOLE.print("[green]Baseline fallback training complete.[/green]")
        return

    result = train_embedding_regressor(
        train_embeddings,
        train_df[data_cfg.target_column].to_numpy(),
        test_embeddings,
        test_df[data_cfg.target_column].to_numpy(),
        config_to_dict(model_cfg.regressor),
        log1p_target=bool(train_cfg.log1p_target),
        train_meta=meta_train,
        test_meta=meta_test,
    )

    run_dir = _create_run_dir(train_cfg, "embed")
    result.save(run_dir)

    predictions_df = test_df[[
        data_cfg.id_column,
        data_cfg.subreddit_column,
        data_cfg.author_column,
        data_cfg.time_column,
        data_cfg.text_column,
        data_cfg.target_column,
    ]].copy()
    predictions_df.rename(columns={data_cfg.target_column: "ups", data_cfg.text_column: "body"}, inplace=True)
    predictions_df["prediction"] = result.y_pred
    predictions_df["prediction_log"] = result.y_pred_log

    predictions_path = run_dir / "predictions.parquet"
    predictions_df.to_parquet(predictions_path, index=False)

    metrics = compute_regression_metrics(predictions_df["ups"], predictions_df["prediction"])
    metrics_path = run_dir / "metrics.json"
    save_metrics(metrics, metrics_path)
    _display_metrics(metrics)

    with mlflow_run(train_cfg, "embed", config_to_dict(model_cfg.regressor)) as run:
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(predictions_path)
        mlflow.log_artifact(metrics_path)
        metadata = {
            "run_dir": str(run_dir),
            "predictions_path": str(predictions_path),
            "metrics": metrics,
            "metrics_path": str(metrics_path),
            "model_type": "embed",
            "mlflow_run_id": run.info.run_id,
            "mlflow_tracking_uri": str(train_cfg.mlflow_tracking_uri),
            "mlflow_experiment": str(train_cfg.experiment_name),
        }

    save_latest_run(metadata)
    CONSOLE.print("[green]Embedding model training complete.[/green]")


@app.command("train-gbdt")
def train_gbdt_cli(
    overrides: List[str] = typer.Option([], "--override", "-o", help=infer_override_help()),
) -> None:
    """Train a LightGBM regressor on meta + embedding features."""
    configure_logging()
    data_cfg = load_config("data", overrides)
    train_cfg = load_config("train", overrides)
    model_cfg = load_config("model_gbdt", overrides)
    embed_cfg = load_config("model_embed", overrides)

    df = _load_dataframe(data_cfg)
    train_df, test_df = _split_dataset(df, data_cfg.target_column, train_cfg.test_size, train_cfg.random_state)

    meta_train, stats = compute_meta_features(
        train_df,
        data_cfg.time_column,
        data_cfg.text_column,
        target_column=data_cfg.target_column,
    )
    meta_test, _ = compute_meta_features(
        test_df,
        data_cfg.time_column,
        data_cfg.text_column,
        target_column=None,
        stats=stats,
    )

    model = load_embedding_model(embed_cfg.model_name, embed_cfg.cache_dir)
    train_embeddings = encode_texts(train_df[data_cfg.text_column], model, batch_size=embed_cfg.batch_size)
    test_embeddings = encode_texts(test_df[data_cfg.text_column], model, batch_size=embed_cfg.batch_size)

    if train_embeddings is None or test_embeddings is None:
        CONSOLE.print("[yellow]Embeddings unavailable. Using meta features only for GBDT.[/yellow]")
        train_embeddings = np.zeros((len(train_df), 0))
        test_embeddings = np.zeros((len(test_df), 0))

    meta_array_train = meta_train.to_numpy(dtype=np.float32)
    meta_array_test = meta_test.to_numpy(dtype=np.float32)
    X_train = np.hstack([train_embeddings, meta_array_train])
    X_test = np.hstack([test_embeddings, meta_array_test])

    embed_feature_names = [f"embed_{i}" for i in range(train_embeddings.shape[1])]
    meta_feature_names = [f"meta_{col}" for col in meta_train.columns]
    feature_names = embed_feature_names + meta_feature_names

    result = train_gbdt(
        X_train,
        train_df[data_cfg.target_column].to_numpy(),
        X_test,
        test_df[data_cfg.target_column].to_numpy(),
        config_to_dict(model_cfg),
        log1p_target=bool(train_cfg.log1p_target),
        quantiles=list(train_cfg.quantiles),
    )

    run_dir = _create_run_dir(train_cfg, "gbdt")
    result.save(run_dir)

    predictions_df = test_df[[
        data_cfg.id_column,
        data_cfg.subreddit_column,
        data_cfg.author_column,
        data_cfg.time_column,
        data_cfg.text_column,
        data_cfg.target_column,
    ]].copy()
    predictions_df.rename(columns={data_cfg.target_column: "ups", data_cfg.text_column: "body"}, inplace=True)
    predictions_df["prediction"] = result.y_pred
    predictions_df["prediction_log"] = result.y_pred_log

    quantile_cols: Dict[str, tuple[str, str]] = {}
    if result.quantile_predictions:
        for name, values in result.quantile_predictions.items():
            predictions_df[f"prediction_{name}"] = values

        def parse_alpha(key: str) -> float:
            part = key.split("_")[-1].replace("p", ".")
            try:
                return float(part)
            except ValueError:
                return float("nan")

        ordered = sorted(result.quantile_predictions.keys(), key=parse_alpha)
        for lower_key, upper_key in zip(ordered, reversed(ordered)):
            if parse_alpha(lower_key) >= parse_alpha(upper_key):
                break
            quantile_cols[f"{lower_key}_to_{upper_key}"] = (
                f"prediction_{lower_key}",
                f"prediction_{upper_key}",
            )

    predictions_path = run_dir / "predictions.parquet"
    predictions_df.to_parquet(predictions_path, index=False)

    metrics = compute_regression_metrics(predictions_df["ups"], predictions_df["prediction"])
    metrics_path = run_dir / "metrics.json"
    save_metrics(metrics, metrics_path)
    _display_metrics(metrics)

    importances = pd.Series(result.model.feature_importances_, index=feature_names)
    importances_path = run_dir / "feature_importances.csv"
    importances.to_csv(importances_path, header=["importance"])

    with mlflow_run(train_cfg, "gbdt", config_to_dict(model_cfg.lightgbm)) as run:
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(predictions_path)
        mlflow.log_artifact(importances_path)
        mlflow.log_artifact(metrics_path)
        metadata = {
            "run_dir": str(run_dir),
            "predictions_path": str(predictions_path),
            "metrics": metrics,
            "metrics_path": str(metrics_path),
            "model_type": "gbdt",
            "mlflow_run_id": run.info.run_id,
            "mlflow_tracking_uri": str(train_cfg.mlflow_tracking_uri),
            "mlflow_experiment": str(train_cfg.experiment_name),
            "importances_path": str(importances_path),
            "quantile_pairs": {k: list(v) for k, v in quantile_cols.items()},
        }

    save_latest_run(metadata)
    CONSOLE.print("[green]GBDT model training complete.[/green]")


@app.command()
def evaluate(
    overrides: List[str] = typer.Option([], "--override", "-o", help=infer_override_help()),
) -> None:
    """Evaluate the most recent run and generate diagnostic figures."""
    configure_logging()
    train_cfg = load_config("train", overrides)
    metadata = load_latest_run()
    predictions_path = Path(metadata["predictions_path"])
    predictions = pd.read_parquet(predictions_path)

    metrics = compute_regression_metrics(predictions["ups"], predictions["prediction"])
    figures_dir = Path(train_cfg.figures_dir)

    importances = None
    ridge_terms = None
    interval_cols = metadata.get("quantile_pairs")

    if metadata.get("importances_path") and Path(metadata["importances_path"]).exists():
        importances_df = pd.read_csv(metadata["importances_path"], index_col=0)
        importances = importances_df.iloc[:, 0]
    if metadata.get("coefficients_path") and Path(metadata["coefficients_path"]).exists():
        ridge_df = pd.read_csv(metadata["coefficients_path"])
        ridge_terms = ridge_df.set_index("feature")["coefficient"]

    generate_figures(predictions, figures_dir, importances, ridge_terms, interval_cols)
    tables = build_tables(predictions)
    run_dir = Path(metadata["run_dir"])
    table_paths = save_tables(tables, run_dir)
    save_metrics(metrics, run_dir / "metrics_eval.json")
    _display_metrics(metrics)

    metadata.update(
        {
            "evaluation_metrics": metrics,
            "tables": table_paths,
            "figures_dir": str(figures_dir),
        }
    )
    save_latest_run(metadata)
    CONSOLE.print("[green]Evaluation complete. Figures available in[/green]" f" {figures_dir}")


@app.command()
def report(
    overrides: List[str] = typer.Option([], "--override", "-o", help=infer_override_help()),
) -> None:
    """Build the polished HTML report for the latest run."""
    configure_logging()
    data_cfg = load_config("data", overrides)
    train_cfg = load_config("train", overrides)
    metadata = load_latest_run()

    predictions = pd.read_parquet(metadata["predictions_path"])
    metrics = metadata.get("evaluation_metrics") or compute_regression_metrics(
        predictions["ups"], predictions["prediction"]
    )

    figures = [
        {"title": "Predicted vs Actual", "src": "../figures/pred_vs_actual.png"},
        {"title": "Residuals vs Fitted", "src": "../figures/residuals_vs_fitted.png"},
        {"title": "Residual Distribution", "src": "../figures/error_distribution.png"},
        {"title": "QQ Plot", "src": "../figures/qq_plot.png"},
        {"title": "Feature Importance", "src": "../figures/feature_importance_gbdt.png"},
        {"title": "Ridge Coefficients", "src": "../figures/coef_ridge_top_terms.png"},
        {"title": "Error by Subreddit", "src": "../figures/error_by_subreddit.png"},
        {"title": "Error by Length", "src": "../figures/error_by_length.png"},
        {"title": "Interval Coverage", "src": "../figures/interval_coverage.png"},
    ]

    tables = {}
    if metadata.get("tables"):
        for name, path in metadata["tables"].items():
            tables[name] = pd.read_csv(path)
    else:
        tables = build_tables(predictions)

    config_summary = _summarise_config(data_cfg, train_cfg, metadata.get("model_type", "unknown"))
    html_path = ensure_dir(Path(train_cfg.html_dir)) / "latest_report.html"
    mlflow_info = {
        "tracking_uri": metadata.get("mlflow_tracking_uri", "local"),
        "experiment": metadata.get("mlflow_experiment", "reddit_upvotes"),
        "run_id": metadata.get("mlflow_run_id", "N/A"),
    }

    build_html_report(html_path, metrics, config_summary, predictions, tables, figures, mlflow_info)
    CONSOLE.print(f"[green]Report generated at[/green] {html_path}")


if __name__ == "__main__":
    app()
