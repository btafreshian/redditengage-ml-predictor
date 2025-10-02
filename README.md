# Reddit Upvote Predictor

A production-ready yet interview-friendly project that predicts the exact Reddit upvote count for a post. The repository focuses on rapid local iteration while maintaining a clear scale-up path for Spark clusters.

## Key features

- **Synthetic bootstrap** – generates a realistic heavy-tailed Reddit sample when no dataset is available. The data generator mirrors the target schema `{id, subreddit, author, created_utc, body, ups}`.
- **Modular feature stack** – text cleaning, meta-features, TF-IDF, and optional sentence-transformer embeddings (with an automatic TF-IDF fallback when offline).
- **Multiple models** – Ridge/Lasso baselines, LightGBM with optional quantile heads, and embedding-based elastic net models, plus a simple ensembling helper.
- **Evaluation suite** – rich diagnostics, SHAP-friendly storage, and polished HTML reports generated with Jinja2. Figures include residual analysis, error breakdowns, and coverage visualisations.
- **Spark path** – PySpark scripts mirror the local workflow for large scale training using HashingTF + IDF features and gradient boosted trees.
- **Experiment tracking** – MLflow logging keeps metrics, parameters, and artefacts discoverable.

## Repository layout

```
.
├── configs/                # Hydra configuration files
├── data/                   # Generated samples and caches (created at runtime)
├── notebooks/              # Interview demo notebook
├── reports/figures/        # Saved plots (generated)
├── reports/html/           # HTML report outputs (generated)
├── src/                    # Library, CLI, and Spark helpers
├── tests/                  # Smoke tests
├── docker/                 # Container recipe
├── Makefile                # Automation helpers
└── requirements.txt        # Python dependencies
```

## Quickstart

```bash
make setup        # create virtualenv and install dependencies
source .venv/bin/activate
make demo         # run prepare → train-embed (fallback safe) → evaluate → report
```

Individual steps are exposed through the Typer CLI:

```bash
python -m src.cli prepare
python -m src.cli train-baseline
python -m src.cli train-embed
python -m src.cli train-gbdt
python -m src.cli evaluate
python -m src.cli report
```

All commands accept Hydra overrides, e.g. `python -m src.cli train-baseline --override data.synthetic_rows=5000`.

## Data pipeline

1. **Ingestion** – `src/data.py` loads an existing JSONL file or synthesises a dataset via `create_synthetic_sample` with configurable size and random seed.
2. **Cleaning** – `src/text_clean.py` strips code fences, URLs, and normalises whitespace. Emoji detection feeds downstream features.
3. **Feature engineering** – `src/features_meta.py` derives temporal, readability, and frequency statistics per author/subreddit while respecting train/test isolation. `src/features_tfidf.py` and `src/embeddings.py` build text representations with caching.
4. **Models** – Baseline Ridge/Lasso (`src/models/baseline.py`), LightGBM (`src/models/gbdt.py`), and embedding regressors (`src/models/embed_reg.py`). Quantile heads deliver prediction intervals.
5. **Evaluation** – `src/eval.py` computes RMSE/MAE/MedianAE/R², per-subreddit/temporal tables, and the required diagnostic plots.
6. **Reporting** – `src/report.py` compiles an HTML dashboard summarising configuration, metrics, and figure thumbnails with MLflow metadata.

## Results snapshot

Once `make demo` completes, the following artefacts are available:

- `reports/figures/*.png` – scatter plots, residual diagnostics, error analyses, and interval coverage (placeholders when a model does not produce the required artefact).
- `reports/html/latest_report.html` – polished report containing metrics, tables, and figure thumbnails.
- `mlruns/` – MLflow experiment tracking data.

## Spark scale-out

For larger datasets, switch to the PySpark utilities:

```bash
python -m src.spark.preprocess_spark INPUT.jsonl OUTPUT.parquet
python -m src.spark.train_spark OUTPUT.parquet spark_model --metrics-output spark_metrics.json
```

`preprocess_spark.py` performs lightweight cleaning, extracts meta features, and produces TF-IDF vectors with HashingTF + IDF. `train_spark.py` fits a `GBTRegressor`, logs metrics, and optionally exports a downsampled CSV for local feature exploration.

## Reproducibility

- Use `make setup` to create an isolated environment (Python 3.10+).
- Hydra ensures every command is reproducible via config overrides.
- Random seeds live in `configs/train.yaml` and `configs/data.yaml`.
- MLflow captures run metadata, metrics, and artefacts for auditability.

## Limitations & future work

- Embedding models fall back to TF-IDF when the sentence-transformer cache is missing; consider pre-downloading the model into `~/.cache/redditengage` for optimal accuracy.
- The synthetic generator approximates Reddit behaviour but should be replaced with real data for production.
- Quantile LightGBM models currently reuse the same feature set; future iterations could include conformal calibration for tighter intervals.
- Spark scripts focus on TF-IDF features. Integrating on-cluster embedding generation (e.g. via UDF + broadcast models) is a natural extension.

## License

The repository intentionally omits a license—add one in downstream forks as required.
