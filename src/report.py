"""HTML reporting utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
from jinja2 import Environment, select_autoescape

from .utils import ensure_dir


TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Reddit Upvote Predictor Report</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, sans-serif; margin: 2rem; color: #1f2933; }
        h1, h2 { color: #0b7285; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }
        th, td { border: 1px solid #d3dce6; padding: 0.5rem 0.75rem; text-align: left; }
        th { background-color: #f1f5f9; }
        .metrics { display: flex; gap: 1rem; flex-wrap: wrap; }
        .metric-card { background: #edf2ff; padding: 1rem; border-radius: 0.5rem; flex: 1 0 10rem; box-shadow: 0 2px 4px rgba(0,0,0,0.08); }
        .figures { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem; }
        .figure-card { border: 1px solid #d3dce6; border-radius: 0.5rem; overflow: hidden; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
        .figure-card img { width: 100%; }
        .figure-card h3 { margin: 0; padding: 0.5rem; background: #f8f9fa; font-size: 1rem; }
        .mlflow { background: #f8f9fa; padding: 1rem; border-radius: 0.5rem; border: 1px solid #d3dce6; }
        .config { columns: 2; column-gap: 2rem; }
        code { background: #f1f3f5; padding: 0.1rem 0.3rem; border-radius: 0.3rem; }
    </style>
</head>
<body>
    <h1>Reddit Upvote Predictor</h1>
    <p>This report summarises the most recent experiment run. It is designed to be data-driven yet lightweight so it can be regenerated quickly for interviews or demos.</p>

    <h2>Metrics</h2>
    <div class="metrics">
        {% for name, value in metrics.items() %}
        <div class="metric-card">
            <h3>{{ name | upper }}</h3>
            <p style="font-size: 1.6rem; margin: 0;">{{ '%.3f' | format(value) }}</p>
        </div>
        {% endfor %}
    </div>

    <h2>Configuration Snapshot</h2>
    <div class="config">
        {% for key, value in config_summary.items() %}
        <p><strong>{{ key }}</strong>: {{ value }}</p>
        {% endfor %}
    </div>

    <h2>Sample Data</h2>
    {{ sample_html | safe }}

    <h2>Per-subreddit performance</h2>
    {{ subreddit_table | safe }}

    <h2>Temporal stability</h2>
    {{ monthly_table | safe }}

    <h2>Figures</h2>
    <div class="figures">
        {% for figure in figures %}
        <div class="figure-card">
            <h3>{{ figure.title }}</h3>
            <img src="{{ figure.src }}" alt="{{ figure.title }}" />
        </div>
        {% endfor %}
    </div>

    <h2>MLflow Run</h2>
    <div class="mlflow">
        <p><strong>Tracking URI:</strong> {{ mlflow.tracking_uri }}</p>
        <p><strong>Experiment:</strong> {{ mlflow.experiment }}</p>
        <p><strong>Run ID:</strong> {{ mlflow.run_id }}</p>
    </div>
</body>
</html>
"""


def build_html_report(
    output_path: Path,
    metrics: Dict[str, float],
    config_summary: Dict[str, str],
    sample_df: pd.DataFrame,
    tables: Dict[str, pd.DataFrame],
    figures: List[Dict[str, str]],
    mlflow_info: Dict[str, str],
) -> Path:
    ensure_dir(output_path.parent)
    env = Environment(autoescape=select_autoescape(["html", "xml"]))
    template = env.from_string(TEMPLATE)

    sample_html = sample_df.head(10).to_html(index=False)
    subreddit_html = tables.get("subreddit", pd.DataFrame()).to_html(index=False)
    monthly_html = tables.get("monthly", pd.DataFrame()).to_html(index=False)

    html = template.render(
        metrics=metrics,
        config_summary=config_summary,
        sample_html=sample_html,
        subreddit_table=subreddit_html,
        monthly_table=monthly_html,
        figures=figures,
        mlflow=mlflow_info,
    )

    output_path.write_text(html, encoding="utf-8")
    return output_path


__all__ = ["build_html_report"]
