"""Data loading utilities including synthetic data generation."""
from __future__ import annotations
import math
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from pandas import DataFrame

from .text_clean import clean_text
from .utils import CONSOLE, ensure_dir


def create_synthetic_sample(n: int = 3000, seed: int = 1337) -> DataFrame:
    """Create a moderately realistic Reddit style synthetic dataset."""
    rng = np.random.default_rng(seed)
    random.seed(seed)

    base_time = datetime.utcnow() - timedelta(days=90)
    subreddits = [
        "python",
        "datascience",
        "machinelearning",
        "learnprogramming",
        "AskReddit",
        "funny",
    ]
    authors = [f"user_{i}" for i in range(120)]

    templates = [
        "I just built a {project} using {tool}! Any tips for improving performance?",
        "What's the best way to learn {topic} in 2024?",
        "Here's a quick tutorial on {topic} with code examples."
        " Feedback appreciated!",
        "Do you prefer using {tool} or {alt_tool} for {task}?",
        "Share your funniest bug involving {tool}! 😂",
        "PSA: {tip} just saved me hours when debugging {task}.",
        "Link to my latest write-up: https://example.com/{project}",
        "{emoji} {emoji} Can't believe this worked with only {lines} lines of code!",
    ]

    topics = [
        "neural networks",
        "data pipelines",
        "web scraping",
        "unit testing",
        "LLMs",
        "APIs",
    ]
    tools = ["Python", "Rust", "Pandas", "Spark", "LightGBM", "TensorFlow"]
    alt_tools = ["R", "Scala", "SQL", "PyTorch", "XGBoost", "CatBoost"]
    tasks = ["feature engineering", "model tuning", "deployment", "ETL", "visualisation"]
    tips = ["log your experiments", "seed everything", "write docstrings", "profile your code"]
    emojis = ["🚀", "🔥", "💡", "🤖", "🙃", "✨"]

    subreddit_multipliers = {
        "python": 1.0,
        "datascience": 1.2,
        "machinelearning": 1.6,
        "learnprogramming": 0.8,
        "AskReddit": 2.0,
        "funny": 2.5,
    }

    rows: List[dict] = []
    for i in range(n):
        subreddit = rng.choice(subreddits)
        author = rng.choice(authors)
        topic = rng.choice(topics)
        tool = rng.choice(tools)
        alt_tool = rng.choice(alt_tools)
        task = rng.choice(tasks)
        tip = rng.choice(tips)
        emoji = rng.choice(emojis)
        template = rng.choice(templates)
        body = template.format(
            project=topic.replace(" ", "-"),
            tool=tool,
            alt_tool=alt_tool,
            topic=topic,
            task=task,
            tip=tip,
            emoji=emoji,
            lines=rng.integers(10, 500),
        )

        created_offset = timedelta(hours=float(rng.integers(0, 24 * 90)), minutes=float(rng.integers(0, 60)))
        created_utc = base_time + created_offset
        base_up = rng.lognormal(mean=math.log(15 + subreddit_multipliers[subreddit]), sigma=1.1)
        author_factor = 1.0 + 0.2 * math.sin(hash(author) % 10)
        body_factor = 1.0 + len(body) / 400.0
        noise = rng.lognormal(mean=0.0, sigma=0.8)
        ups = max(0, int(base_up * subreddit_multipliers[subreddit] * author_factor * body_factor * noise))

        rows.append(
            {
                "id": f"t3_{i:06d}",
                "subreddit": subreddit,
                "author": author,
                "created_utc": int(created_utc.timestamp()),
                "body": body,
                "ups": ups,
            }
        )

    df = pd.DataFrame(rows)
    df["body"] = df["body"].apply(clean_text)
    return df
def load_or_generate(cfg) -> DataFrame:
    raw_path = Path(cfg.raw_path)
    generated_path = Path(cfg.generated_path)

    if raw_path.exists():
        CONSOLE.print(f"[bold green]Loading dataset[/bold green] from {raw_path}")
        df = pd.read_json(raw_path, lines=True)
    elif generated_path.exists():
        CONSOLE.print(f"[bold cyan]Loading cached synthetic dataset[/bold cyan] from {generated_path}")
        df = pd.read_json(generated_path, lines=True)
    elif cfg.use_synthetic:
        CONSOLE.print("[yellow]No dataset found. Generating synthetic sample...[/yellow]")
        df = create_synthetic_sample(cfg.synthetic_rows, cfg.synthetic_seed)
        ensure_dir(generated_path.parent)
        df.to_json(generated_path, lines=True, orient="records")
        CONSOLE.print(f"[green]Synthetic dataset written to[/green] {generated_path}")
    else:
        raise FileNotFoundError(
            f"Expected dataset at {raw_path}, but file is missing and synthetic generation disabled."
        )
    return df


def save_prepared_dataset(df: DataFrame, path: Path) -> Path:
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)
    return path


__all__ = ["create_synthetic_sample", "load_or_generate", "save_prepared_dataset"]
