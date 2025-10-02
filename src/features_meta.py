"""Derive lightweight metadata features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from textstat import textstat

from .text_clean import contains_emoji


@dataclass
class MetaStats:
    subreddit_stats: DataFrame
    author_stats: DataFrame

    def to_dict(self) -> Dict[str, Dict]:
        return {
            "subreddit_stats": self.subreddit_stats.to_dict(orient="index"),
            "author_stats": self.author_stats.to_dict(orient="index"),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Dict]) -> "MetaStats":
        return cls(
            subreddit_stats=pd.DataFrame.from_dict(data.get("subreddit_stats", {}), orient="index"),
            author_stats=pd.DataFrame.from_dict(data.get("author_stats", {}), orient="index"),
        )


def _token_count(text: str) -> int:
    if not isinstance(text, str) or not text:
        return 0
    return len(text.split())


def compute_meta_features(
    df: DataFrame,
    time_column: str,
    text_column: str,
    target_column: str | None = None,
    stats: MetaStats | None = None,
) -> Tuple[DataFrame, MetaStats]:
    result = pd.DataFrame(index=df.index)

    created = pd.to_datetime(df[time_column], unit="s", errors="coerce")
    result["created_hour"] = created.dt.hour.fillna(0).astype(int)
    result["created_weekday"] = created.dt.weekday.fillna(0).astype(int)
    result["created_month"] = created.dt.month.fillna(0).astype(int)

    text = df[text_column].fillna("")
    result["text_length_chars"] = text.str.len()
    result["text_length_tokens"] = text.apply(_token_count)
    result["contains_link"] = text.str.contains("http", regex=False).astype(int)
    result["contains_question"] = text.str.contains(r"\?", regex=True).astype(int)
    result["contains_exclaim"] = text.str.contains(r"!", regex=True).astype(int)
    result["emoji_flag"] = text.apply(lambda x: int(contains_emoji(x)))
    result["caps_ratio"] = text.apply(lambda x: (sum(1 for ch in x if ch.isupper()) / max(len(x), 1)))
    result["digit_ratio"] = text.apply(lambda x: (sum(1 for ch in x if ch.isdigit()) / max(len(x), 1)))
    def _readability(x: str) -> float:
        try:
            return float(textstat.flesch_reading_ease(x or ""))
        except Exception:
            return 0.0

    result["readability"] = text.apply(_readability)

    subreddit_counts = df["subreddit"].value_counts().to_dict()
    author_counts = df["author"].value_counts().to_dict()
    result["subreddit_post_count"] = df["subreddit"].map(subreddit_counts).fillna(0).astype(float)
    result["author_post_count"] = df["author"].map(author_counts).fillna(0).astype(float)

    if target_column and target_column in df.columns:
        subreddit_stats = df.groupby("subreddit")[target_column].agg(["mean", "median", "count"])
        author_stats = df.groupby("author")[target_column].agg(["mean", "median", "count"])
        stats = MetaStats(subreddit_stats=subreddit_stats, author_stats=author_stats)
    elif stats is None:
        stats = MetaStats(subreddit_stats=pd.DataFrame(), author_stats=pd.DataFrame())

    def _lookup(frame: DataFrame, key: str, column: str) -> float:
        if frame.empty or key not in frame.index:
            return float("nan")
        return float(frame.loc[key, column])

    result["subreddit_target_mean"] = df["subreddit"].apply(
        lambda x: _lookup(stats.subreddit_stats, x, "mean")
    ).fillna(result["subreddit_post_count"].median())
    result["author_target_mean"] = df["author"].apply(
        lambda x: _lookup(stats.author_stats, x, "mean")
    ).fillna(result["author_post_count"].median())

    result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return result, stats


__all__ = ["compute_meta_features", "MetaStats"]
