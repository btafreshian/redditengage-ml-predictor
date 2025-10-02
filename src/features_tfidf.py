"""TF-IDF utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import joblib
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfBundle:
    def __init__(
        self,
        word_vectorizer: TfidfVectorizer,
        char_vectorizer: Optional[TfidfVectorizer] = None,
    ) -> None:
        self.word_vectorizer = word_vectorizer
        self.char_vectorizer = char_vectorizer

    def transform(self, texts) -> sparse.csr_matrix:
        matrices = [self.word_vectorizer.transform(texts)]
        if self.char_vectorizer is not None:
            matrices.append(self.char_vectorizer.transform(texts))
        return sparse.hstack(matrices).tocsr()

    def fit_transform(self, texts) -> sparse.csr_matrix:
        matrices = [self.word_vectorizer.fit_transform(texts)]
        if self.char_vectorizer is not None:
            matrices.append(self.char_vectorizer.fit_transform(texts))
        return sparse.hstack(matrices).tocsr()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "word": self.word_vectorizer,
            "char": self.char_vectorizer,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "TfidfBundle":
        data = joblib.load(path)
        return cls(word_vectorizer=data["word"], char_vectorizer=data.get("char"))

    def feature_names(self) -> list[str]:
        names = list(self.word_vectorizer.get_feature_names_out())
        if self.char_vectorizer is not None:
            char_names = [f"char::{n}" for n in self.char_vectorizer.get_feature_names_out()]
            names.extend(char_names)
        return names


def build_tfidf_bundle(cfg: Dict, cache_dir: Path | None = None) -> TfidfBundle:
    cache_dir = Path(cache_dir) if cache_dir else None
    word_vectorizer = TfidfVectorizer(
        lowercase=cfg.get("lowercase", True),
        max_features=cfg.get("max_features"),
        stop_words=cfg.get("stop_words"),
        ngram_range=tuple(cfg.get("ngram_range_word", (1, 1))),
        min_df=cfg.get("min_df", 2),
    )
    char_vectorizer = None
    if cfg.get("use_char", False):
        char_vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=tuple(cfg.get("ngram_range_char", (3, 5))),
            lowercase=False,
            max_features=cfg.get("max_features_char"),
        )
    return TfidfBundle(word_vectorizer=word_vectorizer, char_vectorizer=char_vectorizer)


__all__ = ["TfidfBundle", "build_tfidf_bundle"]
