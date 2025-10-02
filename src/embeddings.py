"""Sentence embedding utilities with offline fallback."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:  # optional heavy imports
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore

from .utils import CONSOLE

LOGGER = logging.getLogger(__name__)


def load_embedding_model(model_name: str, cache_dir: str | None = None) -> Optional["SentenceTransformer"]:
    if SentenceTransformer is None:
        LOGGER.warning("sentence-transformers package is unavailable; falling back to TF-IDF.")
        return None
    cache_path = Path(cache_dir).expanduser() if cache_dir else None
    try:
        model = SentenceTransformer(model_name, cache_folder=str(cache_path) if cache_path else None)
        return model
    except Exception as exc:  # pragma: no cover - network/hf errors
        LOGGER.warning("Failed to load embeddings model %s (%s).", model_name, exc)
        CONSOLE.print(
            "[yellow]Embedding model unavailable. Falling back to TF-IDF features.[/yellow]"
        )
        return None


def encode_texts(
    texts: Iterable[str],
    model: "SentenceTransformer" | None,
    batch_size: int = 32,
    normalize: bool = False,
) -> Optional[np.ndarray]:
    if model is None:
        return None
    try:
        embeddings = model.encode(list(texts), batch_size=batch_size, show_progress_bar=False, normalize_embeddings=normalize)
        return np.asarray(embeddings, dtype=np.float32)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Encoding failed (%s). Falling back to TF-IDF.", exc)
        return None


__all__ = ["load_embedding_model", "encode_texts"]
