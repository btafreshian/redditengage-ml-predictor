"""Fast text cleaning utilities."""
from __future__ import annotations

import re

URL_RE = re.compile(r"https?://\S+")
CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF]")


def strip_code_blocks(text: str) -> str:
    return CODE_BLOCK_RE.sub(" ", text)


def clean_text(text: str, lowercase: bool = True) -> str:
    if not isinstance(text, str):
        return ""
    text = strip_code_blocks(text)
    text = URL_RE.sub(" ", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if lowercase:
        text = text.lower()
    return text


def contains_emoji(text: str) -> bool:
    if not text:
        return False
    return bool(EMOJI_RE.search(text))


__all__ = ["clean_text", "contains_emoji"]
