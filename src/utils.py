"""Utility helpers for configuration management, logging and persistence."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, MutableMapping, Sequence

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.logging import RichHandler

CONSOLE = Console()


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def configure_logging(level: int = logging.INFO) -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=CONSOLE, rich_tracebacks=True, markup=True)],
    )


def _filter_overrides(overrides: Sequence[str], prefix: str) -> List[str]:
    filtered: List[str] = []
    prefix_dot = f"{prefix}."
    for item in overrides:
        if item.startswith(prefix_dot):
            filtered.append(item[len(prefix_dot) :])
    return filtered


def load_config(name: str, overrides: Sequence[str] | None = None) -> DictConfig:
    overrides = overrides or []
    config_dir = get_project_root() / "configs"
    if GlobalHydra().is_initialized():
        GlobalHydra().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=name, overrides=_filter_overrides(overrides, name))
    return cfg


def config_to_dict(cfg: DictConfig) -> Dict:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, data: MutableMapping) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def latest_artifact_path() -> Path:
    return ensure_dir(get_project_root() / "artifacts") / "latest_run.json"


def save_latest_run(metadata: MutableMapping) -> None:
    save_json(latest_artifact_path(), metadata)


def load_latest_run() -> Dict:
    path = latest_artifact_path()
    if not path.exists():
        raise FileNotFoundError("No run metadata found. Train a model first.")
    return load_json(path)


def infer_override_help() -> str:
    return (
        "Hydra style overrides (e.g. --override data.synthetic_rows=500 --override "
        "model_baseline.vectorizer.max_features=1000)."
    )


__all__ = [
    "configure_logging",
    "load_config",
    "config_to_dict",
    "ensure_dir",
    "save_json",
    "load_json",
    "save_latest_run",
    "load_latest_run",
    "infer_override_help",
    "get_project_root",
]
