"""Shared helpers: config loading, seeding, logging, label constants, small IO utilities."""
from __future__ import annotations

import json
import logging
import random
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

# Single source of truth for the 8-class taxonomy (paper, Sec. 3.2). Order defines label ids.
LABELS = ["Joy", "Sadness", "Anger", "Fear", "Anxiety", "Stress", "Surprise", "Neutral"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
NEGATIVE_LABELS = ["Sadness", "Anger", "Fear", "Anxiety", "Stress"]  # set N in Sec. 5.6
NUM_CLASSES = len(LABELS)


def load_config(path: str | Path | None = None) -> dict:
    import yaml

    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(p: str | Path) -> Path:
    """Resolve a (possibly relative) path against the project root."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def path_from_cfg(cfg: dict, key: str, create: bool = False) -> Path:
    p = resolve_path(cfg["paths"][key])
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logger(name: str = "mhnlp", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(level)
    return logger


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def strategy_slug(strategy: str) -> str:
    """'class_weight+augment' -> 'class_weight_augment' (safe for file names)."""
    return re.sub(r"[^a-zA-Z0-9_]+", "_", strategy).strip("_")


def model_dir(cfg: dict, family: str, strategy: str, seed: int | None = None, create: bool = False) -> Path:
    """models/<family>/<strategy>[_seed<k>]  (transformers may use a Drive root, see config)."""
    root = path_from_cfg(cfg, "models_dir")
    name = strategy_slug(strategy) + (f"_seed{seed}" if seed is not None else "")
    d = root / family / name
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


@contextmanager
def timer(label: str = "", logger: logging.Logger | None = None):
    t0 = time.perf_counter()
    yield
    msg = f"{label} took {time.perf_counter() - t0:.2f}s"
    (logger.info if logger else print)(msg)
