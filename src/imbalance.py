"""Class-imbalance mitigation (paper, Sec. 5.4). Everything here touches the TRAINING split only.

Strategies (combine with '+', e.g. "class_weight+augment"):
    baseline      - nothing (control condition)
    class_weight  - loss weight w_c = N / (|E| * n_c)       (changes the objective, not the data)
    oversample    - random duplication of minority-class rows
    augment       - synonym replacement + back-translation of minority rows (src/augmentation.py)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import ID2LABEL, LABELS, NUM_CLASSES, get_logger, load_config

log = get_logger("imbalance")

KNOWN_COMPONENTS = {"baseline", "class_weight", "oversample", "augment"}


def parse_strategy(strategy: str) -> set[str]:
    parts = {p.strip() for p in strategy.split("+") if p.strip()}
    unknown = parts - KNOWN_COMPONENTS
    if unknown:
        raise ValueError(f"Unknown strategy component(s) {unknown}; choose from {sorted(KNOWN_COMPONENTS)}")
    parts.discard("baseline")
    return parts


def compute_class_weights(y, num_classes: int = NUM_CLASSES) -> dict[int, float]:
    """w_c = N / (|E| * n_c). Classes absent from y get the largest observed weight."""
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=num_classes).astype(float)
    n = counts.sum()
    w = {c: n / (num_classes * counts[c]) for c in range(num_classes) if counts[c] > 0}
    fill = max(w.values()) if w else 1.0
    return {c: w.get(c, fill) for c in range(num_classes)}


def target_size(counts: pd.Series, reference: str = "median", ratio: float = 1.0) -> int:
    ref = counts.max() if reference == "max" else int(counts.median())
    return max(int(ref * ratio), 1)


def random_oversample(train_df: pd.DataFrame, cfg: dict | None = None, seed: int = 42) -> pd.DataFrame:
    """Duplicate minority-class rows (with replacement) until each reaches the target size.
    ``max_factor`` caps duplication so a 100-row class is not copied 160x (severe overfitting)."""
    cfg = cfg or load_config()
    oc = cfg["imbalance"]["oversample"]
    counts = train_df["label_id"].value_counts()
    tgt = target_size(counts, oc.get("reference", "median"), oc.get("ratio", 1.0))
    max_factor = oc.get("max_factor", 30)
    rng = np.random.default_rng(seed)
    parts = [train_df]
    for cid, n in counts.items():
        goal = min(tgt, int(n * max_factor))
        if n < goal:
            pool = train_df[train_df["label_id"] == cid]
            extra = pool.iloc[rng.integers(0, n, size=goal - n)].copy()
            extra["source"] = extra["source"].astype(str) + "+oversampled"
            parts.append(extra)
    out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def distribution_table(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Class counts before/after a mitigation step (paper: 'tabulated and reported before and after')."""
    b = before["label_id"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
    a = after["label_id"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
    t = pd.DataFrame({"class": LABELS, "before": b.values, "after": a.values})
    t["share_before_%"] = (100 * t["before"] / t["before"].sum()).round(2)
    t["share_after_%"] = (100 * t["after"] / t["after"].sum()).round(2)
    return t


def prepare_training_data(train_df: pd.DataFrame, strategy: str, cfg: dict | None = None, seed: int = 42
                          ) -> tuple[pd.DataFrame, dict[int, float] | None]:
    """Apply a strategy to the training split. Returns (training_df, class_weights or None).

    Class weights, when requested, are computed on the FINAL (post-resampling) training set.
    Validation and test data must never be passed through here.
    """
    cfg = cfg or load_config()
    parts = parse_strategy(strategy)
    df = train_df
    if "oversample" in parts:
        df = random_oversample(df, cfg, seed)
    if "augment" in parts:
        from src.augmentation import augment_minority  # lazy: heavy optional deps

        df = augment_minority(df, cfg, seed)
    weights = compute_class_weights(df["label_id"]) if "class_weight" in parts else None
    if df is not train_df:
        log.info("Strategy '%s': %d -> %d training rows", strategy, len(train_df), len(df))
        log.info("\n%s", distribution_table(train_df, df).to_string(index=False))
    if weights:
        log.info("Class weights: %s", {ID2LABEL[k]: round(v, 3) for k, v in weights.items()})
    return df, weights
