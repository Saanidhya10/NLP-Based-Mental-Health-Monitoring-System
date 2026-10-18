"""Metrics, confusion matrices, result logging and latency measurement (paper, Sec. 6).

Macro-F1 is the headline metric: every one of the 8 classes counts equally, including classes with
zero predictions (their F1 is 0, which is exactly the penalty we want for ignoring rare emotions).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from src.utils import LABEL2ID, LABELS, ensure_dir, get_logger, path_from_cfg, strategy_slug

log = get_logger("evaluation")


def _to_ids(y: Sequence) -> np.ndarray:
    y = list(y)
    if y and isinstance(y[0], str):
        return np.array([LABEL2ID[v] for v in y])
    return np.asarray(y, dtype=int)


def compute_metrics(y_true: Sequence, y_pred: Sequence, labels: list[str] = LABELS) -> dict:
    yt, yp = _to_ids(y_true), _to_ids(y_pred)
    ids = list(range(len(labels)))
    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=ids, zero_division=0)
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "macro_f1": float(f.mean()),
        "weighted_f1": float(f1_score(yt, yp, labels=ids, average="weighted", zero_division=0)),
        "min_class_f1": float(f.min()),
        "per_class": {
            labels[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i in ids
        },
    }


def per_class_report(y_true, y_pred, labels: list[str] = LABELS) -> pd.DataFrame:
    m = compute_metrics(y_true, y_pred, labels)["per_class"]
    return pd.DataFrame(m).T.rename_axis("class").reset_index()


def confusion(y_true, y_pred, labels: list[str] = LABELS, normalize: str | None = None) -> np.ndarray:
    return confusion_matrix(_to_ids(y_true), _to_ids(y_pred), labels=list(range(len(labels))), normalize=normalize)


def plot_confusion_matrix(cm: np.ndarray, path: str | Path, labels: list[str] = LABELS, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            ax.text(j, i, f"{v:.2f}" if cm.dtype.kind == "f" else int(v), ha="center", va="center",
                    color="white" if v > thresh else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    ensure_dir(Path(path).parent)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def log_run(csv_path: str | Path, model: str, strategy: str, seed: int | None, split: str, metrics: dict,
            extra: dict | None = None) -> None:
    """Append one row (overall + per-class F1) to a results CSV."""
    row = {"model": model, "strategy": strategy, "seed": seed, "split": split}
    row.update({k: v for k, v in metrics.items() if k != "per_class"})
    for c, d in metrics["per_class"].items():
        row[f"f1_{c}"] = d["f1"]
    if extra:
        row.update(extra)
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


def report_run(cfg: dict, model: str, strategy: str, seed: int | None, y_true, y_pred, split: str = "test",
               save_cm: bool = True, extra: dict | None = None) -> dict:
    """Compute metrics, log them to results/model_metrics.csv, optionally save a confusion matrix."""
    m = compute_metrics(y_true, y_pred)
    log_run(path_from_cfg(cfg, "results_dir", create=True) / "model_metrics.csv", model, strategy, seed, split, m, extra)
    log.info("[%s | %s | seed=%s | %s] acc=%.4f macro-F1=%.4f min-class-F1=%.4f",
             model, strategy, seed, split, m["accuracy"], m["macro_f1"], m["min_class_f1"])
    if save_cm and split == "test":
        fig = path_from_cfg(cfg, "figures_dir", create=True) / f"cm_{model}_{strategy_slug(strategy)}_seed{seed}.png"
        plot_confusion_matrix(confusion(y_true, y_pred, normalize="true"), fig, title=f"{model} / {strategy}")
    return m


def aggregate_seeds(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over seeds for each (model, strategy, split) - used for the comparison table."""
    num = [c for c in metrics_df.columns if c not in {"model", "strategy", "seed", "split"}
           and pd.api.types.is_numeric_dtype(metrics_df[c])]
    g = metrics_df.groupby(["model", "strategy", "split"])[num]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std"))
    return out.reset_index()


def imbalance_table(metrics_df: pd.DataFrame, model: str, split: str = "test") -> pd.DataFrame:
    """Per-class F1 per mitigation strategy (mean over seeds) plus delta vs. the baseline row."""
    d = metrics_df[(metrics_df["model"] == model) & (metrics_df["split"] == split)]
    cols = [f"f1_{c}" for c in LABELS] + ["macro_f1", "accuracy"]
    tab = d.groupby("strategy")[cols].mean()
    if "baseline" in tab.index:
        delta = tab.sub(tab.loc["baseline"], axis=1).add_prefix("delta_")
        tab = tab.join(delta[[f"delta_f1_{c}" for c in LABELS] + ["delta_macro_f1"]])
    return tab.reset_index()


def plot_per_class_f1(table: pd.DataFrame, path: str | Path, title: str = "Per-class F1 by strategy") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strategies = table["strategy"].tolist()
    x = np.arange(len(LABELS))
    w = 0.8 / max(len(strategies), 1)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, s in enumerate(strategies):
        vals = [table.loc[table["strategy"] == s, f"f1_{c}"].iloc[0] for c in LABELS]
        ax.bar(x + i * w, vals, w, label=s)
    ax.set_xticks(x + w * (len(strategies) - 1) / 2, LABELS)
    ax.set_ylabel("F1")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    ensure_dir(Path(path).parent)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def hf_compute_metrics(eval_pred) -> dict:
    """compute_metrics callback for the Hugging Face Trainer (drives early stopping on macro-F1)."""
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    m = compute_metrics(labels, preds)
    return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}


def measure_latency(fn: Callable[[str], object], texts: Sequence[str], n_warmup: int = 3, n_runs: int = 50) -> dict:
    """Per-call latency (ms) of ``fn(text)`` over ``n_runs`` calls, cycling through ``texts``."""
    texts = list(texts)
    for i in range(n_warmup):
        fn(texts[i % len(texts)])
    times = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        fn(texts[i % len(texts)])
        times.append((time.perf_counter() - t0) * 1000)
    a = np.array(times)
    return {"mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "n_runs": n_runs}
