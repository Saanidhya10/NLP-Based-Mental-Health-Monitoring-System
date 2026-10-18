"""Classical baselines: TF-IDF + Logistic Regression / LinearSVC / Random Forest (paper, Sec. 5.3).

Hyper-parameters are tuned on the VALIDATION split only (Table 4 grids in config.yaml); the test split
is touched once for the final report.

Usage:
    python -m src.train_classical --model all --strategy baseline
    python -m src.train_classical --model logreg --strategy class_weight --quick
"""
from __future__ import annotations

import argparse
import itertools

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.datasets import TEXT_COLUMN, load_splits
from src.evaluation import compute_metrics, report_run
from src.imbalance import prepare_training_data
from src.utils import get_logger, load_config, model_dir, save_json, set_seed

log = get_logger("train_classical")
MODELS = ["logreg", "svm", "rf"]
TEXT_COL = TEXT_COLUMN["classical"]
VEC_KEYS = {"ngram_range", "max_features"}


def make_vectorizer(p: dict, min_df: int = 2) -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=tuple(p["ngram_range"]), max_features=p["max_features"], min_df=min_df)


def make_classifier(name: str, p: dict, class_weight: dict | None, seed: int):
    if name == "logreg":
        return LogisticRegression(C=p["C"], max_iter=1000, class_weight=class_weight, random_state=seed)
    if name == "svm":
        return LinearSVC(C=p["C"], class_weight=class_weight, max_iter=5000, random_state=seed)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=p["n_estimators"], max_depth=p.get("max_depth"), min_samples_leaf=p.get("min_samples_leaf", 1),
            class_weight=class_weight, n_jobs=-1, random_state=seed,
        )
    raise ValueError(f"Unknown model {name}")


def param_combos(cfg: dict, name: str, quick: bool):
    mapping = {"logreg": "logistic_regression", "svm": "linear_svc", "rf": "random_forest"}
    defaults = cfg["models"]["classical"]["tfidf"]
    model_cfg = cfg["models"]["classical"][mapping[name]]
    grid = {k: [v] for k, v in model_cfg.items()}
    for k in VEC_KEYS:                      # RF grid has no vectoriser keys -> use config defaults
        grid.setdefault(k, [defaults.get(k)])
    keys = list(grid)
    for vals in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, vals))


def train_classical(cfg: dict, name: str, strategy: str = "baseline", seed: int | None = None,
                    quick: bool = False, splits: dict | None = None, save: bool = True) -> dict:
    seed = cfg["project"]["seed"] if seed is None else seed
    set_seed(seed)
    splits = splits or load_splits(cfg)
    train_df, weights = prepare_training_data(splits["train"], strategy, cfg, seed)
    val_df, test_df = splits["val"], splits["test"]
    min_df = cfg["models"]["classical"]["tfidf"].get("min_df", 2)
    y_tr, y_va = train_df["label_id"].to_numpy(), val_df["label_id"].to_numpy()

    # ---- tuning on validation only
    cache, best_f1, best_p = {}, -1.0, None
    for p in param_combos(cfg, name, quick):
        vkey = (tuple(p["ngram_range"]), p["max_features"])
        if vkey not in cache:
            vec = make_vectorizer(p, min_df)
            cache[vkey] = (vec.fit_transform(train_df[TEXT_COL]), vec.transform(val_df[TEXT_COL]))
        X_tr, X_va = cache[vkey]
        clf = make_classifier(name, p, weights, seed).fit(X_tr, y_tr)
        f1 = compute_metrics(y_va, clf.predict(X_va))["macro_f1"]
        log.info("%s %s -> val macro-F1 %.4f", name, p, f1)
        if f1 > best_f1:
            best_f1, best_p = f1, p
    log.info("Best %s params: %s (val macro-F1 %.4f)", name, best_p, best_f1)

    # ---- final fit with best params, then single test evaluation
    pipe = Pipeline([("tfidf", make_vectorizer(best_p, min_df)),
                     ("clf", make_classifier(name, best_p, weights, seed))])
    pipe.fit(train_df[TEXT_COL], y_tr)
    report_run(cfg, name, strategy, seed, y_va, pipe.predict(val_df[TEXT_COL]), split="val", save_cm=False)
    metrics = report_run(cfg, name, strategy, seed, test_df["label_id"], pipe.predict(test_df[TEXT_COL]),
                         split="test")
    if save:
        d = model_dir(cfg, "classical", f"{name}_{strategy}", create=True)
        joblib.dump(pipe, d / "pipeline.joblib")
        save_json({"model": name, "strategy": strategy, "seed": seed, "best_params": best_p,
                   "val_macro_f1": best_f1, "test_macro_f1": metrics["macro_f1"]}, d / "info.json")
    return {"metrics": metrics, "best_params": best_p, "val_macro_f1": best_f1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=MODELS + ["all"])
    ap.add_argument("--strategy", default="baseline")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="skip grid search, use config 'quick' params")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    splits = load_splits(cfg)
    for m in (MODELS if a.model == "all" else [a.model]):
        train_classical(cfg, m, a.strategy, a.seed, a.quick, splits)


if __name__ == "__main__":
    main()
