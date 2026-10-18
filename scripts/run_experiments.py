#!/usr/bin/env python
"""Run the three research experiments that go beyond the plain model comparison.

  imbalance  Train one model under each mitigation (none / class_weight / oversample / synonym /
             backtranslation, optional combos) and several seeds, then score them.
             -> results/imbalance_results.csv, per-class F1 figure, train_distribution_by_condition.csv
  trend      Simulated user timelines -> message predictions -> daily trend + escalation rule.
             -> results/trend_quality.csv
  latency    Single-text inference and aggregation latency per model / device.
             -> results/latency_results.csv
  all        imbalance, then trend, then latency.

Everything is evaluated on the untouched TEST split. Augmentation / oversampling must be applied by the
trainers to the TRAIN split only (src/imbalance.py, src/augmentation.py).

Usage
-----
  python scripts/run_experiments.py imbalance --model distilbert --seeds 42 43 44
  python scripts/run_experiments.py imbalance --model lr --conditions none class_weight oversample
  python scripts/run_experiments.py trend --final-model distilbert:class_weight --users 20
  python scripts/run_experiments.py latency --models lr distilbert roberta --repeats 200
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts

import importlib.util  # noqa: E402

# Load scripts/evaluate.py by path: a plain `import evaluate` would clash with the Hugging Face
# `evaluate` package if it is installed.
_spec = importlib.util.spec_from_file_location("project_evaluate", Path(__file__).resolve().parent / "evaluate.py")
ev = importlib.util.module_from_spec(_spec)
sys.modules["project_evaluate"] = ev
_spec.loader.exec_module(ev)
from train_all import MODELS, run_id, runs_dir, train_one, validate_condition  # noqa: E402

log = logging.getLogger("run_experiments")
DEFAULT_CLASSES = ev.DEFAULT_CLASSES
DEFAULT_NEGATIVE = ["Sadness", "Anger", "Fear", "Anxiety", "Stress"]
DEFAULT_CONDITIONS = ["none", "class_weight", "oversample", "synonym", "backtranslation"]


# ----------------------------------------------------------------------------- shared helpers
def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def cget(cfg: dict, dotted: str, default=None):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def results_dir(cfg: dict) -> Path:
    return ev.abs_path(cget(cfg, "paths.results", "results"))


def best_model_from_comparison(res: Path, fallback: str) -> str:
    f = res / "model_comparison.csv"
    if f.exists():
        d = pd.read_csv(f)
        d = d[d["condition"] == "none"]
        if len(d):
            return d.sort_values("f1_macro", ascending=False).iloc[0]["model"]
    log.warning("No model_comparison.csv yet - defaulting to %s", fallback)
    return fallback


# ----------------------------------------------------------------------------- 1. imbalance
def cmd_imbalance(args, cfg) -> int:
    res = results_dir(cfg)
    config = Path(args.config)
    seeds = args.seeds or [int(cget(cfg, "seed", 42))]
    model = args.model or best_model_from_comparison(res, "distilbert")
    conds = [validate_condition(c) for c in (args.conditions + (args.combos or []))]
    if "none" not in conds:
        conds.insert(0, "none")  # the unmitigated baseline is mandatory for a fair comparison
    rroot = runs_dir(cfg)
    log.info("Imbalance study: model=%s conditions=%s seeds=%s", model, conds, seeds)

    records = []
    for cond in conds:
        for seed in seeds:  # same model + same seed for every condition => fair comparison
            records.append(train_one(model, cond, seed, config, rroot, force=args.force,
                                     dry_run=args.dry_run, subset=args.subset))
    if args.dry_run:
        return 0
    failed = [r["run_id"] for r in records if r["status"] == "failed"]
    if failed:
        log.error("Failed runs: %s", failed)

    # class distribution of the *training* data after each mitigation (paper Sec. 4.3 / 6)
    rows = []
    for cond in conds:
        meta_f = rroot / run_id(model, cond, seeds[0]) / "meta.json"
        if meta_f.exists():
            counts = json.loads(meta_f.read_text()).get("train_class_counts")
            if counts:
                rows.append({"model": model, "condition": cond, **counts})
    if rows:
        out = res / "reports" / "train_distribution_by_condition.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        log.info("Wrote %s", out)
    else:
        log.warning("Trainers did not write meta['train_class_counts'] - distribution-after-mitigation "
                    "table skipped")
    return ev.main(["--config", str(config)] + (["--no-figures"] if args.no_figures else []))


# ----------------------------------------------------------------------------- 2. trend quality
def load_pool(res: Path, final: str) -> pd.DataFrame:
    model, _, cond = final.partition(":")
    cond = cond or "none"
    seed_dirs = sorted((res / "runs").glob(f"{model}__{cond}__seed*"))
    seed_dirs = [d for d in seed_dirs if (d / "predictions.csv").exists()]
    if not seed_dirs:
        raise SystemExit(f"No predictions for {model}:{cond} in {res / 'runs'} - train it first.")
    log.info("Using predictions from %s", seed_dirs[0].name)
    return pd.read_csv(seed_dirs[0] / "predictions.csv")


def cmd_trend(args, cfg) -> int:
    try:
        from src import trends
    except ImportError as e:
        raise SystemExit(f"src/trends.py is required for this experiment ({e})")

    res = results_dir(cfg)
    classes = cget(cfg, "labels.classes", DEFAULT_CLASSES)
    negative = cget(cfg, "labels.negative", DEFAULT_NEGATIVE)
    alpha = float(cget(cfg, "trends.ema_alpha", 0.3))
    tau = float(cget(cfg, "trends.tau", 0.6))
    k = int(cget(cfg, "trends.k_days", 3))
    final = args.final_model or f"{best_model_from_comparison(res, 'lr')}:none"
    pool = load_pool(res, final)
    scenarios = {n: s for n, s in trends.SCENARIOS.items() if not args.scenarios or n in args.scenarios}
    seed0 = int(cget(cfg, "seed", 42))
    pcols = [f"p_{c}" for c in classes]

    rows = []
    for name, scen in scenarios.items():
        for u in range(args.users):
            tl = trends.simulate_timeline(pool, scen, start="2025-01-01", seed=seed0 + u,
                                          user_id=f"{name}-{u}", classes=classes)
            tl = tl.merge(pool[["id", "y_pred"] + pcols], on="id", how="left")
            q = trends.trend_quality(tl, classes=classes, negative=negative, alpha=alpha)
            daily = trends.aggregate_predictions(tl, "D", classes, negative)
            esc = trends.detect_escalation(trends.ema(daily["neg_score"], alpha), tau=tau, k=k)
            first = daily.index[esc.to_numpy()][0] if esc.any() else None
            expected = bool(scen.get("expect_escalation", False))
            rows.append({"scenario": name, "user": u, "n_messages": len(tl), "n_days": len(daily),
                         **q, "escalated": bool(esc.any()), "expected_escalation": expected,
                         "escalation_correct": bool(esc.any()) == expected,
                         "first_escalation_day": str(first) if first is not None else "",
                         "alpha": alpha, "tau": tau, "k_days": k, "model": final})
    out = pd.DataFrame(rows)
    out.to_csv(res / "trend_quality.csv", index=False)

    summ = out.groupby("scenario").agg(
        msg_acc=("message_accuracy", "mean"), day_dom_agree=("daily_dominant_agreement", "mean"),
        neg_corr=("neg_corr", "mean"), neg_mae=("neg_mae", "mean"),
        escalation_rate=("escalated", "mean"), escalation_correct=("escalation_correct", "mean"))
    print(f"\nTrend quality ({final}, alpha={alpha}, tau={tau}, k={k}, users/scenario={args.users}):")
    print(summ.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nNOTE: timelines are SIMULATED from test-set messages in a scripted order (no real user "
          "timelines exist in the datasets). This tests that the module behaves as specified, not clinical validity.")
    return 0


# ----------------------------------------------------------------------------- 3. latency
SAMPLE_TEXTS = [
    "I can't stop worrying about tomorrow's exam.",
    "Had a lovely walk with my sister today, feeling really grateful.",
    "Everything is piling up at work and I feel like I'm drowning in deadlines right now.",
    "ok",
]


def _pctl(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def cmd_latency(args, cfg) -> int:
    try:
        from src.inference import EmotionPredictor
    except ImportError as e:
        raise SystemExit(f"src/inference.py (EmotionPredictor) is required for latency ({e})")
    res = results_dir(cfg)
    classes = cget(cfg, "labels.classes", DEFAULT_CLASSES)
    negative = cget(cfg, "labels.negative", DEFAULT_NEGATIVE)
    devices = ["cpu"]
    try:
        import torch

        if torch.cuda.is_available():
            devices.append("cuda")
    except ImportError:
        pass

    # aggregation cost is independent of the classifier: time it once on a 30-day, 90-message series
    agg_ms = float("nan")
    try:
        from src import trends

        rng = np.random.default_rng(0)
        probs = rng.dirichlet(np.ones(len(classes)), size=90)
        df = pd.DataFrame(probs, columns=[f"p_{c}" for c in classes])
        df["user_id"] = "u"
        df["timestamp"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 30 * 24, 90), unit="h")
        t = []
        for _ in range(20):
            t0 = time.perf_counter()
            d = trends.aggregate_predictions(df, "D", classes, negative)
            trends.ema(d["neg_score"], 0.3)
            t.append((time.perf_counter() - t0) * 1000)
        agg_ms = statistics.median(t)
    except Exception as e:  # noqa: BLE001
        log.warning("Aggregation timing skipped: %s", e)

    rows = []
    for model in args.models:
        for device in devices:
            try:
                pred = EmotionPredictor.from_config(cfg, model=model, device=device)
            except Exception as e:  # noqa: BLE001
                log.warning("[skip] %s on %s: %s", model, device, e)
                continue
            for _ in range(args.warmup):
                pred.predict_proba([SAMPLE_TEXTS[0]])
            times = []
            for i in range(args.repeats):
                text = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]
                t0 = time.perf_counter()
                pred.predict_proba([text])
                if device == "cuda":
                    import torch

                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)
            rows.append({"model": model, "device": device, "n": len(times),
                         "mean_ms": statistics.fmean(times), "median_ms": statistics.median(times),
                         "p95_ms": _pctl(times, 0.95), "aggregation_ms": agg_ms,
                         "end_to_end_ms": statistics.median(times) + (0 if np.isnan(agg_ms) else agg_ms)})
            log.info("%s/%s median %.1f ms", model, device, rows[-1]["median_ms"])
    if not rows:
        log.error("No model could be loaded - nothing measured.")
        return 1
    out = pd.DataFrame(rows)
    out.to_csv(res / "latency_results.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nMeasures preprocessing + inference for ONE text (batch size 1) and aggregation of a 30-day "
          "series. Dashboard rendering time is not included - measure it in the browser.")
    return 0


# ----------------------------------------------------------------------------- cli
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("imbalance", help="mitigation study on one model")
    p.add_argument("--model", choices=list(MODELS), help="default: best baseline model so far")
    p.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    p.add_argument("--combos", nargs="*", help="extra combined conditions, e.g. class_weight+synonym")
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--subset", type=int)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-figures", action="store_true")

    p = sub.add_parser("trend", help="trend quality + escalation behaviour on simulated timelines")
    p.add_argument("--final-model", help="model:condition, default best baseline")
    p.add_argument("--users", type=int, default=20, help="simulated users per scenario")
    p.add_argument("--scenarios", nargs="*")

    p = sub.add_parser("latency", help="inference + aggregation latency")
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--repeats", type=int, default=200)
    p.add_argument("--warmup", type=int, default=10)

    sub.add_parser("all", help="imbalance + trend + latency with defaults")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_cfg(Path(args.config))

    if args.cmd == "all":
        rc = 0
        for sub_argv in (["imbalance"], ["trend"], ["latency"]):
            rc |= main(["--config", args.config] + sub_argv)
        return rc
    return {"imbalance": cmd_imbalance, "trend": cmd_trend, "latency": cmd_latency}[args.cmd](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
