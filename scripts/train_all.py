#!/usr/bin/env python
"""Train every model (or a chosen subset) by calling the src/train_*.py entry points.

Each trainer is run as a subprocess:  python -m src.<trainer> --config C --model M --seed S --imbalance X
and must leave behind (the "run contract"):
  results/runs/<model>__<condition>__seed<seed>/predictions.csv   (test split, with p_<Class> columns)
  results/runs/<run_id>/meta.json

Usage
-----
  python scripts/train_all.py                                  # all 6 models, baseline, seed from config
  python scripts/train_all.py --models lr svm distilbert --seeds 42 43 44
  python scripts/train_all.py --models distilbert --imbalance class_weight
  python scripts/train_all.py --dry-run                        # print commands only
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("train_all")

# model key -> (trainer module, model-family folder under models/)
MODELS = {
    "lr": ("src.train_classical", "classical"),
    "svm": ("src.train_classical", "classical"),
    "rf": ("src.train_classical", "classical"),
    "bilstm": ("src.train_bilstm", "bilstm"),
    "distilbert": ("src.train_transformer", "distilbert"),
    "roberta": ("src.train_transformer", "roberta"),
}
ORDER = ["lr", "svm", "rf", "bilstm", "distilbert", "roberta"]  # cheap -> expensive
CONDITIONS = {"none", "class_weight", "oversample", "synonym", "backtranslation"}


def run_id(model: str, condition: str, seed: int) -> str:
    """Shared naming convention - evaluate.py and run_experiments.py rely on it."""
    return f"{model}__{condition}__seed{seed}"


def validate_condition(cond: str) -> str:
    parts = cond.split("+")
    bad = [p for p in parts if p not in CONDITIONS]
    if bad:
        raise SystemExit(f"Unknown imbalance condition(s) {bad}; choose from {sorted(CONDITIONS)} (join with '+')")
    return cond


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def runs_dir(cfg: dict) -> Path:
    base = Path((cfg.get("paths") or {}).get("results", "results"))
    base = base if base.is_absolute() else ROOT / base
    return base / "runs"


def is_complete(rdir: Path) -> bool:
    return (rdir / "meta.json").exists() and (rdir / "predictions.csv").exists()


def train_one(model: str, condition: str, seed: int, config: Path, rroot: Path, *, force=False,
              dry_run=False, subset: int | None = None, extra: list[str] | None = None) -> dict:
    """Run one training job. Returns a status record. Importable by run_experiments.py."""
    rid = run_id(model, condition, seed)
    rdir = rroot / rid
    rec = {"run_id": rid, "model": model, "condition": condition, "seed": seed}
    if is_complete(rdir) and not force:
        log.info("[skip] %s already complete (use --force to retrain)", rid)
        return {**rec, "status": "skipped", "seconds": 0.0}

    module = MODELS[model][0]
    cmd = [sys.executable, "-m", module, "--config", str(config), "--model", model,
           "--seed", str(seed), "--imbalance", condition]
    if subset:
        cmd += ["--subset", str(subset)]
    cmd += extra or []
    if dry_run:
        print(" ".join(cmd))
        return {**rec, "status": "dry-run", "seconds": 0.0}

    logdir = rroot.parent / "reports" / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    log.info("[run ] %s", rid)
    t0 = time.time()
    with open(logdir / f"{rid}.log", "w") as fh:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    secs = time.time() - t0
    if proc.returncode == 0 and is_complete(rdir):
        log.info("[done] %s in %.0fs", rid, secs)
        return {**rec, "status": "ok", "seconds": round(secs, 1)}
    why = f"exit code {proc.returncode}" if proc.returncode else "no predictions.csv/meta.json written"
    log.error("[FAIL] %s (%s) - see %s", rid, why, logdir / f"{rid}.log")
    return {**rec, "status": "failed", "seconds": round(secs, 1), "detail": why}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=ORDER)
    ap.add_argument("--imbalance", default="none", type=validate_condition,
                    help="none | class_weight | oversample | synonym | backtranslation (join with '+')")
    ap.add_argument("--seeds", nargs="+", type=int, help="default: [seed] from config")
    ap.add_argument("--subset", type=int, help="train on N rows only (debugging / Colab smoke test)")
    ap.add_argument("--force", action="store_true", help="retrain even if results exist")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="everything after --extra is passed to the trainer unchanged")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    config = Path(args.config)
    cfg = load_cfg(config)
    seeds = args.seeds or [int(cfg.get("seed", 42))]
    rroot = runs_dir(cfg)
    rroot.mkdir(parents=True, exist_ok=True)

    if not (ROOT / "data" / "splits" / "test.csv").exists() and not args.dry_run:
        log.warning("data/splits/test.csv not found - run scripts/setup_data.py first")

    models = [m for m in ORDER if m in args.models]
    records = []
    for seed in seeds:  # seed-major: a full seed finishes before the next starts
        for model in models:
            rec = train_one(model, args.imbalance, seed, config, rroot, force=args.force,
                            dry_run=args.dry_run, subset=args.subset, extra=args.extra)
            records.append(rec)
            if rec["status"] == "failed" and args.stop_on_error:
                break
        else:
            continue
        break

    if not args.dry_run:
        out = rroot.parent / "reports" / "train_all_summary.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(out, index=False)
        print("\n" + pd.DataFrame(records)[["run_id", "status", "seconds"]].to_string(index=False))
        print(f"\nSummary: {out}\nNext: python scripts/evaluate.py")
    return 1 if any(r["status"] == "failed" for r in records) else 0


if __name__ == "__main__":
    sys.exit(main())
