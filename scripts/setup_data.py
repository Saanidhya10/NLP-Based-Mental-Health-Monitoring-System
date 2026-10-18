#!/usr/bin/env python
"""Build the unified 8-class dataset and the FROZEN stratified splits.

Pipeline (roadmap Day 1-2):
  1. Load GoEmotions (27 + Neutral) and map to the 8-class taxonomy.
  2. Load Dreaddit; stressed posts -> Stress (truncated to ~128 words).
  3. (optional) Sample Sentiment140 for a coarse-polarity benchmark.
  4. De-duplicate, then create stratified train/val/test splits ONCE.
  5. Save a manifest with file hashes so the test set can never silently change.
  6. (optional) Run src.preprocess to write data/processed/.

Usage
-----
  python scripts/setup_data.py                       # real data
  python scripts/setup_data.py --synthetic 6000      # offline smoke test
  python scripts/setup_data.py --verify              # check frozen splits
  python scripts/setup_data.py --force               # rebuild (changes test set!)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
log = logging.getLogger("setup_data")

DEFAULT_CLASSES = ["Joy", "Sadness", "Anger", "Fear", "Anxiety", "Stress", "Surprise", "Neutral"]

# Fixed BEFORE any model is trained (paper Sec. 4.2). Override via config labels.goemotions_map.
DEFAULT_GOEMOTIONS_MAP = {
    "Joy": ["admiration", "amusement", "approval", "caring", "excitement", "gratitude",
            "joy", "love", "optimism", "pride", "relief"],
    "Sadness": ["disappointment", "embarrassment", "grief", "remorse", "sadness"],
    "Anger": ["anger", "annoyance", "disapproval", "disgust"],
    "Fear": ["fear"],
    "Anxiety": ["nervousness"],
    "Surprise": ["surprise", "realization"],
    "Neutral": ["neutral", "confusion", "curiosity", "desire"],
    "Stress": [],  # populated from Dreaddit only
}

DREADDIT_URL = "http://www.cs.columbia.edu/~eturcan/data/dreaddit.zip"
DREADDIT_HF_IDS = ["asarvazyan/dreaddit", "dreaddit"]  # tried in order if no local copy/URL works


# ----------------------------------------------------------------------------- config helpers
def load_config(path: Path) -> dict:
    if path.exists():
        return yaml.safe_load(path.read_text()) or {}
    log.warning("Config %s not found - using built-in defaults", path)
    return {}


def cfg_get(cfg: dict, dotted: str, default=None):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def resolve(cfg: dict, key: str, default: str, root_override: Path | None, sub: str) -> Path:
    if root_override is not None:
        return root_override / sub
    p = Path(cfg_get(cfg, key, default))
    return p if p.is_absolute() else ROOT / p


# ----------------------------------------------------------------------------- loaders
def load_goemotions(raw_dir: Path, hf_id: str) -> tuple[pd.DataFrame, list[str]]:
    """Return (df[id,text,label_ids], label_names). Local one-hot CSVs win over HF download."""
    local = sorted((raw_dir / "goemotions").glob("goemotions_*.csv"))
    if local:
        log.info("GoEmotions: reading %d local CSV file(s)", len(local))
        df = pd.concat([pd.read_csv(f) for f in local], ignore_index=True)
        meta = {"id", "text", "author", "subreddit", "link_id", "parent_id", "created_utc",
                "rater_id", "example_very_unclear"}
        names = [c for c in df.columns if c not in meta]
        df = df[df.get("example_very_unclear", False) != True]  # noqa: E712
        # raters are separate rows; keep labels chosen by >=2 raters (approximation of majority)
        agg = df.groupby(["id", "text"])[names].sum().reset_index()
        agg["label_ids"] = agg[names].apply(lambda r: [i for i, v in enumerate(r.values) if v >= 2], axis=1)
        return agg[["id", "text", "label_ids"]], names

    from datasets import load_dataset  # lazy: only needed for real data

    log.info("GoEmotions: downloading %s (simplified)", hf_id)
    ds = load_dataset(hf_id, "simplified")
    names = ds["train"].features["labels"].feature.names
    frames = []
    for split in ds:
        d = ds[split].to_pandas()
        d = d.rename(columns={"labels": "label_ids"})
        frames.append(d[["id", "text", "label_ids"]])
    df = pd.concat(frames, ignore_index=True)
    df["label_ids"] = df["label_ids"].apply(list)
    return df, list(names)


def map_goemotions(df: pd.DataFrame, names: list[str], mapping: dict[str, list[str]]) -> pd.DataFrame:
    """27 -> 8. Keep a comment only if all its labels fall in ONE target class; drop ambiguous ones."""
    src2tgt = {src: tgt for tgt, srcs in mapping.items() for src in srcs}
    unmapped = [n for n in names if n not in src2tgt]
    if unmapped:
        log.warning("GoEmotions labels without mapping (rows dropped): %s", unmapped)

    def to_class(ids):
        tg = {src2tgt.get(names[i]) for i in ids}
        return next(iter(tg)) if len(tg) == 1 and None not in tg else None

    out = df.copy()
    out["label"] = out["label_ids"].apply(to_class)
    n0 = len(out)
    out = out.dropna(subset=["label"])
    log.info("GoEmotions: kept %d / %d comments after single-class mapping", len(out), n0)
    out["source"] = "goemotions"
    out["id"] = "go-" + out["id"].astype(str)
    return out[["id", "text", "label", "source"]]


def _read_dreaddit_csvs(files: list[Path]) -> pd.DataFrame:
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def load_dreaddit(raw_dir: Path, local_path: Path | None, url: str) -> pd.DataFrame:
    ddir = raw_dir / "dreaddit"
    ddir.mkdir(parents=True, exist_ok=True)
    if local_path is not None:
        local_path = Path(local_path)
        if local_path.suffix == ".zip":
            zipfile.ZipFile(local_path).extractall(ddir)
        elif local_path.is_file():
            (ddir / local_path.name).write_bytes(local_path.read_bytes())
    files = sorted(ddir.rglob("dreaddit*.csv")) or sorted(ddir.rglob("*.csv"))
    if not files:
        try:
            log.info("Dreaddit: downloading %s", url)
            data = urllib.request.urlopen(url, timeout=60).read()
            zipfile.ZipFile(io.BytesIO(data)).extractall(ddir)
            files = sorted(ddir.rglob("*.csv"))
        except Exception as e:  # noqa: BLE001
            log.warning("Dreaddit URL failed (%s); trying Hugging Face ids", e)
    if files:
        return _read_dreaddit_csvs(files)

    from datasets import load_dataset

    for hf_id in DREADDIT_HF_IDS:
        try:
            ds = load_dataset(hf_id)
            return pd.concat([ds[s].to_pandas() for s in ds], ignore_index=True)
        except Exception as e:  # noqa: BLE001
            log.warning("Dreaddit HF id %s failed: %s", hf_id, e)
    raise SystemExit(
        "Could not obtain Dreaddit. Download it manually (Turcan & McKeown 2019) and rerun with\n"
        f"  --dreaddit-path <dreaddit.zip or CSV>   (or place CSVs in {ddir})"
    )


def map_dreaddit(df: pd.DataFrame, max_words: int) -> pd.DataFrame:
    """Stressed posts -> Stress. Not-stressed posts are DROPPED (not-stressed != Neutral)."""
    df = df[df["label"] == 1].copy()
    df["text"] = df["text"].astype(str).str.split().str[:max_words].str.join(" ")
    idcol = "id" if "id" in df.columns else None
    df["id"] = "dd-" + (df[idcol].astype(str) if idcol else df.index.astype(str))
    df["label"] = "Stress"
    df["source"] = "dreaddit"
    log.info("Dreaddit: %d stressed posts (truncated to %d words)", len(df), max_words)
    return df[["id", "text", "label", "source"]]


def sample_sentiment140(path: Path, n: int, seed: int, out: Path) -> None:
    df = pd.read_csv(path, encoding="latin-1", header=None,
                     names=["target", "id", "date", "flag", "user", "text"])
    df["polarity"] = (df["target"] > 0).astype(int)
    half = n // 2
    s = pd.concat([g.sample(min(half, len(g)), random_state=seed) for _, g in df.groupby("polarity")])
    s[["id", "polarity", "text"]].to_csv(out, index=False)
    log.info("Sentiment140: wrote %d tweets to %s", len(s), out)


# ----------------------------------------------------------------------------- synthetic (smoke tests)
_SYN_VOCAB = {
    "Joy": "happy great love wonderful thrilled grateful amazing delighted proud excited".split(),
    "Sadness": "sad lonely crying hopeless miss lost empty heartbroken down grief".split(),
    "Anger": "angry furious hate annoyed unfair ridiculous mad disgusting rage irritated".split(),
    "Fear": "scared terrified afraid frightened danger panic threat nightmare horror dread".split(),
    "Anxiety": "nervous worried anxious uneasy restless overthinking jittery tense apprehensive racing".split(),
    "Stress": "overwhelmed deadline pressure exhausted burnout workload swamped stressed behind drowning".split(),
    "Surprise": "wow unexpected shocked suddenly unbelievable whoa astonished sudden surprise realized".split(),
    "Neutral": "okay maybe today the meeting tomorrow weather think probably usual normal fine".split(),
}
_SYN_PRIOR = dict(zip(DEFAULT_CLASSES, [0.30, 0.10, 0.10, 0.03, 0.03, 0.03, 0.06, 0.35]))
_FILLER = "i the a to it was just so really this that and my we they about with for".split()


def make_synthetic(n: int, seed: int) -> pd.DataFrame:
    """Toy imbalanced data with weak class signal. FOR PIPELINE TESTING ONLY."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(list(_SYN_PRIOR), size=n, p=list(_SYN_PRIOR.values()))
    rows = []
    for i, lab in enumerate(labels):
        k = rng.integers(1, 3)
        sig = list(rng.choice(_SYN_VOCAB[lab], size=k))
        other = rng.choice(_SYN_VOCAB[rng.choice(list(_SYN_VOCAB))], size=1)  # label noise word
        words = sig + list(other) + list(rng.choice(_FILLER, size=rng.integers(3, 9)))
        rng.shuffle(words)
        rows.append((f"syn-{i}", " ".join(words), lab, "synthetic"))
    return pd.DataFrame(rows, columns=["id", "text", "label", "source"])


# ----------------------------------------------------------------------------- splits
def clean_and_dedupe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    key = df["text"].str.lower()
    conflicting = df.groupby(key)["label"].transform("nunique") > 1
    if conflicting.any():
        log.info("Dropping %d rows whose duplicate text carries conflicting labels", conflicting.sum())
    df = df[~conflicting]
    before = len(df)
    df = df[~key[~conflicting].duplicated()]
    log.info("De-duplicated %d -> %d rows", before, len(df))
    df["n_words"] = df["text"].str.split().str.len()
    return df.reset_index(drop=True)


def stratified_split(df: pd.DataFrame, val: float, test: float, seed: int):
    from sklearn.model_selection import train_test_split

    train_val, te = train_test_split(df, test_size=test, stratify=df["label"], random_state=seed)
    tr, va = train_test_split(train_val, test_size=val / (1 - test),
                              stratify=train_val["label"], random_state=seed)
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(splits_dir: Path, splits: dict, classes: list[str], seed: int, ratios: dict) -> dict:
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "ratios": ratios,
        "classes": classes,
        "files": {},
    }
    for name, d in splits.items():
        f = splits_dir / f"{name}.csv"
        manifest["files"][name] = {
            "rows": len(d),
            "sha256": sha256(f),
            "class_counts": d["label"].value_counts().reindex(classes, fill_value=0).to_dict(),
        }
    (splits_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def verify_splits(splits_dir: Path) -> bool:
    mf = splits_dir / "split_manifest.json"
    if not mf.exists():
        log.error("No manifest at %s", mf)
        return False
    manifest = json.loads(mf.read_text())
    ok = True
    for name, info in manifest["files"].items():
        f = splits_dir / f"{name}.csv"
        good = f.exists() and sha256(f) == info["sha256"]
        log.info("%-6s %s", name, "OK" if good else "MODIFIED / MISSING")
        ok &= good
    return ok


# ----------------------------------------------------------------------------- reporting
def report_distribution(splits: dict, classes: list[str], full: pd.DataFrame, figs: Path, reports: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tbl = pd.DataFrame({k: v["label"].value_counts().reindex(classes, fill_value=0) for k, v in splits.items()})
    tbl["total"] = tbl.sum(axis=1)
    tbl["share_%"] = (100 * tbl["total"] / tbl["total"].sum()).round(2)
    tbl.index.name = "class"
    tbl.to_csv(reports / "class_distribution.csv")

    lens = full.groupby("label")["n_words"].agg(["median", "mean", "max"]).reindex(classes).round(1)
    lens.to_csv(reports / "text_length_by_class.csv")

    order = tbl["total"].sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(order.index, order.values, color="#2a6fb0", height=0.6)
    for b, v in zip(bars, order.values):
        ax.text(v, b.get_y() + b.get_height() / 2, f"  {v:,}", va="center", fontsize=9, color="#333")
    ax.set_title("Class distribution (all splits)", loc="left", fontsize=11)
    ax.set_xlabel("examples")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.xaxis.grid(True, color="#e5e5e5")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(figs / "class_distribution.png", dpi=150)
    plt.close(fig)

    print("\nClass distribution:\n", tbl.to_string())
    print("\nWords per text (median/mean/max):\n", lens.to_string())
    if full["source"].nunique() > 1 and "Stress" in set(full["label"]):
        stress_src = full[full["label"] == "Stress"]["source"].value_counts(normalize=True).round(3).to_dict()
        print(f"\nNOTE: Stress comes from {stress_src}. Stress posts are longer and from a different domain than\n"
              "the other classes, so models can learn 'long text = Stress'. Check this in error analysis.")


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    ap.add_argument("--data-root", type=Path, help="override data/ directory (and results/ next to it)")
    ap.add_argument("--dreaddit-path", type=Path, help="local Dreaddit zip or CSV")
    ap.add_argument("--sentiment140-path", type=Path, help="local Sentiment140 CSV (optional)")
    ap.add_argument("--sentiment140-n", type=int, default=50_000)
    ap.add_argument("--synthetic", type=int, metavar="N", help="generate N toy rows instead of downloading")
    ap.add_argument("--subset", type=int, help="keep only N rows (stratified) for fast debugging")
    ap.add_argument("--skip-preprocess", action="store_true")
    ap.add_argument("--verify", action="store_true", help="only check frozen splits against the manifest")
    ap.add_argument("--force", action="store_true", help="rebuild splits even if frozen (changes the test set)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(Path(args.config))
    seed = int(cfg_get(cfg, "seed", 42))
    classes = cfg_get(cfg, "labels.classes", DEFAULT_CLASSES)
    mapping = cfg_get(cfg, "labels.goemotions_map", DEFAULT_GOEMOTIONS_MAP)
    val_frac = float(cfg_get(cfg, "splits.val_size", 0.1))
    test_frac = float(cfg_get(cfg, "splits.test_size", 0.1))
    max_words = int(cfg_get(cfg, "data.dreaddit_max_words", 128))

    ro = args.data_root
    raw = resolve(cfg, "paths.raw", "data/raw", ro, "raw")
    interim = resolve(cfg, "paths.interim", "data/interim", ro, "interim")
    processed = resolve(cfg, "paths.processed", "data/processed", ro, "processed")
    splits_dir = resolve(cfg, "paths.splits", "data/splits", ro, "splits")
    results = (ro.parent / "results") if ro else resolve(cfg, "paths.results", "results", None, "")
    figs, reports = results / "figures", results / "reports"
    for d in (raw, interim, processed, splits_dir, figs, reports):
        d.mkdir(parents=True, exist_ok=True)

    if args.verify:
        return 0 if verify_splits(splits_dir) else 1

    if (splits_dir / "split_manifest.json").exists() and not args.force:
        log.error("Splits are frozen (%s). Use --verify to check them or --force to rebuild.", splits_dir)
        return 1

    # 1-3. load + map
    if args.synthetic:
        log.warning("SYNTHETIC DATA - for pipeline testing only; never report these numbers.")
        unified = make_synthetic(args.synthetic, seed)
    else:
        go_raw, names = load_goemotions(raw, cfg_get(cfg, "data.goemotions_hf_id",
                                                     "google-research-datasets/go_emotions"))
        go = map_goemotions(go_raw, names, mapping)
        dd = map_dreaddit(load_dreaddit(raw, args.dreaddit_path, cfg_get(cfg, "data.dreaddit_url", DREADDIT_URL)),
                          max_words)
        unified = pd.concat([go, dd], ignore_index=True)
        if args.sentiment140_path:
            sample_sentiment140(args.sentiment140_path, args.sentiment140_n, seed,
                                interim / "sentiment140_sample.csv")

    unknown = set(unified["label"]) - set(classes)
    if unknown:
        raise SystemExit(f"Labels outside configured classes: {unknown}")

    # 4. clean, optionally subset, split ONCE
    unified = clean_and_dedupe(unified)
    if args.subset and args.subset < len(unified):
        frac = args.subset / len(unified)
        unified = pd.concat([g.sample(min(len(g), max(10, int(round(len(g) * frac)))), random_state=seed)
                             for _, g in unified.groupby("label")]).reset_index(drop=True)
        log.info("Debug subset: %d rows", len(unified))
    unified.to_csv(interim / "unified.csv", index=False)
    tr, va, te = stratified_split(unified, val_frac, test_frac, seed)
    splits = {"train": tr, "val": va, "test": te}
    for name, d in splits.items():
        d.to_csv(splits_dir / f"{name}.csv", index=False)
    write_manifest(splits_dir, splits, classes, seed, {"val": val_frac, "test": test_frac})
    log.info("Wrote frozen splits to %s (train %d / val %d / test %d)", splits_dir, len(tr), len(va), len(te))

    report_distribution(splits, classes, unified, figs, reports)

    # 6. preprocessing hook (src/preprocess.py)
    if not args.skip_preprocess:
        try:
            from src.preprocess import preprocess_dataframe
        except ImportError as e:
            log.warning("src.preprocess.preprocess_dataframe not importable (%s) - skipping processed/", e)
        else:
            for name, d in splits.items():
                preprocess_dataframe(d, text_col="text").to_csv(processed / f"{name}.csv", index=False)
            log.info("Wrote preprocessed splits to %s", processed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
