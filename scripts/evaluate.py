#!/usr/bin/env python
"""Turn raw run predictions into the result tables and figures.

Reads   results/runs/<run_id>/{predictions.csv, meta.json}   (one directory per trained run)
Writes  results/model_metrics.csv       one row per run (overall + per-class P/R/F1)
        results/model_comparison.csv    mean +- std over seeds per (model, condition) [+ latency]
        results/imbalance_results.csv   per-class F1 per condition with delta vs. baseline
        results/figures/*.png           model comparison, per-class F1 before/after, confusion matrix
        results/reports/*               classification report + evaluation_summary.md

Only the TEST predictions are scored. Macro-F1 is the headline metric; accuracy is reported but never
used for model selection (paper Sec. 6.1).

Usage
-----
  python scripts/evaluate.py
  python scripts/evaluate.py --final-model distilbert:class_weight
  python scripts/evaluate.py --no-figures
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("evaluate")

DEFAULT_CLASSES = ["Joy", "Sadness", "Anger", "Fear", "Anxiety", "Stress", "Surprise", "Neutral"]
MODEL_LABELS = {"lr": "Logistic Regression", "svm": "SVM", "rf": "Random Forest", "bilstm": "Bi-LSTM",
                "distilbert": "DistilBERT", "roberta": "RoBERTa"}
COND_ORDER = ["none", "class_weight", "oversample", "synonym", "backtranslation"]
# Okabe-Ito (colour-blind safe); fixed order so a condition keeps its colour in every chart.
COND_COLOR = {"none": "#7f7f7f", "class_weight": "#0072B2", "oversample": "#E69F00",
              "synonym": "#009E73", "backtranslation": "#CC79A7"}


# ----------------------------------------------------------------------------- loading / scoring
def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def abs_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_runs(runs_dir: Path) -> list[dict]:
    runs = []
    for meta_f in sorted(runs_dir.glob("*/meta.json")):
        pred_f = meta_f.parent / "predictions.csv"
        if not pred_f.exists():
            log.warning("%s has meta.json but no predictions.csv - skipped", meta_f.parent.name)
            continue
        meta = json.loads(meta_f.read_text())
        meta.setdefault("run_id", meta_f.parent.name)
        runs.append({"meta": meta, "pred": pd.read_csv(pred_f), "dir": meta_f.parent})
    return runs


def score_run(pred: pd.DataFrame, classes: list[str]) -> dict:
    y, p = pred["y_true"], pred["y_pred"]
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=classes, zero_division=0)
    _, _, f1_w, _ = precision_recall_fscore_support(y, p, labels=classes, average="weighted", zero_division=0)
    present = sup > 0  # a class absent from y_true must not be counted as a 0 in macro averages
    out = {
        "n_test": len(pred),
        "accuracy": float((y == p).mean()),
        "precision_macro": float(pr[present].mean()),
        "recall_macro": float(rc[present].mean()),
        "f1_macro": float(f1[present].mean()),
        "f1_weighted": float(f1_w),
        "min_class_f1": float(f1[present].min()),
        "min_class": classes[int(np.where(present, f1, np.inf).argmin())],
    }
    for c, a, b, d, s in zip(classes, pr, rc, f1, sup):
        out.update({f"precision_{c}": a, f"recall_{c}": b, f"f1_{c}": d, f"support_{c}": int(s)})
    return out


def build_metrics(runs: list[dict], classes: list[str]) -> pd.DataFrame:
    rows = []
    for r in runs:
        m = r["meta"]
        missing = {"y_true", "y_pred"} - set(r["pred"].columns)
        if missing:
            raise SystemExit(f"{r['dir']}/predictions.csv lacks columns {missing}")
        rows.append({"run_id": m["run_id"], "model": m["model"], "condition": m.get("condition", "none"),
                     "seed": m.get("seed"), "train_seconds": m.get("train_seconds"),
                     **score_run(r["pred"], classes)})
    df = pd.DataFrame(rows)
    df["_c"] = df["condition"].map(lambda c: COND_ORDER.index(c) if c in COND_ORDER else len(COND_ORDER))
    return df.sort_values(["model", "_c", "condition", "seed"]).drop(columns="_c").reset_index(drop=True)


# ----------------------------------------------------------------------------- aggregation
def build_comparison(metrics: pd.DataFrame, latency_csv: Path) -> pd.DataFrame:
    g = metrics.groupby(["model", "condition"])
    comp = g.agg(n_seeds=("seed", "nunique"),
                 accuracy=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
                 f1_macro=("f1_macro", "mean"), f1_macro_std=("f1_macro", "std"),
                 min_class_f1=("min_class_f1", "mean"), min_class_f1_std=("min_class_f1", "std"),
                 weakest_class=("min_class", lambda s: s.mode().iat[0]),
                 train_seconds=("train_seconds", "mean")).reset_index()
    comp["is_baseline"] = comp["condition"] == "none"
    if latency_csv.exists():
        lat = pd.read_csv(latency_csv)
        key = "median_ms" if "median_ms" in lat.columns else lat.columns[-1]
        dev = lat[lat["device"] == "cpu"] if "device" in lat.columns and (lat["device"] == "cpu").any() else lat
        comp = comp.merge(dev.groupby("model")[key].min().rename("latency_ms_cpu").reset_index(),
                          on="model", how="left")
    return comp.sort_values("f1_macro", ascending=False).reset_index(drop=True)


def build_imbalance(metrics: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Long table: per (model, condition, class) mean/std F1 and delta vs. the same model's baseline."""
    rows = []
    for (model, cond), g in metrics.groupby(["model", "condition"]):
        for c in classes + ["MACRO"]:
            col = "f1_macro" if c == "MACRO" else f"f1_{c}"
            rows.append({"model": model, "condition": cond, "class": c, "f1_mean": g[col].mean(),
                         "f1_std": g[col].std(), "n_seeds": g["seed"].nunique(),
                         "support": int(g[f"support_{c}"].iloc[0]) if c != "MACRO" else int(g["n_test"].iloc[0])})
    long = pd.DataFrame(rows)
    base = long[long["condition"] == "none"].set_index(["model", "class"])
    keys = list(zip(long["model"], long["class"]))
    long["f1_baseline"] = [base["f1_mean"].get(k, np.nan) for k in keys]
    long["delta_vs_baseline"] = long["f1_mean"] - long["f1_baseline"]
    base_std = [base["f1_std"].get(k, np.nan) for k in keys]
    seed_noise = np.fmax(long["f1_std"].fillna(0).to_numpy(), np.nan_to_num(base_std))
    has_var = (long["n_seeds"] > 1).to_numpy()
    # Paper Sec. 6.3: only call a difference real if it exceeds seed-to-seed variability.
    long["exceeds_seed_std"] = np.where(has_var & (long["condition"] != "none"),
                                        long["delta_vs_baseline"].abs() > seed_noise, np.nan)
    return long[long["model"].isin(long[long["condition"] != "none"]["model"].unique())].reset_index(drop=True)


def pick_final(comp: pd.DataFrame, override: str | None) -> tuple[str, str]:
    if override:
        m, _, c = override.partition(":")
        return m, c or "none"
    best = comp.iloc[0]  # sorted by mean macro-F1
    return best["model"], best["condition"]


# ----------------------------------------------------------------------------- figures
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#bbbbbb")
    ax.spines["bottom"].set_color("#bbbbbb")
    ax.tick_params(colors="#444444")


def fig_model_comparison(comp: pd.DataFrame, out: Path):
    import matplotlib.pyplot as plt

    d = comp[comp["condition"] == "none"].sort_values("f1_macro")
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(d) + 1.5))
    y = np.arange(len(d))
    ax.barh(y, d["f1_macro"], xerr=d["f1_macro_std"].fillna(0), color="#0072B2", height=0.55,
            error_kw={"ecolor": "#333", "capsize": 3, "lw": 1})
    for yi, v in zip(y, d["f1_macro"]):
        ax.text(v + 0.012, yi, f"{v:.3f}", va="center", fontsize=9, color="#222")
    ax.set_yticks(y, [MODEL_LABELS.get(m, m) for m in d["model"]])
    ax.set_xlabel("Macro-F1 (test, baseline condition, mean ± std over seeds)")
    ax.set_xlim(0, max(0.05, float((d["f1_macro"] + d["f1_macro_std"].fillna(0)).max()) + 0.08))
    ax.set_title("Model comparison", loc="left", fontsize=11)
    ax.xaxis.grid(True, color="#e8e8e8")
    ax.set_axisbelow(True)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_per_class_mitigation(imb: pd.DataFrame, model: str, classes: list[str], out: Path):
    import matplotlib.pyplot as plt

    d = imb[(imb["model"] == model) & (imb["class"] != "MACRO")]
    conds = [c for c in COND_ORDER if c in set(d["condition"])] + \
            sorted(set(d["condition"]) - set(COND_ORDER))
    if len(conds) < 2:
        return
    fig, ax = plt.subplots(figsize=(10, 4.6))
    w = 0.8 / len(conds)
    x = np.arange(len(classes))
    for i, c in enumerate(conds):
        s = d[d["condition"] == c].set_index("class").reindex(classes)
        ax.bar(x + (i - (len(conds) - 1) / 2) * w, s["f1_mean"], w * 0.9, yerr=s["f1_std"].fillna(0),
               color=COND_COLOR.get(c, "#444"), label=c.replace("_", " "),
               error_kw={"ecolor": "#333", "lw": 0.8, "capsize": 2})
    ax.set_xticks(x, classes)
    ax.set_ylabel("F1 (test)")
    ax.set_ylim(0, 1)
    ax.set_title(f"Per-class F1 by imbalance mitigation - {MODEL_LABELS.get(model, model)}", loc="left", fontsize=11)
    ax.legend(frameon=False, ncol=len(conds), loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=9)
    ax.yaxis.grid(True, color="#e8e8e8")
    ax.set_axisbelow(True)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_confusion(pred: pd.DataFrame, classes: list[str], title: str, out: Path):
    import matplotlib.pyplot as plt

    cm = confusion_matrix(pred["y_true"], pred["y_pred"], labels=classes)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=7.5,
                    color="white" if norm[i, j] > 0.55 else "#222")
    ax.set_xticks(range(len(classes)), classes, rotation=40, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (row-normalised, counts in brackets)\n{title}", loc="left", fontsize=10)
    fig.colorbar(im, fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- report
def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)

    head = "| " + " | ".join(df.columns) + " |\n|" + "|".join("---" for _ in df.columns) + "|\n"
    return head + "\n".join("| " + " | ".join(f(v) for v in row) + " |" for row in df.itertuples(index=False))


def write_summary(path: Path, comp: pd.DataFrame, imb: pd.DataFrame, final: tuple[str, str], classes, n_runs):
    pm = lambda a, s: [f"{x:.3f}" + (f" ± {y:.3f}" if pd.notna(y) else "") for x, y in zip(a, s)]  # noqa: E731
    t5 = pd.DataFrame({"Model": comp["model"].map(lambda m: MODEL_LABELS.get(m, m)),
                       "Condition": comp["condition"], "Seeds": comp["n_seeds"],
                       "Accuracy": pm(comp["accuracy"], comp["accuracy_std"]),
                       "Macro-F1": pm(comp["f1_macro"], comp["f1_macro_std"]),
                       "Min. class F1": pm(comp["min_class_f1"], comp["min_class_f1_std"]),
                       "Weakest class": comp["weakest_class"]})
    if "latency_ms_cpu" in comp:
        t5["Latency (ms, CPU)"] = comp["latency_ms_cpu"].map(lambda v: "" if pd.isna(v) else f"{v:.1f}")
    lines = [f"# Evaluation summary\n\nTest-set results from {n_runs} runs. Headline metric: macro-F1.\n",
             "## Model comparison (paper Table 5)\n", md_table(t5), "",
             f"\n**Selected configuration:** {MODEL_LABELS.get(final[0], final[0])} / `{final[1]}` "
             "(highest mean macro-F1).\n"]
    if not imb.empty:
        for model in imb["model"].unique():
            w = imb[imb["model"] == model].pivot(index="class", columns="condition", values="f1_mean")
            w = w.reindex(classes + ["MACRO"])
            cols = [c for c in COND_ORDER if c in w.columns] + [c for c in w.columns if c not in COND_ORDER]
            w = w[cols].reset_index()
            lines += [f"## Per-class F1 by mitigation - {MODEL_LABELS.get(model, model)}\n", md_table(w), ""]
        flagged = imb[(imb["condition"] != "none") & (imb["class"] != "MACRO")]
        if flagged["exceeds_seed_std"].notna().any():
            ok = flagged[flagged["exceeds_seed_std"] == True]  # noqa: E712
            lines.append(f"{len(ok)} of {int(flagged['exceeds_seed_std'].notna().sum())} per-class deltas exceed "
                         "seed-to-seed std (only those should be called improvements).\n")
        else:
            lines.append("_Single seed only: deltas cannot be separated from run-to-run noise. "
                         "Run `train_all.py --seeds 42 43 44` before drawing conclusions._\n")
    path.write_text("\n".join(lines))


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    ap.add_argument("--results-dir", type=Path, help="default: paths.results from config or ./results")
    ap.add_argument("--final-model", help="override selection, e.g. distilbert:class_weight")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_cfg(Path(args.config))
    classes = (cfg.get("labels") or {}).get("classes", DEFAULT_CLASSES)
    res = args.results_dir or abs_path((cfg.get("paths") or {}).get("results", "results"))
    figs, reports = res / "figures", res / "reports"
    figs.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    runs = load_runs(res / "runs")
    if not runs:
        log.error("No completed runs in %s - run scripts/train_all.py first.", res / "runs")
        return 1
    metrics = build_metrics(runs, classes)
    comp = build_comparison(metrics, res / "latency_results.csv")
    imb = build_imbalance(metrics, classes)
    final = pick_final(comp, args.final_model)

    metrics.to_csv(res / "model_metrics.csv", index=False)
    comp.to_csv(res / "model_comparison.csv", index=False)
    imb.to_csv(res / "imbalance_results.csv", index=False)

    final_runs = [r for r in runs if r["meta"]["model"] == final[0] and r["meta"].get("condition", "none") == final[1]]
    if not final_runs:
        log.error("Final configuration %s:%s has no runs", *final)
        return 1
    pooled = pd.concat([r["pred"] for r in final_runs], ignore_index=True)  # pooled over seeds
    from sklearn.metrics import classification_report
    (reports / f"classification_report_{final[0]}_{final[1]}.txt").write_text(
        classification_report(pooled["y_true"], pooled["y_pred"], labels=classes, digits=3, zero_division=0))

    if not args.no_figures:
        import matplotlib

        matplotlib.use("Agg")
        fig_model_comparison(comp, figs / "model_comparison.png")
        fig_confusion(pooled, classes, f"{MODEL_LABELS.get(final[0], final[0])} / {final[1]} "
                      f"({len(final_runs)} seed(s) pooled)", figs / "confusion_matrix_final.png")
        for model in imb["model"].unique():
            fig_per_class_mitigation(imb, model, classes, figs / f"per_class_f1_mitigation_{model}.png")

    write_summary(reports / "evaluation_summary.md", comp, imb, final, classes, len(runs))
    show = comp[["model", "condition", "n_seeds", "accuracy", "f1_macro", "min_class_f1"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nFinal configuration: {final[0]} / {final[1]}\nWrote tables to {res}, figures to {figs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
