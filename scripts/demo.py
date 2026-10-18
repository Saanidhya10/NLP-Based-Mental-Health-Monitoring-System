#!/usr/bin/env python
"""End-to-end demo: raw text -> emotion -> stored predictions -> mood trend -> supportive suggestion.

Runs a scripted 3-week journal (a person drifting from Joy toward Stress/Sadness) through the real pipeline
(src.inference, src.database, src.trends, src.recommend) and prints what the dashboard would show.
Good for screenshots and for the live demo.

NOT A DIAGNOSTIC TOOL. The journal below is invented for the demo; nothing is stored unless you pass --keep-db.

Usage
-----
  python scripts/demo.py                          # scripted journal, best available model
  python scripts/demo.py --model distilbert --plot
  python scripts/demo.py --text "I can't sleep, my mind keeps racing about tomorrow"
  python scripts/demo.py --app                    # launch the Streamlit dashboard instead
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
log = logging.getLogger("demo")

DEFAULT_CLASSES = ["Joy", "Sadness", "Anger", "Fear", "Anxiety", "Stress", "Surprise", "Neutral"]
DEFAULT_NEGATIVE = ["Sadness", "Anger", "Fear", "Anxiety", "Stress"]

DISCLAIMER = ("This is a wellness-support and early-awareness demo. It is NOT a diagnostic tool and cannot "
              "replace a qualified mental-health professional.")
CONSENT = ("Consent notice: the messages below are invented for this demo. In the real app, text is only "
           "analysed and stored with your explicit consent, and you can delete it at any time.")

# (days before the last day, hour, text) - an invented journal, days 0..20
JOURNAL = [
    (20, 9, "Had a wonderful weekend with friends, I feel so happy and relaxed."),
    (19, 20, "Got the internship offer! I'm thrilled and so grateful."),
    (17, 13, "Lunch with my sister was lovely, laughed all afternoon."),
    (15, 8, "Another ordinary day. Class at ten, then the library."),
    (14, 21, "Feeling pretty good about the semester so far, proud of my progress."),
    (12, 10, "The meeting is moved to Thursday, I'll check the schedule later."),
    (11, 22, "Slightly nervous about the presentation next week, but it should be fine."),
    (10, 9, "Three deadlines landed on the same day and I don't know where to start."),
    (9, 23, "I'm so behind. There's too much to do and I keep falling further back."),
    (8, 15, "Skipped lunch again to finish the report, I feel completely overwhelmed."),
    (7, 7, "Barely slept. My mind keeps racing about everything that could go wrong."),
    (6, 20, "I'm worried I'll fail the course and let everyone down."),
    (5, 11, "Everything feels heavy lately and I can't shake this sadness."),
    (5, 22, "I feel so alone, I haven't talked to anyone in days."),
    (4, 9, "The pressure at work is exhausting, I don't know how much more I can take."),
    (3, 14, "Cried in the bathroom between classes. I just feel empty."),
    (3, 23, "Can't focus on anything. My chest is tight and my thoughts won't stop."),
    (2, 10, "Another day of being swamped and anxious, nothing is getting better."),
    (1, 19, "I feel hopeless about all of it, so tired of feeling like this."),
    (0, 8, "Woke up dreading the day again. I'm exhausted and overwhelmed."),
]


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def cget(cfg: dict, dotted: str, default=None):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def pick_model(cfg: dict, requested: str) -> str:
    if requested != "auto":
        return requested
    comp = ROOT / cget(cfg, "paths.results", "results") / "model_comparison.csv"
    if comp.exists():
        return str(pd.read_csv(comp).sort_values("f1_macro", ascending=False).iloc[0]["model"])
    return "lr"


def bar(v: float, width: int = 20) -> str:
    n = int(round(max(0.0, min(1.0, v)) * width))
    return "#" * n + "." * (width - n)


def print_recommendation(rec: dict) -> None:
    print("\n--- Suggestion " + "-" * 50)
    print(rec.get("message", ""))
    for s in rec.get("suggestions", []):
        print(f"  * {s}")
    if rec.get("escalate"):
        print("\n  It may help to talk with a qualified mental-health professional.")
        for h in rec.get("helplines", []):
            print(f"  - {h.get('name', '')}: {h.get('number', '')}  {h.get('note', '')}".rstrip())
        print("  (Please verify helpline numbers for your region before relying on them.)")
    print(f"\n{rec.get('disclaimer', DISCLAIMER)}")


def run_single(predictor, cfg, text: str) -> None:
    classes = predictor.classes
    p = predictor.predict_proba([text])[0]
    order = np.argsort(p)[::-1][:3]
    print(f'\nText: "{text}"')
    for i in order:
        print(f"  {classes[i]:<9} {p[i]:.2f}  {bar(p[i])}")
    from src.recommend import recommend

    print_recommendation(recommend(classes[order[0]], escalate=False, region=cget(cfg, "recommend.region", "IN")))


def run_journal(args, cfg) -> int:
    from src import trends
    from src.database import MoodDB
    from src.inference import EmotionPredictor
    from src.recommend import recommend

    classes = cget(cfg, "labels.classes", DEFAULT_CLASSES)
    negative = cget(cfg, "labels.negative", DEFAULT_NEGATIVE)
    alpha = float(cget(cfg, "trends.ema_alpha", 0.3))
    tau = float(cget(cfg, "trends.tau", 0.6))
    k = int(cget(cfg, "trends.k_days", 3))

    model = pick_model(cfg, args.model)
    log.info("Loading model: %s", model)
    predictor = EmotionPredictor.from_config(cfg, model=model, device=args.device)
    classes = list(getattr(predictor, "classes", classes))

    last_day = date.today()
    ts = [datetime.combine(last_day - timedelta(days=d), time(hour=h)) for d, h, _ in JOURNAL]
    texts = [t for _, _, t in JOURNAL]
    proba = predictor.predict_proba(texts)
    df = pd.DataFrame(proba, columns=[f"p_{c}" for c in classes])
    df.insert(0, "text", texts)
    df.insert(0, "timestamp", ts)
    df.insert(0, "user_id", "demo-user")
    df["emotion"] = [classes[i] for i in proba.argmax(1)]
    df["confidence"] = proba.max(1)

    # --- store + read back (SQLite) ---
    db_path = Path(args.db) if args.db else ROOT / "results" / "demo_mood.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    db = MoodDB(str(db_path))
    db.insert_predictions(df.drop(columns=["text"]))  # raw text is NOT stored
    stored = db.fetch("demo-user")

    print("\nMessages -> emotion")
    for r in df.itertuples():
        print(f"  {r.timestamp:%d %b %H:%M}  {r.emotion:<9} {r.confidence:.2f}  {r.text[:62]}")

    daily = trends.aggregate_predictions(stored, "D", classes, negative)
    daily["neg_smooth"] = trends.ema(daily["neg_score"], alpha)
    daily["escalate"] = trends.detect_escalation(daily["neg_smooth"], tau=tau, k=k)
    print(f"\nDaily mood trend  (neg = negative-affect score, EMA alpha={alpha}; escalation: smoothed > {tau} "
          f"for {k}+ days)")
    for idx, r in daily.iterrows():
        flag = "  <-- sustained" if r["escalate"] else ""
        print(f"  {str(idx)[:10]}  {r['dominant_emotion']:<9} n={int(r['n_messages'])}  "
              f"{bar(r['neg_smooth'])} {r['neg_smooth']:.2f}{flag}")

    weekly = trends.aggregate_predictions(stored, "W", classes, negative)
    print("\nWeekly summary")
    for idx, r in weekly.iterrows():
        print(f"  week of {str(idx)[:10]}  dominant={r['dominant_emotion']:<9} neg={r['neg_score']:.2f}  "
              f"messages={int(r['n_messages'])}")

    window = int(cget(cfg, "trends.window_days", 14))
    recent = daily.tail(window)
    escalate = bool(recent["escalate"].any())
    rec = recommend(str(recent["dominant_emotion"].iloc[-1]), escalate=escalate,
                    region=cget(cfg, "recommend.region", "IN"))
    print_recommendation(rec)

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.plot(stored["timestamp"], stored[[f"p_{c}" for c in negative]].sum(axis=1), "o", color="#9aa5b1",
                ms=5, label="message")
        ax.plot(pd.to_datetime(daily.index), daily["neg_smooth"], color="#0072B2", lw=2, label="smoothed trend")
        ax.axhline(tau, color="#D55E00", ls="--", lw=1, label=f"threshold ({tau})")
        ax.set_ylim(0, 1)
        ax.set_ylabel("negative-affect score")
        ax.set_title("Mood trend - demo journal (invented data)", loc="left", fontsize=11)
        ax.legend(frameon=False, fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.autofmt_xdate()
        fig.tight_layout()
        out = ROOT / "results" / "figures" / "demo_mood_trend.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"\nSaved {out}")

    if not args.keep_db:
        db_path.unlink(missing_ok=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    ap.add_argument("--model", default="auto", help="lr|svm|rf|bilstm|distilbert|roberta|auto (best by macro-F1)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--text", help="classify one message instead of the scripted journal")
    ap.add_argument("--plot", action="store_true", help="save results/figures/demo_mood_trend.png")
    ap.add_argument("--db", help="SQLite path for the demo (default results/demo_mood.db)")
    ap.add_argument("--keep-db", action="store_true", help="keep the SQLite file after the demo")
    ap.add_argument("--app", action="store_true", help="launch the Streamlit dashboard (app/app.py)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_cfg(Path(args.config))

    print("=" * 72 + f"\n{DISCLAIMER}\n{CONSENT}\n" + "=" * 72)
    if args.app:
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "app.py")], cwd=ROOT)
    if args.text:
        from src.inference import EmotionPredictor

        run_single(EmotionPredictor.from_config(cfg, model=pick_model(cfg, args.model), device=args.device),
                   cfg, args.text)
        return 0
    return run_journal(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
