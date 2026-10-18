"""Backend building blocks for the Streamlit app (no Streamlit imports here, so it is easy to test).

Self-contained on purpose: you can later move these pieces into src/inference.py, src/trends.py,
src/recommend.py and src/database.py and import them from there instead.
"""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data/processed/mood_app.db"

EMOTIONS = ["Joy", "Sadness", "Anger", "Fear", "Anxiety", "Stress", "Surprise", "Neutral"]
NEGATIVE = ["Sadness", "Anger", "Fear", "Anxiety", "Stress"]
PCOLS = [f"p_{e}" for e in EMOTIONS]
NEG_PCOLS = [f"p_{e}" for e in NEGATIVE]

# Verify these before any real deployment - numbers change and differ by region.
HELPLINES = {
    "India - Tele-MANAS (24x7, free)": "14416",
    "India - emergency services": "112",
}

# ---------------------------------------------------------------- parameters
DEFAULT_PARAMS = {"alpha": 0.3, "tau": 0.55, "k": 3, "max_gap_days": 3}


def load_params() -> dict:
    """tau / k / alpha calibrated in notebook 06 (results/trend_params.json), else conservative defaults."""
    p = dict(DEFAULT_PARAMS)
    f = ROOT / "results/trend_params.json"
    if f.exists():
        try:
            p.update({k: v for k, v in json.loads(f.read_text()).items() if k in p})
        except Exception:
            pass
    return p


# ---------------------------------------------------------------- preprocessing
_URL = re.compile(r"https?://\S+|www\.\S+")
_HTML = re.compile(r"<[^>]+>|&\w+;")
_HANDLE = re.compile(r"(?<!\w)@\w+|(?<!\w)/?u/\w+")
_REPEAT = re.compile(r"([!?.])\1{2,}")
_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")
_classical_cache: dict = {}


def preprocess_tf(text: str) -> str:
    """Same light cleaning used for training transformers (notebook 01)."""
    t = _HANDLE.sub(" ", _HTML.sub(" ", _URL.sub(" ", str(text))))
    try:
        import emoji
        t = emoji.demojize(t, delimiters=(" ", " ")).replace("_", " ")
    except ImportError:
        pass
    return _WS.sub(" ", _REPEAT.sub(r"\1\1", t)).strip()


def preprocess_classical(text: str) -> str:
    """Same pipeline used for TF-IDF models: lowercase, stop-words removed (negations kept), lemmatised."""
    if not _classical_cache:
        import nltk
        for pkg in ["stopwords", "wordnet", "omw-1.4"]:
            nltk.download(pkg, quiet=True)
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        neg_base = {"no", "nor", "not", "never", "none", "nothing", "neither", "nobody", "cannot", "don", "didn",
                    "doesn", "isn", "wasn", "aren", "couldn", "shouldn", "wouldn", "hasn", "haven", "hadn", "won",
                    "weren", "needn", "mustn", "shan", "mightn", "ain"}
        sw = set(stopwords.words("english"))
        negations = {w for w in sw if w in neg_base or "n't" in w} | neg_base
        _classical_cache["stop"] = sw - negations
        _classical_cache["lem"] = WordNetLemmatizer()
    t = preprocess_tf(text).lower().replace("\u2019", "'")
    return " ".join(_classical_cache["lem"].lemmatize(w) for w in _TOKEN.findall(t) if w not in _classical_cache["stop"])


# ---------------------------------------------------------------- model
class Predictor:
    """Text -> probability distribution over the 8 emotions. Transformer if trained, else classical LR."""

    def __init__(self):
        self.model = None
        self.pipe = None
        meta = ROOT / "models/final_model.json"
        keys = ([json.loads(meta.read_text())["model"]] if meta.exists() else []) + ["distilbert", "roberta"]
        for key in keys:
            path = ROOT / "models" / key / "final"
            if (path / "config.json").exists():
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                self._torch = torch
                self.tok = AutoTokenizer.from_pretrained(str(path))
                self.model = AutoModelForSequenceClassification.from_pretrained(str(path)).eval()
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model.to(self.device)
                self.name, self.kind = key, "transformer"
                return
        lr_candidates = [
            ROOT / "models/classical/logreg_baseline/pipeline.joblib",
            ROOT / "models/classical/logreg_baseline.joblib",
            ROOT / "models/classical/svm_baseline/pipeline.joblib",
            ROOT / "models/classical/rf_baseline/pipeline.joblib",
        ]
        for lr in lr_candidates:
            if lr.exists():
                import joblib
                self.pipe = joblib.load(lr)
                self.name = lr.parent.name if lr.name == "pipeline.joblib" else lr.stem
                self.kind, self.device = "classical", "cpu"
                return
        raise FileNotFoundError(
            "No trained model found. Run notebooks 02 (classical) or 04 (transformers) first - "
            "expected models/<model>/final/ or models/classical/logreg_baseline/pipeline.joblib.")

    def predict_proba(self, text: str) -> np.ndarray:
        if self.kind == "classical":
            return self.pipe.predict_proba([preprocess_classical(text)])[0]
        enc = self.tok(preprocess_tf(text), truncation=True, max_length=64, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            return self._torch.softmax(self.model(**enc).logits, -1)[0].cpu().numpy()

    @property
    def label(self) -> str:
        return f"{self.name} ({self.kind})"


# ---------------------------------------------------------------- storage (derived scores only, never raw text)
def _con() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS predictions (
        user_id TEXT, timestamp TEXT, emotion TEXT, confidence REAL, sentiment_score REAL, probs TEXT)""")
    return con


def add_prediction(user_id: str, probs: np.ndarray, ts: datetime | None = None) -> None:
    ts = ts or datetime.now()
    probs = np.asarray(probs, float)
    s = float(probs[[EMOTIONS.index(e) for e in NEGATIVE]].sum())
    with _con() as con:
        con.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?)",
                    (user_id, ts.isoformat(timespec="seconds"), EMOTIONS[int(probs.argmax())],
                     float(probs.max()), s, json.dumps([round(float(x), 5) for x in probs])))


def load_history(user_id: str) -> pd.DataFrame:
    with _con() as con:
        df = pd.read_sql("SELECT * FROM predictions WHERE user_id=? ORDER BY timestamp", con, params=(user_id,))
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df[PCOLS] = np.array([json.loads(p) for p in df["probs"]])
    df["s"] = df[NEG_PCOLS].sum(axis=1)
    return df


def delete_history(user_id: str) -> int:
    with _con() as con:
        return con.execute("DELETE FROM predictions WHERE user_id=?", (user_id,)).rowcount


def export_history_csv(user_id: str) -> bytes:
    df = load_history(user_id)
    cols = ["timestamp", "emotion", "confidence", "sentiment_score"] + PCOLS
    return (df[cols] if not df.empty else pd.DataFrame(columns=cols)).to_csv(index=False).encode()


def add_demo_history(user_id: str, days: int = 28, seed: int = 0) -> None:
    """SIMULATED data (random probabilities, no model) drifting from positive to negative mood - for demos only."""
    rng = np.random.default_rng(seed)
    pos = np.array([.45, .03, .02, .02, .02, .03, .13, .30])
    neg = np.array([.02, .30, .08, .05, .20, .27, .01, .07])
    start = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0) - pd.Timedelta(days=days - 1)
    for d in range(days):
        w = d / (days - 1)
        for _ in range(int(rng.integers(1, 4))):
            p = rng.dirichlet(60 * ((1 - w) * pos + w * neg) + 0.3)
            add_prediction(user_id, p, start + pd.Timedelta(days=d, hours=int(rng.integers(0, 12)),
                                                              minutes=int(rng.integers(0, 60))))


# ---------------------------------------------------------------- temporal aggregation
def daily_summary(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Per-day mean probabilities, dominant emotion, negative-affect S and its EMA S~ (days with entries only)."""
    d = df.set_index("timestamp")[PCOLS + ["s"]].resample("D").mean().dropna(subset=["s"])
    d["n_entries"] = df.set_index("timestamp")["s"].resample("D").size().reindex(d.index)
    d["dominant"] = d[PCOLS].idxmax(axis=1).str[2:]
    d["S"] = d["s"]
    d["S_ema"] = d["S"].ewm(alpha=alpha, adjust=False).mean()
    return d


def window_summary(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Weekly ('7D') or monthly ('MS') summaries from message-level probabilities."""
    w = df.set_index("timestamp")[PCOLS + ["s"]].resample(rule).mean().dropna(subset=["s"])
    w["dominant"] = w[PCOLS].idxmax(axis=1).str[2:]
    w["S"] = w["s"]
    return w


def escalation_flags(daily: pd.DataFrame, tau: float, k: int, max_gap_days: int = 3) -> np.ndarray:
    """True on days where S~ has exceeded tau for >= k consecutive entry-days (a gap > max_gap_days resets the run)."""
    run, prev, flags = 0, None, []
    for ts, v in zip(daily.index, daily["S_ema"].values):
        gap = (ts - prev).days if prev is not None else 0
        prev = ts
        run = (run + 1 if gap <= max_gap_days else 1) if v > tau else 0
        flags.append(run >= k)
    return np.array(flags, dtype=bool)


# ---------------------------------------------------------------- rule-based recommendations (non-diagnostic)
SUGGESTIONS = {
    "Stress": ("Try a slow breathing exercise",
               "Stress seems to be showing up in your recent entries. A few rounds of slow breathing "
               "(in for 4 seconds, hold for 4, out for 6) can help your body settle."),
    "Anxiety": ("Try a grounding exercise",
                "Things may feel on edge lately. Name 5 things you can see, 4 you can feel, 3 you can hear, "
                "2 you can smell and 1 you can taste."),
    "Sadness": ("Be gentle with yourself",
                "Your recent entries sound heavy. Reaching out to someone you trust, or spending a little time "
                "on something comforting, can help."),
    "Anger": ("Give yourself some space",
              "There seems to be some frustration lately. A short walk or writing down what is bothering you "
              "can create a bit of distance."),
    "Fear": ("Make the worry smaller",
             "Worry seems present in your entries. Naming the specific worry and one small next step can make "
             "it feel more manageable."),
}
ESCALATION_TITLE = "Consider talking to someone who can support you"
ESCALATION_TEXT = ("Over the past days your entries have shown a persistent low mood. This tool cannot diagnose "
                   "anything, but speaking with a qualified mental-health professional could really help. "
                   "You do not have to wait until things feel worse.")


def recommend(daily: pd.DataFrame, flags: np.ndarray, tau: float) -> dict:
    """Level in {none, suggestion, escalation} based on persistence (not on a single message)."""
    if daily.empty:
        return {"level": "none", "title": "", "text": ""}
    last, dom = daily.iloc[-1], daily.iloc[-1]["dominant"]
    if bool(flags[-1]):
        return {"level": "escalation", "title": ESCALATION_TITLE, "text": ESCALATION_TEXT, "dominant": dom}
    if dom in SUGGESTIONS and last["S_ema"] > 0.7 * tau:
        t, x = SUGGESTIONS[dom]
        return {"level": "suggestion", "title": t, "text": x, "dominant": dom}
    return {"level": "none", "title": "Nothing to flag right now",
            "text": "Your recent entries do not show a persistent pattern of low mood.", "dominant": dom}
