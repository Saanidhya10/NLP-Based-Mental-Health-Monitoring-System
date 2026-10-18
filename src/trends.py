"""Temporal aggregation, smoothing, escalation detection and trend-quality evaluation (paper, Sec. 5.6, 6.4).

Definitions (per window W of messages i with probability vectors p_i):
    dom(W)  = argmax_e mean_i p_i(e)
    s_i     = sum_{e in N} p_i(e),    S(W) = mean_i s_i        N = {Sadness, Anger, Fear, Anxiety, Stress}
    EMA     : S~_t = alpha * S_t + (1 - alpha) * S~_{t-1}
    Escalation: S~_t > tau for >= k consecutive (observed) days within the recent window.

No dataset has real per-user timelines, so quality is tested on SCRIPTED timelines built by sampling labelled
test messages in a known order (``build_scripted_timeline``). This tests that the module behaves as
specified; it does not establish clinical validity (state this in the report).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import (ID2LABEL, LABEL2ID, LABELS, NEGATIVE_LABELS, NUM_CLASSES, get_logger, load_config,
                       path_from_cfg)

log = get_logger("trends")

PROB_COLS = [f"p_{l}" for l in LABELS]
NEG_IDX = [LABEL2ID[l] for l in NEGATIVE_LABELS]
NEG_COLS = [f"p_{l}" for l in NEGATIVE_LABELS]
FREQ = {"daily": "D", "weekly": "W", "monthly": "MS"}


# --------------------------------------------------------------------------- core computations
def negative_scores(probs) -> np.ndarray | float:
    if isinstance(probs, dict):
        return float(sum(probs.get(e, 0.0) for e in NEGATIVE_LABELS))
    arr = np.asarray(probs, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] == NUM_CLASSES:
            return float(arr[NEG_IDX].sum())
        return float(arr.sum())
    return arr[:, NEG_IDX].sum(axis=1)


neg_score = negative_scores
negative_score = negative_scores
compute_negative_score = negative_scores
negative_affect_score = negative_scores


def dominant_emotion(probs) -> str:
    """Return dominant emotion name from a dict, list of dicts, array or DataFrame."""
    if isinstance(probs, pd.DataFrame):
        return current_dominant(probs)
    if isinstance(probs, (list, tuple)):
        if len(probs) == 0:
            return "Neutral"
        if isinstance(probs[0], dict):
            means = {e: sum(p.get(e, 0.0) for p in probs) / len(probs) for e in LABELS}
            return max(means, key=means.get)
        arr = np.asarray(probs, dtype=float)
        mean_p = arr.mean(axis=0)
        return LABELS[int(mean_p.argmax())]
    if isinstance(probs, dict):
        return max(probs, key=probs.get)
    arr = np.asarray(probs, dtype=float)
    return LABELS[int(arr.argmax())]


get_dominant_emotion = dominant_emotion
compute_dominant_emotion = dominant_emotion


def build_prediction_frame(timestamps, probs, user_id=None, texts=None, true_labels=None) -> pd.DataFrame:
    """One row per message: timestamp, p_<emotion> columns, neg_score, emotion, confidence."""
    probs = np.asarray(probs, dtype=float)
    df = pd.DataFrame(probs, columns=PROB_COLS)
    df.insert(0, "timestamp", pd.to_datetime(list(timestamps)))
    if user_id is not None:
        df.insert(0, "user_id", user_id)
    df["neg_score"] = negative_scores(probs)
    df["emotion"] = [ID2LABEL[i] for i in probs.argmax(axis=1)]
    df["confidence"] = probs.max(axis=1)
    if texts is not None:
        df["text"] = list(texts)
    if true_labels is not None:
        df["true_label"] = list(true_labels)
    return df.sort_values("timestamp").reset_index(drop=True)


def aggregate(df: pd.DataFrame, freq: str = "daily") -> pd.DataFrame:
    """Window summary: dominant emotion, negative-affect score S(W), message count and mean probabilities."""
    rule = FREQ.get(freq, freq)
    grouped = df.groupby(pd.Grouper(key="timestamp", freq=rule))
    counts = grouped.size().rename("n_messages")
    mean_p = grouped[PROB_COLS].mean()
    out = mean_p[counts > 0].copy()
    out.insert(0, "n_messages", counts[counts > 0])
    out.insert(0, "neg_score", out[NEG_COLS].sum(axis=1))
    out.insert(0, "dominant_emotion", out[PROB_COLS].idxmax(axis=1).str.replace("p_", "", regex=False))
    return out


def smooth_scores(series, alpha: float = 0.3) -> pd.Series:
    if not isinstance(series, pd.Series):
        series = pd.Series(series, dtype=float)
    return series.ewm(alpha=alpha, adjust=False).mean()


ema = smooth_scores
apply_ema = smooth_scores
exponential_moving_average = smooth_scores


def detect_escalation(smoothed: pd.Series, tau: float, k: int, window_days: int | None = None) -> dict:
    """True when the smoothed score exceeds tau for >= k consecutive observed days in the recent window."""
    if smoothed.empty:
        return {"triggered": False, "longest_run": 0, "current_run": 0, "trigger_date": None, "latest": None}
    s = smoothed
    if window_days:
        s = s[s.index > s.index.max() - pd.Timedelta(days=window_days)]
    above = (s > tau).to_numpy()
    longest = run = 0
    trigger = None
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        longest = max(longest, run)
        if run >= k and trigger is None:
            trigger = s.index[i]
    return {"triggered": longest >= k, "longest_run": int(longest), "current_run": int(run),
            "trigger_date": trigger, "latest": float(s.iloc[-1])}


def current_dominant(df: pd.DataFrame, recent_days: int = 7) -> str:
    recent = df[df["timestamp"] > df["timestamp"].max() - pd.Timedelta(days=recent_days)]
    return ID2LABEL[int(recent[PROB_COLS].mean().to_numpy().argmax())]


def mood_summary(df: pd.DataFrame, cfg: dict | None = None) -> dict:
    """Everything the dashboard and recommender need, from message-level predictions."""
    cfg = cfg or load_config()
    tc = cfg["trends"]
    if df.empty:
        return {"daily": pd.DataFrame(), "weekly": pd.DataFrame(), "monthly": pd.DataFrame(),
                "smoothed": pd.Series(dtype=float), "escalation": detect_escalation(pd.Series(dtype=float), 1, 1),
                "latest_dominant": None, "latest_neg_score": None}
    daily = aggregate(df, "daily")
    smoothed = smooth_scores(daily["neg_score"], tc["alpha"])
    return {
        "daily": daily, "weekly": aggregate(df, "weekly"), "monthly": aggregate(df, "monthly"),
        "smoothed": smoothed,
        "escalation": detect_escalation(smoothed, tc["tau"], tc["k"], tc.get("window_days")),
        "latest_dominant": current_dominant(df, tc.get("recent_days", 7)),
        "latest_neg_score": float(smoothed.iloc[-1]),
    }


# --------------------------------------------------------------------------- scripted timelines
SCENARIOS = ["stable_positive", "drift_to_stress", "isolated_negative", "sustained_negative", "recovery"]


def _p_neg_curve(name: str, days: int) -> np.ndarray:
    d = np.arange(days)
    f = d / max(days - 1, 1)
    if name == "stable_positive":
        return np.full(days, 0.08)
    if name == "drift_to_stress":
        return 0.10 + 0.75 * f
    if name == "isolated_negative":
        p = np.full(days, 0.08)
        p[days // 2] = 1.0
        return p
    if name == "sustained_negative":
        return np.where(d >= days // 4, 0.85, 0.08)
    if name == "recovery":
        return 0.85 - 0.77 * f
    raise ValueError(f"Unknown scenario '{name}'. Choose from {SCENARIOS}")


def _class_weights_for_day(p_neg: float) -> np.ndarray:
    neg_split = {"Stress": 0.4, "Sadness": 0.3, "Anxiety": 0.2, "Anger": 0.1}
    pos_split = {"Joy": 0.5, "Neutral": 0.5}
    w = np.zeros(NUM_CLASSES)
    for k, v in neg_split.items():
        w[LABEL2ID[k]] += p_neg * v
    for k, v in pos_split.items():
        w[LABEL2ID[k]] += (1 - p_neg) * v
    return w


def build_scripted_timeline(scenario: str, pool_df: pd.DataFrame, cfg: dict | None = None, seed: int = 0,
                            user_id: str | None = None, text_col: str = "text_transformer") -> pd.DataFrame:
    """Sample labelled messages (from the TEST split) in a scripted emotional order. Returns
    user_id, timestamp, text, true_label."""
    cfg = cfg or load_config()
    sc = cfg["trends"]["simulation"]
    rng = np.random.default_rng(seed)
    pools = {l: pool_df.loc[pool_df["label"] == l, text_col].astype(str).tolist() for l in LABELS}
    available = np.array([len(pools[l]) > 0 for l in LABELS])
    start = pd.Timestamp(sc["start_date"])
    lo, hi = sc["msgs_per_day"]
    rows = []
    for day, p_neg in enumerate(_p_neg_curve(scenario, sc["days"])):
        w = _class_weights_for_day(float(p_neg)) * available
        w = w / w.sum()
        for _ in range(int(rng.integers(lo, hi + 1))):
            label = LABELS[int(rng.choice(NUM_CLASSES, p=w))]
            text = pools[label][int(rng.integers(len(pools[label])))]
            ts = start + pd.Timedelta(days=day, hours=int(rng.integers(8, 22)), minutes=int(rng.integers(0, 60)))
            rows.append({"user_id": user_id or f"sim_{scenario}_{seed}", "timestamp": ts, "text": text,
                         "true_label": label})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- trend-quality evaluation
def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan")


def evaluate_trend_quality(pred_df: pd.DataFrame, cfg: dict | None = None) -> dict:
    """Compare aggregated predictions against the scripted ground truth (needs a ``true_label`` column).

    * daily_dominant_agreement - aggregated dominant emotion == scripted dominant emotion
    * single_message_agreement - baseline: label of the day's LAST message == scripted dominant (RQ4)
    * daily_polarity_agreement - negative-affect side (S > 0.5) matches
    * neg_score_corr / mae     - predicted vs scripted daily negative share (raw and EMA-smoothed)
    * escalation_match         - predicted trigger == trigger of the same rule applied to the TRUE labels
    """
    cfg = cfg or load_config()
    tc = cfg["trends"]
    onehot = np.eye(NUM_CLASSES)[pred_df["true_label"].map(LABEL2ID).to_numpy()]
    true_df = build_prediction_frame(pred_df["timestamp"], onehot)
    p_daily, t_daily = aggregate(pred_df, "daily"), aggregate(true_df, "daily")
    idx = p_daily.index.intersection(t_daily.index)
    p_daily, t_daily = p_daily.loc[idx], t_daily.loc[idx]

    last_msg = pred_df.assign(day=pred_df["timestamp"].dt.normalize()).groupby("day")["emotion"].last().loc[idx]
    p_s, t_s = smooth_scores(p_daily["neg_score"], tc["alpha"]), smooth_scores(t_daily["neg_score"], tc["alpha"])
    p_esc = detect_escalation(p_s, tc["tau"], tc["k"], tc.get("window_days"))
    t_esc = detect_escalation(t_s, tc["tau"], tc["k"], tc.get("window_days"))
    return {
        "n_messages": len(pred_df), "n_days": len(idx),
        "daily_dominant_agreement": float((p_daily["dominant_emotion"] == t_daily["dominant_emotion"]).mean()),
        "single_message_agreement": float((last_msg.values == t_daily["dominant_emotion"].values).mean()),
        "daily_polarity_agreement": float(((p_daily["neg_score"] > 0.5) == (t_daily["neg_score"] > 0.5)).mean()),
        "neg_score_corr": _corr(p_daily["neg_score"].to_numpy(), t_daily["neg_score"].to_numpy()),
        "neg_score_mae": float((p_daily["neg_score"] - t_daily["neg_score"]).abs().mean()),
        "smoothed_corr": _corr(p_s.to_numpy(), t_s.to_numpy()),
        "escalation_pred": p_esc["triggered"], "escalation_true": t_esc["triggered"],
        "escalation_match": p_esc["triggered"] == t_esc["triggered"],
    }


def run_trend_evaluation(predictor, pool_df: pd.DataFrame, cfg: dict | None = None, seeds=(0, 1, 2),
                         save: bool = True) -> pd.DataFrame:
    """Build every scripted scenario x seed, run ``predictor.predict_proba`` and score the trends.
    ``pool_df`` must be the held-out TEST split (columns: label, text_transformer)."""
    cfg = cfg or load_config()
    rows = []
    for scenario in cfg["trends"]["simulation"]["scenarios"]:
        for seed in seeds:
            tl = build_scripted_timeline(scenario, pool_df, cfg, seed)
            probs = predictor.predict_proba(tl["text"].tolist())
            pf = build_prediction_frame(tl["timestamp"], probs, true_labels=tl["true_label"])
            rows.append({"scenario": scenario, "seed": seed, **evaluate_trend_quality(pf, cfg)})
    res = pd.DataFrame(rows)
    if save:
        out = path_from_cfg(cfg, "results_dir", create=True) / "trend_quality.csv"
        res.to_csv(out, index=False)
        log.info("Saved %s", out)
    return res
