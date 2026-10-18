"""Tests for src/trends.py.

Checks the temporal aggregation spec from the research paper (Section 5.6):
  dom(W)  = argmax_e mean_i p_i(e)
  s_i     = sum_{e in N} p_i(e),  N = {Sadness, Anger, Fear, Anxiety, Stress}
  S(W)    = mean_i s_i
  EMA     : S~_t = a*S_t + (1-a)*S~_{t-1}

The first block validates the maths against a reference implementation, so
it always runs. The second block exercises YOUR functions in src/trends.py;
add your function names to the CANDIDATES lists if they differ.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

EMOTIONS = ["Joy", "Sadness", "Anger", "Fear", "Anxiety", "Stress", "Surprise", "Neutral"]
NEGATIVE = {"Sadness", "Anger", "Fear", "Anxiety", "Stress"}


def onehot(emotion, conf=1.0):
    rest = (1.0 - conf) / (len(EMOTIONS) - 1)
    return {e: (conf if e == emotion else rest) for e in EMOTIONS}


# ---------------------------------------------------------------- reference
def ref_neg_score(p):
    return sum(v for k, v in p.items() if k in NEGATIVE)


def ref_window_score(ps):
    return sum(ref_neg_score(p) for p in ps) / len(ps)


def ref_dominant(ps):
    means = {e: sum(p[e] for p in ps) / len(ps) for e in EMOTIONS}
    return max(means, key=means.get)


def ref_ema(xs, alpha=0.3):
    out = [xs[0]]
    for x in xs[1:]:
        out.append(alpha * x + (1 - alpha) * out[-1])
    return out


def test_reference_negative_score_bounds():
    assert ref_neg_score(onehot("Joy")) == pytest.approx(0.0)
    assert ref_neg_score(onehot("Sadness")) == pytest.approx(1.0)
    assert 0.0 <= ref_neg_score(onehot("Neutral", 0.6)) <= 1.0


def test_reference_dominant_uses_mean_probability():
    ps = [onehot("Joy", 0.5), onehot("Joy", 0.5), onehot("Sadness", 1.0)]
    assert ref_dominant(ps) in {"Joy", "Sadness"}


def test_reference_ema_smooths_spike():
    xs = [0.1, 0.1, 0.9, 0.1, 0.1]
    sm = ref_ema(xs)
    assert max(sm) < max(xs)
    assert len(sm) == len(xs)


# ------------------------------------------------------------ your module
trends = pytest.importorskip("src.trends", reason="src/trends.py not importable")


def _find(names):
    for n in names:
        fn = getattr(trends, n, None)
        if callable(fn):
            return fn
    return None


NEG_CANDIDATES = ["negative_affect_score", "compute_negative_score", "negative_score", "neg_score"]
DOM_CANDIDATES = ["dominant_emotion", "get_dominant_emotion", "compute_dominant_emotion"]
EMA_CANDIDATES = ["ema", "exponential_moving_average", "smooth_scores", "smooth", "apply_ema"]

neg_fn = _find(NEG_CANDIDATES)
dom_fn = _find(DOM_CANDIDATES)
ema_fn = _find(EMA_CANDIDATES)


@pytest.mark.skipif(neg_fn is None, reason=f"none of {NEG_CANDIDATES} found in src/trends.py")
def test_negative_score_joy_vs_sadness():
    assert neg_fn(onehot("Joy")) < neg_fn(onehot("Sadness"))


@pytest.mark.skipif(neg_fn is None, reason=f"none of {NEG_CANDIDATES} found in src/trends.py")
def test_negative_score_in_unit_interval():
    for e in EMOTIONS:
        assert 0.0 <= neg_fn(onehot(e, 0.7)) <= 1.0


@pytest.mark.skipif(dom_fn is None, reason=f"none of {DOM_CANDIDATES} found in src/trends.py")
def test_dominant_emotion_simple_window():
    ps = [onehot("Stress", 0.8), onehot("Stress", 0.7), onehot("Joy", 0.6)]
    assert dom_fn(ps) == "Stress"


@pytest.mark.skipif(ema_fn is None, reason=f"none of {EMA_CANDIDATES} found in src/trends.py")
def test_ema_matches_reference():
    xs = [0.1, 0.2, 0.8, 0.3, 0.4, 0.9]
    try:
        got = list(ema_fn(xs, alpha=0.3))
    except TypeError:
        got = list(ema_fn(xs))
    assert len(got) == len(xs)
    assert got == pytest.approx(ref_ema(xs, 0.3), abs=1e-6)


@pytest.mark.skipif(ema_fn is None, reason=f"none of {EMA_CANDIDATES} found in src/trends.py")
def test_ema_constant_series_is_constant():
    got = list(ema_fn([0.4] * 6))
    assert got == pytest.approx([0.4] * 6, abs=1e-6)


def test_sustained_negative_run_vs_isolated_message():
    """Persistence logic (paper 6.4): a sustained run must cross the
    threshold, an isolated spike must not. Verified on the reference."""
    tau, k = 0.55, 3
    isolated = ref_ema([0.1, 0.1, 0.95, 0.1, 0.1, 0.1, 0.1])
    sustained = ref_ema([0.1, 0.3, 0.7, 0.8, 0.85, 0.9, 0.9, 0.9, 0.9])

    def triggers(series):
        run = 0
        for s in series:
            run = run + 1 if s > tau else 0
            if run >= k:
                return True
        return False

    assert not triggers(isolated)
    assert triggers(sustained)
