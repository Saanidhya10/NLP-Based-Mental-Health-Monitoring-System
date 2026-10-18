"""Tests for src/inference.py.

Checks the message-level classifier interface: f: X -> Delta^|E|
(a probability distribution over the 8 emotions) and argmax labelling.

These tests need a trained model on disk. If none is available they skip
rather than fail. Add your function names to CANDIDATES if they differ.
"""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

inference = pytest.importorskip("src.inference", reason="src/inference.py not importable")

EMOTIONS = ["Joy", "Sadness", "Anger", "Fear", "Anxiety", "Stress", "Surprise", "Neutral"]
CANDIDATES = ["predict", "predict_emotion", "classify", "predict_text", "predict_proba"]


def _find(names):
    for n in names:
        fn = getattr(inference, n, None)
        if callable(fn):
            return fn
    return None


predict = _find(CANDIDATES)
if predict is None:
    pytest.skip(f"none of {CANDIDATES} found in src/inference.py", allow_module_level=True)


def _call(text):
    """Call predict, skipping if no trained model is available."""
    try:
        return predict(text)
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"No trained model available: {e}")


def _label_and_probs(result):
    """Normalise the many shapes a prediction might take."""
    label, probs = None, None
    if isinstance(result, dict):
        label = result.get("label") or result.get("emotion") or result.get("prediction")
        probs = result.get("probs") or result.get("probabilities") or result.get("scores")
        if probs is None and set(EMOTIONS) <= set(result):
            probs = {e: result[e] for e in EMOTIONS}
    elif isinstance(result, tuple) and len(result) >= 2:
        label, probs = result[0], result[1]
    elif isinstance(result, str):
        label = result
    return label, probs


def _as_list(probs):
    if isinstance(probs, dict):
        return [float(probs[e]) for e in EMOTIONS if e in probs]
    return [float(p) for p in probs]


def test_returns_valid_label():
    label, _ = _label_and_probs(_call("I am so happy today"))
    assert label is not None
    assert str(label).capitalize() in EMOTIONS


def test_probabilities_form_distribution():
    _, probs = _label_and_probs(_call("I am worried about my exams"))
    if probs is None:
        pytest.skip("predict() does not return probabilities")
    vals = _as_list(probs)
    assert len(vals) == len(EMOTIONS)
    assert all(0.0 <= v <= 1.0 for v in vals)
    assert sum(vals) == pytest.approx(1.0, abs=1e-3)


def test_label_is_argmax_of_probabilities():
    label, probs = _label_and_probs(_call("This makes me so angry"))
    if probs is None or not isinstance(probs, dict):
        pytest.skip("Need dict of probabilities to verify argmax")
    assert str(label).capitalize() == max(probs, key=probs.get).capitalize()


def test_empty_text_does_not_crash():
    try:
        _call("")
    except (ValueError, AssertionError):
        pass  # explicit rejection of empty input is acceptable


def test_long_text_is_truncated_not_crashing():
    out = _call("I feel stressed about everything. " * 400)
    label, _ = _label_and_probs(out)
    assert label is not None


def test_unicode_and_emoji_input():
    out = _call("so tired 😢 और बहुत उदास हूँ")
    assert _label_and_probs(out)[0] is not None


def test_deterministic_for_same_input():
    a = _label_and_probs(_call("I can't stop worrying"))[0]
    b = _label_and_probs(_call("I can't stop worrying"))[0]
    assert a == b


def test_latency_reasonable():
    """Interactive use (paper 6.5): a single short input should return quickly
    once the model is warm. Generous bound so CPU-only machines pass."""
    _call("warm up")
    start = time.perf_counter()
    _call("I feel okay today")
    assert time.perf_counter() - start < 5.0
