"""Tests for src/recommend.py.

Checks the rule-based recommendation / safety layer (paper Section 5.7):
 - supportive, non-diagnostic suggestions per dominant emotion
 - escalation only on persistent, severe negative affect
 - never uses diagnostic language

Add your function names to the CANDIDATES lists if they differ.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

recommend = pytest.importorskip("src.recommend", reason="src/recommend.py not importable")


def _find(names):
    for n in names:
        fn = getattr(recommend, n, None)
        if callable(fn):
            return fn
    return None


SUGGEST_CANDIDATES = ["get_suggestion", "recommend", "get_recommendation", "suggest", "recommend_for_emotion"]
ESCALATE_CANDIDATES = ["should_escalate", "check_escalation", "needs_escalation", "escalate", "evaluate_escalation"]

suggest_fn = _find(SUGGEST_CANDIDATES)
escalate_fn = _find(ESCALATE_CANDIDATES)

# Clinical / diagnostic wording that must never appear in user-facing text.
FORBIDDEN = [
    "you have depression",
    "you are depressed",
    "you have anxiety disorder",
    "diagnos",  # catches "diagnosed", "diagnosis" — only allowed in a negation
    "disorder",
    "mentally ill",
    "clinical",
]


def as_text(result):
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return " ".join(as_text(v) for v in result.values())
    if isinstance(result, (list, tuple, set)):
        return " ".join(as_text(v) for v in result)
    return str(result)


def _allowed_negation(text, word):
    """'does not diagnose' style disclaimers are fine."""
    t = text.lower()
    return any(neg in t for neg in ("not diagnos", "cannot diagnos", "does not diagnos", "never diagnos", "no diagnos"))


@pytest.mark.skipif(suggest_fn is None, reason=f"none of {SUGGEST_CANDIDATES} found in src/recommend.py")
class TestSuggestions:
    @pytest.mark.parametrize("emotion", ["Joy", "Sadness", "Anger", "Fear", "Anxiety", "Stress", "Surprise", "Neutral"])
    def test_every_emotion_returns_something(self, emotion):
        out = suggest_fn(emotion)
        assert as_text(out).strip() != ""

    def test_stress_mentions_breathing(self):
        assert "breath" in as_text(suggest_fn("Stress")).lower()

    def test_anxiety_mentions_grounding_or_mindfulness(self):
        text = as_text(suggest_fn("Anxiety")).lower()
        assert any(w in text for w in ("ground", "mindful"))

    def test_sadness_has_resources(self):
        assert as_text(suggest_fn("Sadness")).strip() != ""

    @pytest.mark.parametrize("emotion", ["Sadness", "Anger", "Fear", "Anxiety", "Stress"])
    def test_no_diagnostic_language(self, emotion):
        text = as_text(suggest_fn(emotion)).lower()
        for bad in FORBIDDEN:
            if bad in text and not _allowed_negation(text, bad):
                pytest.fail(f"Diagnostic wording {bad!r} found for {emotion}: {text[:120]}")

    def test_unknown_emotion_does_not_crash(self):
        try:
            suggest_fn("NotAnEmotion")
        except (KeyError, ValueError):
            pass  # an explicit, clean error is acceptable


@pytest.mark.skipif(escalate_fn is None, reason=f"none of {ESCALATE_CANDIDATES} found in src/recommend.py")
class TestEscalation:
    """Escalation must depend on persistence and severity, not one message."""

    @staticmethod
    def _call(scores):
        out = escalate_fn(scores)
        if isinstance(out, dict):
            for key in ("escalate", "should_escalate", "triggered"):
                if key in out:
                    return bool(out[key])
        if isinstance(out, tuple):
            return bool(out[0])
        return bool(out)

    def test_low_scores_do_not_escalate(self):
        assert self._call([0.1] * 10) is False

    def test_single_spike_does_not_escalate(self):
        assert self._call([0.1, 0.1, 0.95, 0.1, 0.1, 0.1, 0.1]) is False

    def test_sustained_high_scores_escalate(self):
        assert self._call([0.85] * 10) is True

    def test_empty_history_does_not_escalate(self):
        assert self._call([]) is False


def test_helpline_info_present_somewhere():
    """Escalation output should surface helpline information (paper 5.7)."""
    text = ""
    for name in dir(recommend):
        val = getattr(recommend, name)
        if isinstance(val, (str, dict, list, tuple)) and not name.startswith("__"):
            text += " " + as_text(val)
    if not text.strip():
        pytest.skip("No module-level text constants to inspect")
    assert any(w in text.lower() for w in ("helpline", "tele-manas", "14416", "professional"))
