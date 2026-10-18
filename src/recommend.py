"""Rule-based, NON-DIAGNOSTIC recommendations and safety escalation (paper, Sec. 5.7).

Rules are deliberately transparent rather than learned. Escalation depends on persistence and severity
(see trends.detect_escalation), never on a single message. Wording never names a condition.
``check_wording`` is used in tests/test_recommend.py to enforce that.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from src.utils import LABELS, NEGATIVE_LABELS, load_config

SUGGESTIONS: dict[str, list[str]] = {
    "Stress": [
        "Try box breathing: breathe in for 4 seconds, hold for 4, out for 4, hold for 4. Repeat for a few minutes.",
        "Pick just one small task to do next and set the rest aside for now.",
        "Take a short break away from screens, even five minutes can help.",
    ],
    "Anxiety": [
        "Try a grounding exercise: name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste.",
        "Spend a few minutes on slow, mindful breathing, noticing the air moving in and out.",
        "Write down what is on your mind; seeing it on paper can make it feel more manageable.",
    ],
    "Sadness": [
        "Consider reaching out to a friend or family member you trust, even with a short message.",
        "Gentle activity, like a short walk outside, can lift your energy a little.",
        "Be kind to yourself today; small, comforting routines count.",
    ],
    "Anger": [
        "Give yourself a pause before responding, and take a few slow breaths.",
        "Physical movement, like a brisk walk, can help release built-up tension.",
        "Writing down what is frustrating you can help you sort out what matters most.",
    ],
    "Fear": [
        "Take slow breaths with a long exhale to help your body settle.",
        "Name what is worrying you, then note one small step you could take or one person you could talk to.",
        "Remind yourself of a time you handled something difficult.",
    ],
}
POSITIVE_NOTE = "It sounds like things are going reasonably well lately. Keep doing what supports you."
ESCALATION_TEXT = (
    "Over the past several days your messages have shown a sustained pattern of low or stressful mood. "
    "This tool cannot assess your health, but a qualified mental-health professional can offer proper support. "
    "Please consider reaching out to one, or to someone you trust, soon. You do not have to handle this alone."
)

# Terms that must never appear in user-facing text (no diagnosis, no clinical labels, no alarmism).
BANNED_PATTERNS = [
    r"\bdepress(ed|ion|ive)\b", r"\bdisorder\b", r"\bdiagnos(e|ed|is)\b(?! or)", r"\byou (have|are suffering)\b",
    r"\bsuffer(ing)? from\b", r"\bmentally ill\b", r"\bsuicid", r"\bcrisis\b", r"\bdanger(ous)?\b(?! right)",
]


@dataclass
class Recommendation:
    level: str                      # "none" | "supportive" | "escalation"
    headline: str
    suggestions: list[str] = field(default_factory=list)
    helplines: list[dict] = field(default_factory=list)
    emergency_note: str = ""
    disclaimer: str = ""
    emotion: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def user_text(self) -> str:
        """All user-facing text except the disclaimer/emergency note (those intentionally mention 'diagnose')."""
        return " ".join([self.headline, *self.suggestions])


def check_wording(text: str) -> list[str]:
    """Return the banned patterns found in ``text`` (empty list == OK)."""
    return [p for p in BANNED_PATTERNS if re.search(p, text, flags=re.I)]


def get_helplines(cfg: dict) -> list[dict]:
    rc = cfg.get("recommend", {})
    hl = rc.get("helplines", [])
    if isinstance(hl, list):
        return hl
    if isinstance(hl, dict):
        return list(hl.get(rc.get("region", "default"), hl.get("default", [])))
    return [{"name": "Tele-MANAS", "number": "14416"}]


def get_suggestion(dominant_emotion: str) -> str:
    if dominant_emotion not in LABELS and dominant_emotion not in SUGGESTIONS:
        raise ValueError(f"Unknown emotion: {dominant_emotion}")
    suggs = SUGGESTIONS.get(dominant_emotion, [POSITIVE_NOTE])
    return " ".join(suggs)


suggest = get_suggestion
get_recommendation = get_suggestion
recommend_for_emotion = get_suggestion


def should_escalate(scores, tau: float = 0.55, k: int = 3, alpha: float = 0.3) -> bool:
    if not scores:
        return False
    # calculate EMA
    s = [scores[0]]
    for x in scores[1:]:
        s.append(alpha * x + (1.0 - alpha) * s[-1])
    run = 0
    for val in s:
        run = run + 1 if val > tau else 0
        if run >= k:
            return True
    return False


check_escalation = should_escalate
needs_escalation = should_escalate
escalate = should_escalate
evaluate_escalation = should_escalate


def recommend(dominant_emotion: str | None = None, escalation: dict | None = None, neg_score: float | None = None,
              cfg: dict | None = None) -> Recommendation:
    cfg = cfg or load_config()
    rc = cfg.get("recommend", {})
    tc = cfg.get("trends", {})
    disclaimer = rc.get("disclaimer") or cfg.get("project", {}).get("disclaimer", "")
    emergency_note = rc.get("emergency_note", "If you are in immediate danger, please contact local emergency services.")
    common = {"disclaimer": disclaimer, "emotion": dominant_emotion}

    if escalation and escalation.get("triggered"):
        return Recommendation(
            level="escalation", headline=ESCALATION_TEXT,
            suggestions=SUGGESTIONS.get(dominant_emotion, SUGGESTIONS["Stress"])[:2],
            helplines=get_helplines(cfg), emergency_note=emergency_note, **common)

    if dominant_emotion in NEGATIVE_LABELS and (neg_score is None or neg_score >= tc.get("supportive_min_score", 0.35)):
        return Recommendation(
            level="supportive",
            headline=f"Lately your messages lean towards {dominant_emotion.lower()}. Here are a few things that might help:",
            suggestions=SUGGESTIONS[dominant_emotion], **common)

    return Recommendation(level="none", headline=POSITIVE_NOTE, **common)


def recommend_from_summary(summary: dict, cfg: dict | None = None) -> Recommendation:
    """Convenience wrapper around ``trends.mood_summary`` output."""
    return recommend(summary.get("latest_dominant"), summary.get("escalation"), summary.get("latest_neg_score"), cfg)
