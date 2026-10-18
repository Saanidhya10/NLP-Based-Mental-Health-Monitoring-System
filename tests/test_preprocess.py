"""Tests for src/preprocess.py.

Covers the preprocessing rules from the research paper (Section 5.1):
lowercasing, URL/HTML/handle removal, emoji -> emotion tokens, negation
retention, and stop-word removal for classical pipelines only.

If your function names differ, add them to CANDIDATES below.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

preprocess = pytest.importorskip("src.preprocess", reason="src/preprocess.py not importable")

CANDIDATES = ["clean_text", "preprocess_text", "preprocess", "normalize_text", "clean"]


def _find(module, names):
    for n in names:
        fn = getattr(module, n, None)
        if callable(fn):
            return fn
    return None


clean = _find(preprocess, CANDIDATES)
pytestmark = pytest.mark.skipif(
    clean is None, reason=f"No preprocessing function found; tried {CANDIDATES}"
)


def run(text):
    out = clean(text)
    # Some implementations return a token list; normalise to a string.
    if isinstance(out, (list, tuple)):
        out = " ".join(out)
    return out


def test_returns_string_or_tokens():
    assert isinstance(run("Hello world"), str)


def test_lowercases_text():
    out = run("I Am FEELING Very SAD")
    assert out == out.lower()


def test_removes_urls():
    out = run("check this https://example.com/page?x=1 and www.test.org now")
    assert "http" not in out
    assert "example.com" not in out
    assert "www." not in out


def test_removes_user_handles():
    out = run("@john_doe thanks for the help")
    assert "@john_doe" not in out
    assert "john_doe" not in out


def test_removes_html_artefacts():
    out = run("<p>I feel <b>bad</b></p> &amp; tired")
    assert "<" not in out and ">" not in out
    assert "&amp;" not in out


def test_emoji_converted_not_dropped():
    out = run("so happy 😊")
    assert "😊" not in out
    # demojize gives something like "smiling_face"; the token must survive.
    assert "smil" in out or "face" in out


def test_different_emojis_give_different_tokens():
    assert run("great 😊") != run("great 😢")


def test_negation_is_retained():
    # Paper 5.1: negation words must survive, otherwise "not happy" flips meaning.
    out = run("I am not happy")
    assert "not" in out.split() or "n't" in out or "no" in out.split()
    assert "happy" in out or "happi" in out


def test_extra_whitespace_collapsed():
    out = run("too     many    spaces\n\nand\tlines")
    assert "  " not in out


def test_empty_string_is_safe():
    assert run("") == ""


def test_only_noise_input_is_safe():
    out = run("https://example.com @user")
    assert isinstance(out, str)
    assert out.strip() == ""


def test_stopword_removal_is_optional_flag():
    """Stop-word removal applies to classical pipelines only (paper 5.1)."""
    import inspect

    sig = inspect.signature(clean)
    flag = next(
        (p for p in sig.parameters if "stop" in p.lower() or "classical" in p.lower()),
        None,
    )
    if flag is None:
        pytest.skip("No stop-word / classical flag exposed on the cleaning function")
    text = "this is the book that i am reading"
    with_removal = clean(text, **{flag: True})
    without_removal = clean(text, **{flag: False})
    if isinstance(with_removal, (list, tuple)):
        with_removal = " ".join(with_removal)
        without_removal = " ".join(without_removal)
    assert len(with_removal) <= len(without_removal)
