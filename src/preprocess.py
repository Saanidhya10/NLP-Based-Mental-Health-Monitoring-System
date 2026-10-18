"""Text preprocessing (paper, Sec. 5.1).

Three views of each message are produced, because the model families need different input:

* ``text_classical``   - lowercased, punctuation-normalised, lemmatised, stop-words removed
                         (NEGATIONS ARE KEPT) -> TF-IDF models.
* ``text_seq``         - lowercased, punctuation spaced out, stop-words kept -> Bi-LSTM.
* ``text_transformer`` - light clean only (URLs/handles/HTML removed, emoji -> tokens, casing kept)
                         -> DistilBERT / RoBERTa, whose tokenisers expect natural text.

CLI:  python -m src.preprocess      (data/interim/unified.csv -> data/processed/unified.csv)
"""
from __future__ import annotations

import html
import re
from typing import Iterable

import pandas as pd

from src.utils import get_logger, load_config, path_from_cfg

log = get_logger("preprocess")

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.I)
HTML_TAG_RE = re.compile(r"<[^>]+>")
HANDLE_RE = re.compile(r"(?<!\w)@\w+")
REDDIT_USER_RE = re.compile(r"(?<!\w)/?u/[\w\-]+", re.I)
GOEMO_TOKEN_RE = re.compile(r"\[(NAME|RELIGION)\]")  # GoEmotions anonymisation placeholders
WS_RE = re.compile(r"\s+")

EMOTICONS = {
    ":)": " emoji_smile ", ":-)": " emoji_smile ", ":D": " emoji_grin ",
    ";)": " emoji_wink ", ":(": " emoji_sad ", ":-(": " emoji_sad ", ":'(": " emoji_crying ",
    ":/": " emoji_unsure ", ":O": " emoji_surprised ", "<3": " emoji_heart ",
}

# Negations must survive stop-word removal ("not happy" != "happy").
NEGATIONS = {
    "no", "not", "nor", "never", "none", "nobody", "nothing", "neither", "nowhere", "cannot", "without",
    "don", "didn", "doesn", "isn", "wasn", "weren", "aren", "hasn", "haven", "hadn", "couldn",
    "shouldn", "wouldn", "won", "mustn", "needn", "shan", "mightn", "ain", "n't", "cant", "dont",
}
FALLBACK_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "of", "at", "by", "for", "with", "about",
    "to", "from", "in", "out", "on", "off", "over", "under", "again", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "should",
    "now", "i", "me", "my", "myself", "we", "our", "you", "your", "he", "him", "his", "she", "her", "it",
    "its", "they", "them", "their", "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
    "does", "did", "doing", "as", "until", "because", "into", "through", "during", "before", "after",
    "above", "below", "up", "down", "between", "against",
}


def clean_basic(text: str) -> str:
    """Shared cleaning: HTML, URLs, handles, placeholders, emoji -> text tokens. Keeps casing."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = GOEMO_TOKEN_RE.sub(" ", text)
    text = HANDLE_RE.sub(" ", text)
    text = REDDIT_USER_RE.sub(" ", text)
    for emo, tok in EMOTICONS.items():
        text = text.replace(emo, tok)
    try:
        import emoji

        text = emoji.demojize(text, delimiters=(" emoji_", " "))
    except ImportError:  # emoji package missing: leave unicode emoji in place
        pass
    return WS_RE.sub(" ", text).strip()


def normalize_punct(text: str) -> str:
    """Collapse repeated punctuation, drop stray symbols, space out . , ! ? so str.split() tokenises."""
    text = re.sub(r"([!?.,])\1+", r"\1", text)
    text = re.sub(r"[^\w\s!?.,'\-]", " ", text)
    text = re.sub(r"([!?.,])", r" \1 ", text)
    return WS_RE.sub(" ", text).strip()


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """Preprocess text: cleans HTML, URLs, handles, emojis, collapses whitespace.
    If remove_stopwords=True (classical), stop-words are removed (preserving negations)."""
    if not isinstance(text, str):
        return ""
    base = normalize_punct(clean_basic(text).lower())
    if not remove_stopwords:
        return base
    pre = TextPreprocessor()
    toks = []
    for w in base.split():
        if w in {".", ","}:
            continue
        if w in pre.stopwords and w not in NEGATIONS:
            continue
        toks.append(w)
    return " ".join(toks)


preprocess_text = clean_text
clean = clean_text
normalize_text = clean_text
preprocess = clean_text


class TextPreprocessor:
    def __init__(self, cfg: dict | None = None, lemmatize: bool | None = None):
        cfg = cfg or load_config()
        pc = cfg.get("preprocess", {})
        self.spacy_model = pc.get("spacy_model", "en_core_web_sm")
        self.lemmatize = pc.get("lemmatize", True) if lemmatize is None else lemmatize
        self._nlp = None
        self._wn = None
        self._stop: set[str] | None = None

    # -- resources (lazy) --------------------------------------------------
    @property
    def stopwords(self) -> set[str]:
        if self._stop is None:
            try:
                from nltk.corpus import stopwords

                words = set(stopwords.words("english"))
            except Exception:
                words = set(FALLBACK_STOPWORDS)
            self._stop = {w for w in words if w not in NEGATIONS and "n't" not in w}
        return self._stop

    def _get_nlp(self):
        if self._nlp is None and self.lemmatize:
            try:
                import spacy

                self._nlp = spacy.load(self.spacy_model, disable=["parser", "ner"])
            except Exception:
                log.warning("spaCy model unavailable; falling back to NLTK WordNet lemmatiser (or none).")
                self._nlp = False
        return self._nlp

    def _get_wordnet(self):
        if self._wn is None:
            try:
                from nltk.stem import WordNetLemmatizer

                wn = WordNetLemmatizer()
                wn.lemmatize("tests")  # raises LookupError if corpus missing
                self._wn = wn
            except Exception:
                self._wn = False
        return self._wn

    # -- views ---------------------------------------------------------------
    def transformer(self, text: str) -> str:
        return clean_basic(text)

    def seq(self, text: str) -> str:
        return normalize_punct(clean_basic(text).lower())

    def classical(self, text: str) -> str:
        return self.classical_many([text])[0]

    def classical_many(self, texts: Iterable[str]) -> list[str]:
        base = [normalize_punct(clean_basic(t).lower()) for t in texts]
        nlp = self._get_nlp()
        out = []
        if nlp:
            for doc in nlp.pipe(base, batch_size=256):
                toks = []
                for tok in doc:
                    lemma = tok.lemma_.lower().strip()
                    if not lemma or tok.is_space:
                        continue
                    if tok.is_punct and lemma not in {"!", "?"}:
                        continue
                    if lemma in self.stopwords and lemma not in NEGATIONS:
                        continue
                    toks.append(lemma)
                out.append(" ".join(toks))
            return out
        wn = self._get_wordnet() if self.lemmatize else False
        for s in base:
            toks = []
            for w in s.split():
                if w in {".", ","}:
                    continue
                if w in self.stopwords and w not in NEGATIONS:
                    continue
                toks.append(wn.lemmatize(w) if wn else w)
            out.append(" ".join(toks))
        return out


def preprocess_dataframe(
    df: pd.DataFrame, text_col: str = "text", cfg: dict | None = None, pre: TextPreprocessor | None = None
) -> pd.DataFrame:
    """Add text_transformer / text_seq / text_classical columns; drop rows that end up empty."""
    pre = pre or TextPreprocessor(cfg)
    df = df.copy()
    raw = df[text_col].fillna("").astype(str).tolist()
    df["text_transformer"] = [pre.transformer(t) for t in raw]
    df["text_seq"] = [pre.seq(t) for t in raw]
    df["text_classical"] = pre.classical_many(raw)
    keep = (df["text_transformer"].str.len() > 0) & (df["text_classical"].str.len() > 0)
    dropped = int((~keep).sum())
    if dropped:
        log.info("Dropped %d rows that were empty after preprocessing", dropped)
    return df[keep].reset_index(drop=True)


def main() -> None:
    cfg = load_config()
    src = path_from_cfg(cfg, "data_interim") / "unified.csv"
    dst_dir = path_from_cfg(cfg, "data_processed", create=True)
    df = pd.read_csv(src)
    log.info("Preprocessing %d rows from %s", len(df), src)
    out = preprocess_dataframe(df, cfg=cfg)
    out.to_csv(dst_dir / "unified.csv", index=False)
    log.info("Saved %d rows -> %s", len(out), dst_dir / "unified.csv")


if __name__ == "__main__":
    main()
