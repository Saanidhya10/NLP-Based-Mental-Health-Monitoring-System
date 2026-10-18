"""Text-level augmentation for minority classes (paper, Sec. 5.4).

* synonym replacement (EDA-style, WordNet)   - cheap, lexical variety
* back-translation en -> fr -> en (MarianMT) - paraphrase; slow, so results are cached on disk

Augmented rows are generated from the TRAINING split only and are tagged ``source='augmented_*'``.
Negation words and emotion-bearing emoji tokens are never replaced (they would flip/erase the label).
Use ``export_spotcheck`` to manually review samples, as the paper requires.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocess import NEGATIONS, TextPreprocessor, preprocess_dataframe
from src.utils import ID2LABEL, LABELS, NUM_CLASSES, get_logger, path_from_cfg

log = get_logger("augmentation")


# --------------------------------------------------------------------------- synonym replacement
def _wordnet():
    try:
        from nltk.corpus import wordnet

        wordnet.synsets("test")  # raises LookupError if corpus is not downloaded
        return wordnet
    except Exception as e:
        raise RuntimeError("Synonym augmentation needs NLTK WordNet: python -m nltk.downloader wordnet omw-1.4") from e


def synonyms(word: str, wn) -> list[str]:
    out = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            cand = lemma.name().replace("_", " ").lower()
            if cand != word and cand.isascii() and cand.replace(" ", "").isalpha():
                out.add(cand)
    return sorted(out)


def synonym_replace(text: str, n: int, rng: random.Random, stop: set[str], wn=None) -> str:
    wn = wn or _wordnet()
    words = text.split()
    idx = [i for i, w in enumerate(words)
           if w.isalpha() and w.lower() not in stop and w.lower() not in NEGATIONS and len(w) > 2]
    rng.shuffle(idx)
    done = 0
    for i in idx:
        syns = synonyms(words[i].lower(), wn)
        if syns:
            words[i] = rng.choice(syns)
            done += 1
        if done >= n:
            break
    return " ".join(words)


# --------------------------------------------------------------------------- back-translation
class BackTranslator:
    """English -> pivot -> English with Helsinki-NLP MarianMT models (loaded lazily)."""

    def __init__(self, pivot: str = "fr", device: str | None = None, batch_size: int = 32, max_length: int = 128):
        self.pivot, self.batch_size, self.max_length = pivot, batch_size, max_length
        self.device = device
        self._loaded = False

    def _load(self):
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        fwd, bwd = f"Helsinki-NLP/opus-mt-en-{self.pivot}", f"Helsinki-NLP/opus-mt-{self.pivot}-en"
        self.tok_f, self.m_f = MarianTokenizer.from_pretrained(fwd), MarianMTModel.from_pretrained(fwd).to(self.device)
        self.tok_b, self.m_b = MarianTokenizer.from_pretrained(bwd), MarianMTModel.from_pretrained(bwd).to(self.device)
        self.m_f.eval()
        self.m_b.eval()
        self._loaded = True

    def _translate(self, texts, tok, model):
        import torch

        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)
            with torch.no_grad():
                gen = model.generate(**enc, num_beams=4, max_length=self.max_length)
            out.extend(tok.batch_decode(gen, skip_special_tokens=True))
        return out

    def __call__(self, texts: list[str]) -> list[str]:
        if not self._loaded:
            self._load()
        return self._translate(self._translate(texts, self.tok_f, self.m_f), self.tok_b, self.m_b)


# --------------------------------------------------------------------------- minority augmentation
def _cache_path(cfg: dict, train_df: pd.DataFrame, seed: int) -> Path:
    ac = cfg["augmentation"]
    key = f"{len(train_df)}|{seed}|{ac['methods']}|{ac['reference']}|{ac['ratio']}|{ac['max_new_per_class']}|{ac['synonym_n']}|{ac['pivot_language']}"
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    return path_from_cfg(cfg, "data_interim", create=True) / f"augmented_train_{h}.csv"


def augment_minority(train_df: pd.DataFrame, cfg: dict, seed: int = 42, text_col: str = "text_transformer"
                     ) -> pd.DataFrame:
    """Return train_df plus synthetic rows for classes smaller than the target size."""
    ac = cfg["augmentation"]
    cache = _cache_path(cfg, train_df, seed)
    if ac.get("cache", True) and cache.exists():
        log.info("Loading cached augmentation: %s", cache)
        new = pd.read_csv(cache)
        return pd.concat([train_df, new], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    rng = random.Random(seed)
    counts = train_df["label_id"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
    ref = counts.max() if ac["reference"] == "max" else int(counts.median())
    tgt = max(int(ref * ac["ratio"]), 1)
    methods = list(ac["methods"])
    pre = TextPreprocessor(cfg)
    wn = _wordnet() if "synonym" in methods else None
    bt = BackTranslator(ac.get("pivot_language", "fr"), batch_size=ac.get("backtranslation_batch_size", 32)) \
        if "backtranslation" in methods else None
    existing = set(train_df[text_col].astype(str))
    new_rows = []

    for cid in range(NUM_CLASSES):
        n = int(counts[cid])
        need = min(tgt - n, ac["max_new_per_class"])
        if n == 0 or need <= 0:
            continue
        pool = train_df[train_df["label_id"] == cid][text_col].astype(str).tolist()
        src_texts = [pool[rng.randrange(n)] for _ in range(need)]
        # alternate methods across the sampled source sentences
        assign = [methods[i % len(methods)] for i in range(need)]
        produced: list[tuple[str, str]] = []
        syn_idx = [i for i, m in enumerate(assign) if m == "synonym"]
        bt_idx = [i for i, m in enumerate(assign) if m == "backtranslation"]
        for i in syn_idx:
            produced.append((synonym_replace(src_texts[i], ac["synonym_n"], rng, pre.stopwords, wn), "synonym"))
        if bt_idx:
            translated = bt([src_texts[i] for i in bt_idx])
            produced.extend((t, "backtranslation") for t in translated)
        kept = 0
        for text, how in produced:
            text = text.strip()
            if text and text not in existing and len(text.split()) >= 2:
                existing.add(text)
                new_rows.append({"text": text, "label": ID2LABEL[cid], "label_id": cid,
                                 "source": f"augmented_{how}"})
                kept += 1
        log.info("Class %-8s: %5d -> +%d augmented (target %d)", LABELS[cid], n, kept, tgt)

    if not new_rows:
        log.warning("Augmentation produced no new rows.")
        return train_df
    new = preprocess_dataframe(pd.DataFrame(new_rows), text_col="text", cfg=cfg, pre=pre)
    if ac.get("cache", True):
        new.to_csv(cache, index=False)
    return pd.concat([train_df, new], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def export_spotcheck(augmented_df: pd.DataFrame, cfg: dict, n: int | None = None, seed: int = 42) -> Path:
    """Write a random sample of augmented rows to results/reports/augmentation_spotcheck.csv for manual review
    (back-translation can soften emotional intensity; synonym swaps can change nuance)."""
    n = n or cfg["augmentation"].get("spotcheck_n", 50)
    aug = augmented_df[augmented_df["source"].astype(str).str.startswith("augmented")]
    sample = aug.sample(n=min(n, len(aug)), random_state=seed)[["label", "source", "text_transformer"]]
    sample["label_ok"] = ""   # fill in by hand: y / n
    out = path_from_cfg(cfg, "reports_dir", create=True) / "augmentation_spotcheck.csv"
    sample.to_csv(out, index=False)
    log.info("Spot-check sample written to %s", out)
    return out
