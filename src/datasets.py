"""Dataset loading, label mapping (27 -> 8), stratified splits, and torch Dataset wrappers.

Pipeline (see scripts/setup_data.py):
    build_unified_dataset()  -> data/interim/unified.csv      (text, label, source)
    src.preprocess           -> data/processed/unified.csv    (+ text_classical/text_seq/text_transformer)
    make_splits()            -> data/splits/{train,val,test}.csv   (created ONCE, then frozen)
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import LABEL2ID, LABELS, get_logger, load_config, path_from_cfg, resolve_path

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # keeps classical-only environments working
    torch = None
    Dataset = object

log = get_logger("datasets")

# Which text column each model family consumes.
TEXT_COLUMN = {"classical": "text_classical", "bilstm": "text_seq", "transformer": "text_transformer"}

# Order of the 28 label ids in GoEmotions ("simplified" HF config and the original TSV files).
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
    "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral",
]


# --------------------------------------------------------------------------- label mapping
def build_label_lookup(cfg: dict) -> dict[str, str]:
    """GoEmotions label name -> 8-class name, validated (no label mapped twice, none unmapped)."""
    lookup: dict[str, str] = {}
    for target, sources in cfg["label_mapping"].items():
        if target not in LABEL2ID:
            raise ValueError(f"Unknown target class in label_mapping: {target}")
        for s in sources or []:
            if s in lookup:
                raise ValueError(f"GoEmotions label '{s}' is mapped twice")
            lookup[s] = target
    missing = set(GOEMOTIONS_LABELS) - set(lookup)
    if missing:
        log.warning("GoEmotions labels without a mapping (will be dropped): %s", sorted(missing))
    return lookup


def map_goemotions_ids(label_ids, lookup: dict[str, str]) -> str | None:
    """Return the single mapped class, or None if unmapped or the example straddles several classes."""
    classes = {lookup[GOEMOTIONS_LABELS[i]] for i in label_ids if GOEMOTIONS_LABELS[i] in lookup}
    return next(iter(classes)) if len(classes) == 1 else None


# --------------------------------------------------------------------------- raw loaders
def _goemotions_from_hf() -> pd.DataFrame:
    from datasets import load_dataset

    last = None
    for name in ("google-research-datasets/go_emotions", "go_emotions"):
        try:
            ds = load_dataset(name, "simplified")
            break
        except Exception as e:  # network / hub name changes
            last = e
    else:
        raise RuntimeError(f"Could not load GoEmotions from the Hugging Face hub: {last}")
    frames = [ds[s].to_pandas()[["text", "labels"]] for s in ds]
    df = pd.concat(frames, ignore_index=True)
    df["label_ids"] = df["labels"].apply(lambda x: [int(i) for i in x])
    return df[["text", "label_ids"]]


def _goemotions_from_local(directory: Path) -> pd.DataFrame:
    frames = []
    for fname in ("train.tsv", "dev.tsv", "test.tsv"):
        f = directory / fname
        if not f.exists():
            raise FileNotFoundError(f"Missing {f}. Download GoEmotions TSVs or set data.goemotions_source: hf")
        d = pd.read_csv(f, sep="\t", header=None, names=["text", "labels", "id"])
        d["label_ids"] = d["labels"].astype(str).apply(lambda s: [int(x) for x in s.split(",")])
        frames.append(d[["text", "label_ids"]])
    return pd.concat(frames, ignore_index=True)


def load_goemotions(cfg: dict) -> pd.DataFrame:
    src = cfg["data"].get("goemotions_source", "auto")
    local_dir = resolve_path(cfg["data"]["goemotions_dir"])
    if src == "local" or (src == "auto" and (local_dir / "train.tsv").exists()):
        raw = _goemotions_from_local(local_dir)
    else:
        raw = _goemotions_from_hf()
    lookup = build_label_lookup(cfg)
    raw["label"] = raw["label_ids"].apply(lambda ids: map_goemotions_ids(ids, lookup))
    kept = raw.dropna(subset=["label"])
    log.info("GoEmotions: %d raw -> %d single-class mapped examples", len(raw), len(kept))
    return pd.DataFrame({"text": kept["text"], "label": kept["label"], "source": "goemotions"})


def load_dreaddit(cfg: dict) -> pd.DataFrame:
    """Stressed posts (label==1) -> 'Stress'. Posts are truncated to ``dreaddit_max_words``."""
    d = resolve_path(cfg["data"]["dreaddit_dir"])
    files = [d / "dreaddit-train.csv", d / "dreaddit-test.csv"]
    existing = [f for f in files if f.exists()]
    if not existing:
        raise FileNotFoundError(
            f"No Dreaddit CSVs in {d}. Download dreaddit-train.csv / dreaddit-test.csv (Turcan & McKeown, 2019)."
        )
    df = pd.concat([pd.read_csv(f) for f in existing], ignore_index=True)
    max_words = cfg["data"]["dreaddit_max_words"]
    df["text"] = df["text"].astype(str).apply(lambda t: " ".join(t.split()[:max_words]))
    policy = cfg["data"].get("dreaddit_nonstress_policy", "drop")
    out = df[df["label"] == 1][["text"]].assign(label="Stress")
    if policy == "neutral":
        out = pd.concat([out, df[df["label"] == 0][["text"]].assign(label="Neutral")], ignore_index=True)
    out["source"] = "dreaddit"
    log.info("Dreaddit: %d posts kept (policy for non-stress: %s)", len(out), policy)
    return out.reset_index(drop=True)


def load_sentiment140(cfg: dict, n: int = 50_000, seed: int = 42) -> pd.DataFrame:
    """OPTIONAL coarse polarity data for supplementary pretraining. Never mixed into the 8-class set
    (labels come from emoticon distant supervision and are noisy)."""
    f = resolve_path(cfg["data"]["sentiment140_csv"])
    df = pd.read_csv(f, encoding="latin-1", header=None, names=["target", "id", "date", "flag", "user", "text"])
    df = df.sample(n=min(n, len(df)), random_state=seed)
    df["polarity"] = df["target"].map({0: "negative", 4: "positive"})
    return df[["text", "polarity"]].reset_index(drop=True)


# --------------------------------------------------------------------------- unify + split
def build_unified_dataset(cfg: dict | None = None, save: bool = True) -> pd.DataFrame:
    cfg = cfg or load_config()
    frames = [load_goemotions(cfg), load_dreaddit(cfg)]
    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.split().str.len() >= cfg["data"]["min_words"]]
    df = df.drop_duplicates(subset=["text", "label"])
    df = df.drop_duplicates(subset=["text"], keep=False)  # same text with conflicting labels -> drop all
    df = df.reset_index(drop=True)
    log.info("Unified dataset: %d rows\n%s", len(df), df["label"].value_counts().to_string())
    if save:
        out = path_from_cfg(cfg, "data_interim", create=True) / "unified.csv"
        df.to_csv(out, index=False)
        log.info("Saved %s", out)
    return df


def make_splits(df: pd.DataFrame, cfg: dict | None = None, overwrite: bool = False) -> dict[str, pd.DataFrame]:
    """Stratified train/val/test split. Existing split files are reused unless overwrite=True, so the
    test set is never silently regenerated."""
    cfg = cfg or load_config()
    out_dir = path_from_cfg(cfg, "data_splits", create=True)
    files = {s: out_dir / f"{s}.csv" for s in ("train", "val", "test")}
    if not overwrite and all(f.exists() for f in files.values()):
        log.warning("Splits already exist in %s - reusing them (pass overwrite=True to rebuild).", out_dir)
        return load_splits(cfg)

    df = df.copy()
    df["label_id"] = df["label"].map(LABEL2ID)
    if df["label_id"].isna().any():
        raise ValueError("Unknown labels present in dataframe")
    fr = cfg["data"]["splits"]
    seed = cfg["project"]["seed"]
    train, rest = train_test_split(df, test_size=1 - fr["train"], stratify=df["label_id"], random_state=seed)
    val, test = train_test_split(
        rest, test_size=fr["test"] / (fr["val"] + fr["test"]), stratify=rest["label_id"], random_state=seed
    )
    splits = {"train": train, "val": val, "test": test}
    for name, d in splits.items():
        d.reset_index(drop=True).to_csv(files[name], index=False)
        log.info("%s: %d rows", name, len(d))
    return {k: v.reset_index(drop=True) for k, v in splits.items()}


def load_splits(cfg: dict | None = None, processed: bool = True) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    d_proc = path_from_cfg(cfg, "data_processed")
    d_splits = path_from_cfg(cfg, "data_splits")
    out = {}
    for s in ("train", "val", "test"):
        f = d_proc / f"{s}.csv" if processed else d_splits / f"{s}.csv"
        if not f.exists():
            f = d_splits / f"{s}.csv"
            if not f.exists():
                raise FileNotFoundError(f"{f} not found - run scripts/setup_data.py first")
        df = pd.read_csv(f)
        if "label_id" not in df.columns and "label" in df.columns:
            df["label_id"] = df["label"].map(LABEL2ID)
        out[s] = df
    return out


def class_distribution(df: pd.DataFrame, label_col: str = "label") -> pd.Series:
    return df[label_col].value_counts().reindex(LABELS, fill_value=0)


# --------------------------------------------------------------------------- torch: transformers
class TransformerDataset(Dataset):
    """Tokenises up-front without padding; pair with DataCollatorWithPadding for dynamic padding."""

    def __init__(self, df: pd.DataFrame, tokenizer, text_col: str = "text_transformer", max_len: int = 64):
        enc = tokenizer(df[text_col].astype(str).tolist(), truncation=True, max_length=max_len)
        labels = df["label_id"].astype(int).tolist()
        self.items = []
        for i in range(len(labels)):
            item = {k: enc[k][i] for k in enc.keys()}
            item["labels"] = labels[i]
            self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# --------------------------------------------------------------------------- torch: Bi-LSTM
PAD, UNK = "<pad>", "<unk>"


class Vocabulary:
    def __init__(self, stoi: dict[str, int]):
        self.stoi = stoi
        self.itos = {i: w for w, i in stoi.items()}
        self.pad_idx = stoi[PAD]
        self.unk_idx = stoi[UNK]

    @classmethod
    def from_texts(cls, texts, min_freq: int = 2, max_size: int = 30000) -> "Vocabulary":
        counts = Counter(w for t in texts for w in str(t).split())
        words = [w for w, c in counts.most_common(max_size) if c >= min_freq]
        stoi = {PAD: 0, UNK: 1}
        for w in words:
            stoi[w] = len(stoi)
        return cls(stoi)

    def __len__(self):
        return len(self.stoi)

    def encode(self, text: str, max_len: int) -> tuple[list[int], int]:
        ids = [self.stoi.get(w, self.unk_idx) for w in str(text).split()][:max_len]
        length = max(len(ids), 1)
        if not ids:
            ids = [self.unk_idx]
        return ids + [self.pad_idx] * (max_len - len(ids)), length

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.stoi, f)

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))


class LSTMDataset(Dataset):
    def __init__(self, texts, labels, vocab: Vocabulary, max_len: int = 64):
        enc = [vocab.encode(t, max_len) for t in texts]
        self.ids = torch.tensor([e[0] for e in enc], dtype=torch.long)
        self.lengths = torch.tensor([e[1] for e in enc], dtype=torch.long)
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.ids[i], self.lengths[i], self.labels[i]


def load_glove_matrix(vocab: Vocabulary, path: str | Path, dim: int, seed: int = 42) -> np.ndarray:
    """Embedding matrix initialised from a GloVe text file; unseen words get small random vectors."""
    rng = np.random.default_rng(seed)
    mat = rng.normal(0, 0.1, size=(len(vocab), dim)).astype("float32")
    mat[vocab.pad_idx] = 0.0
    hit = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) == 2:  # word2vec-style header line
                continue
            w = parts[0]
            if w in vocab.stoi and len(parts) - 1 == dim:
                mat[vocab.stoi[w]] = np.asarray(parts[1:], dtype="float32")
                hit += 1
    log.info("GloVe coverage: %d / %d vocab words (%.1f%%)", hit, len(vocab), 100 * hit / max(len(vocab), 1))
    return mat
