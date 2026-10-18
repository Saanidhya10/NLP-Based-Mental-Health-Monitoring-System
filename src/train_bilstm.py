"""Bi-LSTM emotion classifier on GloVe / Word2Vec embeddings (paper, Sec. 5.3).

Usage:
    python -m src.train_bilstm --strategy baseline --seeds 42 43 44
    python -m src.train_bilstm --strategy class_weight --tune
"""
from __future__ import annotations

import argparse
import copy
import itertools

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from src.datasets import TEXT_COLUMN, LSTMDataset, Vocabulary, load_glove_matrix, load_splits
from src.evaluation import compute_metrics, report_run
from src.imbalance import prepare_training_data
from src.utils import (NUM_CLASSES, get_device, get_logger, load_config, model_dir, resolve_path, save_json,
                       set_seed)

log = get_logger("train_bilstm")
TEXT_COL = TEXT_COLUMN["bilstm"]


class BiLSTMClassifier(nn.Module):
    """Embedding -> dropout -> BiLSTM -> concat(final fwd, final bwd hidden) -> dropout -> linear."""

    def __init__(self, vocab_size: int, emb_dim: int, hidden: int, dropout: float, num_classes: int = NUM_CLASSES,
                 pad_idx: int = 0, embeddings: np.ndarray | None = None, freeze: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        if embeddings is not None:
            self.embedding.weight.data.copy_(torch.as_tensor(embeddings))
            self.embedding.weight.requires_grad = not freeze
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, num_classes)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.embedding(ids))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(self.dropout(h))


# ------------------------------------------------------------------ embeddings
def build_embedding_matrix(cfg: dict, kind: str, vocab: Vocabulary, train_texts, seed: int) -> np.ndarray | None:
    bc = cfg["bilstm"]
    dim = bc["embedding_dim"]
    if kind == "random":
        return None
    if kind == "glove":
        path = resolve_path(bc["glove_path"])
        if not path.exists():
            log.warning("GloVe file %s not found - falling back to random init", path)
            return None
        return load_glove_matrix(vocab, path, dim, seed)
    if kind == "word2vec":
        from gensim.models import KeyedVectors, Word2Vec

        rng = np.random.default_rng(seed)
        mat = rng.normal(0, 0.1, size=(len(vocab), dim)).astype("float32")
        mat[vocab.pad_idx] = 0.0
        if bc.get("word2vec_path"):
            kv = KeyedVectors.load_word2vec_format(str(resolve_path(bc["word2vec_path"])),
                                                   binary=str(bc["word2vec_path"]).endswith(".bin"))
            dim_kv = kv.vector_size
            if dim_kv != dim:
                raise ValueError(f"word2vec dim {dim_kv} != bilstm.embedding_dim {dim}")
        else:  # no pretrained file: train skip-gram on the TRAIN split only (no leakage)
            w2v = Word2Vec([str(t).split() for t in train_texts], vector_size=dim, window=5, min_count=1,
                           sg=1, seed=seed, workers=1, epochs=10)
            kv = w2v.wv
        for w, i in vocab.stoi.items():
            if w in kv:
                mat[i] = kv[w]
        return mat
    raise ValueError(f"Unknown embedding kind {kind}")


# ------------------------------------------------------------------ train / eval
@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys, probs = [], []
    for ids, lengths, y in loader:
        logits = model(ids.to(device), lengths)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        ys.append(y.numpy())
    probs, ys = np.concatenate(probs), np.concatenate(ys)
    return ys, probs.argmax(1), probs


def fit(hp: dict, cfg: dict, train_df, val_df, weights, vocab, emb, seed: int, device):
    bc = cfg["bilstm"]
    set_seed(seed)
    tr = LSTMDataset(train_df[TEXT_COL], train_df["label_id"], vocab, bc["max_len"])
    va = LSTMDataset(val_df[TEXT_COL], val_df["label_id"], vocab, bc["max_len"])
    g = torch.Generator().manual_seed(seed)
    tr_dl = DataLoader(tr, batch_size=bc["batch_size"], shuffle=True, generator=g)
    va_dl = DataLoader(va, batch_size=256)

    emb_dim = emb.shape[1] if emb is not None else bc["embedding_dim"]
    model = BiLSTMClassifier(len(vocab), emb_dim, hp["hidden"], hp["dropout"], pad_idx=vocab.pad_idx,
                             embeddings=emb, freeze=bc.get("freeze_embeddings", False)).to(device)
    w = None
    if weights:
        w = torch.tensor([weights.get(c, 1.0) for c in range(NUM_CLASSES)], dtype=torch.float, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=hp["lr"])

    best_f1, best_state, bad = -1.0, None, 0
    for epoch in range(1, bc["epochs"] + 1):
        model.train()
        total = 0.0
        for ids, lengths, y in tr_dl:
            opt.zero_grad()
            loss = loss_fn(model(ids.to(device), lengths), y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(y)
        yv, pv, _ = predict_loader(model, va_dl, device)
        f1 = compute_metrics(yv, pv)["macro_f1"]
        log.info("epoch %2d | train loss %.4f | val macro-F1 %.4f", epoch, total / len(tr), f1)
        if f1 > best_f1:
            best_f1, best_state, bad = f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= bc["patience"]:
                log.info("early stopping at epoch %d", epoch)
                break
    model.load_state_dict(best_state)
    return model, best_f1


def train_bilstm(cfg: dict, strategy: str = "baseline", seed: int | None = None, tune: bool = False,
                 splits: dict | None = None, save: bool = True) -> dict:
    bc = cfg["bilstm"]
    seed = cfg["project"]["seed"] if seed is None else seed
    set_seed(seed)
    device = get_device()
    splits = splits or load_splits(cfg)
    train_df, weights = prepare_training_data(splits["train"], strategy, cfg, seed)
    val_df, test_df = splits["val"], splits["test"]
    vocab = Vocabulary.from_texts(train_df[TEXT_COL], **bc["vocab"])
    log.info("Vocabulary size: %d", len(vocab))

    base = {"hidden": bc["hidden"], "dropout": bc["dropout"], "lr": bc["lr"], "embedding": bc["embedding"]}
    combos = [base]
    if tune:
        g = bc["grid"]
        combos = [dict(zip(g, v)) for v in itertools.product(*g.values())]

    emb_cache, best = {}, (-1.0, None, None, None)
    for hp in combos:
        if hp["embedding"] not in emb_cache:
            emb_cache[hp["embedding"]] = build_embedding_matrix(cfg, hp["embedding"], vocab, train_df[TEXT_COL], seed)
        log.info("Training with %s", hp)
        model, f1 = fit(hp, cfg, train_df, val_df, weights, vocab, emb_cache[hp["embedding"]], seed, device)
        if f1 > best[0]:
            best = (f1, hp, model, emb_cache[hp["embedding"]])
    best_f1, best_hp, model, emb = best

    te = LSTMDataset(test_df[TEXT_COL], test_df["label_id"], vocab, bc["max_len"])
    yt, yp, _ = predict_loader(model, DataLoader(te, batch_size=256), device)
    report_run(cfg, "bilstm", strategy, seed, yt, yp, split="test")
    if save:
        d = model_dir(cfg, "bilstm", strategy, seed, create=True)
        torch.save(model.state_dict(), d / "model.pt")
        vocab.save(d / "vocab.json")
        save_json({**best_hp, "emb_dim": model.embedding.embedding_dim, "max_len": bc["max_len"],
                   "val_macro_f1": best_f1, "seed": seed, "strategy": strategy}, d / "hparams.json")
    return {"val_macro_f1": best_f1, "hparams": best_hp}


@torch.no_grad()
def predict_proba_bilstm(model: BiLSTMClassifier, vocab: Vocabulary, texts, max_len: int, device,
                         batch_size: int = 128) -> np.ndarray:
    """Probabilities for already-preprocessed (``text_seq``-style) strings."""
    model.eval()
    enc = [vocab.encode(t, max_len) for t in texts]
    ids = torch.tensor([e[0] for e in enc]); lengths = torch.tensor([e[1] for e in enc])
    out = []
    for i in range(0, len(texts), batch_size):
        logits = model(ids[i:i + batch_size].to(device), lengths[i:i + batch_size])
        out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="baseline")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    splits = load_splits(cfg)
    for s in a.seeds or cfg["seeds"]:
        train_bilstm(cfg, a.strategy, s, a.tune, splits)


if __name__ == "__main__":
    main()
