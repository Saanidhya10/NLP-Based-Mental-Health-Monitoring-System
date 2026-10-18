"""Unified inference wrapper: load any trained model and get an 8-class probability distribution.

    predictor = EmotionPredictor.from_config(cfg)          # reads cfg['inference']
    predictor.predict("I can't stop worrying about tomorrow")
    predictor.predict_proba(["text 1", "text 2"])          # (n, 8) array, columns ordered as utils.LABELS
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from src.preprocess import TextPreprocessor
from src.utils import LABELS, NUM_CLASSES, get_logger, load_config, load_json, resolve_path, softmax

log = get_logger("inference")


class EmotionPredictor:
    def __init__(self, kind: str, path: str | Path, cfg: dict | None = None, device: str | None = None,
                 batch_size: int = 64):
        self.cfg = cfg or load_config()
        self.kind = kind
        self.path = resolve_path(path)
        self.batch_size = batch_size
        self.pre = TextPreprocessor(self.cfg)
        self._device = device
        self._load()

    @classmethod
    def from_config(cls, cfg: dict | None = None, **kw) -> "EmotionPredictor":
        cfg = cfg or load_config()
        if "inference" in cfg and "kind" in cfg["inference"] and "path" in cfg["inference"]:
            return cls(cfg["inference"]["kind"], cfg["inference"]["path"], cfg, **kw)

        for kind in ["distilbert", "roberta"]:
            p = resolve_path(f"models/{kind}/final")
            if (p / "config.json").exists():
                return cls("transformer", p, cfg, **kw)

        for cand in [
            "models/classical/logreg_baseline/pipeline.joblib",
            "models/classical/logreg_baseline",
            "models/classical/svm_baseline/pipeline.joblib",
            "models/classical/rf_baseline/pipeline.joblib",
        ]:
            p = resolve_path(cand)
            if p.exists():
                return cls("classical", p, cfg, **kw)

        raise FileNotFoundError(
            "No trained model found. Expected trained transformer or classical pipeline in models/."
        )

    # ---------------------------------------------------------------- loading
    def _load(self) -> None:
        if self.kind == "classical":
            f = self.path / "pipeline.joblib" if self.path.is_dir() else self.path
            self.model = joblib.load(f)
        elif self.kind == "bilstm":
            import torch

            from src.datasets import Vocabulary
            from src.train_bilstm import BiLSTMClassifier

            self.device = torch.device(self._device or ("cuda" if torch.cuda.is_available() else "cpu"))
            hp = load_json(self.path / "hparams.json")
            self.vocab = Vocabulary.load(self.path / "vocab.json")
            self.max_len = hp["max_len"]
            self.model = BiLSTMClassifier(len(self.vocab), hp["emb_dim"], hp["hidden"], hp["dropout"],
                                          pad_idx=self.vocab.pad_idx)
            self.model.load_state_dict(torch.load(self.path / "model.pt", map_location=self.device))
            self.model.to(self.device).eval()
        elif self.kind in ("distilbert", "roberta", "transformer"):
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.device = torch.device(self._device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.path))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.path)).to(self.device).eval()
            self.max_len = self.cfg["transformer"]["max_len"]
        else:
            raise ValueError(f"Unknown model kind '{self.kind}'")
        log.info("Loaded %s model from %s", self.kind, self.path)

    # ---------------------------------------------------------------- prediction
    def predict_proba(self, texts) -> np.ndarray:
        texts = [texts] if isinstance(texts, str) else list(texts)
        if self.kind == "classical":
            return self._proba_classical(texts)
        if self.kind == "bilstm":
            return self._proba_bilstm(texts)
        return self._proba_transformer(texts)

    def _proba_classical(self, texts) -> np.ndarray:
        X = self.pre.classical_many(texts)
        if hasattr(self.model, "predict_proba"):
            p = self.model.predict_proba(X)
        else:  # LinearSVC: no probabilities -> softmax over decision scores (uncalibrated)
            p = softmax(self.model.decision_function(X))
        out = np.zeros((len(texts), NUM_CLASSES))
        out[:, np.asarray(self.model.classes_, dtype=int)] = p   # classes_ are label ids (may miss a class)
        return out

    def _proba_bilstm(self, texts) -> np.ndarray:
        from src.train_bilstm import predict_proba_bilstm

        return predict_proba_bilstm(self.model, self.vocab, [self.pre.seq(t) for t in texts], self.max_len,
                                    self.device, self.batch_size)

    def _proba_transformer(self, texts) -> np.ndarray:
        import torch

        clean = [self.pre.transformer(t) or "." for t in texts]
        out = []
        with torch.no_grad():
            for i in range(0, len(clean), self.batch_size):
                enc = self.tokenizer(clean[i:i + self.batch_size], truncation=True, max_length=self.max_len,
                                     padding=True, return_tensors="pt").to(self.device)
                out.append(torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy())
        return np.concatenate(out)

    def predict(self, text: str) -> dict:
        p = self.predict_proba([text])[0]
        i = int(p.argmax())
        return {"emotion": LABELS[i], "confidence": float(p[i]),
                "probabilities": {l: float(v) for l, v in zip(LABELS, p)}}

    def predict_many(self, texts) -> list[dict]:
        P = self.predict_proba(texts)
        return [{"emotion": LABELS[int(p.argmax())], "confidence": float(p.max()),
                 "probabilities": {l: float(v) for l, v in zip(LABELS, p)}} for p in P]


_default_predictor = None


def get_default_predictor(cfg: dict | None = None) -> EmotionPredictor:
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = EmotionPredictor.from_config(cfg)
    return _default_predictor


def predict(text: str) -> dict:
    if text is None or not str(text).strip():
        raise ValueError("Input text cannot be empty")
    return get_default_predictor().predict(str(text))


def predict_proba(texts) -> np.ndarray:
    return get_default_predictor().predict_proba(texts)


predict_emotion = predict
classify = predict
predict_text = predict
