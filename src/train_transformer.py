"""Fine-tune DistilBERT / RoBERTa with a class-weight-aware Hugging Face Trainer (paper, Sec. 5.3-5.5).

Usage (Colab T4 is enough):
    python -m src.train_transformer --model distilbert --strategy baseline --seeds 42 43 44
    python -m src.train_transformer --model roberta --strategy class_weight
    python -m src.train_transformer --model distilbert --tune        # lr x batch-size grid on val macro-F1
"""
from __future__ import annotations

import argparse
import inspect
import itertools
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

from src.datasets import TEXT_COLUMN, TransformerDataset, load_splits
from src.evaluation import hf_compute_metrics, report_run
from src.imbalance import prepare_training_data
from src.utils import (ID2LABEL, LABEL2ID, NUM_CLASSES, get_logger, load_config, model_dir, save_json, set_seed)

log = get_logger("train_transformer")
TEXT_COL = TEXT_COLUMN["transformer"]


class WeightedTrainer(Trainer):
    """Cross-entropy with per-class weights w_c = N / (|E| n_c) (paper, Sec. 5.4)."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        w = self.class_weights.to(outputs.logits.device) if self.class_weights is not None else None
        loss = F.cross_entropy(outputs.logits, labels, weight=w)
        return (loss, outputs) if return_outputs else loss


def _training_args(cfg: dict, out_dir: str, hp: dict, seed: int) -> TrainingArguments:
    tc = cfg["transformer"]
    sig = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"  # renamed in transformers 4.41
    kwargs = dict(
        output_dir=out_dir, num_train_epochs=hp["epochs"], learning_rate=hp["lr"],
        per_device_train_batch_size=hp["batch_size"], per_device_eval_batch_size=128,
        weight_decay=tc["weight_decay"], warmup_ratio=tc["warmup_ratio"], lr_scheduler_type="linear",
        save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="macro_f1",
        greater_is_better=True, save_total_limit=1, seed=seed, data_seed=seed,
        fp16=torch.cuda.is_available(), report_to="none", logging_steps=100,
    )
    kwargs[eval_key] = "epoch"
    return TrainingArguments(**kwargs)


def _run_one(cfg, name, checkpoint, strategy, hp, seed, train_df, val_df, weights, out_dir):
    tc = cfg["transformer"]
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=NUM_CLASSES, id2label=ID2LABEL, label2id=LABEL2ID)
    tr = TransformerDataset(train_df, tok, TEXT_COL, tc["max_len"])
    va = TransformerDataset(val_df, tok, TEXT_COL, tc["max_len"])
    cw = torch.tensor([weights.get(c, 1.0) for c in range(NUM_CLASSES)], dtype=torch.float) if weights else None
    trainer = WeightedTrainer(
        model=model, args=_training_args(cfg, str(out_dir / "checkpoints"), hp, seed),
        train_dataset=tr, eval_dataset=va, data_collator=DataCollatorWithPadding(tok),
        compute_metrics=hf_compute_metrics, class_weights=cw,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=tc["early_stopping_patience"])],
    )
    trainer.train()
    val_f1 = trainer.evaluate()["eval_macro_f1"]
    return trainer, tok, val_f1


def train_transformer(cfg: dict, name: str, strategy: str = "baseline", seed: int | None = None,
                      tune: bool = False, splits: dict | None = None, save: bool = True) -> dict:
    tc = cfg["transformer"]
    checkpoint = tc["models"][name]
    seed = cfg["project"]["seed"] if seed is None else seed
    splits = splits or load_splits(cfg)
    train_df, weights = prepare_training_data(splits["train"], strategy, cfg, seed)
    val_df, test_df = splits["val"], splits["test"]
    out_dir = model_dir(cfg, name, strategy, seed, create=True)

    base = {"lr": tc["lr"], "batch_size": tc["batch_size"], "epochs": tc["epochs"]}
    combos = [base]
    if tune:
        combos = [{"lr": lr, "batch_size": bs, "epochs": tc["epochs"]}
                  for lr, bs in itertools.product(tc["grid"]["lr"], tc["grid"]["batch_size"])]

    best = (-1.0, None, None, None)
    for hp in combos:
        log.info("Fine-tuning %s | strategy=%s | seed=%d | %s", name, strategy, seed, hp)
        trainer, tok, f1 = _run_one(cfg, name, checkpoint, strategy, hp, seed, train_df, val_df, weights, out_dir)
        log.info("val macro-F1 = %.4f", f1)
        if f1 > best[0]:
            best = (f1, hp, trainer, tok)
    best_f1, best_hp, trainer, tok = best

    te = TransformerDataset(test_df, tok, TEXT_COL, tc["max_len"])
    pred = trainer.predict(te)
    y_pred = np.argmax(pred.predictions, axis=-1)
    report_run(cfg, name, strategy, seed, pred.label_ids, y_pred, split="test")

    if save:
        trainer.save_model(str(out_dir))
        tok.save_pretrained(str(out_dir))
        save_json({"model": name, "checkpoint": checkpoint, "strategy": strategy, "seed": seed,
                   "hparams": best_hp, "val_macro_f1": best_f1, "max_len": tc["max_len"]}, out_dir / "info.json")
    shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)  # keep Drive/disk usage small
    return {"val_macro_f1": best_f1, "hparams": best_hp, "out_dir": str(out_dir)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["distilbert", "roberta"])
    ap.add_argument("--strategy", default="baseline")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    splits = load_splits(cfg)
    for s in a.seeds or cfg["seeds"]:
        train_transformer(cfg, a.model, a.strategy, s, a.tune, splits)


if __name__ == "__main__":
    main()
