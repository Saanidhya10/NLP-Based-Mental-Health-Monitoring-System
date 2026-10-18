# results/

Experiment outputs. The CSVs here are **header-only templates** with exactly the columns the notebooks write;
they contain no numbers until you run the experiments. Never fill them by hand.

| File | Written by | Content |
|---|---|---|
| `model_metrics.csv` | notebooks 02, 03, 04 | One row per (model, condition, seed): accuracy, macro P/R/F1, min-class F1, per-class F1, validation macro-F1. Rows are upserted, so re-running replaces the same (model, condition, seed) |
| `model_comparison.csv` | notebook 05 | Paper Table 5: baseline and best-mitigation row per model (mean ± std over seeds, CPU latency) |
| `imbalance_results.csv` | notebook 04 | Best transformer: per-class F1 under baseline / class_weight / oversample / augment / augment+class_weight |
| `trend_quality.csv` | notebook 06 | Aggregated trend vs. scripted ground truth per simulated scenario (message vs. daily vs. EMA error) |
| `latency_results.csv` | notebooks 02, 03, 04, 06 | ms per text per model and device (CPU/GPU), inference-only and end-to-end |
| `figures/` | all notebooks | PNG plots (class distribution, confusion matrices, per-class F1, trends) |
| `reports/` | notebook 05 | e.g. `error_examples.csv` for the write-up |

Also created by the notebooks (not templates): `predictions_<model>_<condition>.csv` (test predictions),
`escalation_eval.csv` and `trend_params.json` (notebook 06; τ, k, α are read by the app).

Run order: 01 → 02 → 03 → 04 → 05 → 06.
