# data/

All data used by the project. Nothing here is written by hand: the folders are filled by
`notebooks/01_eda.ipynb` (and later notebooks). Only public, de-identified datasets are used.

```
data/
├── raw/         downloaded source files (you put Dreaddit CSVs here)
├── interim/     intermediate artifacts, e.g. cached augmented training rows
├── processed/   unified, cleaned dataset + metadata
├── splits/      frozen train / val / test CSVs
└── README.md
```

## Sources
| Dataset | Used for | How to get it |
|---|---|---|
| **GoEmotions** (Google Research, ~58k Reddit comments, 27 emotions + neutral) | Main 8-class training/eval data | Downloaded automatically by notebook 01 (`google-research-datasets/go_emotions`, "simplified") |
| **Dreaddit** (Turcan & McKeown 2019, ~3.5k Reddit posts labelled stressed / not stressed) | The `Stress` class | Not always on the Hugging Face Hub. Download `dreaddit-train.csv` and `dreaddit-test.csv` (e.g. from Kaggle) and place them in `data/raw/` |
| Sentiment140 (optional) | Not used by the notebooks | Skip, or subsample for optional coarse pretraining |

Check each dataset's licence/terms before redistributing any of its files; do not commit raw datasets to a public repo.

## raw/
Only files you add by hand (Dreaddit CSVs). Read-only afterwards.

## interim/
| File | Written by | Meaning |
|---|---|---|
| `augmented_train_s<seed>.csv` | notebook 04 | Synonym-replacement + back-translation rows for minority classes (cached so back-translation runs once). Derived from the **train split only** |

## processed/
| File | Written by | Meaning |
|---|---|---|
| `unified.csv` | notebook 01 | All cleaned, de-duplicated texts with 8-class labels |
| `label_mapping.json` | notebook 01 | 27 → 8 GoEmotions mapping, dropped labels, Dreaddit rule |
| `class_weights.json` | notebook 01 | `w_c = N / (|E|·n_c)` computed on the train split |
| `mood_demo.db` | notebook 06 | SQLite demo of timestamped predictions (scores only, no raw text) |
| `mood_app.db` | app | Per-profile mood scores created by the Streamlit app (scores only, no raw text) |

### Columns of `unified.csv` and of the split files
| Column | Meaning |
|---|---|
| `id` | `go_<id>` or `dd_<n>` |
| `source` | `goemotions` or `dreaddit` |
| `text` | original text (Dreaddit truncated to 128 words) |
| `text_tf` | light cleaning for Bi-LSTM / transformers (URLs, HTML, handles removed; emoji → words; case kept) |
| `text_clean` | classical view: lowercased, stop-words removed (negations kept), lemmatised |
| `label` | one of Joy, Sadness, Anger, Fear, Anxiety, Stress, Surprise, Neutral |
| `label_id` | 0–7 in that order |
| `n_words` | word count of `text_tf` |

## splits/
`train.csv`, `val.csv`, `test.csv` — stratified 80/10/10 with `random_state=42`, **created once and frozen**
(notebook 01 refuses to overwrite them unless `FORCE_REBUILD=True`; do not change this after training starts).

Rules that keep results honest:
- Class weighting, oversampling and augmentation touch the **train split only**.
- Hyper-parameters and model/condition selection use **val** only.
- **test** is used for final reporting only.

## Privacy
Only public datasets are used. App/demo databases keep derived emotion scores and timestamps, never raw user text.
Do not commit anything under `data/` that contains real user entries.
