# models/classical
Scikit-learn `Pipeline(TfidfVectorizer -> classifier)` objects saved with `joblib`, trained on `text_clean`.
Files: `logreg_baseline.joblib` (also the fallback predictor for notebook 06), `svm_*.joblib`, `rf_*.joblib`, `best_params.json`.
Load: `joblib.load("models/classical/logreg_baseline.joblib").predict_proba([preprocess_classical(text)])`
(LinearSVC has no `predict_proba`.)
