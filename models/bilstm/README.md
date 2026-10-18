# models/bilstm
PyTorch state dicts (`bilstm_baseline.pt`, `bilstm_class_weight.pt` = best-validation seed of each condition),
`vocab.json` (index -> token list built from the train split) and `config.json` (hidden size, dropout, embedding, max_len, tokenizer regex).
Rebuild the model with `BiLSTMClassifier` from notebook 03 using the values in `config.json`.
