# Training & optimization (`src/training`)

Owned by the **model training and optimization** workstream (Moukthika Nellutla).

## What this does

- Loads [`xTRam1/safe-guard-prompt-injection`](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection) at the pinned revision
- Stratified validation split from **train only**; official **test** held out for final eval
- Trains finalizable TF-IDF models:
  - `logreg` — logistic regression with balanced class weights
  - `mlp` — MLP with early stopping + sample weights
- Tunes the decision threshold on validation to prioritize **injection recall** (precision floor)
- Reports precision / recall / F1 / confusion matrix / **false-negative rate**

## Not in this package

- Broad model-selection bakeoff
- Hyperparameter search grids

## Run

```bash
pip install -r requirements.txt
PYTHONPATH=. python -m src.training --model logreg --output-dir artifacts/training
PYTHONPATH=. python -m src.training --model mlp --output-dir artifacts/training
```

Notebook: `notebooks/moukthika_train_optimize.ipynb`

Metrics land in `artifacts/training/metrics_*.json`.
