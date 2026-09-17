# Hyperparameter tuning (Moukthika)

Tunes **promising classical TF-IDF classifiers** for Safe-Guard prompt-injection detection.

Sibling workstreams own model-family selection and the final training/optimization package. This module only defines search spaces, runs RandomizedSearchCV on the **train** split, and records best configs + validation metrics.

## Dataset protocol

- Source: [`xTRam1/safe-guard-prompt-injection`](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection) @ `a3a877d608f37b7d20d9945671902df895ecdb46`
- Official **test** split is **never** used for tuning or selection
- From `train`: stratified 80/20 fit vs holdout; 3-fold StratifiedKFold inside the fit portion
- Primary ranking metric: **recall on label 1** (injection)

## Models tuned

| Model | Why included |
| --- | --- |
| `logistic_regression` | Strong linear sparse-text baseline |
| `linear_svc` | Often competitive with LR on TF-IDF |
| `multinomial_nb` | Fast probabilistic BoW baseline |
| `sgd_classifier` | Scalable linear (log_loss / hinge / modified_huber) |
| `random_forest` | Non-linear ensemble check |

## Search space notes

Shared vectorizer knobs (most models):

- `ngram_range`: unigrams vs unigram+bigram
- `min_df` / `max_df`: rare / very common token filters
- `sublinear_tf`: log-scaled TF on/off
- RF also searches `max_features` for vocabulary size caps

Classifier-specific:

- **LR / LinearSVC**: `C`, `class_weight` (None vs balanced), solver/`max_iter` as needed
- **MultinomialNB**: `alpha`, `fit_prior`
- **SGD**: `loss`, `alpha`, `penalty`, `class_weight`
- **RandomForest**: `n_estimators`, `max_depth`, `min_samples_leaf`, `class_weight`

Search: `RandomizedSearchCV` with `n_iter=20` (default), `scoring="recall"`, `random_state=42`.

## Run

From repo root (after installing deps in `requirements.txt`):

```bash
python -m src.hyperparameter_tuning.tune
# or a subset:
python -m src.hyperparameter_tuning.tune --models logistic_regression linear_svc --n-iter 15
```

Artifacts:

- `src/hyperparameter_tuning/artifacts/tuning_results.json` — full params + metrics
- `src/hyperparameter_tuning/artifacts/tuning_summary.md` — human-readable ranking

Also see `notebooks/moukthika_hyperparameter_tuning.ipynb` for an interactive walkthrough.
