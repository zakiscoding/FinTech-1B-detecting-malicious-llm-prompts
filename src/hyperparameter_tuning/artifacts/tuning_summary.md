# Hyperparameter tuning summary

- Dataset: `xTRam1/safe-guard-prompt-injection` @ `a3a877d608f37b7d20d9945671902df895ecdb46`
- Official test split: **held out** (not used).
- Fit/val from train: 6498 / 1625
- Primary metric: recall on injection label 1 (`recall`).

## Best overall (train-holdout)

- Model: `multinomial_nb`
- Holdout recall (1): **0.9898**
- Holdout precision (1): 0.9660
- Holdout F1 (1): 0.9777
- Best CV recall: 0.9867
- Best params: `{"clf__alpha": 0.020237789540485076, "clf__fit_prior": false, "tfidf__max_df": 0.7568350301015521, "tfidf__min_df": 1, "tfidf__ngram_range": [1, 2], "tfidf__sublinear_tf": false}`

## Per-model holdout metrics

| Model | CV recall | Holdout recall(1) | Precision(1) | F1(1) | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| `multinomial_nb` | 0.9867 | 0.9898 | 0.9660 | 0.9777 | 0.9983 |
| `sgd_classifier` | 0.9872 | 0.9877 | 0.9938 | 0.9908 | 0.9998 |
| `linear_svc` | 0.9800 | 0.9836 | 0.9979 | 0.9907 | 0.9998 |
| `logistic_regression` | 0.9749 | 0.9816 | 0.9958 | 0.9886 | 0.9994 |
| `random_forest` | 0.9672 | 0.9652 | 0.9937 | 0.9792 | 0.9988 |
