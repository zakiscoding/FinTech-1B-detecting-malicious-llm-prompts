# Reported official-test metrics

Dataset: `xTRam1/safe-guard-prompt-injection` (revision `a3a877d608f37b7d20d9945671902df895ecdb46`).  
Validation was carved from train (15%, stratified). Thresholds tuned on validation with `min_precision=0.80`.

## Logistic regression (recommended default)

| Setting | Precision | Recall | F1 | FNR | Confusion `[[TN,FP],[FN,TP]]` |
| --- | ---: | ---: | ---: | ---: | --- |
| threshold 0.5 | 0.9969 | 0.9769 | 0.9868 | **0.0231** | `[[1408, 2], [15, 635]]` |
| threshold **0.15** (tuned) | 0.8265 | **0.9969** | 0.9038 | **0.0031** | `[[1274, 136], [2, 648]]` |

False-negative rate at the tuned threshold: **0.31%** (2 missed injections of 650).

## MLP (early stopping)

| Setting | Precision | Recall | F1 | FNR | Confusion `[[TN,FP],[FN,TP]]` |
| --- | ---: | ---: | ---: | ---: | --- |
| threshold 0.5 | 0.9845 | 0.9785 | 0.9815 | **0.0215** | `[[1400, 10], [14, 636]]` |
| threshold **0.20** (tuned) | 0.8811 | **0.9923** | 0.9334 | **0.0077** | `[[1323, 87], [5, 645]]` |

False-negative rate at the tuned threshold: **0.77%** (5 missed injections of 650).

## Takeaway

Threshold tuning trades some precision for substantially lower FNR. For a security filter that prefers catching attacks, use the tuned threshold (or the saved `threshold_*.json` next to the metrics).
