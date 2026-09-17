# Model selection recommendation

**Dataset:** `xTRam1/safe-guard-prompt-injection`
**Top model:** `linear_svc`

## Selection criterion

Maximize validation recall on label 1 (injection), then F1, then precision; minimize false negatives.

## Validation metrics (from train only)

- Precision (label 1): **1.0000**
- Recall (label 1): **0.9840**
- F1 (label 1): **0.9919**
- Confusion: TN=861, FP=0, **FN=6**, TP=369
- False-negative rate: **0.0160**

## Runners-up (validation)

- `keras_ffnn_128_64` — recall=0.9813, f1=0.9892, FN=7
- `keras_ffnn_64` — recall=0.9787, f1=0.9892, FN=8
- `keras_ffnn_256_128` — recall=0.9760, f1=0.9879, FN=9

## Official test smoke check (top model only)

- Precision: 0.9953
- Recall: 0.9862
- F1: 0.9907
- FN=9, FP=3

## Notes

- Validation was carved from the official train split only (stratified).
- Official test split used only for a smoke check of the top candidate.
- Hyperparameter search and final training are out of scope for this slice.
