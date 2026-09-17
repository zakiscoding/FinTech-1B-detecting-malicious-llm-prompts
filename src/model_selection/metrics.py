"""Evaluation helpers emphasizing recall / false negatives for label 1."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def binary_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute selection metrics with FN emphasis for the injection class."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    support_pos = int(tp + fn)
    fnr = float(fn / support_pos) if support_pos else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_negative_rate": fnr,
        "confusion_matrix": cm.tolist(),
    }


def rank_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    """Primary: recall (label 1). Tie-break: F1, then precision, then lower FNR."""
    return (
        float(row["recall"]),
        float(row["f1"]),
        float(row["precision"]),
        -float(row["false_negative_rate"]),
    )
