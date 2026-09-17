"""Classification metrics with explicit false-negative rate for injection class."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

POSITIVE_LABEL = 1


def false_negative_rate(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    pos_label: int = POSITIVE_LABEL,
) -> float:
    """Fraction of true positives missed: FN / (FN + TP)."""
    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))
    mask = y_true_arr == pos_label
    n_pos = int(mask.sum())
    if n_pos == 0:
        return float("nan")
    fn = int(((y_pred_arr[mask]) != pos_label).sum())
    return fn / n_pos


def compute_classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    pos_label: int = POSITIVE_LABEL,
    labels: tuple[int, int] = (0, 1),
) -> Dict[str, Any]:
    """Return precision/recall/F1, confusion matrix, and false-negative rate."""
    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(labels))
    # rows = true [0, 1], cols = pred [0, 1]
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    return {
        "precision": float(
            precision_score(y_true_arr, y_pred_arr, pos_label=pos_label, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true_arr, y_pred_arr, pos_label=pos_label, zero_division=0)
        ),
        "f1": float(f1_score(y_true_arr, y_pred_arr, pos_label=pos_label, zero_division=0)),
        "false_negative_rate": false_negative_rate(
            y_true_arr, y_pred_arr, pos_label=pos_label
        ),
        "confusion_matrix": {
            "labels": list(labels),
            "matrix": cm.tolist(),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "support": {
            "n": int(len(y_true_arr)),
            "n_positive": int((y_true_arr == pos_label).sum()),
            "n_negative": int((y_true_arr != pos_label).sum()),
        },
    }


def format_metrics_report(split_name: str, metrics: Mapping[str, Any]) -> str:
    """Human-readable metrics block for CLI / notebook output."""
    cm = metrics["confusion_matrix"]
    lines = [
        f"=== {split_name} ===",
        f"precision (injection): {metrics['precision']:.4f}",
        f"recall    (injection): {metrics['recall']:.4f}",
        f"f1        (injection): {metrics['f1']:.4f}",
        f"false-negative rate:   {metrics['false_negative_rate']:.4f}",
        f"confusion [[tn, fp], [fn, tp]]: {cm['matrix']}",
        f"TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}",
    ]
    return "\n".join(lines)
