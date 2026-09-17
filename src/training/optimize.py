"""Optimization helpers: class weights and decision-threshold tuning."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from .metrics import compute_classification_metrics


def compute_class_weights(
    y: Iterable[int],
    *,
    classes: Sequence[int] = (0, 1),
) -> Dict[int, float]:
    """Return sklearn-style balanced class weights keyed by label."""
    y_arr = np.asarray(list(y), dtype=int)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.asarray(classes, dtype=int),
        y=y_arr,
    )
    return {int(c): float(w) for c, w in zip(classes, weights)}


def tune_threshold_for_recall(
    y_true: Iterable[int],
    y_proba: Iterable[float],
    *,
    min_precision: float = 0.80,
    candidate_thresholds: Optional[Sequence[float]] = None,
    prefer_high_recall: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """Pick a probability threshold that prioritizes injection recall.

    Strategy
    --------
    1. Sweep candidate thresholds on validation probabilities.
    2. Keep candidates whose precision >= ``min_precision`` (when any exist).
    3. Among survivors, maximize recall; break ties with F1 then lower threshold
       (more aggressive detection) when ``prefer_high_recall`` is True.

    If no candidate meets the precision floor, fall back to the threshold with
    the best F1 (still reporting recall / FNR so the trade-off is visible).
    """
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_proba_arr = np.asarray(list(y_proba), dtype=float)
    if candidate_thresholds is None:
        # Dense grid plus validation score quantiles for coverage.
        grid = np.linspace(0.05, 0.95, 37)
        quantiles = np.unique(np.quantile(y_proba_arr, np.linspace(0.05, 0.95, 19)))
        candidate_thresholds = np.unique(np.concatenate([grid, quantiles]))

    rows = []
    for thr in candidate_thresholds:
        y_pred = (y_proba_arr >= float(thr)).astype(int)
        m = compute_classification_metrics(y_true_arr, y_pred)
        rows.append(
            {
                "threshold": float(thr),
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "false_negative_rate": m["false_negative_rate"],
            }
        )

    feasible = [r for r in rows if r["precision"] >= min_precision]
    pool = feasible if feasible else rows

    def sort_key(row: Dict[str, float]) -> Tuple[float, float, float]:
        # Higher recall / f1 better; lower threshold better when preferring recall.
        thr_term = -row["threshold"] if prefer_high_recall else row["threshold"]
        return (row["recall"], row["f1"], thr_term)

    best = max(pool, key=sort_key)
    return best["threshold"], best
