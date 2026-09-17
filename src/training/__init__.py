"""Training and optimization for Safe-Guard prompt-injection detection."""

from .metrics import compute_classification_metrics, false_negative_rate
from .optimize import compute_class_weights, tune_threshold_for_recall

__all__ = [
    "compute_classification_metrics",
    "compute_class_weights",
    "false_negative_rate",
    "tune_threshold_for_recall",
]
