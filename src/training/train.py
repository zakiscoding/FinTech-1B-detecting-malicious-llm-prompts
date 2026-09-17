"""Train/eval entrypoint for finalizable Safe-Guard detectors.

Example
-------
python -m src.training --model logreg --output-dir artifacts/training
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from .data import DATASET_ID, DATASET_REVISION, make_splits
from .metrics import compute_classification_metrics, format_metrics_report
from .models import FittedModel, ModelName, predict_proba_positive, train_model
from .optimize import tune_threshold_for_recall


def evaluate_with_threshold(
    model: FittedModel,
    X,
    y,
    threshold: float,
) -> Dict[str, Any]:
    proba = predict_proba_positive(model, X)
    y_pred = (proba >= threshold).astype(int)
    return compute_classification_metrics(y, y_pred)


def run_train_eval(
    *,
    model_name: ModelName = "logreg",
    val_size: float = 0.15,
    random_state: int = 42,
    min_precision: float = 0.80,
    output_dir: Optional[Path] = None,
    dataset_id: str = DATASET_ID,
    revision: str = DATASET_REVISION,
) -> Dict[str, Any]:
    """Train, tune threshold on validation, evaluate on official test."""
    splits = make_splits(
        val_size=val_size,
        random_state=random_state,
        dataset_id=dataset_id,
        revision=revision,
    )
    model = train_model(model_name, splits.X_train, splits.y_train, random_state=random_state)

    val_proba = predict_proba_positive(model, splits.X_val)
    threshold, val_tune = tune_threshold_for_recall(
        splits.y_val,
        val_proba,
        min_precision=min_precision,
    )

    default_threshold = 0.5
    val_default = evaluate_with_threshold(model, splits.X_val, splits.y_val, default_threshold)
    val_tuned = evaluate_with_threshold(model, splits.X_val, splits.y_val, threshold)
    test_default = evaluate_with_threshold(model, splits.X_test, splits.y_test, default_threshold)
    test_tuned = evaluate_with_threshold(model, splits.X_test, splits.y_test, threshold)

    report: Dict[str, Any] = {
        "dataset": {"id": dataset_id, "revision": revision},
        "split_sizes": {
            "train": int(len(splits.X_train)),
            "val": int(len(splits.X_val)),
            "test": int(len(splits.X_test)),
        },
        "model": {
            "name": model.name,
            "class_weights": model.class_weights,
            "extra": model.extra,
        },
        "threshold_tuning": {
            "min_precision": min_precision,
            "selected_threshold": threshold,
            "validation_at_selection": val_tune,
        },
        "metrics": {
            "validation_default_0_5": val_default,
            "validation_tuned": val_tuned,
            "test_default_0_5": test_default,
            "test_tuned": test_tuned,
        },
        "notes": [
            "Official HF test split used only for final evaluation.",
            "Validation carved from train via stratified split.",
            "Threshold tuned on validation to prioritize injection recall "
            f"subject to precision >= {min_precision:.2f} when feasible.",
            "False-negative rate = FN / (FN + TP) for label 1 (injection).",
        ],
    }

    print(format_metrics_report("validation @ 0.5", val_default))
    print()
    print(format_metrics_report(f"validation @ {threshold:.4f} (tuned)", val_tuned))
    print()
    print(format_metrics_report("test @ 0.5", test_default))
    print()
    print(format_metrics_report(f"test @ {threshold:.4f} (tuned)", test_tuned))
    print()
    print(
        "Selected threshold "
        f"{threshold:.4f} (val recall={val_tune['recall']:.4f}, "
        f"precision={val_tune['precision']:.4f}, FNR={val_tune['false_negative_rate']:.4f})"
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / f"metrics_{model.name}.json"
        model_path = output_dir / f"model_{model.name}.joblib"
        meta_path = output_dir / f"threshold_{model.name}.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        joblib.dump(
            {
                "pipeline": model.pipeline,
                "threshold": threshold,
                "model_name": model.name,
                "class_weights": model.class_weights,
            },
            model_path,
        )
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "threshold": threshold,
                    "min_precision": min_precision,
                    "validation_at_selection": val_tune,
                },
                f,
                indent=2,
            )
        report["artifacts"] = {
            "metrics": str(metrics_path),
            "model": str(model_path),
            "threshold": str(meta_path),
        }
        print(f"Wrote metrics -> {metrics_path}")
        print(f"Wrote model   -> {model_path}")

    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train a finalizable Safe-Guard prompt-injection detector with "
            "class weights, validation threshold tuning (recall-first), and "
            "optional MLP early stopping."
        )
    )
    p.add_argument(
        "--model",
        choices=["logreg", "mlp"],
        default="logreg",
        help="Estimator to train (default: logreg).",
    )
    p.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Fraction of train used for validation (default: 0.15).",
    )
    p.add_argument(
        "--min-precision",
        type=float,
        default=0.80,
        help="Precision floor when tuning threshold for recall (default: 0.80).",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="RNG seed for split / estimators.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/training"),
        help="Directory for metrics JSON and joblib model.",
    )
    p.add_argument(
        "--dataset-id",
        default=DATASET_ID,
        help="Hugging Face dataset id.",
    )
    p.add_argument(
        "--revision",
        default=DATASET_REVISION,
        help="Pinned dataset revision.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> Dict[str, Any]:
    args = build_parser().parse_args(argv)
    return run_train_eval(
        model_name=args.model,
        val_size=args.val_size,
        random_state=args.random_state,
        min_precision=args.min_precision,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        revision=args.revision,
    )


if __name__ == "__main__":
    main()
