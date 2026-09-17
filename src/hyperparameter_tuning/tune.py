#!/usr/bin/env python3
"""Run RandomizedSearchCV for promising prompt-injection classifiers.

Uses only the Hugging Face *train* split for fit/validation.
The official *test* split is never loaded for scoring here.

Primary objective: maximize recall on injection label 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import (
    DATASET_NAME,
    DATASET_REVISION,
    N_ITER,
    N_JOBS,
    N_SPLITS,
    PRIMARY_SCORING,
    RANDOM_STATE,
    SEARCH_SPACES,
    SECONDARY_METRICS,
    TFIDF_BASE,
    VAL_SIZE,
)

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def load_train_texts_labels() -> tuple[pd.Series, pd.Series]:
    """Load train split only; never touch the official test split for tuning."""
    ds = load_dataset(DATASET_NAME, revision=DATASET_REVISION, split="train")
    df = ds.to_pandas()
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError(f"Unexpected columns: {df.columns.tolist()}")
    df = df.dropna(subset=["text", "label"]).drop_duplicates(subset=["text"])
    df["label"] = df["label"].astype(int)
    if set(df["label"].unique()) - {0, 1}:
        raise ValueError(f"Unexpected labels: {sorted(df['label'].unique())}")
    return df["text"], df["label"]


def make_estimator(name: str) -> Pipeline:
    tfidf = TfidfVectorizer(**TFIDF_BASE)
    if name == "logistic_regression":
        clf = LogisticRegression(random_state=RANDOM_STATE)
    elif name == "linear_svc":
        clf = LinearSVC(random_state=RANDOM_STATE)
    elif name == "multinomial_nb":
        clf = MultinomialNB()
    elif name == "sgd_classifier":
        clf = SGDClassifier(random_state=RANDOM_STATE)
    elif name == "random_forest":
        clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)
    else:
        raise KeyError(f"Unknown model: {name}")
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def decision_or_proba(pipeline: Pipeline, X: pd.Series) -> np.ndarray:
    """Return scores for positive class when available."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    if hasattr(pipeline, "decision_function"):
        return pipeline.decision_function(X)
    # Fallback: hard labels as scores
    return pipeline.predict(X).astype(float)


def holdout_metrics(pipeline: Pipeline, X_val: pd.Series, y_val: pd.Series) -> dict[str, Any]:
    y_pred = pipeline.predict(X_val)
    scores = decision_or_proba(pipeline, X_val)
    metrics: dict[str, Any] = {
        "recall_injection": float(recall_score(y_val, y_pred, pos_label=1, zero_division=0)),
        "precision_injection": float(precision_score(y_val, y_pred, pos_label=1, zero_division=0)),
        "f1_injection": float(f1_score(y_val, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_val, y_pred).tolist(),
        "classification_report": classification_report(
            y_val, y_pred, target_names=["benign_0", "injection_1"], zero_division=0
        ),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_val, scores))
    except ValueError:
        metrics["roc_auc"] = None
    try:
        metrics["average_precision"] = float(average_precision_score(y_val, scores))
    except ValueError:
        metrics["average_precision"] = None
    return metrics


def tune_one(
    name: str,
    X_fit: pd.Series,
    y_fit: pd.Series,
    X_val: pd.Series,
    y_val: pd.Series,
    n_iter: int,
) -> dict[str, Any]:
    pipe = make_estimator(name)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=SEARCH_SPACES[name],
        n_iter=n_iter,
        scoring=PRIMARY_SCORING,
        cv=cv,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        refit=True,
        verbose=1,
        return_train_score=False,
    )
    print(f"\n=== Tuning {name} (n_iter={n_iter}, scoring={PRIMARY_SCORING}) ===", flush=True)
    search.fit(X_fit, y_fit)

    best = search.best_estimator_
    holdout = holdout_metrics(best, X_val, y_val)

    # Summarize top-5 CV trials by mean test score (recall).
    cv_df = pd.DataFrame(search.cv_results_).sort_values("mean_test_score", ascending=False)
    top = []
    for _, row in cv_df.head(5).iterrows():
        top.append(
            {
                "rank": int(row["rank_test_score"]),
                "mean_cv_recall": float(row["mean_test_score"]),
                "std_cv_recall": float(row["std_test_score"]),
                "params": _jsonable(row["params"]),
            }
        )

    return {
        "model": name,
        "best_cv_recall": float(search.best_score_),
        "best_params": _jsonable(search.best_params_),
        "holdout_from_train": holdout,
        "top_cv_trials": top,
        "n_iter": n_iter,
        "cv_folds": N_SPLITS,
        "primary_scoring": PRIMARY_SCORING,
        "secondary_metrics_reported": list(SECONDARY_METRICS),
    }


def run(models: list[str] | None = None, n_iter: int | None = None, out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir) if out_dir else ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    n_iter = n_iter or N_ITER
    model_names = models or list(SEARCH_SPACES.keys())

    X, y = load_train_texts_labels()
    # Inner holdout from train only — official HF test is never used.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X, y, test_size=VAL_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    results: dict[str, Any] = {
        "dataset": DATASET_NAME,
        "revision": DATASET_REVISION,
        "protocol": {
            "official_test_used": False,
            "train_rows_after_dedupe": int(len(X)),
            "fit_rows": int(len(X_fit)),
            "val_rows_from_train": int(len(X_val)),
            "val_size": VAL_SIZE,
            "label_counts_fit": y_fit.value_counts().sort_index().to_dict(),
            "label_counts_val": y_val.value_counts().sort_index().to_dict(),
            "primary_metric": "recall on label 1 (injection)",
            "cv": f"StratifiedKFold(n_splits={N_SPLITS})",
            "search": f"RandomizedSearchCV(n_iter={n_iter})",
        },
        "models": [],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    for name in model_names:
        if name not in SEARCH_SPACES:
            raise KeyError(f"{name} not in SEARCH_SPACES")
        results["models"].append(tune_one(name, X_fit, y_fit, X_val, y_val, n_iter))

    # Rank by holdout injection recall, then F1, then CV recall.
    ranked = sorted(
        results["models"],
        key=lambda m: (
            m["holdout_from_train"]["recall_injection"],
            m["holdout_from_train"]["f1_injection"],
            m["best_cv_recall"],
        ),
        reverse=True,
    )
    results["best_overall"] = {
        "model": ranked[0]["model"],
        "best_params": ranked[0]["best_params"],
        "holdout_from_train": {
            k: ranked[0]["holdout_from_train"][k]
            for k in (
                "recall_injection",
                "precision_injection",
                "f1_injection",
                "roc_auc",
                "average_precision",
                "confusion_matrix",
            )
        },
        "best_cv_recall": ranked[0]["best_cv_recall"],
        "ranking_note": "Ranked by train-holdout recall(label=1), then F1, then CV recall. Official test untouched.",
    }

    out_path = out_dir / "tuning_results.json"
    out_path.write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")

    summary_lines = [
        "# Hyperparameter tuning summary",
        "",
        f"- Dataset: `{DATASET_NAME}` @ `{DATASET_REVISION}`",
        "- Official test split: **held out** (not used).",
        f"- Fit/val from train: {results['protocol']['fit_rows']} / {results['protocol']['val_rows_from_train']}",
        f"- Primary metric: recall on injection label 1 (`{PRIMARY_SCORING}`).",
        "",
        "## Best overall (train-holdout)",
        "",
        f"- Model: `{results['best_overall']['model']}`",
        f"- Holdout recall (1): **{results['best_overall']['holdout_from_train']['recall_injection']:.4f}**",
        f"- Holdout precision (1): {results['best_overall']['holdout_from_train']['precision_injection']:.4f}",
        f"- Holdout F1 (1): {results['best_overall']['holdout_from_train']['f1_injection']:.4f}",
        f"- Best CV recall: {results['best_overall']['best_cv_recall']:.4f}",
        f"- Best params: `{json.dumps(results['best_overall']['best_params'])}`",
        "",
        "## Per-model holdout metrics",
        "",
        "| Model | CV recall | Holdout recall(1) | Precision(1) | F1(1) | ROC-AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in ranked:
        h = m["holdout_from_train"]
        roc = h["roc_auc"] if h["roc_auc"] is not None else float("nan")
        summary_lines.append(
            f"| `{m['model']}` | {m['best_cv_recall']:.4f} | {h['recall_injection']:.4f} | "
            f"{h['precision_injection']:.4f} | {h['f1_injection']:.4f} | {roc:.4f} |"
        )
    summary_path = out_dir / "tuning_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(summary_lines), flush=True)
    print(f"\nWrote {out_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(SEARCH_SPACES.keys()),
        default=None,
        help="Subset of models to tune (default: all).",
    )
    parser.add_argument("--n-iter", type=int, default=N_ITER, help="RandomizedSearchCV iterations.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ARTIFACTS_DIR,
        help="Directory for tuning_results.json and tuning_summary.md.",
    )
    args = parser.parse_args(argv)
    run(models=args.models, n_iter=args.n_iter, out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    # Allow `python -m src.hyperparameter_tuning.tune` from repo root.
    sys.exit(main())
