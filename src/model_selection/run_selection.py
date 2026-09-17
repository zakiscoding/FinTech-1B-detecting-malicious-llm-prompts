#!/usr/bin/env python3
"""Run prompt-injection classifier model selection on Safe-Guard.

Dataset: https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection
- Uses official train/test splits
- Builds stratified validation from train only
- Prioritizes recall on label 1 (injection); reports FN counts

Usage (from repo root):
  python -m src.model_selection.run_selection
  python -m src.model_selection.run_selection --skip-keras
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.model_selection.candidates import (
    available_keras,
    build_sklearn_candidates,
    fit_predict_keras,
    fit_predict_sklearn,
    keras_candidate_specs,
)
from src.model_selection.metrics import binary_report, rank_key

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
DEFAULT_SEED = 42
DEFAULT_VAL_SIZE = 0.15
DEFAULT_MAX_FEATURES = 5000


def load_splits():
    ds = load_dataset("xTRam1/safe-guard-prompt-injection")
    train_df = ds["train"].to_pandas()[["text", "label"]].dropna()
    test_df = ds["test"].to_pandas()[["text", "label"]].dropna()
    train_df["text"] = train_df["text"].astype(str)
    test_df["text"] = test_df["text"].astype(str)
    train_df["label"] = train_df["label"].astype(int)
    test_df["label"] = test_df["label"].astype(int)
    return train_df, test_df


def make_features(train_texts, val_texts, test_texts, max_features: int):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)
    return vectorizer, X_train, X_val, X_test


def run_selection(
    val_size: float = DEFAULT_VAL_SIZE,
    seed: int = DEFAULT_SEED,
    max_features: int = DEFAULT_MAX_FEATURES,
    skip_keras: bool = False,
    write_artifacts: bool = True,
) -> dict:
    train_df, test_df = load_splits()

    X_tr_text, X_va_text, y_tr, y_va = train_test_split(
        train_df["text"],
        train_df["label"],
        test_size=val_size,
        stratify=train_df["label"],
        random_state=seed,
    )
    y_te = test_df["label"].to_numpy()

    _, X_train, X_val, X_test = make_features(
        X_tr_text, X_va_text, test_df["text"], max_features=max_features
    )
    y_tr = y_tr.to_numpy()
    y_va = y_va.to_numpy()

    results: list[dict] = []

    for name, model in build_sklearn_candidates().items():
        y_pred = fit_predict_sklearn(model, X_train, y_tr, X_val)
        metrics = binary_report(y_va, y_pred)
        metrics.update({"model": name, "family": "sklearn", "split": "validation"})
        results.append(metrics)
        _print_row("val", name, metrics)

    if not skip_keras and available_keras():
        for name, spec in keras_candidate_specs().items():
            y_pred = fit_predict_keras(
                spec, X_train, y_tr, X_val, y_va, X_val, name=name
            )
            metrics = binary_report(y_va, y_pred)
            metrics.update(
                {
                    "model": name,
                    "family": "keras",
                    "split": "validation",
                    "arch": spec,
                }
            )
            results.append(metrics)
            _print_row("val", name, metrics)
    elif not skip_keras:
        print("TensorFlow/Keras not available; skipping FFNN candidates.", file=sys.stderr)

    ranked = sorted(
        [r for r in results if r["split"] == "validation"],
        key=rank_key,
        reverse=True,
    )
    top = ranked[0]
    top_name = top["model"]

    test_row = _evaluate_named_on_test(
        top_name, X_train, y_tr, X_val, y_va, X_test, y_te, skip_keras=skip_keras
    )
    if test_row:
        results.append(test_row)
        _print_row("test", top_name, test_row)

    recommendation = {
        "top_model": top_name,
        "selection_criterion": (
            "Maximize validation recall on label 1 (injection), "
            "then F1, then precision; minimize false negatives."
        ),
        "validation_metrics": _metric_slice(top),
        "test_metrics": _metric_slice(test_row) if test_row else None,
        "runners_up": [
            {"model": r["model"], "recall": r["recall"], "f1": r["f1"], "fn": r["fn"]}
            for r in ranked[1:4]
        ],
        "notes": [
            "Validation was carved from the official train split only (stratified).",
            "Official test split used only for a smoke check of the top candidate.",
            "Hyperparameter search and final training are out of scope for this slice.",
        ],
    }

    payload = {
        "dataset": "xTRam1/safe-guard-prompt-injection",
        "n_train_fit": int(len(y_tr)),
        "n_validation": int(len(y_va)),
        "n_test": int(len(y_te)),
        "max_features": max_features,
        "val_size": val_size,
        "seed": seed,
        "results": results,
        "ranked_validation": [
            {"model": r["model"], "recall": r["recall"], "f1": r["f1"], "fn": r["fn"]}
            for r in ranked
        ],
        "recommendation": recommendation,
    }

    if write_artifacts:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        out_json = ARTIFACTS_DIR / "selection_results.json"
        out_md = ARTIFACTS_DIR / "recommendation.md"
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        out_md.write_text(_format_recommendation_md(payload), encoding="utf-8")
        print(f"Wrote {out_json}")
        print(f"Wrote {out_md}")

    return payload


def _metric_slice(row: dict) -> dict:
    keys = (
        "precision",
        "recall",
        "f1",
        "fn",
        "fp",
        "tn",
        "tp",
        "false_negative_rate",
        "confusion_matrix",
    )
    return {k: row[k] for k in keys}


def _print_row(split: str, name: str, metrics: dict) -> None:
    print(
        f"[{split}] {name:22s} recall={metrics['recall']:.4f} "
        f"precision={metrics['precision']:.4f} f1={metrics['f1']:.4f} "
        f"FN={metrics['fn']} FP={metrics['fp']}"
    )


def _evaluate_named_on_test(
    name: str,
    X_train,
    y_tr,
    X_val,
    y_va,
    X_test,
    y_te,
    skip_keras: bool,
):
    sklearn_models = build_sklearn_candidates()
    if name in sklearn_models:
        y_pred = fit_predict_sklearn(sklearn_models[name], X_train, y_tr, X_test)
        metrics = binary_report(y_te, y_pred)
        metrics.update({"model": name, "family": "sklearn", "split": "test"})
        return metrics

    if skip_keras or not available_keras():
        return None
    specs = keras_candidate_specs()
    if name not in specs:
        return None
    y_pred = fit_predict_keras(
        specs[name], X_train, y_tr, X_val, y_va, X_test, name=f"{name}_holdout"
    )
    metrics = binary_report(y_te, y_pred)
    metrics.update({"model": name, "family": "keras", "split": "test"})
    return metrics


def _format_recommendation_md(payload: dict) -> str:
    rec = payload["recommendation"]
    vm = rec["validation_metrics"]
    lines = [
        "# Model selection recommendation",
        "",
        f"**Dataset:** `{payload['dataset']}`",
        f"**Top model:** `{rec['top_model']}`",
        "",
        "## Selection criterion",
        "",
        rec["selection_criterion"],
        "",
        "## Validation metrics (from train only)",
        "",
        f"- Precision (label 1): **{vm['precision']:.4f}**",
        f"- Recall (label 1): **{vm['recall']:.4f}**",
        f"- F1 (label 1): **{vm['f1']:.4f}**",
        f"- Confusion: TN={vm['tn']}, FP={vm['fp']}, **FN={vm['fn']}**, TP={vm['tp']}",
        f"- False-negative rate: **{vm['false_negative_rate']:.4f}**",
        "",
        "## Runners-up (validation)",
        "",
    ]
    for r in rec["runners_up"]:
        lines.append(
            f"- `{r['model']}` — recall={r['recall']:.4f}, f1={r['f1']:.4f}, FN={r['fn']}"
        )
    if rec.get("test_metrics"):
        tm = rec["test_metrics"]
        lines.extend(
            [
                "",
                "## Official test smoke check (top model only)",
                "",
                f"- Precision: {tm['precision']:.4f}",
                f"- Recall: {tm['recall']:.4f}",
                f"- F1: {tm['f1']:.4f}",
                f"- FN={tm['fn']}, FP={tm['fp']}",
            ]
        )
    lines.extend(["", "## Notes", ""])
    for n in rec["notes"]:
        lines.append(f"- {n}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--skip-keras", action="store_true")
    parser.add_argument("--no-artifacts", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    run_selection(
        val_size=args.val_size,
        seed=args.seed,
        max_features=args.max_features,
        skip_keras=args.skip_keras,
        write_artifacts=not args.no_artifacts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
