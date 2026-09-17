"""Finalizable estimators with training-time optimization hooks.

This module intentionally trains a small set of production-oriented models
(TF-IDF + logistic regression, TF-IDF + MLP with early stopping). Broad model
bakeoffs and hyperparameter search grids live in sibling workstreams.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from .optimize import compute_class_weights

ModelName = Literal["logreg", "mlp"]


@dataclass
class FittedModel:
    """Trained sklearn pipeline plus metadata used at eval time."""

    name: ModelName
    pipeline: Pipeline
    class_weights: Dict[int, float]
    extra: Dict[str, Any]


def build_vectorizer(
    max_features: int = 20000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )


def train_logreg(
    X_train,
    y_train,
    *,
    max_features: int = 20000,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> FittedModel:
    """TF-IDF + logistic regression with balanced class weights."""
    weights = compute_class_weights(y_train)
    pipe = Pipeline(
        steps=[
            ("tfidf", build_vectorizer(max_features=max_features)),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    class_weight=weights,
                    max_iter=max_iter,
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    return FittedModel(
        name="logreg",
        pipeline=pipe,
        class_weights=weights,
        extra={"C": C, "max_features": max_features},
    )


def train_mlp(
    X_train,
    y_train,
    *,
    max_features: int = 20000,
    hidden_layer_sizes: tuple[int, ...] = (128, 64),
    max_iter: int = 40,
    early_stopping: bool = True,
    validation_fraction: float = 0.1,
    n_iter_no_change: int = 5,
    random_state: int = 42,
) -> FittedModel:
    """TF-IDF + MLP with early stopping (internal val slice of train).

    Class imbalance is handled via ``sample_weight`` derived from balanced
    class weights, because ``MLPClassifier`` does not accept ``class_weight``.
    """
    weights = compute_class_weights(y_train)
    y_arr = np.asarray(y_train, dtype=int)
    sample_weight = np.asarray([weights[int(y)] for y in y_arr], dtype=float)

    # Fit TF-IDF first so we can pass sample_weight to the MLP step.
    vectorizer = build_vectorizer(max_features=max_features)
    X_vec = vectorizer.fit_transform(X_train)
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=max_iter,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change,
        random_state=random_state,
    )
    clf.fit(X_vec, y_arr, sample_weight=sample_weight)

    pipe = Pipeline(steps=[("tfidf", vectorizer), ("clf", clf)])
    return FittedModel(
        name="mlp",
        pipeline=pipe,
        class_weights=weights,
        extra={
            "hidden_layer_sizes": list(hidden_layer_sizes),
            "max_features": max_features,
            "early_stopping": early_stopping,
            "n_iter_": int(getattr(clf, "n_iter_", max_iter)),
            "best_validation_score_": float(
                getattr(clf, "best_validation_score_", float("nan"))
            ),
        },
    )


def predict_proba_positive(model: FittedModel, X) -> np.ndarray:
    """Return P(label=1) for each row."""
    proba = model.pipeline.predict_proba(X)
    classes = list(model.pipeline.named_steps["clf"].classes_)
    pos_idx = classes.index(1)
    return proba[:, pos_idx]


def train_model(
    name: ModelName,
    X_train,
    y_train,
    **kwargs,
) -> FittedModel:
    if name == "logreg":
        return train_logreg(X_train, y_train, **kwargs)
    if name == "mlp":
        return train_mlp(X_train, y_train, **kwargs)
    raise ValueError(f"Unknown model: {name}")
