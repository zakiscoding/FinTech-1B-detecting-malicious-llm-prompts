"""Candidate classifiers with fixed defaults (no HPO grids).

Sklearn baselines + small Keras FFNN architecture variants for comparison.
Sibling agents own hyperparameter search and full final training.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def available_keras() -> bool:
    try:
        import tensorflow  # noqa: F401

        return True
    except ImportError:
        return False


def build_sklearn_candidates() -> dict[str, Any]:
    """Return name -> unfitted estimator (expects TF-IDF matrices)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC

    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        ),
        "linear_svc": LinearSVC(
            class_weight="balanced",
            dual="auto",
            max_iter=5000,
            random_state=42,
        ),
        "multinomial_nb": MultinomialNB(alpha=1.0),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
    }


def build_ffnn(
    input_dim: int,
    hidden_units: list[int],
    dropout: float = 0.3,
    name: str = "ffnn",
):
    """Build a small Keras feedforward binary classifier (dense TF-IDF input)."""
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(input_dim,), name="tfidf_input")
    x = inputs
    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, activation="relu", name=f"{name}_h{i}")(x)
        if dropout > 0:
            x = layers.Dropout(dropout, name=f"{name}_drop{i}")(x)
    outputs = layers.Dense(1, activation="sigmoid", name=f"{name}_out")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.Recall(name="recall"),
            keras.metrics.Precision(name="precision"),
        ],
    )
    return model


def keras_candidate_specs() -> dict[str, dict[str, Any]]:
    """Architecture variants only — not a hyperparameter search grid."""
    return {
        "keras_ffnn_64": {"hidden_units": [64], "dropout": 0.3, "epochs": 8, "batch_size": 64},
        "keras_ffnn_128_64": {
            "hidden_units": [128, 64],
            "dropout": 0.3,
            "epochs": 8,
            "batch_size": 64,
        },
        "keras_ffnn_256_128": {
            "hidden_units": [256, 128],
            "dropout": 0.4,
            "epochs": 8,
            "batch_size": 64,
        },
    }


def _to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def fit_predict_sklearn(model: Any, X_train, y_train, X_eval) -> np.ndarray:
    model.fit(X_train, y_train)
    return model.predict(X_eval)


def fit_predict_keras(
    spec: dict[str, Any],
    X_train,
    y_train,
    X_val,
    y_val,
    X_eval,
    name: str,
    threshold: float = 0.5,
) -> np.ndarray:
    """Train a Keras FFNN; early-stop on validation recall; predict on X_eval."""
    from tensorflow import keras

    X_train_d = _to_dense(X_train)
    X_val_d = _to_dense(X_val)
    X_eval_d = _to_dense(X_eval)

    model = build_ffnn(
        input_dim=X_train_d.shape[1],
        hidden_units=list(spec["hidden_units"]),
        dropout=float(spec["dropout"]),
        name=name,
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_recall",
            mode="max",
            patience=2,
            restore_best_weights=True,
            verbose=0,
        )
    ]
    model.fit(
        X_train_d,
        y_train,
        validation_data=(X_val_d, y_val),
        epochs=int(spec["epochs"]),
        batch_size=int(spec["batch_size"]),
        verbose=0,
        callbacks=callbacks,
        class_weight={0: 1.0, 1: 2.0},
    )
    proba = model.predict(X_eval_d, verbose=0).ravel()
    return (proba >= threshold).astype(int)
