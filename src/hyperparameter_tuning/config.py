"""Search spaces and constants for prompt-injection hyperparameter tuning.

Scope: tune promising classical TF-IDF classifiers only.
Model-family selection and final training packages live in sibling workstreams.
"""

from __future__ import annotations

from scipy.stats import loguniform, randint, uniform

DATASET_NAME = "xTRam1/safe-guard-prompt-injection"
DATASET_REVISION = "a3a877d608f37b7d20d9945671902df895ecdb46"

# Hold out official HF test; validate only within train.
VAL_SIZE = 0.2
RANDOM_STATE = 42
N_SPLITS = 3  # stratified CV folds inside the train portion of the train split
N_ITER = 20  # RandomizedSearchCV iterations per model
N_JOBS = -1

# Primary ranking metric: recall on injection label 1 (missed attacks cost more).
PRIMARY_SCORING = "recall"
SECONDARY_METRICS = ("precision", "f1", "roc_auc", "average_precision")

# Shared TF-IDF defaults; vectorizer params are also searched per model.
TFIDF_BASE = {
    "lowercase": True,
    "strip_accents": "unicode",
}

SEARCH_SPACES: dict[str, dict] = {
    # Strong linear baseline for sparse text; class_weight for injection recall.
    "logistic_regression": {
        "clf__C": loguniform(1e-2, 1e2),
        "clf__penalty": ["l2"],
        "clf__solver": ["liblinear", "saga"],
        "clf__class_weight": [None, "balanced"],
        "clf__max_iter": [2000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2, 3],
        "tfidf__max_df": uniform(0.7, 0.29),
        "tfidf__sublinear_tf": [True, False],
    },
    # Linear SVM — often competitive with LR on TF-IDF.
    "linear_svc": {
        "clf__C": loguniform(1e-2, 1e2),
        "clf__class_weight": [None, "balanced"],
        "clf__max_iter": [5000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2, 3],
        "tfidf__max_df": uniform(0.7, 0.29),
        "tfidf__sublinear_tf": [True, False],
    },
    # Fast probabilistic baseline; good for sparse bag-of-words.
    "multinomial_nb": {
        "clf__alpha": loguniform(1e-3, 10.0),
        "clf__fit_prior": [True, False],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2, 3],
        "tfidf__max_df": uniform(0.7, 0.29),
        "tfidf__sublinear_tf": [True, False],
    },
    # SGD with log_loss ≈ online logistic regression; hinge ≈ linear SVM.
    "sgd_classifier": {
        "clf__loss": ["log_loss", "hinge", "modified_huber"],
        "clf__alpha": loguniform(1e-6, 1e-2),
        "clf__penalty": ["l2", "l1", "elasticnet"],
        "clf__class_weight": [None, "balanced"],
        "clf__max_iter": [2000],
        "clf__tol": [1e-3],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2, 3],
        "tfidf__max_df": uniform(0.7, 0.29),
        "tfidf__sublinear_tf": [True, False],
    },
    # Non-linear ensemble; smaller tree depth to keep search tractable.
    "random_forest": {
        "clf__n_estimators": randint(100, 401),
        "clf__max_depth": [None, 20, 40, 80],
        "clf__min_samples_leaf": randint(1, 8),
        "clf__class_weight": [None, "balanced", "balanced_subsample"],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2],
        "tfidf__max_df": uniform(0.75, 0.24),
        "tfidf__sublinear_tf": [True, False],
        "tfidf__max_features": [5000, 10000, None],
    },
}
