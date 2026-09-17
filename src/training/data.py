"""Load Safe-Guard prompt-injection data and create a stratified validation split.

The official Hugging Face ``test`` split is reserved for final evaluation only.
Validation rows are carved from ``train``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

DATASET_ID = "xTRam1/safe-guard-prompt-injection"
DATASET_REVISION = "a3a877d608f37b7d20d9945671902df895ecdb46"
TEXT_COL = "text"
LABEL_COL = "label"


@dataclass(frozen=True)
class PromptSplits:
    """Train / validation / held-out test text-label arrays."""

    X_train: pd.Series
    y_train: pd.Series
    X_val: pd.Series
    y_val: pd.Series
    X_test: pd.Series
    y_test: pd.Series


def load_raw_frames(
    dataset_id: str = DATASET_ID,
    revision: str = DATASET_REVISION,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download the pinned dataset revision and return train/test DataFrames."""
    ds = load_dataset(dataset_id, revision=revision)
    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()
    for name, frame in (("train", train_df), ("test", test_df)):
        missing = {TEXT_COL, LABEL_COL} - set(frame.columns)
        if missing:
            raise ValueError(f"{name} split missing columns: {sorted(missing)}")
        if frame[TEXT_COL].isna().any():
            raise ValueError(f"{name} split contains null text")
        labels = set(frame[LABEL_COL].unique())
        if not labels.issubset({0, 1}):
            raise ValueError(f"{name} split has unexpected labels: {labels}")
    return train_df[[TEXT_COL, LABEL_COL]].copy(), test_df[[TEXT_COL, LABEL_COL]].copy()


def make_splits(
    val_size: float = 0.15,
    random_state: int = 42,
    dataset_id: str = DATASET_ID,
    revision: str = DATASET_REVISION,
) -> PromptSplits:
    """Create stratified train/val from train; keep official test untouched."""
    train_df, test_df = load_raw_frames(dataset_id=dataset_id, revision=revision)
    X = train_df[TEXT_COL]
    y = train_df[LABEL_COL].astype(int)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=val_size,
        random_state=random_state,
        stratify=y,
    )
    return PromptSplits(
        X_train=X_train.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        X_test=test_df[TEXT_COL].reset_index(drop=True),
        y_test=test_df[LABEL_COL].astype(int).reset_index(drop=True),
    )
