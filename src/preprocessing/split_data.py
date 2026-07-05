"""
Train/test split for yards-gained holdout evaluation.

Reference: ``docs/modeling_plan.md`` Phase 5.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.schema import SEED

__all__ = [
    "save_split_indices",
    "train_test_split_frame",
]


def train_test_split_frame(
    df: pd.DataFrame,
    target: str,
    *,
    test_size: float = 0.2,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split *df* into train and test feature matrices and target series.

    Uses ``sklearn.model_selection.train_test_split`` on row indices with
    ``random_state=seed`` for deterministic 80/20 (default) partitioning.
    """
    if target not in df.columns:
        raise ValueError(f"target column {target!r} not in dataframe")

    y = df[target]
    X = df.drop(columns=[target])

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    return X_train, X_test, y_train, y_test


def save_split_indices(
    train_idx: np.ndarray | list[int],
    test_idx: np.ndarray | list[int],
    path: Path | str,
) -> Path:
    """Persist train/test index arrays for reproducibility audits."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_idx": np.asarray(train_idx, dtype=int).tolist(),
        "test_idx": np.asarray(test_idx, dtype=int).tolist(),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return out_path


if __name__ == "__main__":
    from src.data.loaders import load_yards_gained_dataset
    from src.data.schema import TARGET_REG

    X, y = load_yards_gained_dataset()
    frame = X.copy()
    frame[TARGET_REG] = y

    X_train, X_test, y_train, y_test = train_test_split_frame(frame, TARGET_REG)
    print("Split smoke OK")
    print(f"  train: {len(y_train):,} rows ({len(y_train) / len(frame):.1%})")
    print(f"  test:  {len(y_test):,} rows ({len(y_test) / len(frame):.1%})")
    print(f"  features: {X_train.shape[1]}")
