"""
Out-of-fold prediction accumulation and parquet-ready frames.

Reference: ``docs/modeling_plan.md`` Phase 2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["accumulate_oof", "build_oof_dataframe"]


def accumulate_oof(
    oof_array: np.ndarray,
    val_idx: np.ndarray,
    preds: np.ndarray,
) -> None:
    """Write validation-fold predictions into a pre-allocated OOF array."""
    oof_array[val_idx] = preds


def build_oof_dataframe(
    y_true: pd.Series | np.ndarray,
    oof_by_model: dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Build a parquet-ready OOF frame.

    Columns: ``y_true``, ``oof_proba_<model_key>`` (or ``oof_pred_<model_key>``
    when values are not probabilities — caller naming convention uses model key).
    """
    frame = pd.DataFrame({"y_true": np.asarray(y_true)})
    for model_key, oof in oof_by_model.items():
        col_name = f"oof_proba_{model_key}"
        frame[col_name] = np.asarray(oof)
    return frame
