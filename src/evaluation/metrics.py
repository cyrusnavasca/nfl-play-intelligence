"""
Classification and regression metrics for modeling CV.

Reference: ``docs/modeling_plan.md`` Phase 2.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

__all__ = [
    "calculate_regression_metrics",
    "classification_metrics",
    "regression_metrics",
]


def classification_metrics(y_true, y_proba) -> dict[str, float]:
    """Binary classification: primary metric ``roc_auc`` only."""
    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_proba, dtype=float)

    return {
        "roc_auc": float(roc_auc_score(y_true_arr, y_proba_arr)),
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """
    Regression metrics in original target space.

    Expects *y_true* and *y_pred* in yards (after inverse target transform).
    Primary selection metric is ``rmse``; ``mae`` and ``r2`` are reported too.
    """
    return calculate_regression_metrics(y_true, y_pred)


def calculate_regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return ``rmse``, ``mae``, and ``r2`` for yards-gained predictions."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
    }
