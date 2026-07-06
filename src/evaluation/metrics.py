"""
Classification and regression metrics for modeling CV.

Reference: ``docs/modeling_plan.md`` Phase 2.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score

__all__ = ["classification_metrics", "regression_metrics"]


def classification_metrics(y_true, y_proba) -> dict[str, float]:
    """Binary classification: primary metric ``roc_auc`` only."""
    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_proba, dtype=float)

    return {
        "roc_auc": float(roc_auc_score(y_true_arr, y_proba_arr)),
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Regression: primary metric ``rmse`` only."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
    }
