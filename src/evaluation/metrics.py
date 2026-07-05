"""
Classification and regression metrics for modeling CV.

Reference: ``docs/modeling_plan.md`` Phase 2.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    median_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

__all__ = ["classification_metrics", "regression_metrics"]


def classification_metrics(y_true, y_proba) -> dict[str, float]:
    """
    Binary classification metrics from true labels and predicted P(positive).

    Primary: ``roc_auc``, ``log_loss``.
    Secondary: ``accuracy``, ``balanced_accuracy``, ``brier_score``.
    """
    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba_arr >= 0.5).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true_arr, y_proba_arr)),
        "log_loss": float(log_loss(y_true_arr, y_proba_arr)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred)),
        "brier_score": float(brier_score_loss(y_true_arr, y_proba_arr)),
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """
    Regression metrics from true and predicted values.

    Primary: ``rmse``, ``mae``.
    Secondary: ``r2``, ``median_absolute_error``.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
        "median_absolute_error": float(
            median_absolute_error(y_true_arr, y_pred_arr)
        ),
    }
