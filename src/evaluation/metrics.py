"""
Classification metrics for modeling CV.

Reference: ``docs/modeling_plan.md`` Phase 2.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

__all__ = [
    "classification_metrics",
]


def classification_metrics(y_true, y_proba) -> dict[str, float]:
    """Binary classification: primary metric ``roc_auc`` only."""
    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_proba, dtype=float)

    return {
        "roc_auc": float(roc_auc_score(y_true_arr, y_proba_arr)),
    }
