"""Regression and classification metric tests."""
from __future__ import annotations

import numpy as np

from src.evaluation.metrics import (
    calculate_regression_metrics,
    classification_metrics,
    regression_metrics,
)


def test_regression_metrics_returns_rmse_mae_r2() -> None:
    y_true = np.array([0.0, 5.0, 10.0, 15.0])
    y_pred = np.array([1.0, 4.0, 11.0, 14.0])

    metrics = regression_metrics(y_true, y_pred)

    assert set(metrics) == {"rmse", "mae", "r2"}
    assert metrics["rmse"] == calculate_regression_metrics(y_true, y_pred)["rmse"]
    assert metrics["mae"] == 1.0
    assert metrics["r2"] > 0.9


def test_calculate_regression_metrics_perfect_prediction() -> None:
    y = np.array([-3.0, 0.0, 7.5, 42.0])
    metrics = calculate_regression_metrics(y, y)

    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0


def test_classification_metrics_returns_roc_auc() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.6, 0.9])
    metrics = classification_metrics(y_true, y_proba)

    assert set(metrics) == {"roc_auc"}
    assert 0.0 <= metrics["roc_auc"] <= 1.0
