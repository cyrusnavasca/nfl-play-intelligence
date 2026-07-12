"""Classification metric tests."""
from __future__ import annotations

import numpy as np

from src.evaluation.metrics import classification_metrics


def test_classification_metrics_returns_roc_auc() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.6, 0.9])
    metrics = classification_metrics(y_true, y_proba)

    assert set(metrics) == {"roc_auc"}
    assert 0.0 <= metrics["roc_auc"] <= 1.0
