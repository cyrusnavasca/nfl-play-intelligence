"""Median imputation helper tests."""
from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer

from src.preprocessing.impute import apply_median_imputer


def test_apply_median_imputer_leaves_non_target_columns() -> None:
    X = pd.DataFrame({"a": [1.0, None, 3.0], "b": [10.0, 20.0, 30.0]})
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X[["a"]])

    out = apply_median_imputer(X, imputer, ["a"])

    assert out["b"].tolist() == [10.0, 20.0, 30.0]
    assert out["a"].notna().all()
