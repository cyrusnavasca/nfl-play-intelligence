"""Feature imputation helpers shared by training and inference."""
from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer

__all__ = ["apply_median_imputer"]


def apply_median_imputer(
    X: pd.DataFrame,
    imputer: SimpleImputer,
    columns: list[str],
) -> pd.DataFrame:
    """Transform *columns* in *X* with a fitted median imputer."""
    if not columns:
        return X

    X_out = X.copy()
    X_out[columns] = imputer.transform(X_out[columns])
    return X_out
