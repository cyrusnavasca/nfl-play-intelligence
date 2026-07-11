"""
Target transforms for yards-gained regression.

Models train on transformed targets; predictions are inverse-transformed back
to yards before metric computation. Fitting uses training data only.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler

__all__ = [
    "TARGET_TRANSFORM_KEYS",
    "TargetTransform",
    "build_target_transform",
    "resolve_target_transform_name",
]

TARGET_TRANSFORM_KEYS: tuple[str, ...] = (
    "none",
    "log",
    "scaled_log",
    "signed_log",
    "yeo_johnson",
)


class TargetTransform(ABC):
    """Fit on train targets, transform for fitting, inverse for predictions."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Registry key for this transform."""

    @abstractmethod
    def fit(self, y: Any) -> TargetTransform:
        """Estimate transform parameters from training targets."""

    @abstractmethod
    def transform(self, y: Any) -> np.ndarray:
        """Map original yards to model target space."""

    @abstractmethod
    def inverse_transform(self, y_transformed: Any) -> np.ndarray:
        """Map model predictions back to yards."""

    def fit_transform(self, y: Any) -> np.ndarray:
        return self.fit(y).transform(y)


class IdentityTransform(TargetTransform):
    """No transform (default)."""

    @property
    def name(self) -> str:
        return "none"

    def fit(self, y: Any) -> IdentityTransform:
        return self

    def transform(self, y: Any) -> np.ndarray:
        return np.asarray(y, dtype=float)

    def inverse_transform(self, y_transformed: Any) -> np.ndarray:
        return np.asarray(y_transformed, dtype=float)


class LogTransform(TargetTransform):
    """
    Offset log: ``log(y + shift)`` with ``shift = 1 - min(y_train)``.

    Handles negative yards (sacks, TFL) by shifting the training minimum to 1
    before taking the log.
    """

    def __init__(self) -> None:
        self._shift: float | None = None

    @property
    def name(self) -> str:
        return "log"

    def fit(self, y: Any) -> LogTransform:
        y_arr = np.asarray(y, dtype=float)
        self._shift = float(1.0 - np.min(y_arr))
        return self

    def transform(self, y: Any) -> np.ndarray:
        if self._shift is None:
            raise RuntimeError("LogTransform.fit must be called before transform")
        y_arr = np.asarray(y, dtype=float)
        return np.log(y_arr + self._shift)

    def inverse_transform(self, y_transformed: Any) -> np.ndarray:
        if self._shift is None:
            raise RuntimeError(
                "LogTransform.fit must be called before inverse_transform"
            )
        y_tf = np.asarray(y_transformed, dtype=float)
        return np.exp(y_tf) - self._shift


class ScaledLogTransform(TargetTransform):
    """``log(y + shift)`` followed by zero-mean unit-variance scaling."""

    def __init__(self) -> None:
        self._log = LogTransform()
        self._scaler = StandardScaler()

    @property
    def name(self) -> str:
        return "scaled_log"

    def fit(self, y: Any) -> ScaledLogTransform:
        y_log = self._log.fit(y).transform(y)
        self._scaler.fit(y_log.reshape(-1, 1))
        return self

    def transform(self, y: Any) -> np.ndarray:
        y_log = self._log.transform(y)
        return self._scaler.transform(y_log.reshape(-1, 1)).ravel()

    def inverse_transform(self, y_transformed: Any) -> np.ndarray:
        y_tf = np.asarray(y_transformed, dtype=float).reshape(-1, 1)
        y_log = self._scaler.inverse_transform(y_tf).ravel()
        return self._log.inverse_transform(y_log)


class SignedLogTransform(TargetTransform):
    """
    Signed log: ``sign(y) * log(1 + |y|)``.

    Handles negative yards without a train-fit shift; inverse is
    ``sign(z) * (exp(|z|) - 1)``.
    """

    @property
    def name(self) -> str:
        return "signed_log"

    def fit(self, y: Any) -> SignedLogTransform:
        return self

    def transform(self, y: Any) -> np.ndarray:
        y_arr = np.asarray(y, dtype=float)
        return np.sign(y_arr) * np.log1p(np.abs(y_arr))

    def inverse_transform(self, y_transformed: Any) -> np.ndarray:
        z = np.asarray(y_transformed, dtype=float)
        return np.sign(z) * np.expm1(np.abs(z))


class YeoJohnsonTransform(TargetTransform):
    """Sklearn Yeo-Johnson power transform with standardization."""

    def __init__(self) -> None:
        self._transformer = PowerTransformer(
            method="yeo-johnson",
            standardize=True,
        )

    @property
    def name(self) -> str:
        return "yeo_johnson"

    def fit(self, y: Any) -> YeoJohnsonTransform:
        y_arr = np.asarray(y, dtype=float).reshape(-1, 1)
        self._transformer.fit(y_arr)
        return self

    def transform(self, y: Any) -> np.ndarray:
        y_arr = np.asarray(y, dtype=float).reshape(-1, 1)
        return self._transformer.transform(y_arr).ravel()

    def inverse_transform(self, y_transformed: Any) -> np.ndarray:
        y_tf = np.asarray(y_transformed, dtype=float).reshape(-1, 1)
        return self._transformer.inverse_transform(y_tf).ravel()


_TRANSFORM_BUILDERS: dict[str, type[TargetTransform]] = {
    "none": IdentityTransform,
    "log": LogTransform,
    "scaled_log": ScaledLogTransform,
    "signed_log": SignedLogTransform,
    "yeo_johnson": YeoJohnsonTransform,
}


def build_target_transform(name: str | None = None) -> TargetTransform:
    """Instantiate a target transform by registry key."""
    key = name or "none"
    if key not in _TRANSFORM_BUILDERS:
        raise ValueError(
            f"unknown target transform {key!r}; "
            f"expected one of {TARGET_TRANSFORM_KEYS}"
        )
    return _TRANSFORM_BUILDERS[key]()


def resolve_target_transform_name(task: str) -> str:
    """Return the active profile's target transform for *task*, or ``none``."""
    from src.utils.experiment_profile import get_active_profile_or_none

    profile = get_active_profile_or_none()
    if profile is None or not profile.has_task(task):  # type: ignore[arg-type]
        return "none"
    return profile.task_target_transform(task)  # type: ignore[arg-type]
