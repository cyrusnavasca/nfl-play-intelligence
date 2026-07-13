"""
Optuna hyperparameter tuning for play-type classifiers.

Wires model builders and shared cross-validation into a Bayesian (TPE) search.
Lives under ``pipelines/`` because it composes both ``models/`` and
``evaluation/`` — the same wiring layer as ``pipelines/train.py``.

A search is driven by a profile's ``tune`` block (study settings) and each
model's ``search`` block (parameter distributions); see
``configs/models/xgboost_tuned.yaml``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split

from src.data.schema import N_FOLDS, SEED
from src.evaluation.cross_validation import cross_validate_classifier, stratified_folds
from src.models import CLASSIFIER_BUILDERS

__all__ = ["TuningResult", "tune_model"]

# Study defaults; overridable via the profile's ``tune`` block.
DEFAULT_N_TRIALS = 30
DEFAULT_CV_FOLDS = 3


@dataclass(frozen=True)
class TuningResult:
    """Outcome of an Optuna study for a single model."""

    model_key: str
    best_params: dict[str, Any]
    best_value: float
    n_trials: int
    trials: pd.DataFrame


def _suggest_params(
    trial: optuna.Trial,
    search_space: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Sample one hyperparameter set from *search_space* for this trial."""
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        dist = spec["type"]
        if dist == "int":
            params[name] = trial.suggest_int(
                name, spec["low"], spec["high"], step=spec.get("step", 1)
            )
        elif dist == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        else:  # categorical
            params[name] = trial.suggest_categorical(name, spec["choices"])
    return params


def _subsample(
    X: pd.DataFrame,
    y: pd.Series,
    n_rows: int | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Stratified subsample for faster search; returns the full data if not set."""
    if not n_rows or n_rows >= len(X):
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=n_rows, random_state=seed, stratify=y
    )
    return X_sub, y_sub


def tune_model(
    model_key: str,
    X: pd.DataFrame,
    y: pd.Series,
    fixed_params: dict[str, Any],
    search_space: dict[str, dict[str, Any]],
    *,
    n_trials: int = DEFAULT_N_TRIALS,
    cv_folds: int = DEFAULT_CV_FOLDS,
    seed: int = SEED,
    subsample_rows: int | None = None,
    metric: str = "roc_auc",
) -> TuningResult:
    """
    Run a TPE (Bayesian) Optuna study maximizing mean CV *metric* for *model_key*.

    Each trial samples from *search_space*, merges with *fixed_params*, and scores
    the estimator with the shared stratified CV loop. Returns the best params
    (fixed + tuned) plus a per-trial frame.
    """
    if model_key not in CLASSIFIER_BUILDERS:
        raise KeyError(f"unknown model key: {model_key!r}")

    X_search, y_search = _subsample(X, y, subsample_rows, seed)
    folds = stratified_folds(y_search, cv_folds, seed)
    impute_cols = X_search.select_dtypes(include=np.number).columns.tolist()

    def objective(trial: optuna.Trial) -> float:
        params = {**fixed_params, **_suggest_params(trial, search_space)}
        model = CLASSIFIER_BUILDERS[model_key](hyperparameters=params)
        result = cross_validate_classifier(
            model, X_search, y_search, folds, impute_numeric_cols=impute_cols
        )
        return float(np.mean([m[metric] for m in result.fold_metrics]))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = {**fixed_params, **study.best_params}
    trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))

    return TuningResult(
        model_key=model_key,
        best_params=best_params,
        best_value=float(study.best_value),
        n_trials=len(study.trials),
        trials=trials,
    )
