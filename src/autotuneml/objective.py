from __future__ import annotations

import time

import numpy as np
import pandas as pd
from optuna.exceptions import TrialPruned
from sklearn.model_selection import KFold, StratifiedKFold

from .config import ExperimentConfig
from .data import ColumnSpec, build_preprocessor, early_stopping_split
from .hardware import resolve_device
from .metrics import compute, needs_proba
from .models import (
    build_model,
    estimate_size,
    fit_estimator,
    fitted_rounds,
    suggest_params,
    supports_eval_set,
)

METRIC_KEYS = ("val_score", "train_time", "model_size")


def _rows(frame, index):
    return frame.iloc[index] if hasattr(frame, "iloc") else frame[index]


def make_splits(X, y, config: ExperimentConfig):
    folds = config.search.cv_folds
    if config.task == "classification":
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.seed)
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=config.seed)
    return list(splitter.split(X, y))


def fit_with_preprocessing(columns: ColumnSpec, config: ExperimentConfig, params: dict,
                           X_train, y_train, seed: int, early_stopping: bool | None = None):
    """Fits a preprocessor and an estimator on the rows it is handed. Any inner
    validation split used for early stopping is carved out of these rows only."""
    if early_stopping is None:
        early_stopping = config.search.early_stopping
    y_array = np.asarray(y_train)

    preprocessor = build_preprocessor(columns)
    X_fit = preprocessor.fit_transform(X_train)

    device = None
    if config.model.name in ("xgboost", "lightgbm"):
        device = resolve_device(config.model.device)
    estimator = build_model(config.model.name, params, seed, config.task, device=device)
    fit_X, fit_y = X_fit, y_array
    X_es = y_es = None
    if early_stopping and supports_eval_set(estimator):
        fit_pos, es_pos = early_stopping_split(y_array, config.task, seed)
        if len(es_pos):
            fit_X, fit_y = X_fit[fit_pos], y_array[fit_pos]
            X_es, y_es = X_fit[es_pos], y_array[es_pos]

    fit_estimator(estimator, fit_X, fit_y, X_es, y_es, early_stopping=early_stopping)
    return preprocessor, estimator


def cross_validate(X, y, columns: ColumnSpec, config: ExperimentConfig, params: dict,
                   on_fold=None, enforce_budget: bool = True) -> dict:
    """Scores one configuration with fold-local preprocessing. `on_fold` sees running
    statistics after every fitted fold and is where reporting and pruning hook in."""
    started = time.perf_counter()
    budget = config.search.trial_time_budget
    metric = config.metric
    scores: list[float] = []
    fit_seconds = 0.0
    bundle = None
    rounds = None

    for fold, (train_idx, valid_idx) in enumerate(make_splits(X, y, config)):
        if enforce_budget and budget is not None and time.perf_counter() - started > budget:
            raise TrialPruned(f"trial passed its {budget}s time budget")

        X_train, X_valid = _rows(X, train_idx), _rows(X, valid_idx)
        y_train, y_valid = _rows(y, train_idx), _rows(y, valid_idx)

        fold_started = time.perf_counter()
        preprocessor, estimator = fit_with_preprocessing(
            columns, config, params, X_train, y_train, seed=config.seed + fold
        )
        fit_seconds += time.perf_counter() - fold_started

        X_valid_t = preprocessor.transform(X_valid)
        prediction = estimator.predict(X_valid_t)
        probability = None
        if needs_proba(metric) and hasattr(estimator, "predict_proba"):
            probability = estimator.predict_proba(X_valid_t)

        score = compute(metric, y_valid, prediction, probability)
        scores.append(float(score))
        bundle = (preprocessor, estimator)
        rounds = fitted_rounds(estimator)

        if on_fold is not None:
            on_fold(fold, {
                "fold": fold,
                "score": float(score),
                "mean_score": float(np.mean(scores)),
                "fit_seconds": fit_seconds,
            })

    if not scores:
        raise RuntimeError("cross-validation produced no folds")

    return {
        "val_score": float(np.mean(scores)),
        "train_time": fit_seconds,
        "model_size": estimate_size(bundle),
        "folds": len(scores),
        "fitted_rounds": rounds,
    }


class Objective:
    def __init__(self, X: pd.DataFrame, y: pd.Series, columns: ColumnSpec,
                 config: ExperimentConfig):
        self.X = X
        self.y = y
        self.columns = columns
        self.config = config
        self.objectives = config.search.objectives
        self.single = len(self.objectives) == 1
        self.space = config.search_space
        self.fixed = dict(config.model.params)

    def __call__(self, trial):
        params = {**self.fixed, **suggest_params(trial, self.space)}
        partial = {"val_score": None, "train_time": 0.0, "model_size": None}

        def on_fold(fold, stats):
            partial["val_score"] = stats["mean_score"]
            partial["train_time"] = stats["fit_seconds"]
            if self.single:
                trial.report(stats["mean_score"], fold)
                if trial.should_prune():
                    raise TrialPruned(
                        f"pruned after fold {fold} with running score {stats['mean_score']:.4f}"
                    )

        try:
            result = cross_validate(self.X, self.y, self.columns, self.config, params, on_fold=on_fold)
        except Exception as exc:
            trial.set_user_attr("metrics", dict(partial))
            trial.set_user_attr("effective_params", params)
            if not isinstance(exc, TrialPruned):
                trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
            raise

        metrics = {key: result[key] for key in METRIC_KEYS}
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("effective_params", params)
        trial.set_user_attr("fitted_rounds", result["fitted_rounds"])

        if self.single:
            return metrics["val_score"]
        return tuple(metrics[objective] for objective in self.objectives)
