from __future__ import annotations

import io
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.base import is_classifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression

MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
    "xgboost",
    "lightgbm",
]

DEFAULT_PARAMS = {
    "logistic_regression": {"C": 1.0, "max_iter": 1000},
    "random_forest": {"n_estimators": 300, "min_samples_leaf": 1},
    "hist_gradient_boosting": {"max_iter": 200, "learning_rate": 0.1},
    "xgboost": {"n_estimators": 300, "learning_rate": 0.1},
    "lightgbm": {"n_estimators": 300, "learning_rate": 0.1},
}

DEFAULT_SPACES = {
    "logistic_regression": {
        "C": [0.01, 10.0],
        "class_weight": ["balanced"],
    },
    "random_forest": {
        "n_estimators": [100, 600],
        "max_depth": [3, 30],
        "min_samples_leaf": [1, 20],
        "max_features": ["sqrt", "log2"],
    },
    "hist_gradient_boosting": {
        "max_iter": [50, 400],
        "learning_rate": [0.01, 0.3],
        "max_leaf_nodes": [15, 127],
        "min_samples_leaf": [5, 60],
        "l2_regularization": [0.0, 1.0],
    },
    "xgboost": {
        "n_estimators": [100, 600],
        "learning_rate": [0.01, 0.3],
        "max_depth": [3, 12],
        "subsample": [0.6, 1.0],
        "colsample_bytree": [0.6, 1.0],
        "min_child_weight": [1, 20],
        "reg_lambda": [0.1, 10.0],
    },
    "lightgbm": {
        "n_estimators": [100, 600],
        "learning_rate": [0.01, 0.3],
        "num_leaves": [15, 127],
        "min_child_samples": [5, 60],
        "subsample": [0.6, 1.0],
        "colsample_bytree": [0.6, 1.0],
    },
}


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def model_available(name: str) -> bool:
    if name == "xgboost":
        try:
            import xgboost  # noqa: F401
        except Exception:
            return False
    elif name == "lightgbm":
        try:
            import lightgbm  # noqa: F401
        except Exception:
            return False
    return True


def validate_space(space: dict, where: str = "search.params") -> None:
    """A two-element numeric list is a closed range, every other list is a set of
    discrete choices."""
    for key, spec in space.items():
        if not isinstance(spec, list):
            if isinstance(spec, (dict, tuple)):
                raise ValueError(f"{where}.{key}: expected a list or a fixed value")
            continue
        if not spec:
            raise ValueError(f"{where}.{key}: empty search space")
        if any(isinstance(v, (dict, tuple)) for v in spec):
            raise ValueError(f"{where}.{key}: nested containers are not allowed")
        numeric = [is_number(v) for v in spec]
        if len(spec) == 2 and all(numeric):
            low, high = spec
            if low > high:
                raise ValueError(f"{where}.{key}: lower bound {low} exceeds upper bound {high}")
        elif any(numeric) and not all(numeric):
            raise ValueError(
                f"{where}.{key}: a two-element numeric list defines a range, "
                "any other list must hold choices of a single kind"
            )


def merge_space(model_name: str, overrides: dict | None) -> dict:
    space = dict(DEFAULT_SPACES.get(model_name, {}))
    space.update(overrides or {})
    validate_space(space)
    return space


def suggest_params(trial, space: dict) -> dict:
    params: dict = {}
    for key, spec in space.items():
        if isinstance(spec, list):
            if len(spec) == 2 and all(is_number(v) for v in spec):
                low, high = spec
                if isinstance(low, int) and isinstance(high, int):
                    params[key] = trial.suggest_int(key, low, high)
                else:
                    params[key] = trial.suggest_float(key, float(low), float(high))
            else:
                params[key] = trial.suggest_categorical(key, list(spec))
        else:
            params[key] = spec
    return params


def _xgb(task: str):
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except Exception as exc:
        raise RuntimeError(
            "xgboost was requested but cannot be imported; install it with `pip install xgboost`"
        ) from exc
    return XGBClassifier() if task == "classification" else XGBRegressor()


def _lightgbm(task: str):
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except Exception as exc:
        raise RuntimeError(
            "lightgbm was requested but cannot be imported; install it with `pip install lightgbm`"
        ) from exc
    return LGBMClassifier() if task == "classification" else LGBMRegressor()


def _sklearn_model(name: str, task: str):
    table = {
        ("classification", "logistic_regression"): LogisticRegression,
        ("classification", "random_forest"): RandomForestClassifier,
        ("classification", "hist_gradient_boosting"): HistGradientBoostingClassifier,
        ("regression", "random_forest"): RandomForestRegressor,
        ("regression", "hist_gradient_boosting"): HistGradientBoostingRegressor,
    }
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model '{name}' (known: {', '.join(MODEL_NAMES)})")
    cls = table.get((task, name))
    if cls is None:
        raise ValueError(f"model '{name}' is not available for {task} tasks")
    return cls()


def device_params(name: str, device: str) -> dict:
    if name == "xgboost":
        return {"device": device}
    if name == "lightgbm":
        return {"device": "gpu" if device == "cuda" else "cpu"}
    return {}


def build_model(name: str, params: dict, seed: int, task: str, device: str | None = None):
    if name == "xgboost":
        est = _xgb(task)
    elif name == "lightgbm":
        est = _lightgbm(task)
    else:
        est = _sklearn_model(name, task)

    known = set(est.get_params(deep=False))
    unknown = sorted(set(params) - known)
    if unknown:
        raise ValueError(f"unknown parameters for {name}: {', '.join(unknown)}")
    est.set_params(**params)
    if device is not None:
        if name in ("xgboost", "lightgbm"):
            est.set_params(**device_params(name, device))
        elif device == "cuda":
            raise RuntimeError(
                f"model '{name}' runs on the CPU only; device 'cuda' does not apply to it"
            )
    if "random_state" in est.get_params(deep=False):
        est.set_params(random_state=seed)
    return est


def supports_eval_set(est) -> bool:
    return "early_stopping_rounds" in est.get_params(deep=False)


def fit_estimator(est, X, y, X_es=None, y_es=None, early_stopping: bool = True):
    params = est.get_params(deep=False)
    if early_stopping and supports_eval_set(est):
        if X_es is None or len(X_es) == 0:
            est.set_params(early_stopping_rounds=None)
            est.fit(X, y)
        else:
            if not params.get("early_stopping_rounds"):
                est.set_params(early_stopping_rounds=25)
            est.fit(X, y, eval_set=[(X_es, y_es)], verbose=False)
    elif early_stopping and "early_stopping" in params:
        est.set_params(early_stopping=True, n_iter_no_change=15)
        est.fit(X, y)
    else:
        est.fit(X, y)
    return est


def fitted_rounds(est) -> int | None:
    for attr in ("best_iteration_", "n_iter_", "n_estimators_", "best_n_iterations"):
        value = getattr(est, attr, None)
        if isinstance(value, (int, np.integer)):
            return int(value)
    return None


def estimate_size(obj) -> int:
    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    return buffer.tell()


@dataclass
class FittedModel:
    preprocessor: object
    estimator: object
    task: str
    model_name: str
    params: dict
    metric: str
    cv_score: float | None = None

    @property
    def classifier(self) -> bool:
        return is_classifier(self.estimator)

    def predict(self, X) -> np.ndarray:
        return self.estimator.predict(self.preprocessor.transform(X))

    def predict_proba(self, X) -> np.ndarray | None:
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(self.preprocessor.transform(X))
        return None

    def predict_all(self, X) -> tuple[np.ndarray, np.ndarray | None]:
        return self.predict(X), self.predict_proba(X)

    def size_bytes(self) -> int:
        return estimate_size(self)
