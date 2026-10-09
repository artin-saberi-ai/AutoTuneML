from __future__ import annotations

from sklearn import metrics as skm

MAXIMIZE = "maximize"
MINIMIZE = "minimize"

METRIC_DIRECTIONS = {
    "accuracy": MAXIMIZE,
    "balanced_accuracy": MAXIMIZE,
    "f1": MAXIMIZE,
    "roc_auc": MAXIMIZE,
    "log_loss": MINIMIZE,
    "rmse": MINIMIZE,
    "mae": MINIMIZE,
    "r2": MAXIMIZE,
}

CLASSIFICATION_METRICS = ["accuracy", "balanced_accuracy", "f1", "roc_auc", "log_loss"]
REGRESSION_METRICS = ["rmse", "mae", "r2"]

METRIC_NAMES = list(METRIC_DIRECTIONS)

DEFAULT_METRIC = {"classification": "accuracy", "regression": "rmse"}

REPORT_METRICS = {
    "classification": ["accuracy", "balanced_accuracy", "f1", "roc_auc", "log_loss"],
    "regression": ["rmse", "mae", "r2"],
}


def metric_names_for(task: str) -> list[str]:
    if task == "classification":
        return list(CLASSIFICATION_METRICS)
    if task == "regression":
        return list(REGRESSION_METRICS)
    raise ValueError(f"unknown task '{task}'")


def metric_direction(name: str) -> str:
    try:
        return METRIC_DIRECTIONS[name]
    except KeyError:
        raise ValueError(f"unknown metric '{name}'") from None


def needs_proba(name: str) -> bool:
    return name in ("roc_auc", "log_loss")


def compute(name: str, y_true, y_pred, y_proba=None) -> float:
    if name == "accuracy":
        return float(skm.accuracy_score(y_true, y_pred))
    if name == "balanced_accuracy":
        return float(skm.balanced_accuracy_score(y_true, y_pred))
    if name == "f1":
        average = "binary" if len(set(y_true)) <= 2 else "macro"
        return float(skm.f1_score(y_true, y_pred, average=average, zero_division=0))
    if name == "roc_auc":
        if y_proba is None:
            raise ValueError("roc_auc requires predicted probabilities")
        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            scores = y_proba[:, 1]
        else:
            scores = y_proba
        return float(skm.roc_auc_score(y_true, scores, multi_class="ovr", average="macro"))
    if name == "log_loss":
        if y_proba is None:
            raise ValueError("log_loss requires predicted probabilities")
        return float(skm.log_loss(y_true, y_proba, labels=sorted(set(y_true))))
    if name == "rmse":
        return float(skm.root_mean_squared_error(y_true, y_pred))
    if name == "mae":
        return float(skm.mean_absolute_error(y_true, y_pred))
    if name == "r2":
        return float(skm.r2_score(y_true, y_pred))
    raise ValueError(f"unknown metric '{name}'")


def compute_report(task: str, y_true, y_pred, y_proba=None) -> dict[str, float]:
    report = {}
    for name in REPORT_METRICS[task]:
        if name == "roc_auc" and y_proba is None:
            continue
        if name == "roc_auc" and task == "classification" and len(set(y_true)) < 2:
            continue
        try:
            report[name] = compute(name, y_true, y_pred, y_proba)
        except ValueError:
            continue
    return report
