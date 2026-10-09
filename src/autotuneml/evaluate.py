from __future__ import annotations

import joblib

from .config import ExperimentConfig
from .data import load_saved_holdout
from .metrics import compute_report
from .models import FittedModel
from .tracking import ExperimentStore, collect_versions, utc_now


def score_frame(model: FittedModel, X, y) -> dict:
    prediction = model.predict(X)
    probability = model.predict_proba(X) if model.classifier else None
    return compute_report(model.task, y, prediction, probability)


def run_evaluate(experiment: str, root: str = "experiments") -> dict:
    """Scores the stored model once on the held-out test split. This is the only
    command besides train that is allowed to read test rows."""
    store = ExperimentStore.open(root, experiment)
    if not store.has_model:
        raise FileNotFoundError(
            f"experiment '{experiment}' has no model.joblib; run `train` first"
        )
    if not (store.holdout_dir / "test.csv").exists():
        raise FileNotFoundError(f"experiment '{experiment}' has no saved holdout split")

    config: ExperimentConfig = store.load_config()
    holdout = load_saved_holdout(
        store.holdout_dir,
        config.data.target or "target",
        config.data.impute,
        config.data.scale,
    )
    model: FittedModel = joblib.load(store.model_path)

    report = {
        "experiment": store.name,
        "evaluated_at": utc_now(),
        "model": config.model.name,
        "metric": config.metric,
        "test": score_frame(model, holdout.X_test, holdout.y_test),
        "rows": {"test": holdout.test_size},
        "versions": collect_versions(),
    }
    store.write_json("eval_report.json", report)
    return report
