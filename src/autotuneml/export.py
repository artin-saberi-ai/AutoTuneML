from __future__ import annotations

import json
import shutil
from pathlib import Path

from .tracking import ExperimentStore, collect_versions, utc_now


def run_export(experiment: str, root: str = "experiments", out: str | None = None) -> dict:
    store = ExperimentStore.open(root, experiment)
    if not store.has_model:
        raise FileNotFoundError(
            f"experiment '{experiment}' has no model.joblib; run `train` first"
        )

    config = store.load_config()
    target = Path(out) if out else Path("artifacts") / experiment
    target.mkdir(parents=True, exist_ok=True)
    model_file = target / "model.joblib"
    shutil.copyfile(store.model_path, model_file)

    params: dict = {}
    cv = None
    test = None
    if store.path("train_report.json").exists():
        report = store.read_json("train_report.json")
        params = report.get("params") or {}
        cv = report.get("cv")
        test = report.get("test")
    elif store.has_summary:
        best = store.read_summary().get("best") or {}
        params = best.get("params") or {}

    metadata = {
        "experiment": store.name,
        "exported_at": utc_now(),
        "model": config.model.name,
        "task": config.task,
        "metric": config.metric,
        "params": params,
        "cv": cv,
        "test": test,
        "source": str(store.model_path),
        "artifact": str(model_file),
        "versions": collect_versions(),
    }
    with (target / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, default=str)
        handle.write("\n")
    return metadata
