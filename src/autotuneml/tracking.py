from __future__ import annotations

import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .config import ExperimentConfig, dump_config, load_config
from .hardware import collect_hardware

CONFIG_FILE = "config.yaml"
META_FILE = "meta.json"
TRIALS_FILE = "trials.jsonl"
SUMMARY_FILE = "summary.json"
HOLDOUT_DIR = "holdout"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def collect_versions() -> dict:
    versions = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "sklearn", "optuna", "joblib"):
        try:
            module = __import__(package)
            versions[package] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[package] = "not installed"
    return versions


class ExperimentStore:
    def __init__(self, root: str | Path, name: str):
        if not name or "/" in name or name in (".", ".."):
            raise ValueError(f"invalid experiment name '{name}'")
        self.root = Path(root)
        self.name = name
        self.dir = self.root / name

    @classmethod
    def create(cls, root, name, config: ExperimentConfig, overwrite: bool = False) -> "ExperimentStore":
        store = cls(root, name)
        if store.dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"experiment '{name}' already exists at {store.dir}; "
                    "pass --overwrite to replace it or choose another name"
                )
            shutil.rmtree(store.dir)
        store.dir.mkdir(parents=True, exist_ok=True)
        store.save_config(config)
        store.write_json(META_FILE, {
            "experiment": name,
            "created_at": utc_now(),
            "seed": config.seed,
            "config_hash": config.config_hash(),
            "config_source": config.source_path,
            "versions": collect_versions(),
            "hardware": collect_hardware(),
        })
        return store

    @classmethod
    def open(cls, root, name) -> "ExperimentStore":
        store = cls(root, name)
        if not store.dir.exists():
            raise FileNotFoundError(f"experiment '{name}' does not exist under {Path(root)}")
        return store

    def path(self, filename: str) -> Path:
        return self.dir / filename

    @property
    def holdout_dir(self) -> Path:
        return self.dir / HOLDOUT_DIR

    @property
    def model_path(self) -> Path:
        return self.dir / "model.joblib"

    def save_config(self, config: ExperimentConfig) -> None:
        dump_config(config, self.path(CONFIG_FILE))

    def load_config(self) -> ExperimentConfig:
        return load_config(self.path(CONFIG_FILE))

    def write_json(self, filename: str, payload: dict) -> Path:
        target = self.path(filename)
        with target.open("w") as handle:
            json.dump(payload, handle, indent=2, default=str)
            handle.write("\n")
        return target

    def read_json(self, filename: str) -> dict:
        target = self.path(filename)
        if not target.exists():
            raise FileNotFoundError(f"{target} does not exist")
        with target.open() as handle:
            return json.load(handle)

    def append_trial(self, record: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.path(TRIALS_FILE).open("a") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def read_trials(self) -> list[dict]:
        target = self.path(TRIALS_FILE)
        if not target.exists():
            return []
        trials = []
        with target.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    trials.append(json.loads(line))
        return trials

    @property
    def has_trials(self) -> bool:
        return self.path(TRIALS_FILE).exists() and bool(self.read_trials())

    def write_summary(self, summary: dict) -> Path:
        return self.write_json(SUMMARY_FILE, summary)

    def read_summary(self) -> dict:
        return self.read_json(SUMMARY_FILE)

    @property
    def has_summary(self) -> bool:
        return self.path(SUMMARY_FILE).exists()

    @property
    def has_model(self) -> bool:
        return self.model_path.exists()


def trial_to_record(trial) -> dict:
    """Converts an optuna trial into the flat record stored in trials.jsonl."""
    metrics = trial.user_attrs.get("metrics") or {}
    params = trial.user_attrs.get("effective_params") or dict(trial.params)
    start = getattr(trial, "datetime_start", None)
    complete = getattr(trial, "datetime_complete", None)
    duration = (complete - start).total_seconds() if start and complete else None
    error = trial.user_attrs.get("error") or getattr(trial, "error", None)
    state = trial.state.name if hasattr(trial.state, "name") else str(trial.state)
    return {
        "trial": trial.number,
        "status": state,
        "params": params,
        "val_score": metrics.get("val_score"),
        "train_time": metrics.get("train_time"),
        "model_size": metrics.get("model_size"),
        "duration": duration,
        "fitted_rounds": trial.user_attrs.get("fitted_rounds"),
        "error": str(error) if error is not None else None,
        "finished_at": complete.isoformat(timespec="seconds") if complete else None,
    }


def list_experiments(root: str | Path) -> list[str]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(
        entry.name for entry in root.iterdir()
        if entry.is_dir() and (entry / SUMMARY_FILE).exists()
    )
