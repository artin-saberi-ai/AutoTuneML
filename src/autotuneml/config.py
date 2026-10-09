from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .hardware import DEVICE_CHOICES
from .metrics import DEFAULT_METRIC, METRIC_NAMES, metric_names_for
from .models import MODEL_NAMES, build_model, merge_space, model_available, validate_space
from .selection import OBJECTIVE_DIRECTIONS, objective_directions

CONSTRAINT_KEYS = ("val_score", "train_time", "model_size")


class ConfigError(ValueError):
    pass


@dataclass
class DataConfig:
    source: str
    target: str | None = None
    test_size: float = 0.2
    impute: str = "median"
    scale: bool = True


@dataclass
class ModelConfig:
    name: str
    params: dict = field(default_factory=dict)
    device: str = "auto"


@dataclass
class SearchConfig:
    n_trials: int = 30
    timeout_seconds: float | None = None
    trial_time_budget: float | None = None
    cv_folds: int = 5
    objectives: list[str] = field(default_factory=lambda: ["val_score"])
    sampler: str = "auto"
    pruner: str = "median"
    early_stopping: bool = True
    params: dict = field(default_factory=dict)


@dataclass
class SelectionConfig:
    weights: dict = field(default_factory=dict)
    require_constraints: bool = True


@dataclass
class ExperimentConfig:
    name: str
    seed: int
    task: str
    metric: str
    data: DataConfig
    model: ModelConfig
    search: SearchConfig
    constraints: dict = field(default_factory=dict)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    source_path: str | None = None

    @property
    def directions(self) -> dict[str, str]:
        return objective_directions(self.search.objectives, self.metric)

    @property
    def search_space(self) -> dict:
        return merge_space(self.model.name, self.search.params)

    def to_dict(self) -> dict:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _build_data(raw: dict) -> DataConfig:
    if not isinstance(raw, dict):
        raise ConfigError("data: the data section is required")
    source = raw.get("source")
    if not source or not isinstance(source, str):
        raise ConfigError("data.source: a dataset path or sklearn:<name> reference is required")
    test_size = raw.get("test_size", 0.2)
    if not isinstance(test_size, (int, float)) or isinstance(test_size, bool) or not 0 < test_size < 1:
        raise ConfigError("data.test_size: must be a fraction between 0 and 1")
    impute = raw.get("impute", "median")
    if impute not in ("median", "mean", "most_frequent", "constant"):
        raise ConfigError(f"data.impute: unknown strategy '{impute}'")
    return DataConfig(
        source=source,
        target=raw.get("target"),
        test_size=float(test_size),
        impute=impute,
        scale=bool(raw.get("scale", True)),
    )


def _build_model(raw: dict) -> ModelConfig:
    if not isinstance(raw, dict):
        raise ConfigError("model: the model section is required")
    name = raw.get("name")
    if not name or not isinstance(name, str):
        raise ConfigError("model.name: a model name is required")
    if name not in MODEL_NAMES:
        raise ConfigError(f"model.name: unknown model '{name}' (known: {', '.join(MODEL_NAMES)})")
    if not model_available(name):
        raise ConfigError(f"model.name: '{name}' is not installed in this environment")
    params = raw.get("params") or {}
    if not isinstance(params, dict):
        raise ConfigError("model.params: expected a mapping")
    device = raw.get("device", "auto")
    if device not in DEVICE_CHOICES:
        raise ConfigError(f"model.device: unknown device '{device}' (use one of {', '.join(DEVICE_CHOICES)})")
    return ModelConfig(name=name, params=params, device=device)


def _build_search(raw: dict, model_name: str, task: str, metric: str) -> SearchConfig:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError("search: expected a mapping")

    n_trials = raw.get("n_trials", 30)
    if not isinstance(n_trials, int) or isinstance(n_trials, bool) or n_trials < 1:
        raise ConfigError("search.n_trials: must be a positive integer")

    cv_folds = raw.get("cv_folds", 5)
    if not isinstance(cv_folds, int) or isinstance(cv_folds, bool) or cv_folds < 2:
        raise ConfigError("search.cv_folds: must be an integer of at least 2")

    objectives = raw.get("objectives") or ["val_score"]
    if not isinstance(objectives, list) or not objectives:
        raise ConfigError("search.objectives: expected a non-empty list")
    for objective in objectives:
        if objective not in OBJECTIVE_DIRECTIONS:
            raise ConfigError(
                f"search.objectives: unknown objective '{objective}' "
                f"(known: {', '.join(OBJECTIVE_DIRECTIONS)})"
            )
    if len(set(objectives)) != len(objectives):
        raise ConfigError("search.objectives: duplicate objectives are not allowed")

    for field_name in ("timeout_seconds", "trial_time_budget"):
        value = raw.get(field_name)
        if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0):
            raise ConfigError(f"search.{field_name}: must be a positive number")

    sampler = raw.get("sampler", "auto")
    if sampler not in ("auto", "tpe", "random", "nsga2"):
        raise ConfigError(f"search.sampler: unknown sampler '{sampler}'")
    pruner = raw.get("pruner", "median")
    if pruner not in ("median", "none"):
        raise ConfigError(f"search.pruner: unknown pruner '{pruner}'")

    space = raw.get("params") or {}
    if not isinstance(space, dict):
        raise ConfigError("search.params: expected a mapping")
    try:
        validate_space(space)
    except ValueError as exc:
        raise ConfigError(str(exc)) from None

    return SearchConfig(
        n_trials=n_trials,
        timeout_seconds=raw.get("timeout_seconds"),
        trial_time_budget=raw.get("trial_time_budget"),
        cv_folds=cv_folds,
        objectives=objectives,
        sampler=sampler,
        pruner=pruner,
        early_stopping=bool(raw.get("early_stopping", True)),
        params=space,
    )


def _build_constraints(raw) -> dict:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError("constraints: expected a mapping of metric -> {min|max: value}")
    constraints = {}
    for metric, spec in raw.items():
        if metric not in CONSTRAINT_KEYS:
            raise ConfigError(f"constraints: unknown metric '{metric}' (known: {', '.join(CONSTRAINT_KEYS)})")
        if not isinstance(spec, dict) or not spec:
            raise ConfigError(f"constraints.{metric}: expected a mapping with 'min' and/or 'max'")
        entry = {}
        for bound, value in spec.items():
            if bound not in ("min", "max"):
                raise ConfigError(f"constraints.{metric}: unknown bound '{bound}' (use min or max)")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ConfigError(f"constraints.{metric}.{bound}: expected a number")
            entry[bound] = float(value)
        if "min" in entry and "max" in entry and entry["min"] > entry["max"]:
            raise ConfigError(f"constraints.{metric}: min exceeds max")
        constraints[metric] = entry
    return constraints


def _build_selection(raw) -> SelectionConfig:
    if raw is None:
        return SelectionConfig()
    if not isinstance(raw, dict):
        raise ConfigError("selection: expected a mapping")
    weights = raw.get("weights") or {}
    if not isinstance(weights, dict):
        raise ConfigError("selection.weights: expected a mapping of objective -> number")
    for key, value in weights.items():
        if key not in OBJECTIVE_DIRECTIONS:
            raise ConfigError(f"selection.weights: unknown objective '{key}'")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ConfigError(f"selection.weights.{key}: expected a non-negative number")
    return SelectionConfig(weights={k: float(v) for k, v in weights.items()},
                           require_constraints=bool(raw.get("require_constraints", True)))


def from_dict(raw: dict, source_path: str | None = None) -> ExperimentConfig:
    if not isinstance(raw, dict):
        raise ConfigError("config: expected a mapping at the top level")

    name = raw.get("name")
    if not name or not isinstance(name, str):
        raise ConfigError("name: an experiment name is required")

    seed = raw.get("seed", 17)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ConfigError("seed: must be a non-negative integer")

    task = raw.get("task", "classification")
    if task not in ("classification", "regression"):
        raise ConfigError(f"task: unknown task '{task}' (use classification or regression)")

    metric = raw.get("metric") or DEFAULT_METRIC[task]
    if metric not in METRIC_NAMES:
        raise ConfigError(f"metric: unknown metric '{metric}' (known: {', '.join(METRIC_NAMES)})")
    if metric not in metric_names_for(task):
        raise ConfigError(f"metric: '{metric}' is not valid for {task} tasks")

    data = _build_data(raw.get("data"))
    model = _build_model(raw.get("model"))
    search = _build_search(raw.get("search"), model.name, task, metric)
    constraints = _build_constraints(raw.get("constraints"))
    selection = _build_selection(raw.get("selection"))

    config = ExperimentConfig(
        name=name,
        seed=seed,
        task=task,
        metric=metric,
        data=data,
        model=model,
        search=search,
        constraints=constraints,
        selection=selection,
        source_path=source_path,
    )
    try:
        space = config.search_space
        build_model(model.name, model.params, 0, task)
        fixed = {key: value for key, value in space.items() if not isinstance(value, list)}
        if fixed:
            build_model(model.name, fixed, 0, task)
    except (ValueError, RuntimeError) as exc:
        raise ConfigError(str(exc)) from None
    return config


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with path.open() as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        raise ConfigError(f"config file is empty: {path}")
    return from_dict(raw, source_path=str(path))


def dump_config(config: ExperimentConfig, path: str | Path) -> None:
    with Path(path).open("w") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
