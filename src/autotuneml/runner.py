from __future__ import annotations

from pathlib import Path

import joblib
import optuna
from optuna import logging as optuna_logging
from optuna.pruners import MedianPruner, NopPruner
from optuna.samplers import NSGAIISampler, RandomSampler, TPESampler

from .config import ExperimentConfig
from .data import Holdout, load_saved_holdout, prepare, save_holdout
from .evaluate import score_frame
from .metrics import metric_direction
from .models import DEFAULT_PARAMS, FittedModel
from .objective import Objective, cross_validate, fit_with_preprocessing
from .selection import is_feasible, select_best
from .tracking import ExperimentStore, collect_versions, trial_to_record, utc_now


def build_sampler(config: ExperimentConfig):
    choice = config.search.sampler
    if choice == "auto":
        choice = "nsga2" if len(config.search.objectives) > 1 else "tpe"
    if choice == "tpe":
        return TPESampler(seed=config.seed)
    if choice == "random":
        return RandomSampler(seed=config.seed)
    return NSGAIISampler(seed=config.seed)


def build_pruner(config: ExperimentConfig):
    if len(config.search.objectives) > 1 or config.search.pruner == "none":
        return NopPruner()
    return MedianPruner(n_startup_trials=5, n_warmup_steps=1)


def get_holdout(store: ExperimentStore, config: ExperimentConfig) -> Holdout:
    saved = store.holdout_dir / "train.csv"
    if saved.exists():
        target = config.data.target or "target"
        return load_saved_holdout(store.holdout_dir, target, config.data.impute, config.data.scale)
    holdout = prepare(
        config.data.source,
        config.data.target,
        config.data.test_size,
        config.seed,
        config.task,
        config.data.impute,
        config.data.scale,
    )
    save_holdout(holdout, store.holdout_dir)
    return holdout


def default_params(config: ExperimentConfig) -> dict:
    return dict(config.model.params) or dict(DEFAULT_PARAMS[config.model.name])


def run_baseline(holdout: Holdout, config: ExperimentConfig, params: dict | None = None) -> dict:
    params = default_params(config) if params is None else params
    result = cross_validate(holdout.X_train, holdout.y_train, holdout.columns, config, params,
                            enforce_budget=False)
    return {
        "params": params,
        "val_score": result["val_score"],
        "train_time": result["train_time"],
        "model_size": result["model_size"],
    }


def resolve_params(config: ExperimentConfig, from_experiment: str | None,
                   root: str = "experiments") -> dict:
    params = default_params(config)
    if from_experiment:
        source = ExperimentStore.open(root, from_experiment)
        if not source.has_summary:
            raise ValueError(
                f"experiment '{from_experiment}' has no summary.json; run optimize first"
            )
        best = source.read_summary().get("best") or {}
        params.update(best.get("params") or {})
    return params


def build_summary(config: ExperimentConfig, records: list[dict], baseline: dict,
                  holdout: Holdout) -> dict:
    objectives = config.search.objectives
    directions = config.directions
    constraints = config.constraints if config.selection.require_constraints else {}

    selection = select_best(records, objectives, directions, constraints, config.selection.weights)
    fallback = selection.trial is None
    if fallback:
        selection = select_best(records, objectives, directions, None, config.selection.weights)

    best = selection.trial
    if best is None:
        raise ValueError("the search produced no completed trial to select from")

    states: dict[str, int] = {}
    for record in records:
        states[record["status"]] = states.get(record["status"], 0) + 1

    score_direction = metric_direction(config.metric)
    baseline_score = baseline["val_score"]
    best_score = best["val_score"]
    if score_direction == "maximize":
        improvement = best_score - baseline_score
    else:
        improvement = baseline_score - best_score

    selected = {
        key: best.get(key)
        for key in ("trial", "status", "params", "val_score", "train_time",
                    "model_size", "duration", "fitted_rounds")
    }
    selected["constraints_satisfied"] = is_feasible(best, config.constraints)
    selected["selected_without_constraints"] = fallback

    return {
        "experiment": config.name,
        "model": config.model.name,
        "task": config.task,
        "metric": config.metric,
        "metric_direction": score_direction,
        "objectives": objectives,
        "directions": directions,
        "seed": config.seed,
        "finished_at": utc_now(),
        "holdout": {
            "train_rows": holdout.train_size,
            "test_rows": holdout.test_size,
            "test_size": config.data.test_size,
        },
        "n_trials": len(records),
        "states": states,
        "baseline": baseline,
        "best": selected,
        "pareto": [
            {key: record.get(key) for key in ("trial", *objectives)}
            for record in selection.front
        ],
        "selection": {
            "rule": selection.rule,
            "weights": config.selection.weights,
            "constraints": config.constraints,
            "require_constraints": config.selection.require_constraints,
        },
        "improvement_over_baseline": improvement,
        "versions": collect_versions(),
    }


def run_optimize(config: ExperimentConfig, root: str = "experiments", trials: int | None = None,
                 timeout: float | None = None, overwrite: bool = False) -> dict:
    n_trials = trials if trials is not None else config.search.n_trials
    timeout_seconds = timeout if timeout is not None else config.search.timeout_seconds

    store = ExperimentStore.create(root, config.name, config, overwrite=overwrite)
    holdout = get_holdout(store, config)
    baseline = run_baseline(holdout, config)

    optuna_logging.set_verbosity(optuna_logging.WARNING)
    study = optuna.create_study(
        directions=[config.directions[o] for o in config.search.objectives],
        sampler=build_sampler(config),
        pruner=build_pruner(config),
    )

    def persist(study_, trial):
        store.append_trial(trial_to_record(trial))

    objective = Objective(holdout.X_train, holdout.y_train, holdout.columns, config)
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_seconds,
        catch=(Exception,),
        callbacks=[persist],
        show_progress_bar=False,
    )

    summary = build_summary(config, store.read_trials(), baseline, holdout)
    store.write_summary(summary)
    return summary


def fit_final_model(holdout: Holdout, config: ExperimentConfig, params: dict) -> FittedModel:
    preprocessor, estimator = fit_with_preprocessing(
        holdout.columns, config, params, holdout.X_train, holdout.y_train, seed=config.seed
    )
    return FittedModel(
        preprocessor=preprocessor,
        estimator=estimator,
        task=config.task,
        model_name=config.model.name,
        params=params,
        metric=config.metric,
    )


def run_train(config: ExperimentConfig, root: str = "experiments",
              from_experiment: str | None = None) -> dict:
    store = _store_for_training(config, root)
    config = store.load_config()
    holdout = get_holdout(store, config)
    params = resolve_params(config, from_experiment, root)

    baseline = run_baseline(holdout, config, params)
    model = fit_final_model(holdout, config, params)
    model.cv_score = baseline["val_score"]
    joblib.dump(model, store.model_path)

    report = {
        "experiment": store.name,
        "trained_at": utc_now(),
        "model": config.model.name,
        "params": params,
        "cv": {
            "metric": config.metric,
            "val_score": baseline["val_score"],
            "train_time": baseline["train_time"],
            "model_size": model.size_bytes(),
        },
        "test": score_frame(model, holdout.X_test, holdout.y_test),
        "rows": {"train": holdout.train_size, "test": holdout.test_size},
        "from_experiment": from_experiment,
        "versions": collect_versions(),
    }
    store.write_json("train_report.json", report)
    return report


def _store_for_training(config: ExperimentConfig, root: str) -> ExperimentStore:
    existing = Path(root) / config.name
    if existing.exists():
        store = ExperimentStore.open(root, config.name)
        if store.path("config.yaml").exists():
            return store
        raise FileExistsError(f"{existing} exists but holds no config.yaml")
    return ExperimentStore.create(root, config.name, config)
