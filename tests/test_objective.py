import optuna
import pytest
from optuna.trial import TrialState

import autotuneml.objective as objective_module
import autotuneml.runner as runner_module
from autotuneml.data import load_saved_holdout, prepare
from autotuneml.objective import Objective, cross_validate
from autotuneml.runner import run_optimize
from autotuneml.tracking import ExperimentStore


def build_holdout(config):
    return prepare(config.data.source, config.data.target, 0.25, config.seed, config.task)


def test_cross_validate_reports_every_metric(make_config):
    config = make_config()
    holdout = build_holdout(config)
    result = cross_validate(holdout.X_train, holdout.y_train, holdout.columns, config,
                            {"C": 1.0, "max_iter": 500})
    assert set(result) == {"val_score", "train_time", "model_size", "folds", "fitted_rounds"}
    assert 0.0 <= result["val_score"] <= 1.0
    assert result["train_time"] > 0
    assert result["model_size"] > 0
    assert result["folds"] == 3


def test_single_objective_returns_one_value(make_config):
    config = make_config(search={"cv_folds": 3})
    holdout = build_holdout(config)
    study = optuna.create_study(direction="maximize")
    study.optimize(Objective(holdout.X_train, holdout.y_train, holdout.columns, config),
                   n_trials=2, catch=(Exception,))

    assert [trial.state for trial in study.trials] == [TrialState.COMPLETE, TrialState.COMPLETE]
    assert isinstance(study.best_value, float)
    metrics = study.best_trial.user_attrs["metrics"]
    assert metrics["model_size"] > 0
    assert set(metrics) == {"val_score", "train_time", "model_size"}


def test_multi_objective_returns_a_tuple(make_config):
    config = make_config(search={"cv_folds": 3,
                                 "objectives": ["val_score", "train_time", "model_size"]})
    holdout = build_holdout(config)
    study = optuna.create_study(directions=["maximize", "minimize", "minimize"])
    study.optimize(Objective(holdout.X_train, holdout.y_train, holdout.columns, config),
                   n_trials=2, catch=(Exception,))

    trial = study.trials[0]
    assert trial.state == TrialState.COMPLETE
    assert len(trial.values) == 3
    assert trial.values[1] > 0 and trial.values[2] > 0


def test_trial_is_pruned_when_time_budget_is_spent(make_config):
    config = make_config(search={"cv_folds": 3, "trial_time_budget": 0.001})
    holdout = build_holdout(config)
    study = optuna.create_study(direction="maximize")
    study.optimize(Objective(holdout.X_train, holdout.y_train, holdout.columns, config),
                   n_trials=1, catch=(Exception,))

    trial = study.trials[0]
    assert trial.state == TrialState.PRUNED
    assert trial.user_attrs["metrics"]["model_size"] is None


def test_pruner_rejects_bad_trials_after_first_report(make_config):
    class AlwaysPrune(optuna.pruners.BasePruner):
        def prune(self, study, trial):
            return True

    config = make_config(search={"cv_folds": 3})
    holdout = build_holdout(config)
    study = optuna.create_study(direction="maximize", pruner=AlwaysPrune())
    study.optimize(Objective(holdout.X_train, holdout.y_train, holdout.columns, config),
                   n_trials=1, catch=(Exception,))

    assert study.trials[0].state == TrialState.PRUNED


def test_failing_trial_is_recorded_and_search_continues(make_config, monkeypatch, root):
    real_cross_validate = objective_module.cross_validate
    calls = {"count": 0}

    def flaky(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("boom")
        return real_cross_validate(*args, **kwargs)

    monkeypatch.setattr(objective_module, "cross_validate", flaky)

    config = make_config(name="flaky", search={"cv_folds": 3, "n_trials": 2})
    summary = run_optimize(config, root=root, trials=2)

    records = ExperimentStore.open(root, "flaky").read_trials()
    failed = [r for r in records if r["status"] == "FAIL"]
    finished = [r for r in records if r["status"] == "COMPLETE"]

    assert summary["states"].get("FAIL") == 1
    assert len(failed) == 1 and len(finished) == 1
    assert "boom" in failed[0]["error"]
    assert failed[0]["params"]


def test_optimize_never_passes_test_rows_to_the_objective(make_config, classification_csv,
                                                          monkeypatch, root):
    seen = []
    real_objective = runner_module.Objective

    class SpyObjective(real_objective):
        def __init__(self, X, y, columns, config):
            seen.append(set(X.index))
            super().__init__(X, y, columns, config)

    monkeypatch.setattr(runner_module, "Objective", SpyObjective)

    config = make_config(
        name="isolation",
        data={"source": str(classification_csv), "target": "target", "test_size": 0.3},
        search={"cv_folds": 3},
    )
    run_optimize(config, root=root, trials=2)

    holdout = load_saved_holdout(ExperimentStore.open(root, "isolation").holdout_dir, "target")
    train_index = set(holdout.X_train.index)
    test_index = set(holdout.X_test.index)

    assert len(seen) == 1
    assert seen[0] == train_index
    assert seen[0] & test_index == set()
    assert train_index | test_index == set(range(120))


def test_budget_pruning_works_in_multi_objective_search(make_config, root):
    config = make_config(
        name="multi",
        search={"cv_folds": 3, "objectives": ["val_score", "train_time"],
                "trial_time_budget": 0.001},
    )
    with pytest.raises(ValueError, match="no completed trial"):
        run_optimize(config, root=root, trials=2)

    records = ExperimentStore.open(root, "multi").read_trials()
    assert [record["status"] for record in records] == ["PRUNED", "PRUNED"]
    assert records[0]["duration"] is not None


def test_regression_objective_minimizes_metric(make_config, root):
    config = make_config(
        name="reg",
        source="sklearn:diabetes",
        task="regression",
        metric="rmse",
        model={"name": "random_forest", "params": {"n_estimators": 50}},
        search={"cv_folds": 3, "params": {"max_depth": [2, 6]}},
    )
    summary = run_optimize(config, root=root, trials=2)
    assert summary["metric_direction"] == "minimize"
    assert summary["directions"]["val_score"] == "minimize"
    assert summary["best"]["val_score"] > 0
    assert summary["improvement_over_baseline"] == pytest.approx(
        summary["baseline"]["val_score"] - summary["best"]["val_score"]
    )
