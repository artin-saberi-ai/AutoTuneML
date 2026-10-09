import pytest

from autotuneml.models import (
    DEFAULT_SPACES,
    build_model,
    device_params,
    fit_estimator,
    merge_space,
    model_available,
    suggest_params,
    validate_space,
)


def test_numeric_pair_becomes_a_range():
    study_space = {"n_estimators": [100, 600], "learning_rate": [0.01, 0.3]}
    validate_space(study_space)
    assert study_space == {"n_estimators": [100, 600], "learning_rate": [0.01, 0.3]}


def test_list_choices_are_accepted():
    validate_space({"max_features": ["sqrt", "log2"], "penalty": ["l2", None]})


def test_rejects_empty_space():
    with pytest.raises(ValueError, match="empty search space"):
        validate_space({"max_depth": []})


def test_rejects_inverted_range():
    with pytest.raises(ValueError, match="lower bound"):
        validate_space({"max_depth": [20, 3]})


def test_rejects_mixed_numeric_and_text():
    with pytest.raises(ValueError, match="single kind"):
        validate_space({"max_depth": [3, "none"]})


def test_rejects_nested_containers():
    with pytest.raises(ValueError, match="nested"):
        validate_space({"grid": [{"a": 1}]})


def test_merge_space_overrides_defaults():
    merged = merge_space("random_forest", {"n_estimators": [10, 50]})
    assert merged["n_estimators"] == [10, 50]
    assert merged["max_depth"] == DEFAULT_SPACES["random_forest"]["max_depth"]


def test_merge_space_without_defaults_only_keeps_overrides():
    assert merge_space("logistic_regression", None) == DEFAULT_SPACES["logistic_regression"]
    assert merge_space("not_a_model", {"a": 1}) == {"a": 1}


def test_suggest_params_keeps_fixed_values_and_suggests_ranges():
    space = {"n_estimators": [10, 50], "learning_rate": [0.01, 0.3],
             "n_jobs": -1, "criterion": "entropy"}

    class Recorder:
        def __init__(self):
            self.suggested = {}

        def suggest_int(self, name, low, high):
            self.suggested[name] = ("int", low, high)
            return low

        def suggest_float(self, name, low, high):
            self.suggested[name] = ("float", low, high)
            return low

        def suggest_categorical(self, name, choices):
            self.suggested[name] = ("categorical", tuple(choices))
            return choices[0]

    trial = Recorder()
    params = suggest_params(trial, space)

    assert params["n_jobs"] == -1
    assert params["criterion"] == "entropy"
    assert params["n_estimators"] == 10
    assert params["learning_rate"] == 0.01
    assert trial.suggested["n_estimators"] == ("int", 10, 50)
    assert trial.suggested["learning_rate"] == ("float", 0.01, 0.3)
    assert "n_jobs" not in trial.suggested
    assert "criterion" not in trial.suggested


def test_build_model_rejects_unknown_parameter():
    with pytest.raises(ValueError, match="unknown parameters"):
        build_model("logistic_regression", {"n_estimators": 10}, seed=1, task="classification")


def test_build_model_rejects_unknown_model():
    with pytest.raises(ValueError, match="unknown model"):
        build_model("resnet", {}, seed=1, task="classification")


def test_build_model_rejects_model_from_wrong_task():
    with pytest.raises(ValueError, match="not available for regression"):
        build_model("logistic_regression", {}, seed=1, task="regression")


def test_build_model_seeds_the_estimator():
    first = build_model("random_forest", {"n_estimators": 10}, seed=4, task="classification")
    second = build_model("random_forest", {"n_estimators": 10}, seed=4, task="classification")
    assert first.get_params()["random_state"] == 4
    assert second.get_params()["random_state"] == 4


def test_sklearn_models_report_available():
    assert model_available("random_forest") is True
    assert model_available("hist_gradient_boosting") is True


def test_early_stopping_stops_boosting_before_max_iter():
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 5))
    y = rng.integers(0, 2, size=400)
    X_es = rng.normal(size=(60, 5))
    y_es = rng.integers(0, 2, size=60)

    booster = build_model("hist_gradient_boosting", {"max_iter": 300, "learning_rate": 0.3},
                          seed=1, task="classification")
    fit_estimator(booster, X, y, X_es, y_es, early_stopping=True)
    assert 15 < booster.n_iter_ < 300

    forest = build_model("random_forest", {"n_estimators": 10}, seed=1, task="classification")
    fit_estimator(forest, X, y, X_es, y_es, early_stopping=True)
    assert forest.n_estimators == 10


def test_device_params_mapping():
    assert device_params("xgboost", "cuda") == {"device": "cuda"}
    assert device_params("xgboost", "cpu") == {"device": "cpu"}
    assert device_params("lightgbm", "cuda") == {"device": "gpu"}
    assert device_params("lightgbm", "cpu") == {"device": "cpu"}
    assert device_params("random_forest", "cuda") == {}


def test_build_model_ignores_cpu_device():
    est = build_model("random_forest", {}, seed=1, task="classification", device="cpu")
    assert est is not None


def test_build_model_rejects_cuda_for_cpu_only_models():
    with pytest.raises(RuntimeError, match="CPU only"):
        build_model("random_forest", {}, seed=1, task="classification", device="cuda")
