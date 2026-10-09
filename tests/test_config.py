import pytest

from autotuneml.config import ConfigError, dump_config, from_dict, load_config


def minimal(**overrides):
    raw = {
        "name": "unit",
        "seed": 3,
        "task": "classification",
        "metric": "accuracy",
        "data": {"source": "sklearn:iris", "test_size": 0.2},
        "model": {"name": "logistic_regression"},
        "search": {"n_trials": 4, "cv_folds": 3},
    }
    raw.update(overrides)
    return raw


def test_valid_config_roundtrip(tmp_path):
    config = from_dict(minimal())
    assert config.name == "unit"
    assert config.search.n_trials == 4
    assert config.constraints == {}
    assert config.selection.require_constraints is True

    path = tmp_path / "config.yaml"
    dump_config(config, path)
    loaded = load_config(path)
    expected = config.to_dict()
    expected.pop("source_path")
    actual = loaded.to_dict()
    actual.pop("source_path")
    assert actual == expected


def test_defaults_are_applied():
    raw = minimal()
    del raw["search"]
    del raw["metric"]
    config = from_dict(raw)
    assert config.metric == "accuracy"
    assert config.search.n_trials == 30
    assert config.search.cv_folds == 5
    assert config.search.objectives == ["val_score"]


def test_search_space_merges_model_defaults():
    config = from_dict(minimal(search={"params": {"C": [0.5, 2.0]}}))
    space = config.search_space
    assert space["C"] == [0.5, 2.0]
    assert "class_weight" in space


def test_rejects_unknown_metric():
    with pytest.raises(ConfigError, match="unknown metric"):
        from_dict(minimal(metric="area_under_curve"))


def test_rejects_metric_from_other_task():
    with pytest.raises(ConfigError, match="not valid for classification"):
        from_dict(minimal(metric="rmse"))


def test_rejects_unknown_model():
    with pytest.raises(ConfigError, match="unknown model"):
        from_dict(minimal(model={"name": "deep_belief_net"}))


def test_rejects_unknown_task():
    with pytest.raises(ConfigError, match="unknown task"):
        from_dict(minimal(task="clustering"))


def test_rejects_missing_data_section():
    raw = minimal()
    del raw["data"]
    with pytest.raises(ConfigError, match="data"):
        from_dict(raw)


def test_rejects_bad_test_size():
    with pytest.raises(ConfigError, match="test_size"):
        from_dict(minimal(data={"source": "sklearn:iris", "test_size": 1.4}))


def test_rejects_inverted_search_range():
    with pytest.raises(ConfigError, match="lower bound"):
        from_dict(minimal(search={"params": {"C": [10.0, 0.1]}}))


def test_rejects_mixed_search_entry():
    with pytest.raises(ConfigError, match="single kind"):
        from_dict(minimal(search={"params": {"C": [0.1, "high"]}}))


def test_rejects_unknown_objective():
    with pytest.raises(ConfigError, match="unknown objective"):
        from_dict(minimal(search={"objectives": ["val_score", "energy_bill"]}))


def test_rejects_duplicate_objectives():
    with pytest.raises(ConfigError, match="duplicate"):
        from_dict(minimal(search={"objectives": ["val_score", "val_score"]}))


def test_rejects_typo_in_fixed_model_params():
    with pytest.raises(ConfigError, match="n_estimatos"):
        from_dict(minimal(model={"name": "random_forest", "params": {"n_estimatos": 10}}))


def test_rejects_bad_constraints():
    with pytest.raises(ConfigError, match="min exceeds max"):
        from_dict(minimal(constraints={"train_time": {"min": 10.0, "max": 1.0}}))
    with pytest.raises(ConfigError, match="unknown metric"):
        from_dict(minimal(constraints={"gpu_hours": {"max": 1.0}}))
    with pytest.raises(ConfigError, match="unknown bound"):
        from_dict(minimal(constraints={"train_time": {"target": 3.0}}))


def test_rejects_negative_selection_weight():
    with pytest.raises(ConfigError, match="non-negative"):
        from_dict(minimal(selection={"weights": {"val_score": -1.0}}))


def test_rejects_missing_config_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml")


def test_config_hash_tracks_content():
    first = from_dict(minimal())
    second = from_dict(minimal(seed=99))
    assert first.config_hash() == from_dict(minimal()).config_hash()
    assert first.config_hash() != second.config_hash()


def test_directions_follow_metric():
    config = from_dict(minimal(metric="accuracy"))
    assert config.directions["val_score"] == "maximize"
    regression = from_dict(minimal(task="regression", metric="rmse",
                                   model={"name": "random_forest"}))
    assert regression.directions["val_score"] == "minimize"


def test_device_defaults_to_auto():
    assert from_dict(minimal()).model.device == "auto"


def test_rejects_unknown_device():
    with pytest.raises(ConfigError, match="unknown device"):
        from_dict(minimal(model={"name": "random_forest", "device": "tpu"}))
