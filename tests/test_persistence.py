from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from autotuneml.tracking import (
    ExperimentStore,
    list_experiments,
    trial_to_record,
    utc_now,
)


def make_trial(number=0, status="COMPLETE", params=None, metrics=None, error=None,
               fitted_rounds=None, started=None, finished=None):
    started = started or datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    finished = finished or started + timedelta(seconds=4, milliseconds=500)
    return SimpleNamespace(
        number=number,
        state=SimpleNamespace(name=status),
        params=params or {"C": 0.5},
        user_attrs={
            "metrics": metrics if metrics is not None else {
                "val_score": 0.91, "train_time": 1.25, "model_size": 4096,
            },
            "effective_params": {"C": 0.5, "max_iter": 500},
            "fitted_rounds": fitted_rounds,
            "error": error,
        },
        error=None,
        datetime_start=started,
        datetime_complete=finished,
    )


def sample_record(**kwargs):
    trial = make_trial(**kwargs)
    return trial_to_record(trial)


def test_create_writes_config_and_meta(make_config, root):
    config = make_config(name="stored")
    store = ExperimentStore.create(root, "stored", config)

    assert store.dir.exists()
    assert store.path("config.yaml").exists()
    meta = store.read_json("meta.json")
    assert meta["experiment"] == "stored"
    assert meta["seed"] == config.seed
    assert "versions" in meta
    assert store.load_config().name == "stored"


def test_create_refuses_to_overwrite_without_flag(make_config, root):
    config = make_config(name="stored")
    ExperimentStore.create(root, "stored", config)

    with pytest.raises(FileExistsError, match="already exists"):
        ExperimentStore.create(root, "stored", config)

    replaced = ExperimentStore.create(root, "stored", config, overwrite=True)
    assert replaced.read_trials() == []
    assert not replaced.path("summary.json").exists()


def test_open_missing_experiment(root):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        ExperimentStore.open(root, "ghost")


def test_invalid_experiment_name(root):
    with pytest.raises(ValueError, match="invalid experiment name"):
        ExperimentStore.open(root, "../escape")


def test_trials_append_and_read_back(make_config, root):
    store = ExperimentStore.create(root, "stored", make_config(name="stored"))
    assert store.read_trials() == []
    assert store.has_trials is False

    store.append_trial(sample_record(number=0))
    store.append_trial(sample_record(number=1, status="PRUNED",
                                     metrics={"val_score": None, "train_time": 0.4,
                                              "model_size": None}))
    store.append_trial(sample_record(number=2, status="FAIL",
                                     metrics={"val_score": None, "train_time": 0.0,
                                              "model_size": None},
                                     error="ValueError: broken"))

    trials = store.read_trials()
    assert len(trials) == 3
    assert trials[0]["val_score"] == 0.91
    assert trials[0]["duration"] == pytest.approx(4.5)
    assert trials[1]["model_size"] is None
    assert trials[2]["status"] == "FAIL"
    assert trials[2]["error"] == "ValueError: broken"
    assert store.has_trials is True


def test_record_keeps_effective_params_and_rounds():
    record = sample_record(fitted_rounds=37)
    assert record["params"] == {"C": 0.5, "max_iter": 500}
    assert record["fitted_rounds"] == 37
    assert record["finished_at"] == "2026-01-01T10:00:04+00:00"


def test_record_without_metrics_is_tolerated():
    trial = make_trial()
    trial.user_attrs = {}
    trial.error = RuntimeError("upstream failure")
    record = trial_to_record(trial)
    assert record["val_score"] is None
    assert record["error"] == "upstream failure"


def test_summary_roundtrip(make_config, root):
    store = ExperimentStore.create(root, "stored", make_config(name="stored"))
    summary = {"experiment": "stored", "best": {"val_score": 0.9}, "states": {"COMPLETE": 1}}
    assert store.has_summary is False

    store.write_summary(summary)
    assert store.has_summary is True
    assert store.read_summary() == summary


def test_read_summary_missing(make_config, root):
    store = ExperimentStore.create(root, "stored", make_config(name="stored"))
    with pytest.raises(FileNotFoundError, match="summary.json"):
        store.read_summary()


def test_list_experiments_only_returns_summarised(make_config, root):
    finished = ExperimentStore.create(root, "with_summary", make_config(name="with_summary"))
    finished.write_summary({"experiment": "with_summary"})
    ExperimentStore.create(root, "no_summary", make_config(name="no_summary"))

    assert list_experiments(root) == ["with_summary"]
    assert list_experiments(root / "nowhere") == []


def test_reports_are_stored_as_json(make_config, root):
    store = ExperimentStore.create(root, "stored", make_config(name="stored"))
    store.write_json("train_report.json", {"test": {"accuracy": 0.9}})
    assert store.read_json("train_report.json")["test"]["accuracy"] == 0.9
    assert store.has_model is False


def test_utc_now_is_iso_format():
    stamp = utc_now()
    assert "T" in stamp
    assert datetime.fromisoformat(stamp).tzinfo is not None
