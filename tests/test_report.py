import importlib.util

import pytest

from autotuneml.report import build_report_data, render_markdown, run_report
from autotuneml.runner import run_optimize, run_train

MPL = importlib.util.find_spec("matplotlib") is not None


def test_report_requires_a_finished_search(make_config, root):
    from autotuneml.tracking import ExperimentStore
    ExperimentStore.create(root, "empty", make_config(name="empty"))
    with pytest.raises(ValueError, match="no summary"):
        run_report("empty", root=root)


def test_report_without_model_marks_test_as_pending(make_config, root):
    config = make_config(name="search_only")
    run_optimize(config, root=root, trials=2)

    data = build_report_data("search_only", root=root)
    assert data["test"] is None
    assert data["per_class"] is None
    assert data["confusion"] is None
    assert len(data["top_trials"]) == 2

    text = render_markdown(data)
    assert "No test evaluation recorded" in text
    assert "Top trials" in text
    assert "Hardware" in text


def test_report_with_model_contains_test_and_confusion(make_config, root):
    config = make_config(name="full")
    run_optimize(config, root=root, trials=2)
    run_train(config, root=root)

    target = run_report("full", root=root)
    assert target.name == "report.md"
    text = target.read_text()

    assert "Per-class scores" in text
    assert "Confusion matrix" in text
    assert "true \\ pred" in text
    assert "Pareto front" in text
    assert "Improvement over baseline" in text
    if MPL:
        assert (target.parent / "history.png").exists()
        assert (target.parent / "tradeoff.png").exists()
        assert "history.png" in text


def test_report_without_plots_when_backend_is_missing(make_config, root, monkeypatch):
    import autotuneml.report as report_module

    monkeypatch.setattr(report_module, "_try_pyplot", lambda: None)

    config = make_config(name="noplots")
    run_optimize(config, root=root, trials=2)
    target = run_report("noplots", root=root)

    assert not (target.parent / "history.png").exists()
    assert "history.png" not in target.read_text()


def test_report_writes_to_custom_directory(make_config, root, tmp_path):
    config = make_config(name="custom")
    run_optimize(config, root=root, trials=2)

    out = tmp_path / "reports"
    target = run_report("custom", root=root, out=str(out))
    assert target == out / "report.md"
    assert target.exists()


def test_regression_report_has_no_confusion_matrix(make_config, root):
    config = make_config(
        name="reg",
        source="sklearn:diabetes",
        task="regression",
        metric="rmse",
        model={"name": "random_forest", "params": {"n_estimators": 20}},
        search={"cv_folds": 2, "params": {"max_depth": [2, 5]}},
    )
    run_optimize(config, root=root, trials=2)
    run_train(config, root=root)

    text = run_report("reg", root=root).read_text()
    assert "rmse" in text
    assert "Per-class scores" not in text
    assert "Confusion matrix" not in text
