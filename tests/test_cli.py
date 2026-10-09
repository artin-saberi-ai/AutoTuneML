from pathlib import Path

from autotuneml.cli import main
from autotuneml.config import dump_config


def write_config(tmp_path, make_config, name="cli", **kwargs):
    config = make_config(name=name, **kwargs)
    path = tmp_path / f"{name}.yaml"
    dump_config(config, path)
    return path


def test_full_workflow_through_the_cli(tmp_path, make_config, classification_csv, capsys):
    root = tmp_path / "experiments"
    config_path = write_config(
        tmp_path, make_config,
        data={"source": str(classification_csv), "target": "target", "test_size": 0.25},
    )

    assert main(["optimize", "--config", str(config_path), "--root", str(root),
                 "--trials", "3"]) == 0
    assert "best" in capsys.readouterr().out

    assert main(["train", "--config", str(config_path), "--root", str(root),
                 "--from-experiment", "cli"]) == 0
    assert "test metrics" in capsys.readouterr().out

    assert main(["evaluate", "cli", "--root", str(root)]) == 0
    assert main(["compare", "--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "cli" in out and "val_score" in out

    artifacts = tmp_path / "artifacts"
    assert main(["export", "cli", "--root", str(root), "--out", str(artifacts)]) == 0

    assert (root / "cli" / "trials.jsonl").exists()
    assert (root / "cli" / "summary.json").exists()
    assert (root / "cli" / "model.joblib").exists()
    assert (root / "cli" / "holdout" / "test.csv").exists()
    assert (artifacts / "model.joblib").exists()
    assert (artifacts / "metadata.json").exists()


def test_optimize_refuses_to_overwrite_an_existing_experiment(tmp_path, make_config, capsys):
    root = tmp_path / "experiments"
    config_path = write_config(tmp_path, make_config, name="once")

    assert main(["optimize", "--config", str(config_path), "--root", str(root),
                 "--trials", "2"]) == 0
    capsys.readouterr()

    assert main(["optimize", "--config", str(config_path), "--root", str(root),
                 "--trials", "2"]) == 2
    assert "already exists" in capsys.readouterr().err

    assert main(["optimize", "--config", str(config_path), "--root", str(root),
                 "--trials", "2", "--overwrite"]) == 0


def test_evaluate_requires_a_trained_model(tmp_path, make_config, capsys):
    root = tmp_path / "experiments"
    config_path = write_config(tmp_path, make_config, name="untrained")

    assert main(["optimize", "--config", str(config_path), "--root", str(root),
                 "--trials", "2"]) == 0
    capsys.readouterr()

    assert main(["evaluate", "untrained", "--root", str(root)]) == 2
    assert "run `train` first" in capsys.readouterr().err


def test_bad_config_returns_an_error_code(tmp_path, capsys):
    missing = tmp_path / "nope.yaml"
    assert main(["optimize", "--config", str(missing)]) == 2
    assert "error:" in capsys.readouterr().err


def test_compare_lists_every_summarised_experiment(tmp_path, make_config, capsys):
    root = tmp_path / "experiments"
    first = write_config(tmp_path, make_config, name="alpha")
    second = write_config(tmp_path, make_config, name="beta")

    assert main(["optimize", "--config", str(first), "--root", str(root), "--trials", "2"]) == 0
    assert main(["optimize", "--config", str(second), "--root", str(root), "--trials", "2"]) == 0
    capsys.readouterr()

    assert main(["compare", "--root", str(root)]) == 0
    output = capsys.readouterr().out
    assert "alpha" in output and "beta" in output

    assert main(["compare", "alpha", "--root", str(root)]) == 0
    assert "alpha" in capsys.readouterr().out

    assert main(["compare", "missing", "--root", str(root)]) == 2
    assert "unknown experiments" in capsys.readouterr().err


def test_name_override_renames_the_experiment(tmp_path, make_config, capsys):
    root = tmp_path / "experiments"
    config_path = write_config(tmp_path, make_config, name="original")

    assert main(["optimize", "--config", str(config_path), "--name", "renamed",
                 "--root", str(root), "--trials", "2"]) == 0
    capsys.readouterr()

    assert (root / "renamed" / "summary.json").exists()
    assert not (root / "original").exists()
    assert main(["compare", "--root", str(root)]) == 0
    assert "renamed" in capsys.readouterr().out


def test_report_command_writes_a_report(tmp_path, make_config, capsys):
    root = tmp_path / "experiments"
    config_path = write_config(tmp_path, make_config, name="reported")

    assert main(["optimize", "--config", str(config_path), "--root", str(root),
                 "--trials", "2"]) == 0
    assert main(["train", "--config", str(config_path), "--root", str(root)]) == 0
    capsys.readouterr()

    assert main(["report", "reported", "--root", str(root)]) == 0
    assert "report.md" in capsys.readouterr().out
    assert (root / "reported" / "report.md").exists()

    assert main(["report", "missing", "--root", str(root)]) == 2
    assert "error:" in capsys.readouterr().err
