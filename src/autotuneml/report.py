from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .metrics import metric_direction
from .selection import sort_records
from .tracking import ExperimentStore, utc_now

TOP_TRIALS = 10


def _test_metrics(store: ExperimentStore) -> dict | None:
    for filename in ("train_report.json", "eval_report.json"):
        if store.path(filename).exists():
            return store.read_json(filename).get("test")
    return None


def _classification_detail(model, holdout) -> tuple[list[dict] | None, dict | None]:
    if not model.classifier:
        return None, None
    y_true = list(holdout.y_test)
    prediction = list(model.predict(holdout.X_test))
    labels = sorted(set(y_true))
    if len(labels) < 2:
        return None, None

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, prediction, labels=labels, zero_division=0
    )
    per_class = [
        {
            "class": label,
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    ]
    detail = {
        "labels": labels,
        "matrix": confusion_matrix(y_true, prediction, labels=labels).tolist(),
    }
    return per_class, detail


def _try_pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def make_plots(records: list[dict], pareto: list[dict], directory) -> list[str]:
    plt = _try_pyplot()
    complete = [r for r in records if r.get("status") == "COMPLETE" and r.get("val_score") is not None]
    if plt is None or len(complete) < 2:
        return []

    numbers = [r["trial"] for r in complete]
    scores = [r["val_score"] for r in complete]
    figure, axes = plt.subplots(figsize=(7, 3.4))
    axes.scatter(numbers, scores, s=18)
    axes.set_xlabel("trial")
    axes.set_ylabel("validation score")
    axes.set_title("Validation score per trial")
    figure.tight_layout()
    history = directory / "history.png"
    figure.savefig(history, dpi=110)
    plt.close(figure)

    pareto_trials = {row["trial"] for row in pareto}
    times = [r["train_time"] or 0.0 for r in complete]
    colors = ["tab:red" if r["trial"] in pareto_trials else "tab:blue" for r in complete]
    figure, axes = plt.subplots(figsize=(7, 3.4))
    axes.scatter(times, scores, s=18, c=colors)
    axes.set_xlabel("train time (s)")
    axes.set_ylabel("validation score")
    axes.set_title("Score vs training cost (red: pareto front)")
    figure.tight_layout()
    tradeoff = directory / "tradeoff.png"
    figure.savefig(tradeoff, dpi=110)
    plt.close(figure)

    return [history.name, tradeoff.name]


def build_report_data(experiment: str, root: str = "experiments") -> dict:
    store = ExperimentStore.open(root, experiment)
    if not store.has_summary:
        raise ValueError(f"experiment '{experiment}' has no summary.json; run optimize first")

    config = store.load_config()
    summary = store.read_summary()
    records = store.read_trials()
    meta = store.read_json("meta.json") if store.path("meta.json").exists() else {}

    direction = metric_direction(config.metric)
    complete = [r for r in records if r.get("status") == "COMPLETE"
                and r.get("val_score") is not None]
    top = sort_records(complete, "val_score", direction)[:TOP_TRIALS]

    per_class = confusion = None
    test_rows = (summary.get("holdout") or {}).get("test_rows")
    if store.has_model and (store.holdout_dir / "test.csv").exists():
        from .data import load_saved_holdout
        holdout = load_saved_holdout(
            store.holdout_dir, config.data.target or "target",
            config.data.impute, config.data.scale,
        )
        model = joblib.load(store.model_path)
        test_rows = holdout.test_size
        if config.task == "classification":
            per_class, confusion = _classification_detail(model, holdout)

    return {
        "experiment": store.name,
        "generated_at": utc_now(),
        "model": config.model.name,
        "task": config.task,
        "metric": config.metric,
        "metric_direction": direction,
        "seed": config.seed,
        "created_at": meta.get("created_at"),
        "hardware": meta.get("hardware"),
        "versions": meta.get("versions") or summary.get("versions"),
        "holdout": {"train_rows": (summary.get("holdout") or {}).get("train_rows"),
                    "test_rows": test_rows},
        "baseline": summary.get("baseline"),
        "best": summary.get("best"),
        "improvement": summary.get("improvement_over_baseline"),
        "selection": summary.get("selection"),
        "states": summary.get("states"),
        "n_trials": summary.get("n_trials"),
        "pareto": summary.get("pareto") or [],
        "test": _test_metrics(store),
        "per_class": per_class,
        "confusion": confusion,
        "top_trials": [
            {key: row.get(key) for key in ("trial", "val_score", "train_time",
                                           "model_size", "params")}
            for row in top
        ],
        "plots": [],
    }


def _table(headers: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines


def _format_score(value) -> str:
    return "-" if value is None else f"{value:.4f}"


def _format_cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(data: dict) -> str:
    best = data["best"] or {}
    baseline = data["baseline"] or {}
    lines = [
        f"# Experiment report: {data['experiment']}",
        "",
        f"Generated {data['generated_at']} by `autotuneml report`.",
        "",
        "## Setup",
        "",
    ]
    lines += _table(
        ["model", "task", "metric", "direction", "seed", "trials", "train rows", "test rows"],
        [[data["model"], data["task"], data["metric"], data["metric_direction"], data["seed"],
          data["n_trials"], (data["holdout"] or {}).get("train_rows"),
          (data["holdout"] or {}).get("test_rows")]],
    )

    lines += ["", "## Baseline vs selected", ""]
    lines += _table(
        ["", "trial", data["metric"], "train time (s)", "model size (bytes)"],
        [
            ["baseline", "-", _format_score(baseline.get("val_score")),
             _format_score(baseline.get("train_time")), baseline.get("model_size") or "-"],
            ["selected", best.get("trial"), _format_score(best.get("val_score")),
             _format_score(best.get("train_time")), best.get("model_size") or "-"],
        ],
    )
    lines.append(f"\nImprovement over baseline: {_format_score(data['improvement'])}.")

    selection = data.get("selection") or {}
    lines += ["", "## Selection", ""]
    lines.append(f"Rule: `{selection.get('rule')}`. "
                 f"Weights: `{selection.get('weights') or {}}`. "
                 f"Constraints: `{selection.get('constraints') or {}}`.")
    lines.append(f"Constraints satisfied: {best.get('constraints_satisfied')}.")
    if best.get("selected_without_constraints"):
        lines.append("No trial satisfied the constraints; the best unconstrained trial was taken.")
    lines.append(f"Best parameters: `{best.get('params') or {}}`.")

    states = data.get("states") or {}
    lines += ["", "## Trials", "",
              f"States: {', '.join(f'{key}={value}' for key, value in states.items())}."]
    if data["plots"]:
        lines += ["", "![score history](history.png)",
                  "", "![score vs cost](tradeoff.png)"]
    lines += ["", "### Top trials", ""]
    lines += _table(
        ["trial", data["metric"], "train time (s)", "model size (bytes)", "params"],
        [[row["trial"], _format_score(row["val_score"]), _format_score(row["train_time"]),
          row["model_size"], row["params"]] for row in data["top_trials"]],
    )

    if data["pareto"]:
        objectives = [key for key in data["pareto"][0] if key != "trial"]
        lines += ["", "### Pareto front", ""]
        lines += _table(["trial", *objectives],
                        [[row["trial"], *[_format_cell(row.get(o)) for o in objectives]]
                         for row in data["pareto"]])

    lines += ["", "## Test set", ""]
    test = data.get("test")
    if test:
        lines += _table(["metric", "value"],
                        [[name, _format_score(value)] for name, value in test.items()])
    else:
        lines.append("No test evaluation recorded yet; run `train` or `evaluate` first.")

    if data.get("per_class"):
        lines += ["", "### Per-class scores", ""]
        lines += _table(["class", "precision", "recall", "f1", "support"],
                        [[row["class"], row["precision"], row["recall"],
                          row["f1"], row["support"]] for row in data["per_class"]])
    if data.get("confusion"):
        confusion = data["confusion"]
        labels = confusion["labels"]
        lines += ["", "### Confusion matrix", "",
                  "Rows are true labels, columns are predicted labels.", ""]
        lines += _table(["true \\ pred", *[str(label) for label in labels]],
                        [[str(labels[i]), *row] for i, row in enumerate(confusion["matrix"])])

    hardware = data.get("hardware") or {}
    if hardware:
        cpu = hardware.get("cpu") or {}
        gpu = hardware.get("gpu") or {}
        devices = ", ".join(d.get("name", "?") for d in gpu.get("devices", [])) or "-"
        lines += ["", "## Hardware", ""]
        lines += _table(["cpu", "arch", "memory (GB)", "gpu present", "gpu devices", "accelerator"],
                        [[cpu.get("count"), cpu.get("arch"), cpu.get("memory_gb"),
                          gpu.get("present"), devices, hardware.get("accelerator")]])

    versions = data.get("versions") or {}
    if versions:
        lines += ["", "## Versions", ""]
        lines += _table(["package", "version"],
                        [[name, version] for name, version in versions.items()])

    return "\n".join(lines) + "\n"


def run_report(experiment: str, root: str = "experiments", out: str | None = None):
    store = ExperimentStore.open(root, experiment)
    data = build_report_data(experiment, root)

    directory = Path(out) if out else store.dir
    directory.mkdir(parents=True, exist_ok=True)
    data["plots"] = make_plots(store.read_trials(), data["pareto"], directory)

    target = directory / "report.md"
    target.write_text(render_markdown(data))
    return target
