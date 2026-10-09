from __future__ import annotations

import argparse
import sys

from . import __version__
from .compare import run_compare
from .config import ConfigError, load_config
from .evaluate import run_evaluate
from .export import run_export
from .report import run_report
from .runner import run_optimize, run_train


def format_size(value) -> str:
    if value is None:
        return "-"
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def format_params(params: dict) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(params.items())) or "-"


def print_metrics(metrics: dict, indent: str = "  ") -> None:
    for name, value in metrics.items():
        if value is None:
            print(f"{indent}{name:<20} -")
        else:
            print(f"{indent}{name:<20} {value:.4f}")


def cmd_optimize(args) -> int:
    config = load_config(args.config)
    if args.name:
        config.name = args.name
    summary = run_optimize(
        config,
        root=args.root,
        trials=args.trials,
        timeout=args.timeout,
        overwrite=args.overwrite,
    )
    best = summary["best"]
    baseline = summary["baseline"]
    states = ", ".join(f"{key.lower()}={value}" for key, value in summary["states"].items())

    print(f"experiment : {summary['experiment']}")
    print(f"model      : {summary['model']} ({summary['task']})")
    print(f"metric     : {summary['metric']} ({summary['metric_direction']})")
    print(f"trials     : {summary['n_trials']} ({states})")
    print(f"baseline   : {baseline['val_score']:.4f} in {baseline['train_time']:.2f}s "
          f"({format_size(baseline['model_size'])})")
    print(f"best       : {best['val_score']:.4f} in {best['train_time']:.2f}s "
          f"({format_size(best['model_size'])}) trial={best['trial']}")
    print(f"gain       : {summary['improvement_over_baseline']:+.4f} over baseline")
    print(f"params     : {format_params(best['params'])}")
    print(f"constraints: {'satisfied' if best['constraints_satisfied'] else 'not satisfied'}")
    print(f"pareto     : {len(summary['pareto'])} non-dominated trial(s)")
    print(f"storage    : {args.root}/{summary['experiment']}")
    return 0


def cmd_train(args) -> int:
    config = load_config(args.config)
    if args.name:
        config.name = args.name
    report = run_train(config, root=args.root, from_experiment=args.from_experiment)

    print(f"experiment : {report['experiment']}")
    print(f"model      : {report['model']}")
    print(f"params     : {format_params(report['params'])}")
    print(f"cv {report['cv']['metric']:<6}: {report['cv']['val_score']:.4f} "
          f"in {report['cv']['train_time']:.2f}s ({format_size(report['cv']['model_size'])})")
    print(f"test rows  : {report['rows']['test']}")
    print("test metrics:")
    print_metrics(report["test"])
    return 0


def cmd_evaluate(args) -> int:
    report = run_evaluate(args.experiment, root=args.root)
    print(f"experiment : {report['experiment']}")
    print(f"model      : {report['model']}")
    print(f"test rows  : {report['rows']['test']}")
    print("test metrics:")
    print_metrics(report["test"])
    return 0


def cmd_compare(args) -> int:
    table = run_compare(root=args.root, names=args.experiments or None)
    print(table)
    return 0


def cmd_report(args) -> int:
    target = run_report(args.experiment, root=args.root, out=args.out)
    print(f"experiment : {args.experiment}")
    print(f"report     : {target}")
    return 0


def cmd_export(args) -> int:
    metadata = run_export(args.experiment, root=args.root, out=args.out)
    print(f"experiment : {metadata['experiment']}")
    print(f"model      : {metadata['model']}")
    print(f"artifact   : {metadata['artifact']}")
    print(f"metadata   : {metadata['artifact'].replace('model.joblib', 'metadata.json')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autotuneml",
        description="Configuration-driven hyperparameter search with a strict holdout protocol.",
    )
    parser.add_argument("--version", action="version", version=f"autotuneml {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser(
        "train",
        help="fit a model with fixed parameters and score it on the held-out test split",
    )
    train.add_argument("--config", required=True, help="experiment configuration file")
    train.add_argument("--name", help="experiment name (defaults to the config name)")
    train.add_argument("--from-experiment", help="reuse parameters selected by another experiment")
    train.add_argument("--root", default="experiments", help="experiment storage root")
    train.set_defaults(func=cmd_train)

    optimize = commands.add_parser("optimize", help="search hyperparameter configurations")
    optimize.add_argument("--config", required=True, help="experiment configuration file")
    optimize.add_argument("--name", help="experiment name (defaults to the config name)")
    optimize.add_argument("--trials", type=int, help="override search.n_trials")
    optimize.add_argument("--timeout", type=float, help="override search.timeout_seconds")
    optimize.add_argument("--root", default="experiments", help="experiment storage root")
    optimize.add_argument("--overwrite", action="store_true", help="replace an existing experiment")
    optimize.set_defaults(func=cmd_optimize)

    evaluate = commands.add_parser(
        "evaluate",
        help="score a stored model once on the held-out test split",
    )
    evaluate.add_argument("experiment", help="experiment name")
    evaluate.add_argument("--root", default="experiments", help="experiment storage root")
    evaluate.set_defaults(func=cmd_evaluate)

    compare = commands.add_parser("compare", help="compare experiments by their summary")
    compare.add_argument("experiments", nargs="*", help="experiment names (default: all)")
    compare.add_argument("--root", default="experiments", help="experiment storage root")
    compare.set_defaults(func=cmd_compare)

    report = commands.add_parser("report", help="write a markdown report for an experiment")
    report.add_argument("experiment", help="experiment name")
    report.add_argument("--root", default="experiments", help="experiment storage root")
    report.add_argument("--out", help="output directory (default: the experiment folder)")
    report.set_defaults(func=cmd_report)

    export = commands.add_parser("export", help="copy a trained model into the artifacts folder")
    export.add_argument("experiment", help="experiment name")
    export.add_argument("--root", default="experiments", help="experiment storage root")
    export.add_argument("--out", help="destination directory (default: artifacts/<experiment>)")
    export.set_defaults(func=cmd_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
