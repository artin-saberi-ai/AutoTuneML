from __future__ import annotations

from .metrics import MAXIMIZE, metric_direction
from .tracking import ExperimentStore, list_experiments


def collect_rows(root: str = "experiments", names: list[str] | None = None) -> list[dict]:
    available = list_experiments(root)
    if names:
        missing = [name for name in names if name not in available]
        if missing:
            raise ValueError(
                f"unknown experiments: {', '.join(missing)} (no summary.json under {root})"
            )
    else:
        names = available

    rows = []
    for name in names:
        summary = ExperimentStore.open(root, name).read_summary()
        best = summary.get("best") or {}
        rows.append({
            "experiment": name,
            "model": summary.get("model"),
            "metric": summary.get("metric"),
            "trials": summary.get("n_trials"),
            "done": (summary.get("states") or {}).get("COMPLETE", 0),
            "val_score": best.get("val_score"),
            "train_time": best.get("train_time"),
            "model_size": best.get("model_size"),
            "constraints": "yes" if best.get("constraints_satisfied") else "no",
            "params": best.get("params") or {},
            "_direction": metric_direction(summary.get("metric", "accuracy")),
        })

    def sort_key(row):
        value = row.get("val_score")
        if value is None:
            return float("-inf") if row["_direction"] == MAXIMIZE else float("inf")
        return value if row["_direction"] == MAXIMIZE else -value

    return sorted(rows, key=sort_key, reverse=True)


def format_table(rows: list[dict]) -> str:
    if not rows:
        return "no experiments with a summary were found"

    def score_text(row):
        value = row.get("val_score")
        return "-" if value is None else f"{value:.4f}"

    def time_text(row):
        value = row.get("train_time")
        return "-" if value is None else f"{value:.2f}s"

    def size_text(row):
        value = row.get("model_size")
        return "-" if value is None else f"{value:,}"

    def params_text(row):
        text = ", ".join(f"{k}={v}" for k, v in sorted(row.get("params", {}).items()))
        return text if len(text) <= 56 else text[:53] + "..."

    columns = [
        ("experiment", lambda r: str(r["experiment"])),
        ("model", lambda r: str(r.get("model") or "-")),
        ("metric", lambda r: str(r.get("metric") or "-")),
        ("trials", lambda r: str(r.get("trials") or 0)),
        ("done", lambda r: str(r.get("done") or 0)),
        ("val_score", score_text),
        ("train_time", time_text),
        ("model_size", size_text),
        ("fit_ok", lambda r: str(r.get("constraints") or "-")),
        ("best params", params_text),
    ]

    rendered = [(row, [render(row) for _, render in columns]) for row in rows]
    widths = []
    for index, (header, _) in enumerate(columns):
        longest = max([len(header)] + [len(cells[index]) for _, cells in rendered])
        widths.append(longest)

    def line(cells):
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

    header = line([name for name, _ in columns])
    separator = line(["-" * width for width in widths])
    body = [line(cells) for _, cells in rendered]
    return "\n".join([header, separator, *body])


def run_compare(root: str = "experiments", names: list[str] | None = None) -> str:
    return format_table(collect_rows(root, names))
