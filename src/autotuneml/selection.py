from __future__ import annotations

from dataclasses import dataclass, field

from .metrics import MAXIMIZE, MINIMIZE, metric_direction

OBJECTIVE_DIRECTIONS = {
    "val_score": MAXIMIZE,
    "train_time": MINIMIZE,
    "model_size": MINIMIZE,
}

OBJECTIVE_LABELS = {
    "val_score": "validation score",
    "train_time": "train time (s)",
    "model_size": "model size (bytes)",
}


def objective_directions(objectives: list[str], metric: str) -> dict[str, str]:
    directions = {}
    for objective in objectives:
        if objective == "val_score":
            directions[objective] = metric_direction(metric)
        else:
            directions[objective] = OBJECTIVE_DIRECTIONS[objective]
    return directions


@dataclass
class Selection:
    trial: dict | None
    feasible: bool
    front: list[dict] = field(default_factory=list)
    rule: str = "single_objective"


def is_feasible(record: dict, constraints: dict) -> bool:
    for metric, spec in (constraints or {}).items():
        value = record.get(metric)
        if value is None:
            return False
        if "min" in spec and value < spec["min"]:
            return False
        if "max" in spec and value > spec["max"]:
            return False
    return True


def pareto_front(trials: list[dict], objectives: list[str], directions: dict[str, str]) -> list[dict]:
    front = []
    for candidate in trials:
        dominated = False
        for other in trials:
            if other is candidate:
                continue
            if _dominates(other, candidate, objectives, directions):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front


def _dominates(a: dict, b: dict, objectives: list[str], directions: dict[str, str]) -> bool:
    better_or_equal, strictly_better = True, False
    for objective in objectives:
        av, bv = a.get(objective), b.get(objective)
        if av is None or bv is None:
            return False
        left, right = (av, bv) if directions[objective] == MAXIMIZE else (bv, av)
        if left < right:
            better_or_equal = False
        elif left > right:
            strictly_better = True
    return better_or_equal and strictly_better


def _normalized(records: list[dict], objective: str, direction: str) -> list[float]:
    values = [float(r[objective]) for r in records]
    low, high = min(values), max(values)
    if high == low:
        return [1.0 for _ in values]
    if direction == MAXIMIZE:
        return [(v - low) / (high - low) for v in values]
    return [(high - v) / (high - low) for v in values]


def weighted_scores(records: list[dict], objectives: list[str], directions: dict[str, str],
                    weights: dict[str, float] | None) -> list[float]:
    active_weights = {o: float((weights or {}).get(o, 1.0)) for o in objectives}
    if not any(active_weights.values()):
        active_weights = {o: 1.0 for o in objectives}

    normalized = {o: _normalized(records, o, directions[o]) for o in objectives}
    scores = []
    for index in range(len(records)):
        total = sum(active_weights[o] * normalized[o][index] for o in objectives)
        scores.append(total)
    return scores


def select_best(trials: list[dict], objectives: list[str], directions: dict[str, str],
                constraints: dict | None = None, weights: dict[str, float] | None = None) -> Selection:
    """Picks the trial that wins on the configured objectives while satisfying the
    configured constraints. Infeasible trials are never selected."""
    complete = [
        t for t in trials
        if t.get("status") == "COMPLETE" and all(t.get(o) is not None for o in objectives)
    ]
    if not complete:
        raise ValueError("no completed trials are available for selection")

    front = pareto_front(complete, objectives, directions)
    feasible = [t for t in complete if is_feasible(t, constraints or {})]
    if not feasible:
        return Selection(trial=None, feasible=False, front=front, rule="infeasible")

    if len(objectives) == 1:
        objective = objectives[0]
        if directions[objective] == MAXIMIZE:
            best = max(feasible, key=lambda r: float(r[objective]))
        else:
            best = min(feasible, key=lambda r: float(r[objective]))
        return Selection(trial=best, feasible=True, front=front, rule="single_objective")

    scores = weighted_scores(feasible, objectives, directions, weights)
    best = feasible[scores.index(max(scores))]
    return Selection(trial=best, feasible=True, front=front, rule="weighted_sum")


def sort_records(trials: list[dict], objective: str = "val_score",
                 direction: str = MAXIMIZE) -> list[dict]:
    records = [t for t in trials if t.get(objective) is not None]
    reverse = direction == MAXIMIZE
    return sorted(records, key=lambda r: float(r[objective]), reverse=reverse)
