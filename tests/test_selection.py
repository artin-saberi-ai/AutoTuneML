import pytest

from autotuneml.selection import (
    is_feasible,
    objective_directions,
    pareto_front,
    select_best,
    sort_records,
    weighted_scores,
)


def record(trial, score, seconds, size, status="COMPLETE"):
    return {
        "trial": trial,
        "status": status,
        "val_score": score,
        "train_time": seconds,
        "model_size": size,
    }


DIRECTIONS = {"val_score": "maximize", "train_time": "minimize", "model_size": "minimize"}
OBJECTIVES = ["val_score", "train_time", "model_size"]


def test_directions_follow_the_metric():
    assert objective_directions(["val_score"], "accuracy")["val_score"] == "maximize"
    assert objective_directions(["val_score"], "rmse")["val_score"] == "minimize"
    assert objective_directions(["train_time"], "rmse")["train_time"] == "minimize"


def test_single_objective_picks_the_best_score():
    trials = [record(0, 0.70, 3.0, 1000), record(1, 0.92, 5.0, 9000), record(2, 0.81, 1.0, 500)]
    selection = select_best(trials, ["val_score"], {"val_score": "maximize"})
    assert selection.trial["trial"] == 1
    assert selection.rule == "single_objective"
    assert selection.feasible is True


def test_single_objective_respects_minimizing_metric():
    trials = [record(0, 12.5, 3.0, 1000), record(1, 9.1, 5.0, 9000)]
    selection = select_best(trials, ["val_score"], {"val_score": "minimize"})
    assert selection.trial["trial"] == 1


def test_incomplete_trials_are_ignored():
    trials = [
        record(0, 0.99, 1.0, 100, status="PRUNED"),
        record(1, 0.98, 1.0, 100, status="FAIL"),
        record(2, 0.70, 1.0, 100),
    ]
    selection = select_best(trials, ["val_score"], {"val_score": "maximize"})
    assert selection.trial["trial"] == 2


def test_no_completed_trials_raises():
    trials = [record(0, None, None, None, status="PRUNED")]
    with pytest.raises(ValueError, match="no completed trials"):
        select_best(trials, ["val_score"], {"val_score": "maximize"})


def test_constraints_filter_infeasible_trials():
    trials = [record(0, 0.99, 60.0, 100), record(1, 0.95, 5.0, 100)]
    constraints = {"train_time": {"max": 10.0}}
    selection = select_best(trials, ["val_score"], {"val_score": "maximize"}, constraints)
    assert selection.trial["trial"] == 1
    assert selection.feasible is True


def test_minimum_bound_constraint():
    trials = [record(0, 0.60, 1.0, 100), record(1, 0.85, 2.0, 100)]
    selection = select_best(trials, ["val_score"], {"val_score": "maximize"},
                            {"val_score": {"min": 0.8}})
    assert selection.trial["trial"] == 1


def test_all_infeasible_returns_no_trial():
    trials = [record(0, 0.99, 60.0, 100), record(1, 0.95, 40.0, 100)]
    selection = select_best(trials, ["val_score"], {"val_score": "maximize"},
                            {"train_time": {"max": 10.0}})
    assert selection.trial is None
    assert selection.feasible is False
    assert selection.rule == "infeasible"
    assert len(selection.front) > 0


def test_missing_metric_makes_trial_infeasible():
    trial = record(0, None, 1.0, 100)
    assert is_feasible(trial, {"val_score": {"min": 0.5}}) is False
    assert is_feasible(trial, {}) is True


def test_pareto_front_keeps_only_non_dominated_trials():
    trials = [
        record(0, 0.90, 10.0, 1000),   # best score, slowest, biggest
        record(1, 0.80, 1.0, 1000),    # fastest, dominated by trial 2? no: score lower, time same
        record(2, 0.85, 5.0, 500),     # middle
        record(3, 0.70, 20.0, 2000),   # dominated by everyone
    ]
    front = {t["trial"] for t in pareto_front(trials, OBJECTIVES, DIRECTIONS)}
    assert 3 not in front
    assert {0, 2} <= front


def test_pareto_front_single_objective_is_one_trial():
    trials = [record(0, 0.5, 9.0, 900), record(1, 0.9, 9.0, 900), record(2, 0.7, 9.0, 900)]
    front = pareto_front(trials, ["val_score"], {"val_score": "maximize"})
    assert [t["trial"] for t in front] == [1]


def test_weighted_sum_prefers_the_heavily_weighted_objective():
    trials = [record(0, 0.99, 30.0, 5000), record(1, 0.90, 1.0, 100)]

    accuracy_wins = weighted_scores(trials, OBJECTIVES, DIRECTIONS,
                                    {"val_score": 1.0, "train_time": 0.0, "model_size": 0.0})
    speed_wins = weighted_scores(trials, OBJECTIVES, DIRECTIONS,
                                 {"val_score": 0.0, "train_time": 1.0, "model_size": 0.0})

    assert accuracy_wins[0] > accuracy_wins[1]
    assert speed_wins[1] > speed_wins[0]


def test_weighted_sum_degrades_to_uniform_weights():
    trials = [record(0, 0.99, 30.0, 5000), record(1, 0.90, 1.0, 100)]
    uniform = weighted_scores(trials, OBJECTIVES, DIRECTIONS, {})
    explicit = weighted_scores(trials, OBJECTIVES, DIRECTIONS,
                               {"val_score": 1.0, "train_time": 1.0, "model_size": 1.0})
    assert uniform == explicit


def test_equal_metric_values_do_not_divide_by_zero():
    trials = [record(0, 0.5, 1.0, 100), record(1, 0.5, 1.0, 100)]
    scores = weighted_scores(trials, OBJECTIVES, DIRECTIONS, None)
    assert scores == [3.0, 3.0]


def test_multi_objective_selection_uses_weights():
    trials = [record(0, 0.99, 30.0, 5000), record(1, 0.90, 1.0, 100)]

    accuracy_first = select_best(trials, OBJECTIVES, DIRECTIONS, None,
                                 {"val_score": 1.0, "train_time": 0.0, "model_size": 0.0})
    speed_first = select_best(trials, OBJECTIVES, DIRECTIONS, None,
                              {"val_score": 0.0, "train_time": 1.0, "model_size": 1.0})

    assert accuracy_first.trial["trial"] == 0
    assert speed_first.trial["trial"] == 1
    assert accuracy_first.rule == "weighted_sum"


def test_sort_records_orders_by_direction():
    trials = [record(0, 0.6, 2.0, 10), record(1, 0.9, 1.0, 10), record(2, 0.7, 3.0, 10)]
    descending = sort_records(trials, "val_score", "maximize")
    ascending = sort_records(trials, "train_time", "minimize")
    assert [t["trial"] for t in descending] == [1, 2, 0]
    assert [t["trial"] for t in ascending] == [1, 0, 2]
