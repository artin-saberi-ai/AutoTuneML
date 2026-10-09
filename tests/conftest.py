import pandas as pd
import pytest
from sklearn.datasets import make_classification

from autotuneml.config import from_dict


@pytest.fixture
def root(tmp_path):
    return tmp_path / "experiments"


@pytest.fixture
def classification_csv(tmp_path):
    X, y = make_classification(n_samples=120, n_features=6, n_informative=4, random_state=0)
    frame = pd.DataFrame(X, columns=[f"f{index}" for index in range(X.shape[1])])
    frame["target"] = y
    path = tmp_path / "toy.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def make_config():
    def make(name="exp", source="sklearn:iris", task="classification", metric="accuracy",
             seed=5, data=None, model=None, search=None, constraints=None, selection=None):
        raw = {
            "name": name,
            "seed": seed,
            "task": task,
            "metric": metric,
            "data": data if data is not None else {"source": source, "test_size": 0.25},
            "model": model if model is not None else {"name": "logistic_regression"},
            "search": {"n_trials": 2, "cv_folds": 3, **(search or {})},
        }
        if constraints is not None:
            raw["constraints"] = constraints
        if selection is not None:
            raw["selection"] = selection
        return from_dict(raw)

    return make
