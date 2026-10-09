from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SKLEARN_SOURCES = {
    "breast_cancer": load_breast_cancer,
    "wine": load_wine,
    "diabetes": load_diabetes,
    "digits": load_digits,
    "iris": load_iris,
}


@dataclass
class ColumnSpec:
    numeric: list[str]
    categorical: list[str]
    target: str
    impute: str = "median"
    scale: bool = True


@dataclass
class Holdout:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    columns: ColumnSpec

    @property
    def train_size(self) -> int:
        return len(self.X_train)

    @property
    def test_size(self) -> int:
        return len(self.X_test)


def load_frame(source: str) -> tuple[pd.DataFrame, str]:
    """Returns the full dataset frame and the name of its target column."""
    if source.startswith("sklearn:"):
        name = source.split(":", 1)[1]
        if name not in SKLEARN_SOURCES:
            known = ", ".join(sorted(SKLEARN_SOURCES))
            raise ValueError(f"unknown sklearn dataset '{name}' (available: {known})")
        bunch = SKLEARN_SOURCES[name]()
        frame = pd.DataFrame(bunch.data, columns=bunch.feature_names)
        frame["target"] = bunch.target
        return frame, "target"

    path = Path(source)
    if not path.exists():
        raise ValueError(f"data source not found: {source}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path), None
    return pd.read_csv(path), None


def build_columns(frame: pd.DataFrame, target: str | None, impute: str, scale: bool) -> ColumnSpec:
    if target is None:
        if "target" in frame.columns:
            target = "target"
        else:
            raise ValueError("no target column configured and none found in the dataset")
    if target not in frame.columns:
        raise ValueError(f"target column '{target}' is not present in the dataset")

    numeric, categorical = [], []
    for column in frame.columns:
        if column == target:
            continue
        dtype = frame[column].dtype
        if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
            numeric.append(column)
        else:
            categorical.append(column)

    if not numeric and not categorical:
        raise ValueError("dataset has no feature columns")

    return ColumnSpec(
        numeric=numeric,
        categorical=categorical,
        target=target,
        impute=impute,
        scale=scale,
    )


def make_holdout(
    frame: pd.DataFrame,
    columns: ColumnSpec,
    test_size: float,
    seed: int,
    task: str,
) -> Holdout:
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

    feature_frame = frame[columns.numeric + columns.categorical]
    target = frame[columns.target]
    stratify = target if task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        feature_frame,
        target,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    return Holdout(X_train, X_test, y_train, y_test, columns)


def prepare(source: str, target: str | None, test_size: float, seed: int, task: str,
            impute: str = "median", scale: bool = True) -> Holdout:
    frame, frame_target = load_frame(source)
    columns = build_columns(frame, target or frame_target, impute, scale)
    return make_holdout(frame, columns, test_size, seed, task)


def build_preprocessor(columns: ColumnSpec) -> ColumnTransformer:
    transformers = []
    if columns.numeric:
        steps = [("impute", SimpleImputer(strategy=columns.impute))]
        if columns.scale:
            steps.append(("scale", StandardScaler()))
        transformers.append(("num", Pipeline(steps), columns.numeric))
    if columns.categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            columns.categorical,
        ))
    if not transformers:
        raise ValueError("no columns available for preprocessing")
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def early_stopping_split(y: pd.Series | np.ndarray, task: str, seed: int,
                         fraction: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Splits training indices into a fit part and a small inner validation part
    used only to watch early stopping, never to report a score."""
    n = len(y)
    n_val = max(2, int(math.ceil(n * fraction)))
    if n - n_val < 20:
        return np.arange(n), np.array([], dtype=int)

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    if task == "classification" and len(np.unique(y)) > 1:
        val = []
        classes = np.unique(y)
        per_class = max(1, n_val // len(classes))
        y_arr = np.asarray(y)
        for cls in classes:
            cls_idx = order[y_arr[order] == cls]
            val.extend(cls_idx[:per_class])
        val = np.array(sorted(set(val))[:n_val], dtype=int)
        if len(val) < 2:
            val = order[:n_val]
    else:
        val = order[:n_val]
    val_set = set(int(i) for i in val)
    train = np.array([i for i in range(n) if i not in val_set], dtype=int)
    return train, val


def save_holdout(holdout: Holdout, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    train = holdout.X_train.copy()
    train[holdout.columns.target] = holdout.y_train
    test = holdout.X_test.copy()
    test[holdout.columns.target] = holdout.y_test
    train.to_csv(directory / "train.csv", index=True, index_label="row_id")
    test.to_csv(directory / "test.csv", index=True, index_label="row_id")


def load_saved_holdout(directory: Path, target: str, impute: str = "median",
                       scale: bool = True) -> Holdout:
    train = pd.read_csv(directory / "train.csv", index_col="row_id")
    test = pd.read_csv(directory / "test.csv", index_col="row_id")
    columns = build_columns(train, target, impute, scale)
    return Holdout(
        X_train=train.drop(columns=[target]),
        X_test=test.drop(columns=[target]),
        y_train=train[target],
        y_test=test[target],
        columns=columns,
    )
