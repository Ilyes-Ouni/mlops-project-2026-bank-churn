"""Train the champion sklearn pipeline (RandomForest by default)."""

from __future__ import annotations

from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bank_churn.config import MODEL_PATH, TARGET_COLUMN


def make_estimator(model_cfg: dict[str, Any]) -> Any:
    name = str(model_cfg.get("name", "random_forest"))
    random_state = int(model_cfg.get("random_state", 42))
    if name == "logreg":
        return LogisticRegression(
            max_iter=int(model_cfg.get("max_iter", 1000)),
            class_weight=model_cfg.get("class_weight", "balanced"),
            random_state=random_state,
        )
    if name != "random_forest":
        raise ValueError(f"Unsupported model.name={name!r} for the DVC train stage")
    return RandomForestClassifier(
        n_estimators=int(model_cfg.get("n_estimators", 200)),
        max_depth=int(model_cfg.get("max_depth", 8)),
        min_samples_leaf=int(model_cfg.get("min_samples_leaf", 5)),
        class_weight=model_cfg.get("class_weight", "balanced"),
        n_jobs=-1,
        random_state=random_state,
    )


def build_pipeline(
    numeric: list[str],
    categorical: list[str],
    model_cfg: dict[str, Any],
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", make_estimator(model_cfg)),
        ]
    )


def split_xy(
    frame: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    features = frame.loc[:, [*numeric, *categorical]]
    target = frame[TARGET_COLUMN].astype(int)
    return features, target


def train_model(
    train_frame: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    model_cfg: dict[str, Any],
) -> Pipeline:
    features, target = split_xy(train_frame, numeric, categorical)
    pipeline = build_pipeline(numeric, categorical, model_cfg)
    pipeline.fit(features, target)
    return pipeline


def save_model(pipeline: Pipeline, path=MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
