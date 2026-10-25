"""Split the engineered churn table into train/test (DVC preprocess stage)."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from bank_churn.config import (
    FEATURE_META_PATH,
    PROCESSED_DATA_DIR,
    TARGET_COLUMN,
    TEST_PATH,
    TRAIN_PATH,
    load_params,
)
from bank_churn.data.load import load_raw
from bank_churn.features.build import build_features


def split_train_test(
    frame: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=frame[TARGET_COLUMN],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def run_preprocess(
    params: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = params or load_params()
    data_cfg = params.get("data", {})
    raw = load_raw(data_cfg.get("raw_filename", "churn.csv"))
    engineered, meta = build_features(raw, params=params)
    train, test = split_train_test(
        engineered,
        test_size=float(data_cfg.get("test_size", 0.2)),
        random_state=int(data_cfg.get("random_state", 42)),
    )
    meta["n_train"] = int(len(train))
    meta["n_test"] = int(len(test))
    meta["test_size"] = float(data_cfg.get("test_size", 0.2))
    meta["random_state"] = int(data_cfg.get("random_state", 42))

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    FEATURE_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return train, test
