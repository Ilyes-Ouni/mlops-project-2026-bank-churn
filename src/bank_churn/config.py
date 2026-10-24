"""Shared paths and constants."""

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "metrics"
PARAMS_PATH = PROJECT_ROOT / "params.yaml"

TARGET_COLUMN = "churn"
ID_COLUMN = "customer_id"

TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_PATH = PROCESSED_DATA_DIR / "test.csv"
FEATURE_META_PATH = PROCESSED_DATA_DIR / "feature_meta.json"
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = METRICS_DIR / "metrics.json"
ROC_PATH = METRICS_DIR / "roc.json"
CONFUSION_PATH = METRICS_DIR / "confusion_matrix.json"


def load_params() -> dict[str, Any]:
    with PARAMS_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)
