"""Shared paths and constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
PARAMS_PATH = PROJECT_ROOT / "params.yaml"

TARGET_COLUMN = "churn"
ID_COLUMN = "customer_id"
