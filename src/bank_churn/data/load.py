"""Load raw churn data tracked by DVC."""

from pathlib import Path

import pandas as pd

from bank_churn.config import RAW_DATA_DIR


def raw_csv_path(filename: str = "churn.csv") -> Path:
    return RAW_DATA_DIR / filename


def load_raw(filename: str = "churn.csv") -> pd.DataFrame:
    path = raw_csv_path(filename)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python scripts/download_raw_data.py` then `dvc checkout`."
        )
    return pd.read_csv(path)
