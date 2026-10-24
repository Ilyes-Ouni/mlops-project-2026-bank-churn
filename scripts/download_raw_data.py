"""Download the Bank Customer Churn raw CSV (Kaggle mirror)."""

from __future__ import annotations

import urllib.request
from pathlib import Path

# Same file as Kaggle: pentakrishnakishore/bank-customer-churn-data
SOURCE_URL = (
    "https://raw.githubusercontent.com/krishnakishore999/"
    "Churn_Prediction/main/churn_prediction.csv"
)
KAGGLE_PAGE = "https://www.kaggle.com/datasets/pentakrishnakishore/bank-customer-churn-data"

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw" / "churn.csv"


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from {SOURCE_URL}")
    print(f"Kaggle card: {KAGGLE_PAGE}")
    urllib.request.urlretrieve(SOURCE_URL, DEST)
    print(f"Wrote {DEST} ({DEST.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
