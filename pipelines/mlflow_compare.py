"""Log LogReg / RandomForest / XGBoost runs to the local MLflow store."""

from __future__ import annotations

import json

import pandas as pd

from bank_churn.config import FEATURE_META_PATH, METRICS_DIR, TEST_PATH, TRAIN_PATH, load_params
from bank_churn.data.preprocess import run_preprocess
from bank_churn.models.tracking import compare_models, tracking_uri


def _load_splits():
    if not TRAIN_PATH.exists() or not TEST_PATH.exists() or not FEATURE_META_PATH.exists():
        run_preprocess()
    meta = json.loads(FEATURE_META_PATH.read_text(encoding="utf-8"))
    return pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH), meta


def main() -> None:
    params = load_params()
    train, test, meta = _load_splits()
    ranking = compare_models(
        train,
        test,
        numeric=meta["numeric_features"],
        categorical=meta["categorical_features"],
        params=params,
    )
    payload = {"tracking_uri": tracking_uri(params), "ranking": ranking}
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "mlflow_ranking.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("Open the UI with: mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")
    print("Then browse http://127.0.0.1:5000 (experiment bank-churn)")


if __name__ == "__main__":
    main()
