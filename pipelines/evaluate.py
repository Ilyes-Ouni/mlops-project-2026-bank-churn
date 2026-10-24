"""DVC stage: score the hold-out split and write metrics/plots."""

import json

import pandas as pd
import joblib

from bank_churn.config import FEATURE_META_PATH, MODEL_PATH, TEST_PATH
from bank_churn.models.evaluate import compute_metrics, downsample_roc, persist_reports


def main() -> None:
    meta = json.loads(FEATURE_META_PATH.read_text(encoding="utf-8"))
    test = pd.read_csv(TEST_PATH)
    pipeline = joblib.load(MODEL_PATH)
    report = compute_metrics(
        pipeline,
        test,
        numeric=meta["numeric_features"],
        categorical=meta["categorical_features"],
    )
    report["roc"] = downsample_roc(report["roc"])
    persist_reports(report)
    print(json.dumps(report["metrics"]))


if __name__ == "__main__":
    main()
