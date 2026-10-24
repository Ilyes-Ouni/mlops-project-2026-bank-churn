"""DVC stage: fit the sklearn pipeline on the train split."""

import json

import pandas as pd

from bank_churn.config import FEATURE_META_PATH, TRAIN_PATH, load_params
from bank_churn.models.train import save_model, train_model


def main() -> None:
    params = load_params()
    meta = json.loads(FEATURE_META_PATH.read_text(encoding="utf-8"))
    train = pd.read_csv(TRAIN_PATH)
    model_cfg = dict(params.get("model", {}))
    model_cfg.setdefault("random_state", params.get("data", {}).get("random_state", 42))
    pipeline = train_model(
        train,
        numeric=meta["numeric_features"],
        categorical=meta["categorical_features"],
        model_cfg=model_cfg,
    )
    save_model(pipeline)
    print(f"Trained {model_cfg.get('name', 'random_forest')} on {len(train)} rows")


if __name__ == "__main__":
    main()
