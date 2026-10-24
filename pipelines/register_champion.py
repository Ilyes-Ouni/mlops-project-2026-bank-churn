"""Promote the best eligible MLflow run to the Model Registry."""

from __future__ import annotations

import json

from bank_churn.config import METRICS_DIR
from bank_churn.models.registry import register_champion


def main() -> None:
    result = register_champion()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "registry.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Load with: mlflow.sklearn.load_model({result['load_uri']!r})")


if __name__ == "__main__":
    main()
