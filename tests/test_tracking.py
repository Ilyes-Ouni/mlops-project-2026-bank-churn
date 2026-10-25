from pathlib import Path

import pandas as pd

from bank_churn.features.build import build_features
from bank_churn.models.tracking import compare_models


def _tiny_churn(n: int = 40) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "customer_id": i,
                "branch_code": i % 3,
                "city": 100 + (i % 2),
                "vintage": 10 + i,
                "age": 25 + (i % 30),
                "dependents": i % 3,
                "gender": "Male" if i % 2 == 0 else "Female",
                "occupation": "salaried" if i % 3 else "self_employed",
                "customer_nw_category": 1 + (i % 3),
                "current_balance": float(i * 3),
                "previous_month_end_balance": float(i * 2),
                "average_monthly_balance_prevQ": float(i * 2.5),
                "average_monthly_balance_prevQ2": float(i * 2.2),
                "current_month_credit": float(i),
                "previous_month_credit": float(max(i - 1, 0)),
                "current_month_debit": float(i % 7),
                "previous_month_debit": float(i % 5),
                "current_month_balance": float(i * 2),
                "previous_month_balance": float(i),
                "last_transaction": "2019-12-01",
                "churn": int(i % 4 == 0),
            }
        )
    return pd.DataFrame(rows)


def test_compare_models_logs_two_runs(tmp_path: Path) -> None:
    params = {
        "data": {"random_state": 42},
        "mlflow": {
            "tracking_uri": str(tmp_path / "mlflow.db"),
            "experiment": "bank-churn-test",
            "candidates": [
                {"name": "logreg", "max_iter": 200, "class_weight": "balanced"},
                {
                    "name": "random_forest",
                    "n_estimators": 20,
                    "max_depth": 3,
                    "class_weight": "balanced",
                },
            ],
        },
    }
    frame, _ = build_features(_tiny_churn(), params={"features": {}})
    train = frame.iloc[:30].reset_index(drop=True)
    test = frame.iloc[30:].reset_index(drop=True)
    ranking = compare_models(
        train,
        test,
        numeric=[
            c
            for c in train.columns
            if c not in {"churn", "gender", "occupation", "customer_nw_category"}
        ],
        categorical=["gender", "occupation", "customer_nw_category"],
        params=params,
    )
    assert len(ranking) == 2
    assert {row["name"] for row in ranking} == {"logreg", "random_forest"}
    assert all("f1" in row and "roc_auc" in row for row in ranking)
    assert (tmp_path / "mlflow.db").exists()
