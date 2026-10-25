import pandas as pd

from bank_churn.features.build import build_features
from bank_churn.models.evaluate import compute_metrics, downsample_roc
from bank_churn.models.train import make_estimator, train_model


def _tiny() -> pd.DataFrame:
    rows = []
    for i in range(24):
        rows.append(
            {
                "customer_id": i,
                "branch_code": 1,
                "city": 10,
                "vintage": 10 + i,
                "age": 30 + i,
                "dependents": i % 3,
                "gender": "Male" if i % 2 == 0 else "Female",
                "occupation": "salaried",
                "customer_nw_category": 2,
                "current_balance": float(i * 10),
                "previous_month_end_balance": float(i * 9),
                "average_monthly_balance_prevQ": float(i * 8),
                "average_monthly_balance_prevQ2": float(i * 7),
                "current_month_credit": float(i),
                "previous_month_credit": float(i),
                "current_month_debit": float(i % 5),
                "previous_month_debit": float(i % 4),
                "current_month_balance": float(i * 8),
                "previous_month_balance": float(i * 7),
                "last_transaction": "2019-12-01",
                "churn": int(i % 3 == 0),
            }
        )
    return pd.DataFrame(rows)


def test_make_estimator_logreg_and_forest() -> None:
    assert make_estimator({"name": "logreg"}).__class__.__name__ == "LogisticRegression"
    assert (
        make_estimator({"name": "random_forest", "n_estimators": 10}).n_estimators == 10
    )


def test_train_and_evaluate_on_tiny_frame() -> None:
    engineered, _ = build_features(_tiny(), params={"features": {}})
    numeric = [
        c
        for c in engineered.columns
        if c not in {"churn", "gender", "occupation", "customer_nw_category"}
    ]
    categorical = ["gender", "occupation", "customer_nw_category"]
    train = engineered.iloc[:16]
    test = engineered.iloc[16:]
    pipeline = train_model(
        train,
        numeric,
        categorical,
        {"name": "logreg", "max_iter": 200, "class_weight": "balanced"},
    )
    report = compute_metrics(pipeline, test, numeric, categorical)
    assert 0.0 <= report["metrics"]["f1"] <= 1.0
    assert 0.0 <= report["metrics"]["roc_auc"] <= 1.0
    assert report["metrics"]["n_test"] == len(test)
    assert len(report["confusion_matrix"]) == 4


def test_downsample_roc_keeps_endpoints() -> None:
    points = [{"fpr": i / 400, "tpr": i / 400} for i in range(401)]
    slim = downsample_roc(points, max_points=20)
    assert len(slim) == 20
    assert slim[0] == points[0]
    assert slim[-1] == points[-1]
