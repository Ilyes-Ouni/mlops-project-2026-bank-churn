import pandas as pd

from bank_churn.config import TARGET_COLUMN, load_params
from bank_churn.data.preprocess import split_train_test
from bank_churn.features.build import add_recency_days, build_features, clean_raw


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "branch_code": [10, 11, 12, 13],
            "city": [100, None, 102, 103],
            "vintage": [10, 20, 30, 40],
            "age": [25, 40, 55, 33],
            "dependents": [None, 2, 0, 1],
            "gender": ["Male", None, "Female", "Male"],
            "occupation": ["salaried", "self_employed", None, "student"],
            "customer_nw_category": [2, 3, 1, 2],
            "current_balance": [100.0, 50.0, 0.0, 80.0],
            "previous_month_end_balance": [90.0, 40.0, 5.0, 70.0],
            "average_monthly_balance_prevQ": [95.0, 45.0, 4.0, 75.0],
            "average_monthly_balance_prevQ2": [88.0, 42.0, 6.0, 72.0],
            "current_month_credit": [10.0, 5.0, 0.0, 8.0],
            "previous_month_credit": [8.0, 4.0, 1.0, 7.0],
            "current_month_debit": [3.0, 20.0, 40.0, 5.0],
            "previous_month_debit": [2.0, 15.0, 30.0, 4.0],
            "current_month_balance": [97.0, 35.0, -10.0, 83.0],
            "previous_month_balance": [90.0, 40.0, 5.0, 70.0],
            "last_transaction": ["2019-12-01", "NaT", "2019-11-01", "2019-12-15"],
            "churn": [0, 1, 1, 0],
        }
    )


def test_clean_raw_imputes_known_gaps() -> None:
    cleaned = clean_raw(_sample_frame())
    assert cleaned["dependents"].isna().sum() == 0
    assert cleaned.loc[1, "gender"] == "unknown"
    assert cleaned.loc[2, "occupation"] == "unknown"


def test_recency_days_uses_shared_reference() -> None:
    frame, reference = add_recency_days(_sample_frame())
    assert reference == pd.Timestamp("2019-12-15")
    assert frame.loc[0, "recency_days"] == 14
    assert frame["recency_days"].isna().sum() == 0


def test_build_features_drops_identifiers() -> None:
    params = load_params()
    engineered, meta = build_features(_sample_frame(), params=params)
    assert "customer_id" not in engineered.columns
    assert "last_transaction" not in engineered.columns
    assert "recency_days" in engineered.columns
    assert TARGET_COLUMN in engineered.columns
    assert meta["n_rows"] == 4


def test_stratified_split_is_deterministic() -> None:
    frame = pd.concat([_sample_frame(), _sample_frame()], ignore_index=True)
    first_train, first_test = split_train_test(frame, test_size=0.25, random_state=42)
    second_train, second_test = split_train_test(frame, test_size=0.25, random_state=42)
    pd.testing.assert_frame_equal(first_train, second_train)
    pd.testing.assert_frame_equal(first_test, second_test)
    assert set(first_test["churn"]) <= {0, 1}
