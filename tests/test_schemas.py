import pytest
from pydantic import ValidationError

from bank_churn.serving.schemas import BatchPredictRequest, ChurnFeatures, PredictResponse


def _valid_payload() -> dict:
    return {
        "vintage": 20,
        "age": 41,
        "dependents": 1,
        "current_balance": 1200.5,
        "previous_month_end_balance": 1100.0,
        "average_monthly_balance_prevQ": 1150.0,
        "average_monthly_balance_prevQ2": 1080.0,
        "current_month_credit": 200.0,
        "previous_month_credit": 180.0,
        "current_month_debit": 90.0,
        "previous_month_debit": 70.0,
        "current_month_balance": 1180.0,
        "previous_month_balance": 1100.0,
        "recency_days": 12,
        "gender": "Male",
        "occupation": "salaried",
        "customer_nw_category": 2,
    }


def test_valid_payload_accepts_numeric_category() -> None:
    features = ChurnFeatures.model_validate(_valid_payload())
    frame = features.to_dataframe()
    assert list(frame.columns)[-3:] == ["gender", "occupation", "customer_nw_category"]
    assert features.customer_nw_category == "2"


def test_missing_required_field_is_rejected() -> None:
    payload = _valid_payload()
    del payload["current_balance"]
    with pytest.raises(ValidationError):
        ChurnFeatures.model_validate(payload)


def test_out_of_range_age_is_rejected() -> None:
    payload = _valid_payload()
    payload["age"] = 200
    with pytest.raises(ValidationError):
        ChurnFeatures.model_validate(payload)


def test_unknown_keys_are_rejected() -> None:
    payload = _valid_payload()
    payload["customer_id"] = 99
    with pytest.raises(ValidationError):
        ChurnFeatures.model_validate(payload)


def test_null_gender_becomes_unknown() -> None:
    payload = _valid_payload()
    payload["gender"] = None
    assert ChurnFeatures.model_validate(payload).gender == "unknown"


def test_batch_requires_at_least_one_instance() -> None:
    with pytest.raises(ValidationError):
        BatchPredictRequest.model_validate({"instances": []})
    batch = BatchPredictRequest.model_validate({"instances": [_valid_payload(), _valid_payload()]})
    assert len(batch.to_dataframe()) == 2


def test_predict_response_bounds() -> None:
    ok = PredictResponse(churn_probability=0.62, churn_label=1)
    assert ok.model_name == "bank-churn"
    with pytest.raises(ValidationError):
        PredictResponse(churn_probability=1.5, churn_label=1)
