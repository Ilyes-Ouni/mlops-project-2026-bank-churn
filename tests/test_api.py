import joblib
import pytest
from fastapi.testclient import TestClient

from bank_churn.features.build import build_features
from bank_churn.models.train import train_model
from bank_churn.serving.app import app
from bank_churn.serving.model import reset_serving_model
from helpers import tiny_churn_frame, valid_payload


@pytest.fixture()
def client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    engineered, _ = build_features(tiny_churn_frame(), params={"features": {}})
    numeric = [
        c
        for c in engineered.columns
        if c not in {"churn", "gender", "occupation", "customer_nw_category"}
    ]
    categorical = ["gender", "occupation", "customer_nw_category"]
    pipeline = train_model(
        engineered,
        numeric,
        categorical,
        {"name": "logreg", "max_iter": 200, "class_weight": "balanced"},
    )
    path = tmp_path / "model.joblib"
    joblib.dump(pipeline, path)
    monkeypatch.setenv("BANK_CHURN_MODEL_PATH", str(path))
    reset_serving_model()
    with TestClient(app) as test_client:
        yield test_client
    reset_serving_model()


def test_health_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_valid_payload(client: TestClient) -> None:
    response = client.post("/predict", json=valid_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["churn_label"] in {0, 1}
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["latency_ms"] is not None


def test_predict_rejects_bad_schema(client: TestClient) -> None:
    payload = valid_payload()
    payload["age"] = -3
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_batch(client: TestClient) -> None:
    response = client.post("/predict/batch", json={"instances": [valid_payload(), valid_payload()]})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2
