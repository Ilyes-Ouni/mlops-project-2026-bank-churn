"""FastAPI service: /health, /predict, /predict/batch."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from bank_churn.serving.model import (
    load_serving_model,
    predict_proba,
    resolve_model_path,
)
from bank_churn.serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ChurnFeatures,
    PredictResponse,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        load_serving_model()
    except Exception:
        pass
    yield


app = FastAPI(
    title="Bank Churn API",
    description="Inference for bank customer churn. Target latency < 150 ms per request.",
    version="0.1.0",
    lifespan=lifespan,
)


def _to_response(probability: float) -> PredictResponse:
    return PredictResponse(
        churn_probability=round(probability, 6),
        churn_label=1 if probability >= 0.5 else 0,
    )


@app.get("/health")
def health() -> dict[str, object]:
    loaded = True
    try:
        load_serving_model()
    except Exception:
        loaded = False
    return {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "model_path": (
            str(resolve_model_path())
            if resolve_model_path()
            else "mlflow://bank-churn@champion"
        ),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: ChurnFeatures) -> PredictResponse:
    started = time.perf_counter()
    try:
        probability = predict_proba(payload.to_dataframe())[0]
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Model unavailable: {exc}"
        ) from exc
    response = _to_response(probability)
    response.latency_ms = round((time.perf_counter() - started) * 1000, 2)
    return response


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
    try:
        scores = predict_proba(payload.to_dataframe())
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Model unavailable: {exc}"
        ) from exc
    return BatchPredictResponse(predictions=[_to_response(score) for score in scores])
