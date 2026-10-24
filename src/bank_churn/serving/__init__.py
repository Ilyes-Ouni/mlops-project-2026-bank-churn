"""FastAPI inference service."""

from bank_churn.serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ChurnFeatures,
    PredictResponse,
)

__all__ = [
    "BatchPredictRequest",
    "BatchPredictResponse",
    "ChurnFeatures",
    "PredictResponse",
]
