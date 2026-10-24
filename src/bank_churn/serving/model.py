"""Load the serving model from a joblib path, then MLflow if needed."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

import joblib
from sklearn.pipeline import Pipeline

from bank_churn.config import MODEL_PATH

_MODEL: Pipeline | None = None
_LOCK = Lock()


def resolve_model_path() -> Path | None:
    env = os.getenv("BANK_CHURN_MODEL_PATH")
    if env:
        path = Path(env)
        return path if path.exists() else None
    if MODEL_PATH.exists():
        return MODEL_PATH
    return None


def load_serving_model(force: bool = False) -> Pipeline:
    global _MODEL
    if _MODEL is not None and not force:
        return _MODEL
    with _LOCK:
        if _MODEL is not None and not force:
            return _MODEL
        path = resolve_model_path()
        if path is not None:
            _MODEL = joblib.load(path)
            return _MODEL
        from bank_churn.models.registry import load_champion

        _MODEL = load_champion()
        return _MODEL


def reset_serving_model() -> None:
    global _MODEL
    _MODEL = None


def predict_proba(frame) -> list[float]:
    model = load_serving_model()
    return [float(p) for p in model.predict_proba(frame)[:, 1]]
