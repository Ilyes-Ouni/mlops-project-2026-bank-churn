"""Training, evaluation, and MLflow registry helpers."""

from bank_churn.models.evaluate import compute_metrics, persist_reports
from bank_churn.models.registry import load_champion, register_champion, select_champion
from bank_churn.models.tracking import compare_models, configure_tracking
from bank_churn.models.train import build_pipeline, save_model, train_model

__all__ = [
    "build_pipeline",
    "compare_models",
    "compute_metrics",
    "configure_tracking",
    "load_champion",
    "persist_reports",
    "register_champion",
    "save_model",
    "select_champion",
    "train_model",
]
