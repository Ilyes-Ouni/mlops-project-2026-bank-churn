"""MLflow Tracking helpers for the bank-churn experiment."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import joblib
import mlflow
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from bank_churn.config import PROJECT_ROOT, load_params
from bank_churn.models.evaluate import compute_metrics
from bank_churn.models.train import split_xy, train_model

DEFAULT_EXPERIMENT = "bank-churn"
DEFAULT_CANDIDATES: tuple[dict[str, Any], ...] = (
    {"name": "logreg", "max_iter": 1000, "class_weight": "balanced"},
    {
        "name": "random_forest",
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_leaf": 5,
        "class_weight": "balanced",
    },
    {
        "name": "xgboost",
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "scale_pos_weight": 4.4,
    },
)


def tracking_uri(params: dict[str, Any] | None = None) -> str:
    """Return a SQLite tracking URI (file-store backends are deprecated in recent MLflow)."""
    cfg = (params or load_params()).get("mlflow", {})
    raw = str(cfg.get("tracking_uri", "sqlite:///mlflow.db"))
    if raw.startswith(("http://", "https://")):
        return raw
    db_path = raw.removeprefix("sqlite:///")
    path = Path(db_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.suffix != ".db":
        path.mkdir(parents=True, exist_ok=True)
        path = path / "mlflow.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.resolve().as_posix()}"


def configure_tracking(params: dict[str, Any] | None = None) -> str:
    params = params or load_params()
    uri = tracking_uri(params)
    mlflow.set_tracking_uri(uri)
    experiment = params.get("mlflow", {}).get("experiment", DEFAULT_EXPERIMENT)
    mlflow.set_experiment(experiment)
    return experiment


def log_run(
    *,
    train_frame,
    test_frame,
    numeric: list[str],
    categorical: list[str],
    model_cfg: dict[str, Any],
    experiment: str | None = None,
) -> dict[str, Any]:
    name = str(model_cfg.get("name", "model"))
    pipeline = train_model(train_frame, numeric, categorical, model_cfg)
    report = compute_metrics(pipeline, test_frame, numeric, categorical)
    metrics = report["metrics"]

    features, _ = split_xy(train_frame, numeric, categorical)
    signature = infer_signature(features.head(20), pipeline.predict_proba(features.head(20)))

    with mlflow.start_run(run_name=name):
        if experiment:
            mlflow.set_tag("experiment", experiment)
        mlflow.set_tag("dataset", "bank-customer-churn")
        mlflow.log_param("model_name", name)
        for key, value in model_cfg.items():
            mlflow.log_param(key, value)
        mlflow.log_metrics(metrics)
        mlflow.log_dict(report["confusion_matrix"], "confusion_matrix.json")
        mlflow.log_dict(report["metrics"], "metrics.json")
        _log_sklearn_model(pipeline, signature, name)

    return {"name": name, "run_metrics": metrics, "pipeline": pipeline}


def _log_sklearn_model(pipeline: Pipeline, signature, name: str) -> None:
    kwargs = {
        "signature": signature,
        "serialization_format": "cloudpickle",
    }
    try:
        mlflow.sklearn.log_model(pipeline, name="model", **kwargs)
    except TypeError:
        mlflow.sklearn.log_model(pipeline, artifact_path="model", **kwargs)
    with TemporaryDirectory() as tmp:
        joblib_path = Path(tmp) / f"{name}.joblib"
        joblib.dump(pipeline, joblib_path)
        mlflow.log_artifact(str(joblib_path), artifact_path="joblib")


def compare_models(
    train_frame,
    test_frame,
    numeric: list[str],
    categorical: list[str],
    candidates: list[dict[str, Any]] | None = None,
    params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    params = params or load_params()
    experiment = configure_tracking(params)
    specs = candidates or list(params.get("mlflow", {}).get("candidates", DEFAULT_CANDIDATES))
    results: list[dict[str, Any]] = []
    for model_cfg in specs:
        cfg = dict(model_cfg)
        cfg.setdefault("random_state", params.get("data", {}).get("random_state", 42))
        try:
            logged = log_run(
                train_frame=train_frame,
                test_frame=test_frame,
                numeric=numeric,
                categorical=categorical,
                model_cfg=cfg,
                experiment=experiment,
            )
        except ImportError:
            if cfg.get("name") in {"xgboost", "xgb"}:
                fallback = dict(cfg)
                fallback["name"] = "hist_gbdt"
                logged = log_run(
                    train_frame=train_frame,
                    test_frame=test_frame,
                    numeric=numeric,
                    categorical=categorical,
                    model_cfg=fallback,
                    experiment=experiment,
                )
            else:
                raise
        results.append({"name": logged["name"], **logged["run_metrics"]})
    return sorted(results, key=lambda row: (row["f1"], row["roc_auc"]), reverse=True)
