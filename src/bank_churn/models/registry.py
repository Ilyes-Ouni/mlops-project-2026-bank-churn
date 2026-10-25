"""MLflow Model Registry: pick, promote, and load the champion."""

from __future__ import annotations

from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from bank_churn.config import load_params
from bank_churn.models.tracking import configure_tracking

DEFAULT_MODEL_NAME = "bank-churn"
DEFAULT_MIN_F1 = 0.55
DEFAULT_MIN_ROC_AUC = 0.80
DEFAULT_ALIAS = "champion"


def registry_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or load_params()
    cfg = params.get("registry", {})
    return {
        "model_name": cfg.get("model_name", DEFAULT_MODEL_NAME),
        "min_f1": float(cfg.get("min_f1", DEFAULT_MIN_F1)),
        "min_roc_auc": float(cfg.get("min_roc_auc", DEFAULT_MIN_ROC_AUC)),
        "alias": cfg.get("alias", DEFAULT_ALIAS),
        "experiment": params.get("mlflow", {}).get("experiment", "bank-churn"),
    }


def select_champion(
    rows: list[dict[str, Any]],
    min_f1: float = DEFAULT_MIN_F1,
    min_roc_auc: float = DEFAULT_MIN_ROC_AUC,
) -> dict[str, Any] | None:
    """Highest F1 among runs that clear both synopsis thresholds."""
    eligible = [
        row
        for row in rows
        if float(row.get("f1", 0)) >= min_f1
        and float(row.get("roc_auc", 0)) >= min_roc_auc
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda row: (float(row["f1"]), float(row["roc_auc"])))


def list_scored_runs(experiment: str) -> list[dict[str, Any]]:
    client = MlflowClient()
    stored = client.get_experiment_by_name(experiment)
    if stored is None:
        return []
    runs = client.search_runs(
        [stored.experiment_id],
        order_by=["metrics.f1 DESC"],
        max_results=50,
    )
    rows: list[dict[str, Any]] = []
    for run in runs:
        if "f1" not in run.data.metrics or "roc_auc" not in run.data.metrics:
            continue
        rows.append(
            {
                "run_id": run.info.run_id,
                "name": run.data.params.get("model_name") or run.info.run_name,
                "f1": float(run.data.metrics["f1"]),
                "roc_auc": float(run.data.metrics["roc_auc"]),
            }
        )
    return rows


def _promote(
    client: MlflowClient, model_name: str, version: str, alias: str
) -> list[str]:
    path: list[str] = []
    for stage in ("Staging", "Production"):
        try:
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=(stage == "Production"),
            )
            path.append(stage)
        except Exception:
            # MLflow 3 prefers aliases over stages.
            continue
    try:
        client.set_registered_model_alias(model_name, alias, version)
        path.append(f"alias:{alias}")
    except Exception:
        pass
    if "Production" not in path:
        try:
            client.set_registered_model_alias(model_name, "production", version)
            path.append("alias:production")
        except Exception:
            pass
    return path


def register_champion(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or load_params()
    configure_tracking(params)
    cfg = registry_config(params)
    rows = list_scored_runs(cfg["experiment"])
    winner = select_champion(rows, min_f1=cfg["min_f1"], min_roc_auc=cfg["min_roc_auc"])
    if winner is None:
        raise ValueError(
            f"No run meets F1>={cfg['min_f1']} and ROC-AUC>={cfg['min_roc_auc']}. "
            "Run `python pipelines/mlflow_compare.py` first."
        )

    model_uri = f"runs:/{winner['run_id']}/model"
    registered = mlflow.register_model(model_uri, cfg["model_name"])
    client = MlflowClient()
    client.update_registered_model(
        name=cfg["model_name"],
        description=(
            "Bank churn champion. Promotion rule: highest F1 among runs with "
            f"F1 >= {cfg['min_f1']} and ROC-AUC >= {cfg['min_roc_auc']}."
        ),
    )
    client.update_model_version(
        name=cfg["model_name"],
        version=str(registered.version),
        description=(
            f"{winner['name']} | F1={winner['f1']:.4f} | ROC-AUC={winner['roc_auc']:.4f}"
        ),
    )
    path = _promote(client, cfg["model_name"], str(registered.version), cfg["alias"])
    return {
        "model_name": cfg["model_name"],
        "version": str(registered.version),
        "run_id": winner["run_id"],
        "candidate": winner["name"],
        "f1": winner["f1"],
        "roc_auc": winner["roc_auc"],
        "promotion": path,
        "load_uri": f"models:/{cfg['model_name']}@{cfg['alias']}",
    }


def load_champion(params: dict[str, Any] | None = None):
    """Load the Production / alias champion for serving."""
    params = params or load_params()
    configure_tracking(params)
    cfg = registry_config(params)
    uris = (
        f"models:/{cfg['model_name']}@{cfg['alias']}",
        f"models:/{cfg['model_name']}@production",
        f"models:/{cfg['model_name']}/Production",
    )
    last_error: Exception | None = None
    for uri in uris:
        try:
            return mlflow.sklearn.load_model(uri)
        except Exception as exc:  # noqa: BLE001 — try the next serving URI
            last_error = exc
    raise RuntimeError(
        f"Could not load champion {cfg['model_name']}. "
        "Register it with `python pipelines/register_champion.py`."
    ) from last_error
