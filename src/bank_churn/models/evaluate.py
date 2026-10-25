"""Evaluate a fitted pipeline and write DVC metrics / plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from bank_churn.config import CONFUSION_PATH, METRICS_PATH, ROC_PATH
from bank_churn.models.train import split_xy


def compute_metrics(
    pipeline: Pipeline,
    test_frame: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
) -> dict[str, Any]:
    features, target = split_xy(test_frame, numeric, categorical)
    predicted = pipeline.predict(features)
    if hasattr(pipeline, "predict_proba"):
        scores = pipeline.predict_proba(features)[:, 1]
    else:
        scores = predicted.astype(float)

    fpr, tpr, _ = roc_curve(target, scores)
    matrix = confusion_matrix(target, predicted, labels=[0, 1])
    return {
        "metrics": {
            "f1": float(f1_score(target, predicted, zero_division=0)),
            "roc_auc": float(roc_auc_score(target, scores)),
            "precision": float(precision_score(target, predicted, zero_division=0)),
            "recall": float(recall_score(target, predicted, zero_division=0)),
            "n_test": int(len(target)),
        },
        "roc": [
            {"fpr": float(fp), "tpr": float(tp)}
            for fp, tp in zip(fpr, tpr, strict=True)
        ],
        "confusion_matrix": [
            {"actual": 0, "predicted": 0, "count": int(matrix[0, 0])},
            {"actual": 0, "predicted": 1, "count": int(matrix[0, 1])},
            {"actual": 1, "predicted": 0, "count": int(matrix[1, 0])},
            {"actual": 1, "predicted": 1, "count": int(matrix[1, 1])},
        ],
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def persist_reports(report: dict[str, Any]) -> None:
    write_json(METRICS_PATH, report["metrics"])
    write_json(ROC_PATH, report["roc"])
    write_json(CONFUSION_PATH, report["confusion_matrix"])


def downsample_roc(
    points: list[dict[str, float]], max_points: int = 200
) -> list[dict[str, float]]:
    if len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points, dtype=int)
    return [points[i] for i in idx]
