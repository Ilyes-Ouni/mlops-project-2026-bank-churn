"""Feature engineering for bank customer churn."""

from __future__ import annotations

from typing import Any

import pandas as pd

from bank_churn.config import TARGET_COLUMN

DEFAULT_DROP = ("customer_id", "branch_code", "city")
DEFAULT_CATEGORICAL = ("gender", "occupation", "customer_nw_category")
DEFAULT_NUMERIC_RAW = (
    "vintage",
    "age",
    "dependents",
    "current_balance",
    "previous_month_end_balance",
    "average_monthly_balance_prevQ",
    "average_monthly_balance_prevQ2",
    "current_month_credit",
    "previous_month_credit",
    "current_month_debit",
    "previous_month_debit",
    "current_month_balance",
    "previous_month_balance",
)


def clean_raw(frame: pd.DataFrame) -> pd.DataFrame:
    """Impute known missingness from the EDA note."""
    out = frame.copy()
    if "dependents" in out.columns:
        out["dependents"] = pd.to_numeric(out["dependents"], errors="coerce").fillna(0)
    for column in ("gender", "occupation"):
        if column in out.columns:
            out[column] = out[column].fillna("unknown").astype(str).str.strip()
            out.loc[out[column].isin(("", "nan", "None")), column] = "unknown"
    if "customer_nw_category" in out.columns:
        out["customer_nw_category"] = out["customer_nw_category"].astype(str)
    return out


def add_recency_days(
    frame: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Turn `last_transaction` into days since a fixed reference date."""
    out = frame.copy()
    dates = pd.to_datetime(out["last_transaction"], errors="coerce")
    if reference_date is None:
        reference_date = dates.max()
        if pd.isna(reference_date):
            reference_date = pd.Timestamp("2019-12-31")
    recency = (reference_date - dates).dt.days
    fill_value = float(recency.median()) if recency.notna().any() else 0.0
    out["recency_days"] = recency.fillna(fill_value)
    return out, pd.Timestamp(reference_date)


def drop_unused(frame: pd.DataFrame, drop_columns: list[str] | None = None) -> pd.DataFrame:
    columns = list(drop_columns or DEFAULT_DROP)
    if "last_transaction" in frame.columns:
        columns.append("last_transaction")
    return frame.drop(columns=[c for c in columns if c in frame.columns])


def model_feature_lists(params: dict[str, Any] | None = None) -> tuple[list[str], list[str]]:
    features = (params or {}).get("features", {})
    numeric = list(features.get("numeric", [*DEFAULT_NUMERIC_RAW, "recency_days"]))
    categorical = list(features.get("categorical", DEFAULT_CATEGORICAL))
    return numeric, categorical


def select_model_frame(
    frame: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    include_target: bool = True,
) -> pd.DataFrame:
    columns = [*numeric, *categorical]
    if include_target and TARGET_COLUMN in frame.columns:
        columns.append(TARGET_COLUMN)
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"Missing expected columns: {missing}")
    return frame.loc[:, columns]


def build_features(
    frame: pd.DataFrame,
    params: dict[str, Any] | None = None,
    reference_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Full preprocess transform used by the DVC stage and by tests."""
    features_cfg = (params or {}).get("features", {})
    drop_columns = list(features_cfg.get("drop_columns", DEFAULT_DROP))
    numeric, categorical = model_feature_lists(params)

    cleaned = clean_raw(frame)
    with_recency, ref = add_recency_days(cleaned, reference_date=reference_date)
    slim = drop_unused(with_recency, drop_columns=drop_columns)
    model_frame = select_model_frame(slim, numeric, categorical)

    meta = {
        "reference_date": ref.strftime("%Y-%m-%d"),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "target": TARGET_COLUMN,
        "n_rows": int(len(model_frame)),
    }
    return model_frame, meta
