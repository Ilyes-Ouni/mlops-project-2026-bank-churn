"""Pydantic contracts for churn inference (single + batch)."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

NUMERIC_FIELDS = (
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
    "recency_days",
)
CATEGORICAL_FIELDS = ("gender", "occupation", "customer_nw_category")


class ChurnFeatures(BaseModel):
    """Model-ready customer snapshot used by `/predict`."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    vintage: float = Field(ge=0)
    age: float = Field(ge=0, le=120)
    dependents: float = Field(ge=0)
    current_balance: float
    previous_month_end_balance: float
    average_monthly_balance_prevQ: float
    average_monthly_balance_prevQ2: float
    current_month_credit: float
    previous_month_credit: float
    current_month_debit: float
    previous_month_debit: float
    current_month_balance: float
    previous_month_balance: float
    recency_days: float = Field(ge=0)
    gender: str = Field(min_length=1)
    occupation: str = Field(min_length=1)
    customer_nw_category: str = Field(min_length=1)

    @field_validator("gender", "occupation", mode="before")
    @classmethod
    def fill_unknown(cls, value: object) -> str:
        if value is None or str(value).strip() in {"", "nan", "None"}:
            return "unknown"
        return str(value)

    @field_validator("customer_nw_category", mode="before")
    @classmethod
    def category_as_str(cls, value: object) -> str:
        if value is None:
            raise ValueError("customer_nw_category is required")
        return str(value)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [self.model_dump()], columns=[*NUMERIC_FIELDS, *CATEGORICAL_FIELDS]
        )


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: list[ChurnFeatures] = Field(min_length=1)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.concat(
            [row.to_dataframe() for row in self.instances], ignore_index=True
        )


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    churn_probability: float = Field(ge=0, le=1)
    churn_label: Literal[0, 1]
    model_name: str = "bank-churn"
    latency_ms: float | None = None


class BatchPredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    predictions: list[PredictResponse] = Field(min_length=1)
