"""Shared test fixtures for churn frames and API payloads."""

import pandas as pd


def valid_payload() -> dict:
    return {
        "vintage": 20,
        "age": 41,
        "dependents": 1,
        "current_balance": 1200.5,
        "previous_month_end_balance": 1100.0,
        "average_monthly_balance_prevQ": 1150.0,
        "average_monthly_balance_prevQ2": 1080.0,
        "current_month_credit": 200.0,
        "previous_month_credit": 180.0,
        "current_month_debit": 90.0,
        "previous_month_debit": 70.0,
        "current_month_balance": 1180.0,
        "previous_month_balance": 1100.0,
        "recency_days": 12,
        "gender": "Male",
        "occupation": "salaried",
        "customer_nw_category": 2,
    }


def tiny_churn_frame(n: int = 24) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "customer_id": i,
                "branch_code": 1,
                "city": 10,
                "vintage": 10 + i,
                "age": 30 + i,
                "dependents": i % 3,
                "gender": "Male" if i % 2 == 0 else "Female",
                "occupation": "salaried",
                "customer_nw_category": 2,
                "current_balance": float(i * 10),
                "previous_month_end_balance": float(i * 9),
                "average_monthly_balance_prevQ": float(i * 8),
                "average_monthly_balance_prevQ2": float(i * 7),
                "current_month_credit": float(i),
                "previous_month_credit": float(i),
                "current_month_debit": float(i % 5),
                "previous_month_debit": float(i % 4),
                "current_month_balance": float(i * 8),
                "previous_month_balance": float(i * 7),
                "last_transaction": "2019-12-01",
                "churn": int(i % 3 == 0),
            }
        )
    return pd.DataFrame(rows)
