# EDA — Bank Customer Churn

Source: [Kaggle — Bank Customer Churn Data](https://www.kaggle.com/datasets/pentakrishnakishore/bank-customer-churn-data)  
File: `data/raw/churn.csv` (DVC) — **28 382 rows × 21 columns**.

## Target

| Value | Meaning | Share |
| --- | --- | --- |
| `churn = 1` | Customer became dormant / left | 18.5% |
| `churn = 0` | Active | 81.5% |

Imbalanced binary classification. Primary metrics: **F1 (class 1)** and **ROC-AUC**, not raw accuracy.

## Feature groups

**Numeric (≥5):** `vintage`, `age`, `dependents`, `current_balance`, `previous_month_end_balance`, `average_monthly_balance_prevQ`, `average_monthly_balance_prevQ2`, `current_month_credit`, `previous_month_credit`, `current_month_debit`, `previous_month_debit`, `current_month_balance`, `previous_month_balance`.

**Categorical (≥2):** `gender` (Male 16 548 / Female 11 309 / missing 525), `occupation` (self_employed, salaried, student, retired, company), `customer_nw_category` (1 high / 2 medium / 3 low).

**Identifiers (drop from model):** `customer_id`, `branch_code` (high cardinality, keep optional as category later).

**Time:** `last_transaction` (361 distinct dates, some `NaT`). Use recency in days, not the raw timestamp as a category.

## Missingness

| Column | Nulls |
| --- | --- |
| dependents | 2 463 |
| city | 803 |
| gender | 525 |
| occupation | 80 |

No missing target. Impute dependents with 0, occupation/gender with `"unknown"`, city with a dedicated missing code.

## Leakage check

Balances and monthly credit/debit are **current / previous-month** snapshots used to predict next-quarter dormancy. They are valid predictors, not post-event labels. Do **not** train on `customer_id`. Do not use future dates after `last_transaction`.

## Proposed production-drift variables

- `current_balance`, `current_month_credit`, `current_month_debit` (KS + PSI)
- `occupation`, `customer_nw_category` (χ²)
- `city` mix and recency of `last_transaction`

Alert if PSI ≥ 0.20 on a key numeric feature or global `drift_share` ≥ 30%.
