# MLOps 2026 — Bank Customer Churn

End-to-end MLOps platform for **bank customer churn prediction** (module project).

## Dataset

[Bank Customer Churn Data (Kaggle)](https://www.kaggle.com/datasets/pentakrishnakishore/bank-customer-churn-data) (~28k customers).

## Stack

| Pillar | Tools |
| --- | --- |
| Craftsmanship | `src/`, Poetry, Ruff/Black/Flake8, pre-commit |
| Data versioning | DVC + remote storage |
| Experiments | MLflow Tracking & Model Registry |
| Serving | FastAPI + Pydantic + Docker |
| CI/CD | Pytest, GitHub Actions, CML |
| Orchestration | Prefect 3.x |
| Observability | Evidently (KS / χ² / PSI), Prometheus |

## Quick start

```bash
git clone https://github.com/Ilyes-Ouni/mlops-project-2026-bank-churn.git
cd mlops-project-2026-bank-churn
# Coming in Jalon 1–2: poetry install && dvc pull && pytest
```

## Project tracking

Linear project: [MLOps 2026 — Bank Customer Churn](https://linear.app/loxara/project/mlops-2026-bank-customer-churn-a238a69e503e)

## License

Course project — see dataset license on Kaggle.
