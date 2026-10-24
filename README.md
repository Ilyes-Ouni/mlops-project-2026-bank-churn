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
python -m pip install -e ".[dev]"
pre-commit install
python scripts/download_raw_data.py
dvc checkout
pytest
```

Raw data is versioned with DVC (`data/raw/churn.csv.dvc`). The CSV itself is not in Git. Source: [Kaggle — Bank Customer Churn Data](https://www.kaggle.com/datasets/pentakrishnakishore/bank-customer-churn-data).

Poetry is also supported (`poetry install`).

## Project tracking

Linear project: [MLOps 2026 — Bank Customer Churn](https://linear.app/mlops-2026/project/mlops-2026-bank-customer-churn-057761a7741e)

## License

Course project — see dataset license on Kaggle.
