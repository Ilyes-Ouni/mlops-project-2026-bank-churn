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
dvc repro
pytest
```

`dvc repro` runs `preprocess → train → evaluate`. Metrics land in `metrics/metrics.json` (F1, ROC-AUC, precision, recall). Processed CSVs and `models/model.joblib` stay in the DVC cache, not in Git.

## MLflow Tracking

Compare Logistic Regression, Random Forest, and XGBoost (falls back to `HistGradientBoosting` if XGBoost is missing):

```bash
python pipelines/mlflow_compare.py
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) and select the `bank-churn` experiment. Each run logs params, F1 / ROC-AUC / precision / recall, the fitted sklearn pipeline, and `confusion_matrix.json`.

### Model Registry

A run becomes **champion** only if **F1 ≥ 0.55** and **ROC-AUC ≥ 0.80** (synopsis targets). Among eligible runs, the highest F1 wins. Promotion path: Staging → Production, plus alias `champion`.

```bash
python pipelines/register_champion.py
```

Serving loads `models:/bank-churn@champion` (fallback: `models:/bank-churn/Production`).

Raw data is versioned with DVC (`data/raw/churn.csv.dvc`). The CSV itself is not in Git. Source: [Kaggle — Bank Customer Churn Data](https://www.kaggle.com/datasets/pentakrishnakishore/bank-customer-churn-data).

Poetry is also supported (`poetry install`).

## Project tracking

Linear project: [MLOps 2026 — Bank Customer Churn](https://linear.app/mlops-2026/project/mlops-2026-bank-customer-churn-057761a7741e)

## License

Course project — see dataset license on Kaggle.
