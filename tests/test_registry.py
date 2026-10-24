import pytest

from bank_churn.models.registry import select_champion


def test_select_champion_prefers_highest_f1_above_thresholds() -> None:
    winner = select_champion(
        [
            {"name": "logreg", "f1": 0.47, "roc_auc": 0.75},
            {"name": "random_forest", "f1": 0.59, "roc_auc": 0.84},
            {"name": "xgboost", "f1": 0.61, "roc_auc": 0.84},
        ],
        min_f1=0.55,
        min_roc_auc=0.80,
    )
    assert winner is not None
    assert winner["name"] == "xgboost"


def test_select_champion_rejects_high_f1_with_weak_auc() -> None:
    winner = select_champion(
        [{"name": "overfit", "f1": 0.90, "roc_auc": 0.70}],
        min_f1=0.55,
        min_roc_auc=0.80,
    )
    assert winner is None


def test_select_champion_empty() -> None:
    assert select_champion([]) is None


def test_register_champion_requires_eligible_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    from bank_churn.models import registry

    monkeypatch.setattr(registry, "configure_tracking", lambda params=None: "bank-churn")
    monkeypatch.setattr(registry, "list_scored_runs", lambda experiment: [])
    with pytest.raises(ValueError, match="No run meets"):
        registry.register_champion({"mlflow": {"experiment": "bank-churn"}, "registry": {}})
