from bank_churn import __version__
from bank_churn.config import TARGET_COLUMN


def test_package_version() -> None:
    assert __version__ == "0.1.0"


def test_target_column() -> None:
    assert TARGET_COLUMN == "churn"
