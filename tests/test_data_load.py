from pathlib import Path

import pytest

from bank_churn.data import load_raw, raw_csv_path


def test_raw_csv_path() -> None:
    assert raw_csv_path().name == "churn.csv"


@pytest.mark.skipif(not Path("data/raw/churn.csv").exists(), reason="raw data not pulled")
def test_load_raw_shape() -> None:
    frame = load_raw()
    assert len(frame) == 28382
    assert "churn" in frame.columns
    assert "last_transaction" in frame.columns
