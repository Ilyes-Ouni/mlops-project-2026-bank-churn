"""DVC stage: raw CSV → engineered train/test splits."""

from bank_churn.data.preprocess import run_preprocess


def main() -> None:
    train, test = run_preprocess()
    print(f"Wrote train={len(train)} test={len(test)}")


if __name__ == "__main__":
    main()
