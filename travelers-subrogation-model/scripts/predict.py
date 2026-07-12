"""
Score a test CSV with a saved model package and write a submission file.

Usage
-----
    python scripts/predict.py --test-csv data/Testing_TriGuard.csv \
        --model outputs/models/final_model.pkl \
        --output outputs/submission.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIG, PATHS  # noqa: E402
from src.model_io import load_model_package, predict_with_package  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Generate subrogation predictions")
    parser.add_argument("--test-csv", type=Path, default=PATHS["test"])
    parser.add_argument("--model", type=Path, default=PATHS["model_package"])
    parser.add_argument("--output", type=Path, default=PATHS["submission"])
    args = parser.parse_args()

    if not args.test_csv.exists():
        raise FileNotFoundError(f"Test data not found at {args.test_csv}")
    if not args.model.exists():
        raise FileNotFoundError(f"Model package not found at {args.model}. Run scripts/train.py first.")

    package = load_model_package(args.model)
    df_test = pd.read_csv(args.test_csv)

    id_col = CONFIG["id_col"]
    claim_ids = df_test[id_col]
    X_test = df_test.drop(columns=[id_col], errors="ignore")

    preds, probs = predict_with_package(package, X_test)

    submission = pd.DataFrame({
        id_col: claim_ids,
        CONFIG["target_col"]: preds,
    })
    submission.to_csv(args.output, index=False)

    print(f"\nSaved predictions to: {args.output}")
    print(f"Predicted positive rate: {submission[CONFIG['target_col']].mean():.2%}")
    print(submission.head())


if __name__ == "__main__":
    main()
