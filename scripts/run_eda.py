"""
Run the data-quality report described in the competition deck (slides 3-7):
liability/subrogation relationship, vehicle-year glitch, license-age issue,
driver-age anomalies, and claim day-of-week mismatch.

Usage
-----
    python scripts/run_eda.py --train-csv data/Training_TriGuard.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PATHS, PLOT_DIR  # noqa: E402
from src.data_quality import run_full_report  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run the data-quality diagnostic report")
    parser.add_argument("--train-csv", type=Path, default=PATHS["train"])
    args = parser.parse_args()

    if not args.train_csv.exists():
        raise FileNotFoundError(f"Training data not found at {args.train_csv}")

    df = pd.read_csv(args.train_csv)
    run_full_report(df, plot_dir=PLOT_DIR)


if __name__ == "__main__":
    main()
