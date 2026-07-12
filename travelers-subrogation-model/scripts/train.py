"""
End-to-end training script.

Usage
-----
    python scripts/train.py --trials 100 --cv-folds 10

Runs, in order:
  1. Load + prepare data
  2. Optuna hyperparameter search (LightGBM, tuned scale_pos_weight, tuned threshold)
  3. Train final base model on the full training set
  4. Calibrate probabilities (isotonic by default)
  5. Select a conservative decision threshold via repeated-CV + 1-SE rule
  6. Save the full model package to outputs/models/final_model.pkl
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIG, NUMERIC_FEATURES, CATEGORICAL_FEATURES, PATHS  # noqa: E402
from src.pipeline import MLPipeline  # noqa: E402
from src.calibration import ModelCalibrator  # noqa: E402
from src.threshold import ThresholdOptimizer  # noqa: E402
from src.model_io import save_model_package  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Train the subrogation prediction model")
    parser.add_argument("--trials", type=int, default=CONFIG["optuna_trials"])
    parser.add_argument("--cv-folds", type=int, default=CONFIG["optuna_cv_folds"])
    parser.add_argument("--threshold-option", choices=["A", "B", "raw"],
                         default=CONFIG["threshold_selection"])
    parser.add_argument("--calibration-method", choices=["isotonic", "sigmoid"],
                         default=CONFIG["calibration_method"])
    parser.add_argument("--train-csv", type=Path, default=PATHS["train"])
    parser.add_argument("--output", type=Path, default=PATHS["model_package"])
    args = parser.parse_args()

    if not args.train_csv.exists():
        raise FileNotFoundError(
            f"Training data not found at {args.train_csv}. "
            "Place Training_TriGuard.csv in the data/ directory (see data/README.md)."
        )

    df = pd.read_csv(args.train_csv)

    pipeline = MLPipeline(
        df,
        target_col=CONFIG["target_col"],
        id_col=CONFIG["id_col"],
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        random_state=CONFIG["random_state"],
    )
    pipeline.prepare_data()

    # --- Step 1: hyperparameter + threshold search ---------------------
    study = pipeline.optimize(n_trials=args.trials, cv_folds=args.cv_folds)

    # --- Step 2: train final base model on full data --------------------
    best_params = study.best_params.copy()
    _, X_pre = pipeline.train_final_model(best_params)
    pipeline.plot_feature_importance(top_n=20, save_path=PATHS["feature_importance_plot"])

    # --- Step 3: calibrate ------------------------------------------------
    calibrator = ModelCalibrator(method=args.calibration_method, cv=CONFIG["calibration_cv"],
                                  random_state=CONFIG["random_state"])
    calibrator.calibrate(pipeline.best_model, X_pre, pipeline.y)
    calibrator.evaluate(X_pre, pipeline.y)

    # --- Step 4: conservative threshold selection (1-SE rule) -----------
    # NOTE: `calibrator.calibrated_model` was fit on the full training set, so
    # scoring it on CV folds carved from that same set is not a strictly
    # clean holdout (this mirrors the original competition submission). For a
    # leakage-free variant, pass `refit_per_fold=True` with an *unfitted*
    # calibrated-classifier template instead -- slower, but methodologically
    # cleaner; see the docstring in src/threshold.py.
    optimizer = ThresholdOptimizer(
        n_splits=CONFIG["threshold_cv_splits"],
        n_repeats=CONFIG["threshold_cv_repeats"],
        random_state=CONFIG["random_state"],
        threshold_range=CONFIG["threshold_range"],
        threshold_step=CONFIG["threshold_step"],
    )
    optimizer.optimize(X_pre, pipeline.y, calibrator.calibrated_model, refit_per_fold=False)
    optimizer.plot(save_path=PATHS["threshold_plot"])
    final_threshold = optimizer.get_threshold(option=args.threshold_option)

    # --- Step 5: persist everything --------------------------------------
    save_model_package(
        pipeline=pipeline,
        calibrator=calibrator,
        threshold=final_threshold,
        best_params=best_params,
        cv_f1_score=study.best_value,
        filename=args.output,
    )

    print(f"\nDone. Final threshold={final_threshold:.4f}, CV F1={study.best_value:.4f}")


if __name__ == "__main__":
    main()
