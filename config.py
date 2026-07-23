"""
Central configuration for the Travelers subrogation-prediction project.

Edit DATA_DIR / OUTPUT_DIR for your local environment, or override any of
these values with environment variables / CLI flags in the scripts.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
MODEL_DIR = OUTPUT_DIR / "models"

for d in (DATA_DIR, OUTPUT_DIR, PLOT_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

PATHS = {
    "train": DATA_DIR / "Training_TriGuard.csv",
    "test": DATA_DIR / "Testing_TriGuard.csv",
    "submission": OUTPUT_DIR / "submission_calibrated_1se.csv",
    "model_package": MODEL_DIR / "final_model.pkl",
    "calibration_plot": PLOT_DIR / "calibration_curves.png",
    "threshold_plot": PLOT_DIR / "threshold_1se_rule.png",
    "precision_recall_plot": PLOT_DIR / "precision_recall_tradeoff.png",
    "feature_importance_plot": PLOT_DIR / "feature_importance.png",
    "dow_confusion_plot": PLOT_DIR / "dow_confusion_matrix.png",
    "license_age_plot": PLOT_DIR / "license_age_issue.png",
}

# ---------------------------------------------------------------------------
# Modeling configuration
# ---------------------------------------------------------------------------
CONFIG = {
    "random_state": 42,
    "target_col": "subrogation",
    "id_col": "claim_number",
    # Optuna
    "optuna_trials": 100,
    "optuna_cv_folds": 10,
    # Threshold search (1-SE rule)
    "threshold_range": (0.10, 0.90),
    "threshold_step": 0.01,
    "threshold_cv_splits": 5,
    "threshold_cv_repeats": 5,          # 5 x 5 = 25 iterations
    "threshold_selection": "A",         # "A" = highest mean F1 in 1-SE band (0.315 in our run)
                                         # "B" = most stable (lowest std) in 1-SE band
    # Calibration
    "calibration_method": "isotonic",   # or "sigmoid" (Platt scaling)
    "calibration_cv": 5,
    # Data-quality bounds applied during feature engineering
    "driver_age_bounds": (15, 100),
}

# Raw numeric / categorical columns present in the Travelers TriGuard dataset,
# *before* feature engineering adds derived columns.
NUMERIC_FEATURES = [
    "year_of_born", "safety_rating", "annual_income",
    "past_num_of_claims", "liab_prct", "claim_est_payout",
    "vehicle_made_year", "vehicle_price", "vehicle_weight",
    "age_of_DL", "vehicle_mileage",
]

CATEGORICAL_FEATURES = [
    "gender", "email_or_tel_available", "high_education_ind",
    "address_change_ind", "living_status", "zip_code", "claim_day_of_week",
    "accident_site", "witness_present_ind", "channel",
    "policy_report_filed_ind", "vehicle_category", "vehicle_color",
    "accident_type", "in_network_bodyshop",
]
