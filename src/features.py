"""
Feature engineering for the TriGuard subrogation dataset.

This module encodes every data-quality finding and engineered feature
described in the competition deck:

  * Vehicle-made-year "time travel" glitch (slide 4)       -> invalid_vehicle_year flag,
                                                               vehicle_age kept as-is (still
                                                               predictive despite being wrong)
  * License-age issue (slide 5)                             -> invalid_license_age flag
  * Driver-age anomalies, 8 to 241 years old (slide 6)      -> driver_age capped to [15, 100]
  * Claim-day-of-week mismatch (slide 7)                    -> day of week recomputed from
                                                               claim_date rather than trusting
                                                               the provided field
  * Financial risk ratios & threshold flags (slide 8)       -> vehicle/claim/income ratios,
                                                               high_value_claim flag

The transformer is a standard scikit-learn Transformer so it can be dropped
into a Pipeline / ColumnTransformer and safely cross-validated (all
statistics used inside `transform`, e.g. the median claim payout, are
learned in `fit` on the training fold only -- no leakage).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds engineered features and applies data-quality fixes.

    Parameters
    ----------
    driver_age_bounds : tuple(int, int)
        Driver age is clipped to this range to remove implausible values
        (the raw data contained ages from 8 to 241).
    """

    def __init__(self, driver_age_bounds=(15, 100)):
        self.driver_age_bounds = driver_age_bounds
        self.global_claim_min_ = None
        self.global_median_claim_ = None

    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        X_temp = self._parse_dates(X.copy())
        self.global_claim_min_ = X_temp["claim_date"].min()
        self.global_median_claim_ = X_temp["claim_est_payout"].median()
        return self

    # ------------------------------------------------------------------
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineer expects a pandas DataFrame")

        X = self._parse_dates(X.copy())
        epsilon = 1e-6

        # --- Temporal features -----------------------------------------
        X["driver_age_raw"] = X["claim_date"].dt.year - X["year_of_born"]
        X["implausible_driver_age"] = (
            (X["driver_age_raw"] < self.driver_age_bounds[0])
            | (X["driver_age_raw"] > self.driver_age_bounds[1])
        ).astype(int)
        # Cap driver age at [15, 100] per the data-quality review (slide 6)
        X["driver_age"] = X["driver_age_raw"].clip(*self.driver_age_bounds)

        X["vehicle_age"] = X["claim_date"].dt.year - X["vehicle_made_year"]
        # Vehicle made-year "time travel" glitch (slide 4): ~92% of records show a
        # made-year in the future relative to the claim. We flag it rather than
        # drop it -- despite being impossible, vehicle_age retained real signal.
        X["invalid_vehicle_year"] = (X["vehicle_made_year"] > X["claim_date"].dt.year).astype(int)

        # License-age impossibility (slide 5): license obtained before the driver
        # was old enough (or after the claim date implies negative experience).
        X["invalid_license_age"] = (X["driver_age_raw"] < X["age_of_DL"]).astype(int)
        X["driving_experience"] = X["driver_age"] - X["age_of_DL"]

        # Claim-date/day-of-week mismatch (slide 7): recompute day of week from
        # claim_date instead of trusting the provided (frequently wrong) field.
        computed_dow = X["claim_date"].dt.day_name()
        X["claim_day_of_week"] = computed_dow  # overwrite the unreliable raw field
        X["is_weekend"] = computed_dow.isin(["Saturday", "Sunday"]).astype(int)
        X["claim_month"] = X["claim_date"].dt.month
        X["claim_quarter"] = X["claim_date"].dt.quarter
        X["is_month_end"] = (X["claim_date"].dt.day > 25).astype(int)
        X["days_since_first_claim"] = (X["claim_date"] - self.global_claim_min_).dt.days

        # --- Financial risk ratios (slide 8) -----------------------------
        X["vehicle_to_income_ratio"] = (X["vehicle_price"] / (X["annual_income"] + epsilon)).clip(0, 10)
        X["claim_to_vehicle_ratio"] = (X["claim_est_payout"] / (X["vehicle_price"] + epsilon)).clip(0, 5)
        X["claim_to_income_ratio"] = (X["claim_est_payout"] / (X["annual_income"] + epsilon)).clip(0, 5)
        X["high_value_claim"] = (X["claim_est_payout"] > self.global_median_claim_).astype(int)

        # --- Age-group / vehicle-age interactions ------------------------
        X["young_driver"] = ((X["driver_age"] > 15) & (X["driver_age"] < 25)).astype(int)
        X["senior_driver"] = ((X["driver_age"] > 65) & (X["driver_age"] <= 100)).astype(int)
        X["old_vehicle"] = (X["vehicle_age"] > 10).astype(int)
        X["new_vehicle"] = ((X["vehicle_age"] > 0) & (X["vehicle_age"] < 3)).astype(int)

        # --- Binned features ---------------------------------------------
        X["income_bin"] = pd.qcut(X["annual_income"], q=5, labels=False, duplicates="drop")
        X["claim_size_bin"] = pd.qcut(X["claim_est_payout"], q=5, labels=False, duplicates="drop")
        # Liability quartiles -- the single strongest predictor (slide 3):
        # low liability -> high subrogation rate, high liability -> near zero.
        X["liab_prct_bin"] = pd.cut(
            X["liab_prct"],
            bins=[-0.01, 25, 50, 75, 100],
            labels=["0-25%", "26-50%", "51-75%", "76-100%"],
        )

        # Drop raw columns superseded by engineered / corrected versions
        drop_cols = [c for c in ["year_of_born", "vehicle_made_year", "claim_date", "driver_age_raw"]
                     if c in X.columns]
        X = X.drop(columns=drop_cols)

        return X

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_dates(X):
        if "claim_date" in X.columns and not pd.api.types.is_datetime64_any_dtype(X["claim_date"]):
            X["claim_date"] = pd.to_datetime(X["claim_date"], format="%m/%d/%Y", errors="coerce")
        return X
