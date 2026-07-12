"""
End-to-end modeling pipeline: data prep -> Optuna-tuned LightGBM -> final fit.

Model choice (LightGBM over XGBoost/CatBoost) rationale, per the competition
deck: comparable-to-better F1, 5-10x faster hyperparameter search, ~50% lower
memory footprint, and native handling of 20+ categorical features without
one-hot encoding.
"""

import contextlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from joblib import parallel_backend
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.features import FeatureEngineer
from src.preprocessing import build_preprocessor

warnings.filterwarnings("ignore")


class MLPipeline:
    """Orchestrates feature engineering, preprocessing, and LightGBM tuning.

    Class imbalance (77% non-subrogation / 23% subrogation) is handled by
    letting Optuna tune `scale_pos_weight` directly, rather than fixing it
    to the naive class-ratio value -- this lets the search trade off
    precision/recall along with every other hyperparameter.
    """

    def __init__(self, df, target_col="subrogation", id_col="claim_number",
                 numeric_features=None, categorical_features=None, random_state=42):
        self.df = df
        self.target_col = target_col
        self.id_col = id_col
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        self.random_state = random_state

        self.fe = FeatureEngineer()
        self.preprocessor = None
        self.best_model = None
        self.best_threshold = None
        self.X = None
        self.y = None

    # ------------------------------------------------------------------
    def prepare_data(self):
        """Drop rows with missing target and the raw ID column; split X/y."""
        df_clean = self.df.copy()
        df_clean = df_clean.dropna(subset=[self.target_col])
        df_clean = df_clean.drop(columns=[self.id_col], errors="ignore")

        self.X = df_clean.drop(columns=[self.target_col]).reset_index(drop=True)
        self.y = df_clean[self.target_col].reset_index(drop=True)

        print(f"Dataset prepared: {len(self.X)} rows, {self.X.shape[1]} raw columns")
        print(f"Class distribution: {self.y.value_counts(normalize=True).round(3).to_dict()}")
        return self.X, self.y

    # ------------------------------------------------------------------
    def create_preprocessor(self, X_fe):
        self.preprocessor = build_preprocessor(X_fe, self.numeric_features, self.categorical_features)
        return self.preprocessor

    # ------------------------------------------------------------------
    def objective(self, trial, cv_folds=10):
        """Optuna objective: mean CV F1 for a candidate hyperparameter set + threshold."""
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 20, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 150),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 100, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 100, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0, 1),
            # Tuned rather than fixed to the raw 77/23 ratio -- lets the search
            # find the imbalance-handling weight that actually maximizes F1.
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 6.0),
            "random_state": self.random_state,
            "verbosity": -1,
            "n_jobs": 1,
        }
        threshold = trial.suggest_float("threshold", 0.25, 0.75)

        cv_scores = {"f1": [], "auc": [], "precision": [], "recall": []}
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        for fold, (train_idx, valid_idx) in enumerate(cv.split(self.X, self.y)):
            X_tr_raw = self.X.iloc[train_idx].reset_index(drop=True)
            X_val_raw = self.X.iloc[valid_idx].reset_index(drop=True)
            y_tr = self.y.iloc[train_idx].reset_index(drop=True)
            y_val = self.y.iloc[valid_idx].reset_index(drop=True)

            X_tr_fe = self.fe.fit_transform(X_tr_raw)
            X_val_fe = self.fe.transform(X_val_raw)

            preprocessor = build_preprocessor(X_tr_fe, self.numeric_features, self.categorical_features)
            X_tr_pre = preprocessor.fit_transform(X_tr_fe)
            X_val_pre = preprocessor.transform(X_val_fe)

            model = LGBMClassifier(**params)
            with contextlib.redirect_stdout(None):
                model.fit(
                    X_tr_pre, y_tr,
                    eval_set=[(X_val_pre, y_val)],
                    eval_metric="binary_logloss",
                    callbacks=[early_stopping(stopping_rounds=100), log_evaluation(period=0)],
                )

            probs = model.predict_proba(X_val_pre)[:, 1]
            preds = (probs > threshold).astype(int)

            cv_scores["f1"].append(f1_score(y_val, preds, zero_division=0))
            cv_scores["auc"].append(roc_auc_score(y_val, probs))
            cv_scores["precision"].append(precision_score(y_val, preds, zero_division=0))
            cv_scores["recall"].append(recall_score(y_val, preds, zero_division=0))

            trial.report(np.mean(cv_scores["f1"]), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("mean_auc", float(np.mean(cv_scores["auc"])))
        trial.set_user_attr("mean_precision", float(np.mean(cv_scores["precision"])))
        trial.set_user_attr("mean_recall", float(np.mean(cv_scores["recall"])))
        trial.set_user_attr("std_f1", float(np.std(cv_scores["f1"])))

        return float(np.mean(cv_scores["f1"]))

    # ------------------------------------------------------------------
    def optimize(self, n_trials=100, cv_folds=10, timeout=None):
        print(f"\nStarting Optuna search: {n_trials} trials, {cv_folds}-fold CV")
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
        )
        with parallel_backend("threading"):
            study.optimize(
                lambda trial: self.objective(trial, cv_folds=cv_folds),
                n_trials=n_trials, timeout=timeout, show_progress_bar=True,
            )

        print("\n" + "=" * 60)
        print("OPTUNA SEARCH RESULTS")
        print("=" * 60)
        print(f"Best F1:        {study.best_value:.4f}")
        print(f"Best params:    {study.best_params}")
        print("=" * 60)
        return study

    # ------------------------------------------------------------------
    def train_final_model(self, params):
        """Fit feature engineering + preprocessing + LightGBM on the full dataset."""
        params = dict(params)
        threshold = params.pop("threshold", 0.5)
        self.best_threshold = threshold

        X_fe = self.fe.fit_transform(self.X)
        self.preprocessor = self.create_preprocessor(X_fe)
        X_pre = self.preprocessor.fit_transform(X_fe)

        self.best_model = LGBMClassifier(**params)
        self.best_model.fit(X_pre, self.y)
        print("Final base model trained on full dataset.")
        return self.best_model, X_pre

    # ------------------------------------------------------------------
    def transform(self, X_new):
        """Apply fitted feature engineering + preprocessing (no model call)."""
        X_fe = self.fe.transform(X_new)
        return self.preprocessor.transform(X_fe)

    def predict(self, X_new, threshold=None):
        if self.best_model is None:
            raise ValueError("Call train_final_model() first.")
        threshold = threshold if threshold is not None else self.best_threshold
        X_pre = self.transform(X_new)
        probs = self.best_model.predict_proba(X_pre)[:, 1]
        preds = (probs > threshold).astype(int)
        return preds, probs

    # ------------------------------------------------------------------
    def plot_feature_importance(self, top_n=20, save_path=None):
        """Reproduces slide 11: top-N LightGBM feature importances.

        Top predictors typically found: liability %, accident type/site,
        witness presence, education level, mileage/vehicle age, and the
        data-quality flags (invalid_license_age, invalid_vehicle_year).
        """
        if self.best_model is None:
            raise ValueError("Call train_final_model() first.")

        importances = self.best_model.feature_importances_
        feat_names = self.preprocessor.get_feature_names_out()
        feat_df = (
            pd.DataFrame({"feature": feat_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(top_n)
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feat_df["feature"], feat_df["importance"])
        ax.invert_yaxis()
        ax.set_title(f"Top {top_n} Feature Importances")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
            print(f"Saved: {save_path}")
        return feat_df
