"""
Conservative classification-threshold selection via the 1-SE rule.

Rather than picking the single threshold that maximizes F1 on a validation
set (which can overfit to that particular split), we run Repeated Stratified
K-Fold CV across a grid of thresholds, then apply the "1 standard error"
rule borrowed from regularization-path selection (e.g. glmnet's
`lambda.1se`): among all thresholds whose mean F1 is within one standard
error of the best mean F1, pick a more conservative one rather than the
raw maximizer.

Two selection strategies are supported once you're in the 1-SE band:
  * "A" - highest mean F1 in the band (performance-focused)
  * "B" - lowest std-dev in the band (stability-focused)

Option "A" is used by default.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold


class ThresholdOptimizer:
    """Finds a stable classification threshold using repeated stratified CV + 1-SE rule.

    Parameters
    ----------
    n_splits, n_repeats : int
        E.g. 5 splits x 5 repeats = 25 total CV iterations.
    threshold_range, threshold_step : tuple, float
        Grid of thresholds to evaluate.
    """

    def __init__(self, n_splits=5, n_repeats=5, random_state=42,
                 threshold_range=(0.10, 0.90), threshold_step=0.01):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.threshold_range = threshold_range
        self.threshold_step = threshold_step
        self.thresholds = np.arange(threshold_range[0], threshold_range[1], threshold_step)
        self.results = None

    # ------------------------------------------------------------------
    def optimize(self, X, y, estimator, refit_per_fold=True, verbose=True):
        """Run the CV threshold search.

        Parameters
        ----------
        X : array-like or DataFrame
            Already feature-engineered and preprocessed features.
        y : array-like
            Binary target.
        estimator : sklearn-compatible classifier
            If `refit_per_fold=True` this is treated as an *unfitted*
            estimator template and cloned + refit on every training fold
            (use this for an uncalibrated base model). If `refit_per_fold=False`,
            this must already be a fitted (e.g. calibrated) model and is
            reused as-is to score every validation fold, which is the
            correct mode for evaluating a model that was calibrated out-of-fold
            upstream.
        """
        y_array = np.asarray(y)
        X_values = X.values if hasattr(X, "values") else np.asarray(X)

        rskf = RepeatedStratifiedKFold(
            n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state
        )
        n_folds = self.n_splits * self.n_repeats
        n_thresh = len(self.thresholds)

        f1_matrix = np.zeros((n_thresh, n_folds))
        precision_matrix = np.zeros((n_thresh, n_folds))
        recall_matrix = np.zeros((n_thresh, n_folds))

        if verbose:
            print(f"Threshold search: {n_folds} folds ({self.n_splits}x{self.n_repeats}) "
                  f"x {n_thresh} thresholds")

        for fold_i, (train_idx, valid_idx) in enumerate(rskf.split(X_values, y_array)):
            X_val, y_val = X_values[valid_idx], y_array[valid_idx]

            if refit_per_fold:
                model = clone(estimator)
                model.fit(X_values[train_idx], y_array[train_idx])
            else:
                model = estimator  # already fitted / calibrated

            probs = model.predict_proba(X_val)[:, 1]

            for i, t in enumerate(self.thresholds):
                preds = (probs > t).astype(int)
                f1_matrix[i, fold_i] = f1_score(y_val, preds, zero_division=0)
                precision_matrix[i, fold_i] = precision_score(y_val, preds, zero_division=0)
                recall_matrix[i, fold_i] = recall_score(y_val, preds, zero_division=0)

        self.results = self._aggregate(f1_matrix, precision_matrix, recall_matrix, n_folds)
        if verbose:
            self._print_results()
        return self.results

    # ------------------------------------------------------------------
    def _aggregate(self, f1_matrix, precision_matrix, recall_matrix, n_folds):
        f1_means = f1_matrix.mean(axis=1)
        f1_stds = f1_matrix.std(axis=1, ddof=1)
        f1_ses = f1_stds / np.sqrt(n_folds)

        best_idx = int(np.argmax(f1_means))
        max_mean_f1 = f1_means[best_idx]
        se_at_best = f1_ses[best_idx]
        one_se_lower_bound = max_mean_f1 - se_at_best

        candidates = np.where(f1_means >= one_se_lower_bound)[0]
        selected_idx_A = candidates[np.argmax(f1_means[candidates])]   # performance-focused
        selected_idx_B = candidates[np.argmin(f1_stds[candidates])]    # stability-focused

        return {
            "thresholds": self.thresholds,
            "f1_means": f1_means,
            "f1_stds": f1_stds,
            "f1_ses": f1_ses,
            "precision_means": precision_matrix.mean(axis=1),
            "recall_means": recall_matrix.mean(axis=1),
            "best_raw_threshold": self.thresholds[best_idx],
            "best_raw_f1": max_mean_f1,
            "best_raw_std": f1_stds[best_idx],
            "one_se_lower_bound": one_se_lower_bound,
            "candidate_thresholds": self.thresholds[candidates],
            "recommended_threshold_A": self.thresholds[selected_idx_A],
            "recommended_f1_A": f1_means[selected_idx_A],
            "recommended_std_A": f1_stds[selected_idx_A],
            "recommended_threshold_B": self.thresholds[selected_idx_B],
            "recommended_f1_B": f1_means[selected_idx_B],
            "recommended_std_B": f1_stds[selected_idx_B],
        }

    # ------------------------------------------------------------------
    def get_threshold(self, option="A"):
        if self.results is None:
            raise ValueError("Call optimize() first.")
        if option == "A":
            return self.results["recommended_threshold_A"]
        if option == "B":
            return self.results["recommended_threshold_B"]
        if option == "raw":
            return self.results["best_raw_threshold"]
        raise ValueError("option must be 'A', 'B', or 'raw'")

    # ------------------------------------------------------------------
    def _print_results(self):
        r = self.results
        print("\n" + "=" * 70)
        print("THRESHOLD OPTIMIZATION RESULTS (Repeated Stratified CV + 1-SE rule)")
        print("=" * 70)
        print(f"Raw best F1:            {r['best_raw_f1']:.4f} at threshold {r['best_raw_threshold']:.3f}")
        print(f"1-SE lower bound:       {r['one_se_lower_bound']:.4f}")
        print(f"Candidate range:        [{r['candidate_thresholds'].min():.3f}, "
              f"{r['candidate_thresholds'].max():.3f}] ({len(r['candidate_thresholds'])} thresholds)")
        print(f"Recommended (A, perf):  {r['recommended_threshold_A']:.3f} "
              f"(F1={r['recommended_f1_A']:.4f} +/- {r['recommended_std_A']:.4f})")
        print(f"Recommended (B, stable):{r['recommended_threshold_B']:.3f} "
              f"(F1={r['recommended_f1_B']:.4f} +/- {r['recommended_std_B']:.4f})")
        print("=" * 70)

    # ------------------------------------------------------------------
    def plot(self, save_path=None):
        """Two-panel plot: F1-vs-threshold with 1-SE band, and precision/recall tradeoff."""
        if self.results is None:
            raise ValueError("Call optimize() first.")
        r = self.results

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.errorbar(r["thresholds"], r["f1_means"], yerr=r["f1_ses"],
                     fmt="-o", capsize=3, markersize=3, alpha=0.6, label="Mean F1 +/- SE")
        ax1.axvline(r["best_raw_threshold"], color="C1", linestyle="--",
                    label=f"Raw best = {r['best_raw_threshold']:.3f}")
        ax1.axvline(r["recommended_threshold_A"], color="C2", linestyle=":", linewidth=2,
                    label=f"1SE (A) = {r['recommended_threshold_A']:.3f}")
        ax1.axvline(r["recommended_threshold_B"], color="C3", linestyle="-.", linewidth=2,
                    label=f"1SE (B) = {r['recommended_threshold_B']:.3f}")
        ax1.axhline(r["one_se_lower_bound"], color="gray", linestyle=":", alpha=0.5,
                    label="1-SE lower bound")
        ax1.set_xlabel("Threshold")
        ax1.set_ylabel("Mean F1")
        ax1.set_title("Threshold selection with 1-SE rule")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        ax2.plot(r["thresholds"], r["precision_means"], label="Precision", linewidth=2)
        ax2.plot(r["thresholds"], r["recall_means"], label="Recall", linewidth=2)
        ax2.plot(r["thresholds"], r["f1_means"], label="F1", linewidth=2, alpha=0.7)
        ax2.axvline(r["recommended_threshold_A"], color="black", linestyle="--",
                    label=f"Selected = {r['recommended_threshold_A']:.3f}")
        ax2.set_xlabel("Threshold")
        ax2.set_ylabel("Score")
        ax2.set_title("Precision-recall tradeoff")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"Saved: {save_path}")
        return fig
