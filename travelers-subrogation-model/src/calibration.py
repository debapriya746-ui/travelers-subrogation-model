"""
Probability calibration.

A raw LightGBM score is not a true probability -- it's optimized for ranking,
not for "does this number mean what a business stakeholder thinks it means."
We calibrate with cross-validated isotonic regression (or Platt/sigmoid
scaling) so the output can be trusted as an actual likelihood, which matters
for business interpretability and downstream case-prioritization decisions.
"""

import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss


class ModelCalibrator:
    """Wraps `CalibratedClassifierCV` with evaluation/plotting helpers.

    Parameters
    ----------
    method : {"isotonic", "sigmoid"}
        Isotonic regression is more flexible (non-parametric) and was used
        for the final model; sigmoid (Platt scaling) is more robust with
        less data and is offered as a fallback.
    cv : int
        Number of CV folds used internally to produce out-of-fold calibrated
        probabilities (avoids calibrating on the same data the base model saw).
    """

    def __init__(self, method="isotonic", cv=5, random_state=42):
        self.method = method
        self.cv = cv
        self.random_state = random_state
        self.base_model = None
        self.calibrated_model = None

    def calibrate(self, base_model, X, y, verbose=True):
        if verbose:
            print(f"Calibrating with method='{self.method}', cv={self.cv} ...")
        self.base_model = base_model
        self.calibrated_model = CalibratedClassifierCV(
            estimator=base_model, method=self.method, cv=self.cv, n_jobs=-1
        )
        self.calibrated_model.fit(X, y)
        if verbose:
            print("Calibration complete.")
        return self.calibrated_model

    def evaluate(self, X, y, n_bins=10, verbose=True):
        if self.calibrated_model is None:
            raise ValueError("Call calibrate() first.")

        base_probs = self.base_model.predict_proba(X)[:, 1]
        calib_probs = self.calibrated_model.predict_proba(X)[:, 1]

        results = {
            "base_brier": brier_score_loss(y, base_probs),
            "calibrated_brier": brier_score_loss(y, calib_probs),
            "base_logloss": log_loss(y, base_probs),
            "calibrated_logloss": log_loss(y, calib_probs),
            "base_curve": calibration_curve(y, base_probs, n_bins=n_bins, strategy="uniform"),
            "calib_curve": calibration_curve(y, calib_probs, n_bins=n_bins, strategy="uniform"),
        }
        results["brier_improvement"] = results["base_brier"] - results["calibrated_brier"]
        results["logloss_improvement"] = results["base_logloss"] - results["calibrated_logloss"]

        if verbose:
            print(f"Brier score:  base={results['base_brier']:.4f}  "
                  f"calibrated={results['calibrated_brier']:.4f}  "
                  f"(improvement {results['brier_improvement']:+.4f})")
            print(f"Log loss:     base={results['base_logloss']:.4f}  "
                  f"calibrated={results['calibrated_logloss']:.4f}  "
                  f"(improvement {results['logloss_improvement']:+.4f})")
        return results

    def plot_calibration_curve(self, X, y, n_bins=10, save_path=None):
        base_probs = self.base_model.predict_proba(X)[:, 1]
        calib_probs = self.calibrated_model.predict_proba(X)[:, 1]
        base_frac_pos, base_mean_pred = calibration_curve(y, base_probs, n_bins=n_bins, strategy="uniform")
        calib_frac_pos, calib_mean_pred = calibration_curve(y, calib_probs, n_bins=n_bins, strategy="uniform")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax1.plot(base_mean_pred, base_frac_pos, "s-", label="Base model")
        ax1.plot(calib_mean_pred, calib_frac_pos, "o-", label=f"Calibrated ({self.method})")
        ax1.set_xlabel("Mean predicted probability")
        ax1.set_ylabel("Fraction of positives")
        ax1.set_title("Calibration curve")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.hist(base_probs, bins=30, alpha=0.5, label="Base model", density=True)
        ax2.hist(calib_probs, bins=30, alpha=0.5, label=f"Calibrated ({self.method})", density=True)
        ax2.set_xlabel("Predicted probability")
        ax2.set_ylabel("Density")
        ax2.set_title("Probability distribution")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"Saved: {save_path}")
        return fig

    def predict_proba(self, X):
        if self.calibrated_model is None:
            raise ValueError("Call calibrate() first.")
        return self.calibrated_model.predict_proba(X)[:, 1]
