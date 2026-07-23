"""
Save/load the complete model artifact needed for production predictions:
feature engineer + preprocessor + base model + calibrator + decision threshold.
"""

import pickle

from src.preprocessing import cast_categorical_dtypes


def save_model_package(pipeline, calibrator, threshold, best_params, cv_f1_score, filename):
    """Bundle everything a downstream scorer needs into one pickle file.

    Parameters
    ----------
    pipeline : src.pipeline.MLPipeline
        Must already have `.fe` and `.preprocessor` fitted.
    calibrator : src.calibration.ModelCalibrator
        Must already be calibrated.
    threshold : float
        Selected decision threshold (see src.threshold.ThresholdOptimizer).
    best_params : dict
        Optuna's best hyperparameters, for reference/reproducibility.
    cv_f1_score : float
        Best CV F1 achieved, stored for reference.
    """
    package = {
        "feature_engineer": pipeline.fe,
        "preprocessor": pipeline.preprocessor,
        "calibrated_model": calibrator.calibrated_model,
        "threshold": threshold,
        "best_params": best_params,
        "cv_f1_score": cv_f1_score,
    }
    with open(filename, "wb") as f:
        pickle.dump(package, f)
    print(f"Model package saved to '{filename}' (threshold={threshold:.4f}, "
          f"cv_f1={cv_f1_score:.4f})")


def load_model_package(filename):
    with open(filename, "rb") as f:
        package = pickle.load(f)
    print(f"Model package loaded from '{filename}' "
          f"(threshold={package['threshold']:.4f}, cv_f1={package['cv_f1_score']:.4f})")
    return package


def predict_with_package(package, X_new):
    """Run the full inference path: feature engineering -> preprocessing -> calibrated model."""
    X_fe = package["feature_engineer"].transform(X_new)
    X_pre = package["preprocessor"].transform(X_fe)
    X_pre = cast_categorical_dtypes(X_pre, package["preprocessor"])
    probs = package["calibrated_model"].predict_proba(X_pre)[:, 1]
    preds = (probs > package["threshold"]).astype(int)
    return preds, probs
