"""
Preprocessing pipeline built on top of the engineered features.

LightGBM handles missing values and non-linearities natively, so preprocessing
is intentionally light: numeric columns pass straight through (imputation
happens implicitly via LightGBM's native NaN handling upstream in most
columns, with a safety-net imputer here), and categoricals are ordinal
encoded so LightGBM can treat them as native categorical splits.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


def build_preprocessor(X_fe, numeric_features=None, categorical_features=None):
    """Build a ColumnTransformer for the feature-engineered dataframe.

    `FeatureEngineer` adds a number of derived columns (ratios, flags, bins)
    on top of the raw dataset, so rather than maintaining an exhaustive list
    of every engineered column name, numeric/categorical membership is
    inferred from dtype. Pass explicit `numeric_features` / `categorical_features`
    lists only if you want to force specific raw columns into one bucket.

    Parameters
    ----------
    X_fe : pd.DataFrame
        A sample of feature-engineered data used to infer column roles.
    numeric_features : list[str], optional
    categorical_features : list[str], optional
    """
    inferred_categorical = set(c for c in X_fe.columns if X_fe[c].dtype.kind not in "biuf")
    inferred_categorical |= set(categorical_features or []) & set(X_fe.columns)
    inferred_numeric = [c for c in X_fe.columns if c not in inferred_categorical]

    numeric_after_fe = sorted(inferred_numeric)
    cat_after_fe = sorted(inferred_categorical)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_after_fe),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), cat_after_fe),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    # Stashed so every caller (CV folds, final fit, saved-model inference) can
    # find the categorical columns without re-deriving them from dtype - the
    # ordinal-encoded output is float64, so dtype alone no longer tells you
    # which columns are categorical after this point.
    preprocessor.cat_features_ = cat_after_fe

    return preprocessor


def cast_categorical_dtypes(X_pre, preprocessor):
    """Cast a preprocessor's ordinal-encoded categorical columns to pandas
    'category' dtype.

    OrdinalEncoder outputs plain float64 columns, and LightGBM's sklearn API
    only auto-detects categoricals from category-dtype DataFrame columns.
    Without this cast (and passing `categorical_feature` to `fit`), every
    categorical column gets split on as if it were an arbitrary ordered
    integer instead of using LightGBM's native categorical splits.
    """
    cat_cols = preprocessor.cat_features_
    if cat_cols:
        X_pre = X_pre.astype({col: "category" for col in cat_cols})
    return X_pre
