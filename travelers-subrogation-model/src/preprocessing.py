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

    return preprocessor
