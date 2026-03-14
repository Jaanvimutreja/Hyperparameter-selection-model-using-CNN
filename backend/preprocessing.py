"""
preprocessing.py
-----------------
Dataset cleaning and preprocessing pipeline.
Handles missing values, categorical encoding, normalization,
and outlier clipping before meta-feature extraction.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from backend.logger import get_logger

logger = get_logger(__name__)


def handle_missing_values(X, strategy="zero"):
    """
    Replace missing / infinite values.

    Parameters
    ----------
    X : np.ndarray
    strategy : str
        'zero'  → replace with 0
        'mean'  → replace with column mean
        'median' → replace with column median

    Returns
    -------
    X_clean : np.ndarray
    """
    X = np.array(X, dtype=np.float64)
    n_missing = np.isnan(X).sum() + np.isinf(X).sum()

    if n_missing > 0:
        logger.info(f"Handling {n_missing} missing/inf values (strategy={strategy})")

    # Replace Inf first
    X = np.where(np.isinf(X), np.nan, X)

    if strategy == "zero":
        X = np.nan_to_num(X, nan=0.0)
    elif strategy == "mean":
        col_means = np.nanmean(X, axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
    elif strategy == "median":
        col_medians = np.nanmedian(X, axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_medians, inds[1])
    else:
        X = np.nan_to_num(X, nan=0.0)

    return X


def encode_categorical_features(df):
    """
    Label-encode all categorical (object / category) columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    df_encoded : pd.DataFrame
    encoders   : dict[str, LabelEncoder]
    """
    df = df.copy()
    encoders = {}

    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) == "category":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            logger.debug(f"Encoded column '{col}' ({le.classes_.shape[0]} classes)")

    return df, encoders


def encode_target(y):
    """
    Label-encode the target array if it is non-numeric.

    Returns
    -------
    y_encoded : np.ndarray of int
    encoder   : LabelEncoder or None
    """
    if hasattr(y, "dtype") and (y.dtype == object or str(y.dtype) == "category"):
        le = LabelEncoder()
        y_encoded = le.fit_transform(np.asarray(y).astype(str))
        logger.debug(f"Encoded target ({len(le.classes_)} classes)")
        return y_encoded.astype(np.int64), le

    return np.asarray(y, dtype=np.int64), None


def normalize_features(X, method="standard"):
    """
    Normalize feature matrix.

    Parameters
    ----------
    X : np.ndarray
    method : str
        'standard' → zero mean, unit variance
        'minmax'   → scale to [0, 1]

    Returns
    -------
    X_norm : np.ndarray
    """
    if method == "standard":
        scaler = StandardScaler()
        return scaler.fit_transform(X)
    elif method == "minmax":
        xmin = X.min(axis=0)
        xmax = X.max(axis=0)
        denom = xmax - xmin
        denom[denom == 0] = 1.0
        return (X - xmin) / denom
    else:
        return X


def clip_outliers(X, lower=1, upper=99):
    """
    Clip feature values to [lower, upper] percentiles per column.
    """
    lo = np.percentile(X, lower, axis=0)
    hi = np.percentile(X, upper, axis=0)
    return np.clip(X, lo, hi)


def clean_dataset(X, y, normalize=True, clip=True):
    """
    Full preprocessing pipeline for a raw dataset.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
    y : np.ndarray or pd.Series

    Returns
    -------
    X_clean : np.ndarray
    y_clean : np.ndarray
    """
    # If DataFrame, encode categorical columns first
    if isinstance(X, pd.DataFrame):
        X, _ = encode_categorical_features(X)
        X = X.values.astype(np.float64)

    # Encode target
    y, _ = encode_target(y)

    # Handle missing values
    X = handle_missing_values(X, strategy="mean")

    # Clip outliers
    if clip and X.shape[0] > 10:
        X = clip_outliers(X)

    # Normalize
    if normalize:
        X = normalize_features(X, method="standard")

    logger.info(f"Preprocessed: {X.shape[0]} samples × {X.shape[1]} features, "
                f"{len(np.unique(y))} classes")

    return X, y


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    X_clean, y_clean = clean_dataset(X, y)
    print(f"Clean shape: {X_clean.shape}, target classes: {np.unique(y_clean)}")
