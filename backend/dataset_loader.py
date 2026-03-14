"""
dataset_loader.py
-----------------
Loads 50+ tabular datasets from scikit-learn and OpenML.
Supports categorization by size (small/medium/large).
Caches OpenML downloads for fast re-use.
"""

import os
import warnings
import pickle
import numpy as np
import pandas as pd
from sklearn import datasets as sk_datasets
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample

from backend.config import (
    TRAIN_DATASETS, TEST_DATASETS, ALL_DATASETS,
    DATASET_CACHE_DIR, MAX_DATASET_SAMPLES,
)

# ---------------------------------------------------------------------------
# OpenML dataset ID mapping  (name -> OpenML dataset id)
# ---------------------------------------------------------------------------
OPENML_IDS = {
    # Small
    "glass":              41,
    "seeds":              1499,
    "ionosphere":         59,
    "vehicle":            54,
    "sonar":              40,
    "zoo":                62,
    "ecoli":              39,
    "vertebral":          1523,
    "dermatology":        35,
    "haberman":           43,
    "balance_scale":      11,
    "blood_transfusion":  1464,
    "liver":              1480,
    "hayes_roth":         329,
    "teaching":           48,
    "user_knowledge":     1508,
    "planning_relax":     460,
    "diabetes":           37,
    "splice":             46,
    "sick":               38,
    "colic":              27,
    # Medium
    "banknote":           1462,
    "car":                21,
    "segment":            36,
    "satimage":           182,
    "optdigits":          28,
    "pendigits":          32,
    "waveform":           60,
    "page_blocks":        30,
    "mfeat_factors":      12,
    "steel_plates":       1504,
    "yeast":              181,
    "abalone":            183,
    "credit_german":      31,
    "vowel":              307,
    "wall_robot":         1497,
    "kr_vs_kp":           3,
    "mfeat_pixel":        20,
    "phishing":           4534,
    "eeg":                1471,
    # Large
    "letter":             6,
    "magic":              1120,
    "shuttle":            40685,
    "nursery":            26,
    "mushroom":           24,
    "electricity":        151,
    "nomao":              1486,
    # Test
    "heart":              1510,
    "titanic":            40945,
    "adult":              1590,
    "spambase":           44,
    "credit_australian":  40981,
    "mfeat_morphological": 18,
    "tic_tac_toe":        50,
    "cylinder_bands":     6332,
    "climate_model":      40994,
    "monks1":             333,
}

# Size categories
SMALL_DATASETS = [
    "iris", "wine", "breast_cancer", "glass", "seeds",
    "ionosphere", "vehicle", "sonar", "zoo", "ecoli",
    "vertebral", "dermatology", "haberman", "balance_scale",
    "blood_transfusion", "liver", "hayes_roth", "teaching",
    "user_knowledge", "planning_relax", "diabetes", 
    "splice", "sick", "colic",
]
MEDIUM_DATASETS = [
    "banknote", "car", "segment", "satimage", "optdigits",
    "pendigits", "waveform", "page_blocks", "mfeat_factors",
    "steel_plates", "yeast", "abalone", "credit_german",
    "vowel", "wall_robot", "kr_vs_kp", "mfeat_pixel", 
    "phishing", "eeg",
]
LARGE_DATASETS = [
    "letter", "magic", "shuttle", "nursery", "mushroom",
    "electricity", "nomao",
]


def _cache_path(name: str) -> str:
    return os.path.join(DATASET_CACHE_DIR, f"{name}.pkl")


def _load_sklearn(name: str):
    """Load a dataset bundled with scikit-learn."""
    loaders = {
        "iris":          sk_datasets.load_iris,
        "wine":          sk_datasets.load_wine,
        "breast_cancer": sk_datasets.load_breast_cancer,
    }
    loader = loaders.get(name)
    if loader is None:
        return None
    bunch = loader()
    return bunch.data, bunch.target


def _load_openml(name: str):
    """Load a dataset from OpenML by its registered ID."""
    dataset_id = OPENML_IDS.get(name)
    if dataset_id is None:
        return None

    # Check cache first
    cache = _cache_path(name)
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)

    try:
        from sklearn.datasets import fetch_openml
        bunch = fetch_openml(data_id=dataset_id, as_frame=True, parser="auto")
        df = bunch.data.copy()
        target = bunch.target.copy()

        # Encode categorical features
        for col in df.columns:
            if df[col].dtype == object or str(df[col].dtype) == "category":
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        # Encode target
        if target.dtype == object or str(target.dtype) == "category":
            le = LabelEncoder()
            target = le.fit_transform(target.astype(str))

        X = df.values.astype(np.float64)
        y = np.array(target, dtype=np.int64)

        # Replace NaN / Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Cache for next time
        with open(cache, "wb") as f:
            pickle.dump((X, y), f)

        return X, y
    except Exception as e:
        warnings.warn(f"Could not load '{name}' from OpenML (id={dataset_id}): {e}")
        return None


def load_dataset(name: str):
    """
    Load a single dataset by name.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)
    """
    result = _load_sklearn(name)
    if result is not None:
        return result

    result = _load_openml(name)
    if result is not None:
        return result

    raise ValueError(f"Unknown dataset: {name}")


def load_all_datasets(names=None, max_samples=MAX_DATASET_SAMPLES):
    """
    Load multiple datasets with optional subsampling.

    Parameters
    ----------
    names : list[str] or None
        Dataset names to load. Defaults to ALL_DATASETS.
    max_samples : int or None
        Cap dataset size — subsample if larger.

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray]]
    """
    if names is None:
        names = ALL_DATASETS

    data = {}
    for name in names:
        try:
            X, y = load_dataset(name)

            # Subsample large datasets
            if max_samples and X.shape[0] > max_samples:
                indices = resample(
                    np.arange(X.shape[0]),
                    n_samples=max_samples,
                    stratify=y,
                    random_state=42,
                )
                X = X[indices]
                y = y[indices]

            # Standardize features
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            data[name] = (X, y)
            print(f"  ✓ {name:25s}  samples={X.shape[0]:5d}  features={X.shape[1]:3d}  classes={len(np.unique(y))}")
        except Exception as e:
            warnings.warn(f"Skipping '{name}': {e}")

    return data


def load_small_datasets():
    """Load small datasets (<1000 rows)."""
    return load_all_datasets(SMALL_DATASETS)


def load_medium_datasets():
    """Load medium datasets (1k-10k rows)."""
    return load_all_datasets(MEDIUM_DATASETS)


def load_large_datasets():
    """Load large datasets (>10k rows, subsampled)."""
    return load_all_datasets(LARGE_DATASETS)


def get_dataset_category(name: str) -> str:
    """Get the size category of a dataset."""
    if name in SMALL_DATASETS:
        return "small"
    elif name in MEDIUM_DATASETS:
        return "medium"
    elif name in LARGE_DATASETS:
        return "large"
    return "unknown"


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading all datasets...\n")
    datasets = load_all_datasets()
    print(f"\nLoaded {len(datasets)} datasets successfully.")
