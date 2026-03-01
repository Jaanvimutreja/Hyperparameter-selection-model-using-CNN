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
    "anneal": 2,
    "labor": 4,
    "arrhythmia": 5,
    "audiology": 7,
    "autos": 9,
    "lymph": 10,
    "mfeat_fourier": 14,
    "breast_w": 15,
    "mfeat_karhunen": 16,
    "mfeat_zernike": 22,
    "cmc": 23,
    "credit_approval": 29,
    "credit_g": 31,
    "postoperative_patient_data": 34,
    "soybean": 42,
    "tae": 48,
    "heart_c": 49,
    "heart_h": 51,
    "heart_statlog": 53,
    "hepatitis": 55,
    "vote": 56,
    "hypothyroid": 57,
    "waveform_5000": 60,
    "bng(tic_tac_toe)": 137,
    "molecular_biology_promoters": 164,
    "primary_tumor": 171,
    "kropt": 184,
    "baseball": 185,
    "braziltourism": 186,
    "eucalyptus": 188,
    "bng(breast_w)": 251,
    "meta_all_arff": 275,
    "meta_batchincremental_arff": 276,
    "meta_ensembles_arff": 277,
    "meta_instanceincremental_arff": 278,
    "meta_stream_intervals_arff": 279,
    "flags": 285,
    "mammography": 310,
    "oil_spill": 311,
    "scene": 312,
    "spectrometer": 313,
    "yeast_ml8": 316,
    "bridges": 327,
    "monks_problems_1": 333,
    "monks_problems_2": 334,
    "monks_problems_3": 335,
    "spect": 336,
    "spectf": 337,
    "grub_damage": 338,
    "squash_stored": 340,
    "squash_unstored": 342,
    "white_clover": 343,
    "aids": 346,
    "webdata_wxa": 350,
    "internet_usage": 372,
    "unix_user_data": 373,
    "syskillwebert_biomedical": 374,
    "japanesevowels": 375,
    "syskillwebert_sheep": 376,
    "synthetic_control": 377,
    "ipums_la_99_small": 378,
    "syskillwebert_goats": 379,
    "syskillwebert_bands": 380,
    "ipums_la_98_small": 381,
    "ipums_la_97_small": 382,
    "analcatdata_broadway": 443,
    "analcatdata_boxing2": 444,
    "prnn_crabs": 446,
    "analcatdata_boxing1": 448,
    "analcatdata_homerun": 449,
    "analcatdata_lawsuit": 450,
    "irish": 451,
    "analcatdata_broadwaymult": 452,
    "analcatdata_bondrate": 453,
    "analcatdata_halloffame": 454,
    "cars": 455,
    "analcatdata_authorship": 458,
    "analcatdata_asbestos": 459,
    "analcatdata_reviewer": 460,
    "analcatdata_creditscore": 461,
    "backache": 463,
    "prnn_synth": 464,
    "analcatdata_cyyoung8092": 465,
    "schizo": 466,
    "analcatdata_japansolvent": 467,
    "confidence": 468,
    "analcatdata_dmft": 469,
    "profb": 470,
    "lupus": 472,
    "cjs": 473,
    "analcatdata_marketing": 474,
    "analcatdata_germangss": 475,
    "analcatdata_bankruptcy": 476,
    "fl2000": 477,
    "analcatdata_cyyoung9302": 479,
    "prnn_viruses": 480,
    "biomed": 481,
    "colleges_aaup": 488,
    "rmftsa_sleepdata": 679,
    "sleuth_ex2016": 682,
}

# Size categories
SMALL_DATASETS = [
    "iris", "wine", "breast_cancer", "glass", "seeds",
    "ionosphere", "vehicle", "sonar", "zoo", "ecoli",
    "vertebral", "dermatology", "haberman", "balance_scale",
    "blood_transfusion", "liver", "hayes_roth", "teaching",
    "user_knowledge", "planning_relax", "diabetes", 
    "splice", "sick", "colic",
    "anneal", "labor", "arrhythmia", "audiology", "autos", "lymph", "breast_w", "credit_approval", "credit_g", "postoperative_patient_data", "soybean", "tae", "heart_c", "heart_h", "heart_statlog", "hepatitis", "vote", "molecular_biology_promoters", "primary_tumor", "braziltourism", "eucalyptus", "meta_all_arff", "meta_batchincremental_arff", "meta_ensembles_arff", "meta_instanceincremental_arff", "flags", "oil_spill", "spectrometer", "bridges", "monks_problems_1", "monks_problems_2", "monks_problems_3", "spect", "spectf", "grub_damage", "squash_stored", "squash_unstored", "white_clover", "aids", "syskillwebert_biomedical", "syskillwebert_sheep", "synthetic_control", "syskillwebert_goats", "syskillwebert_bands", "analcatdata_broadway", "analcatdata_boxing2", "prnn_crabs", "analcatdata_boxing1", "analcatdata_homerun", "analcatdata_lawsuit", "irish", "analcatdata_broadwaymult", "analcatdata_bondrate", "cars", "analcatdata_authorship", "analcatdata_asbestos", "analcatdata_reviewer", "analcatdata_creditscore", "backache", "prnn_synth", "analcatdata_cyyoung8092", "schizo", "analcatdata_japansolvent", "confidence", "analcatdata_dmft", "profb", "lupus", "analcatdata_marketing", "analcatdata_germangss", "analcatdata_bankruptcy", "fl2000", "analcatdata_cyyoung9302", "prnn_viruses", "biomed", "sleuth_ex2016"
]
MEDIUM_DATASETS = [
    "banknote", "car", "segment", "satimage", "optdigits",
    "pendigits", "waveform", "page_blocks", "mfeat_factors",
    "steel_plates", "yeast", "abalone", "credit_german",
    "vowel", "wall_robot", "kr_vs_kp", "mfeat_pixel", 
    "phishing", "eeg",
    "mfeat_fourier", "mfeat_karhunen", "mfeat_zernike", "cmc", "hypothyroid", "waveform_5000", "baseball", "scene", "yeast_ml8", "unix_user_data", "japanesevowels", "ipums_la_99_small", "ipums_la_98_small", "ipums_la_97_small", "analcatdata_halloffame", "cjs", "colleges_aaup", "rmftsa_sleepdata"
]
LARGE_DATASETS = [
    "letter", "magic", "shuttle", "nursery", "mushroom",
    "electricity", "nomao",
    "bng(tic_tac_toe)", "kropt", "bng(breast_w)", "meta_stream_intervals_arff", "mammography", "webdata_wxa", "internet_usage"
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


def _load_local_csv(name: str):
    """Load a dataset from the local offline CSV directory if it exists."""
    csv_path = os.path.join(os.path.dirname(DATASET_CACHE_DIR), "100_datasets", f"{name}.csv")
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)
        if "target" not in df.columns:
            return None
        
        target = df.pop("target")

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

        # Save to pickle cache to speed up subsequent loads
        cache = _cache_path(name)
        if not os.path.exists(cache):
            with open(cache, "wb") as f:
                pickle.dump((X, y), f)

        return X, y
    except Exception as e:
        warnings.warn(f"Could not load '{name}' from local CSV: {e}")
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

    result = _load_local_csv(name)
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
