"""
feature_extraction.py
---------------------
Extract rich meta-features from datasets using pymfe + custom features.
Produces a 20×20 meta-feature matrix for CNN input.

Feature groups:
  - general, statistical, info-theory (pymfe)
  - landmarking, model-based (pymfe)
  - custom: class imbalance, PCA variance, correlation stats
"""

import warnings
import numpy as np
from pymfe.mfe import MFE

from backend.config import META_FEATURE_GROUPS, MATRIX_SIZE, META_FEATURE_LENGTH

FEATURE_LENGTH = META_FEATURE_LENGTH  # 400


def _extract_pymfe_features(X, y):
    """Extract features from all configured pymfe groups."""
    all_values = []
    all_names = []

    for group in META_FEATURE_GROUPS:
        try:
            mfe = MFE(groups=[group], suppress_warnings=True)
            mfe.fit(X, y)
            names, values = mfe.extract(suppress_warnings=True)
            values = np.array(values, dtype=np.float64)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            all_values.extend(values.tolist())
            all_names.extend([f"{group}.{n}" for n in names])
        except Exception:
            pass

    return np.array(all_values, dtype=np.float32), all_names


def _extract_custom_features(X, y):
    """Extract additional custom meta-features."""
    features = []
    names = []

    n_samples, n_features = X.shape
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)

    # --- Class imbalance ---
    class_ratios = counts / counts.sum()
    imbalance_ratio = counts.max() / max(counts.min(), 1)
    entropy = -np.sum(class_ratios * np.log2(class_ratios + 1e-10))

    features.extend([imbalance_ratio, entropy, n_classes, n_samples, n_features,
                      n_samples / max(n_features, 1)])
    names.extend(["imbalance_ratio", "class_entropy", "n_classes", "n_samples",
                   "n_features", "sample_feature_ratio"])

    # --- Per-feature statistics ---
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    for stat_fn, stat_name in [
        (np.mean, "global_mean"), (np.std, "global_std"),
        (np.median, "global_median"), (np.min, "global_min"),
        (np.max, "global_max"),
    ]:
        features.append(float(stat_fn(means)))
        names.append(f"mean_{stat_name}")
        features.append(float(stat_fn(stds)))
        names.append(f"std_{stat_name}")

    # Skewness and kurtosis (per feature, then aggregate)
    from scipy import stats as sp_stats
    skews = sp_stats.skew(X, axis=0, nan_policy="omit")
    kurts = sp_stats.kurtosis(X, axis=0, nan_policy="omit")
    skews = np.nan_to_num(skews, 0.0)
    kurts = np.nan_to_num(kurts, 0.0)

    for agg, agg_name in [(np.mean, "mean"), (np.std, "std"), (np.max, "max")]:
        features.append(float(agg(skews)))
        names.append(f"skewness_{agg_name}")
        features.append(float(agg(kurts)))
        names.append(f"kurtosis_{agg_name}")

    # --- Correlation ---
    if n_features > 1:
        try:
            corr = np.corrcoef(X, rowvar=False)
            corr = np.nan_to_num(corr, 0.0)
            upper = corr[np.triu_indices_from(corr, k=1)]
            features.extend([
                float(np.mean(np.abs(upper))),
                float(np.std(upper)),
                float(np.max(np.abs(upper))),
            ])
            names.extend(["corr_mean_abs", "corr_std", "corr_max_abs"])
        except Exception:
            features.extend([0.0, 0.0, 0.0])
            names.extend(["corr_mean_abs", "corr_std", "corr_max_abs"])
    else:
        features.extend([0.0, 0.0, 0.0])
        names.extend(["corr_mean_abs", "corr_std", "corr_max_abs"])

    # --- PCA explained variance ---
    try:
        from sklearn.decomposition import PCA
        n_comp = min(10, n_features, n_samples)
        pca = PCA(n_components=n_comp)
        pca.fit(X)
        ev = pca.explained_variance_ratio_
        features.extend([
            float(ev[0]),
            float(np.sum(ev[:3])) if len(ev) >= 3 else float(np.sum(ev)),
            float(np.sum(ev)),
        ])
        names.extend(["pca_var_1", "pca_var_top3", "pca_var_total"])
    except Exception:
        features.extend([0.0, 0.0, 0.0])
        names.extend(["pca_var_1", "pca_var_top3", "pca_var_total"])

    # --- Landmarking (simple) ---
    try:
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import cross_val_score
        dt = DecisionTreeClassifier(max_depth=3, random_state=42)
        dt_score = cross_val_score(dt, X, y, cv=min(3, n_samples), scoring="accuracy")
        features.append(float(np.mean(dt_score)))
        names.append("dt_accuracy_d3")
    except Exception:
        features.append(0.0)
        names.append("dt_accuracy_d3")

    try:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score
        knn = KNeighborsClassifier(n_neighbors=min(3, n_samples - 1))
        knn_score = cross_val_score(knn, X, y, cv=min(3, n_samples), scoring="accuracy")
        features.append(float(np.mean(knn_score)))
        names.append("knn3_accuracy")
    except Exception:
        features.append(0.0)
        names.append("knn3_accuracy")

    return np.array(features, dtype=np.float32), names


def extract_meta_features(X, y):
    """
    Extract all meta-features (pymfe + custom).

    Returns
    -------
    vector : np.ndarray of shape (FEATURE_LENGTH,)
    names  : list[str]
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        pymfe_vec, pymfe_names = _extract_pymfe_features(X, y)
        custom_vec, custom_names = _extract_custom_features(X, y)

    vec = np.concatenate([pymfe_vec, custom_vec])
    names = pymfe_names + custom_names

    # Pad or truncate to FEATURE_LENGTH
    if len(vec) >= FEATURE_LENGTH:
        vec = vec[:FEATURE_LENGTH]
        names = names[:FEATURE_LENGTH]
    else:
        pad = FEATURE_LENGTH - len(vec)
        vec = np.concatenate([vec, np.zeros(pad, dtype=np.float32)])
        names.extend([f"pad_{i}" for i in range(pad)])

    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return vec, names


def reshape_to_matrix(vector):
    """Reshape feature vector to MATRIX_SIZE × MATRIX_SIZE."""
    return vector[:FEATURE_LENGTH].reshape(MATRIX_SIZE, MATRIX_SIZE)


def normalize_features(vector):
    """Min-max normalize to [0, 1]."""
    vmin, vmax = vector.min(), vector.max()
    if vmax - vmin < 1e-10:
        return np.zeros_like(vector)
    return (vector - vmin) / (vmax - vmin)


def extract_and_reshape(X, y):
    """
    Full pipeline: extract → normalize → reshape to 2D matrix.

    Returns
    -------
    matrix : np.ndarray of shape (MATRIX_SIZE, MATRIX_SIZE)
    names  : list[str]
    """
    vec, names = extract_meta_features(X, y)
    vec = normalize_features(vec)
    matrix = reshape_to_matrix(vec)
    return matrix, names


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    matrix, names = extract_and_reshape(X, y)
    print(f"Matrix shape: {matrix.shape}")
    print(f"Feature count: {len(names)}")
    print(f"Value range: [{matrix.min():.4f}, {matrix.max():.4f}]")
