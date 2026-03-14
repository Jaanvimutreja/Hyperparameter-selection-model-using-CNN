"""
hyperparameter_search.py
------------------------
Evaluates all algorithm+hyperparameter configurations on a dataset.
Uses algorithm_space.py for the full multi-algorithm search space.
"""

import warnings
import numpy as np
from sklearn.model_selection import cross_val_score

from backend.config import HP_CV_FOLDS, HP_SCORING
from backend.algorithm_space import (
    CONFIG_REGISTRY, NUM_CONFIGS,
    build_classifier, get_config_by_index, config_label,
    ALGORITHM_NAMES, ALGO_RANGES,
)

# Re-export for backward compatibility
PARAM_GRID = CONFIG_REGISTRY


def get_index_by_config(algo_or_C, params_or_gamma=None):
    """
    Backward-compatible config lookup.

    Can be called as:
      get_index_by_config("SVM", {"C": 1, "gamma": 0.1})
      get_index_by_config(1, 0.1)  # legacy SVM-only form
    """
    if isinstance(algo_or_C, str):
        # New form: algo name + params dict
        algo = algo_or_C
        params = params_or_gamma
        for cfg in CONFIG_REGISTRY:
            if cfg["algo"] == algo and cfg["params"] == params:
                return cfg["index"]
        raise ValueError(f"Config not found: {algo} {params}")
    else:
        # Legacy form: C, gamma for SVM
        C_val = algo_or_C
        gamma_val = params_or_gamma
        for cfg in CONFIG_REGISTRY:
            if cfg["algo"] == "SVM" and cfg["params"].get("C") == C_val and cfg["params"].get("gamma") == gamma_val:
                return cfg["index"]
        raise ValueError(f"SVM config not found: C={C_val}, gamma={gamma_val}")


def evaluate_all_configs(X, y, cv=HP_CV_FOLDS, scoring=HP_SCORING):
    """
    Evaluate ALL configurations on a dataset.

    Returns
    -------
    best_index : int
    scores     : dict[int, float]  {config_index: mean_accuracy}
    """
    scores = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for cfg in CONFIG_REGISTRY:
            idx = cfg["index"]
            try:
                clf = build_classifier(idx)
                cv_scores = cross_val_score(clf, X, y, cv=cv, scoring=scoring)
                scores[idx] = float(np.mean(cv_scores))
            except Exception:
                scores[idx] = 0.0

    best_index = max(scores, key=scores.get)
    return best_index, scores


def evaluate_algorithm(X, y, algo_name, cv=HP_CV_FOLDS):
    """Evaluate all configs for a single algorithm."""
    from backend.algorithm_space import get_algo_configs
    configs = get_algo_configs(algo_name)
    scores = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for cfg in configs:
            try:
                clf = build_classifier(cfg["index"])
                cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
                scores[cfg["index"]] = float(np.mean(cv_scores))
            except Exception:
                scores[cfg["index"]] = 0.0

    best_index = max(scores, key=scores.get) if scores else 0
    return best_index, scores


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler

    X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)

    print(f"Evaluating {NUM_CONFIGS} configs on Iris...\n")
    best_idx, scores = evaluate_all_configs(X, y)

    for idx in sorted(scores, key=scores.get, reverse=True)[:10]:
        cfg = get_config_by_index(idx)
        print(f"  [{idx:2d}] {config_label(idx):40s}  acc={scores[idx]:.4f}")

    print(f"\nBest: [{best_idx}] {config_label(best_idx)}  acc={scores[best_idx]:.4f}")
