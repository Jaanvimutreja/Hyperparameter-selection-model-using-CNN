"""
algorithm_space.py
------------------
Defines the complete algorithm + hyperparameter configuration space.

Supports 5 algorithms:
  - SVM (RBF kernel)
  - RandomForest
  - XGBoost (or GradientBoosting fallback)
  - LogisticRegression
  - KNN

Each (algorithm, hyperparam_combo) gets a unique index.
The CNN predicts which index is best for a given dataset.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# ---------------------------------------------------------------------------
# Configuration registry
# ---------------------------------------------------------------------------

# Each config: { "algo", "name", "params", index }
# Total = 36 configs

CONFIG_REGISTRY = []


def _add_configs(algo_name, param_dicts):
    """Add configurations for an algorithm to the registry."""
    for params in param_dicts:
        CONFIG_REGISTRY.append({
            "algo": algo_name,
            "params": params,
            "index": len(CONFIG_REGISTRY),
        })


# --- SVM (9 configs) ---
_add_configs("SVM", [
    {"C": c, "gamma": g}
    for c in [0.1, 1, 10]
    for g in [0.01, 0.1, 1]
])

# --- RandomForest (9 configs) ---
_add_configs("RandomForest", [
    {"n_estimators": n, "max_depth": d}
    for n in [50, 100, 200]
    for d in [5, 10, None]
])

# --- GradientBoosting / XGBoost (9 configs) ---
_add_configs("GradientBoosting", [
    {"n_estimators": n, "max_depth": d, "learning_rate": lr}
    for n in [50, 100]
    for d in [3, 5]
    for lr in [0.05, 0.1]
] + [
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05},
])

# --- LogisticRegression (4 configs) ---
_add_configs("LogisticRegression", [
    {"C": 0.01, "solver": "lbfgs", "max_iter": 2000},
    {"C": 0.1,  "solver": "lbfgs", "max_iter": 2000},
    {"C": 1.0,  "solver": "lbfgs", "max_iter": 2000},
    {"C": 10.0, "solver": "lbfgs", "max_iter": 2000},
])

# --- KNN (5 configs) ---
_add_configs("KNN", [
    {"n_neighbors": 3, "weights": "uniform"},
    {"n_neighbors": 5, "weights": "uniform"},
    {"n_neighbors": 7, "weights": "uniform"},
    {"n_neighbors": 5, "weights": "distance"},
    {"n_neighbors": 10, "weights": "distance"},
])

NUM_CONFIGS = len(CONFIG_REGISTRY)  # should be 36

# Algorithm index ranges (for display/grouping)
ALGO_RANGES = {}
for cfg in CONFIG_REGISTRY:
    algo = cfg["algo"]
    if algo not in ALGO_RANGES:
        ALGO_RANGES[algo] = {"start": cfg["index"], "end": cfg["index"]}
    ALGO_RANGES[algo]["end"] = cfg["index"]

ALGORITHM_NAMES = list(ALGO_RANGES.keys())


def get_config_by_index(index: int) -> dict:
    """Get full config (algo + params) by index."""
    return CONFIG_REGISTRY[index]


def get_index_by_config(algo: str, params: dict) -> int:
    """Find the index of a specific (algo, params) config."""
    for cfg in CONFIG_REGISTRY:
        if cfg["algo"] == algo and cfg["params"] == params:
            return cfg["index"]
    raise ValueError(f"Config not found: {algo} {params}")


def get_algo_configs(algo_name: str) -> list:
    """Get all configs for a specific algorithm."""
    return [c for c in CONFIG_REGISTRY if c["algo"] == algo_name]


def build_classifier(index: int):
    """Build a scikit-learn classifier from a config index."""
    cfg = CONFIG_REGISTRY[index]
    algo = cfg["algo"]
    params = cfg["params"]

    if algo == "SVM":
        return SVC(kernel="rbf", max_iter=5000, **params)
    elif algo == "RandomForest":
        return RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    elif algo == "GradientBoosting":
        return GradientBoostingClassifier(random_state=42, **params)
    elif algo == "LogisticRegression":
        return LogisticRegression(random_state=42, **params)
    elif algo == "KNN":
        return KNeighborsClassifier(**params)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def config_label(index: int) -> str:
    """Human-readable label for a config."""
    cfg = CONFIG_REGISTRY[index]
    algo = cfg["algo"]
    params = cfg["params"]
    param_str = ", ".join(f"{k}={v}" for k, v in params.items()
                          if k not in ("solver", "max_iter"))
    return f"{algo}({param_str})"


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Total configs: {NUM_CONFIGS}")
    print(f"\nAlgorithm ranges:")
    for algo, r in ALGO_RANGES.items():
        count = r["end"] - r["start"] + 1
        print(f"  {algo:25s}  indices {r['start']:2d}-{r['end']:2d}  ({count} configs)")

    print(f"\nAll configs:")
    for cfg in CONFIG_REGISTRY:
        print(f"  [{cfg['index']:2d}] {config_label(cfg['index'])}")
