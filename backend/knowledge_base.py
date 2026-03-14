"""
knowledge_base.py
-----------------
Persistent meta-knowledge base storing:
  - Dataset meta-features
  - Best algorithm + hyperparameters
  - Performance metrics

Enables warm-start recommendations and grows as new datasets are processed.
"""

import os
import json
import numpy as np
from backend.config import KNOWLEDGE_BASE_PATH
from backend.logger import get_logger

logger = get_logger(__name__)


def load_knowledge_base(path=KNOWLEDGE_BASE_PATH):
    """Load the knowledge base from disk."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_knowledge_base(kb, path=KNOWLEDGE_BASE_PATH):
    """Save the knowledge base to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, default=str)
    logger.info(f"Knowledge base saved: {len(kb)} entries → {path}")


def add_entry(kb, dataset_name, features, best_index, best_algo,
              best_params, best_accuracy, all_scores=None):
    """
    Add or update a dataset entry in the knowledge base.

    Parameters
    ----------
    kb            : dict
    dataset_name  : str
    features      : np.ndarray or list  (flattened meta-feature vector)
    best_index    : int
    best_algo     : str
    best_params   : dict
    best_accuracy : float
    all_scores    : dict or None
    """
    if isinstance(features, np.ndarray):
        features = features.tolist()

    kb[dataset_name] = {
        "features": features,
        "best_index": int(best_index),
        "best_algo": best_algo,
        "best_params": best_params,
        "best_accuracy": float(best_accuracy),
        "all_scores": all_scores,
    }


def get_entry(kb, dataset_name):
    """Get a dataset entry from the knowledge base."""
    return kb.get(dataset_name)


def get_all_features(kb):
    """Get all feature vectors as a 2D numpy array."""
    names = []
    features = []
    for name, entry in kb.items():
        names.append(name)
        features.append(entry["features"])
    if features:
        return names, np.array(features, dtype=np.float32)
    return [], np.array([], dtype=np.float32)


def get_summary(kb):
    """Get a summary of the knowledge base."""
    algo_counts = {}
    for entry in kb.values():
        algo = entry.get("best_algo", "unknown")
        algo_counts[algo] = algo_counts.get(algo, 0) + 1

    return {
        "total_datasets": len(kb),
        "algo_distribution": algo_counts,
        "mean_accuracy": float(np.mean([e["best_accuracy"] for e in kb.values()])) if kb else 0.0,
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    kb = load_knowledge_base()
    summary = get_summary(kb)
    print(f"Knowledge base: {summary['total_datasets']} datasets")
    print(f"Algorithm distribution: {summary['algo_distribution']}")
    print(f"Mean accuracy: {summary['mean_accuracy']:.4f}")
