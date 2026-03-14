"""
dataset_similarity.py
---------------------
Compute similarity between datasets using their meta-feature vectors.
Supports cosine similarity and Euclidean distance.
"""

import numpy as np
from backend.config import SIMILARITY_K


def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def euclidean_distance(a, b):
    """Euclidean distance between two vectors."""
    return float(np.linalg.norm(a - b))


def find_nearest_datasets(query_features, knowledge_base, k=SIMILARITY_K, metric="cosine"):
    """
    Find the k most similar datasets from the knowledge base.

    Parameters
    ----------
    query_features : np.ndarray  (flattened meta-feature vector)
    knowledge_base : dict  {dataset_name: {"features": [...], "best_index": int, ...}}
    k : int
    metric : "cosine" or "euclidean"

    Returns
    -------
    list[dict]  sorted by similarity (best first)
        [{name, similarity, best_index, best_algo, best_accuracy}, ...]
    """
    similarities = []

    for name, entry in knowledge_base.items():
        stored = np.array(entry["features"], dtype=np.float32)

        if metric == "cosine":
            sim = cosine_similarity(query_features, stored)
        else:
            sim = -euclidean_distance(query_features, stored)  # negate so higher = better

        similarities.append({
            "name": name,
            "similarity": sim,
            "best_index": entry.get("best_index"),
            "best_algo": entry.get("best_algo", "unknown"),
            "best_accuracy": entry.get("best_accuracy", 0.0),
        })

    # Sort by similarity descending
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return similarities[:k]


def similarity_based_recommendation(query_features, knowledge_base, k=SIMILARITY_K):
    """
    Recommend config based on nearest datasets (majority vote).

    Returns
    -------
    recommended_index : int or None
    nearest           : list[dict]
    """
    nearest = find_nearest_datasets(query_features, knowledge_base, k=k)

    if not nearest:
        return None, []

    # Majority vote on best_index weighted by similarity
    votes = {}
    for entry in nearest:
        idx = entry["best_index"]
        weight = max(entry["similarity"], 0.01)
        votes[idx] = votes.get(idx, 0) + weight

    recommended = max(votes, key=votes.get) if votes else None
    return recommended, nearest
