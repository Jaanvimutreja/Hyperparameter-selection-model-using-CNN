"""
metrics.py
----------
Evaluation metrics for meta-learning recommendation system.
"""

import numpy as np
from sklearn.model_selection import cross_val_score
from backend.algorithm_space import build_classifier, config_label
from backend.config import EVAL_CV_FOLDS


def recommendation_accuracy(true_indices, pred_indices):
    """Fraction of datasets where the predicted config matches the true best."""
    correct = sum(1 for t, p in zip(true_indices, pred_indices) if t == p)
    return correct / max(len(true_indices), 1)


def mean_reciprocal_rank(true_indices, all_probabilities):
    """
    MRR: average of 1/rank of the true best config in the predicted ranking.
    """
    mrr = 0.0
    for true_idx, probs in zip(true_indices, all_probabilities):
        ranking = np.argsort(probs)[::-1]
        rank = int(np.where(ranking == true_idx)[0][0]) + 1
        mrr += 1.0 / rank
    return mrr / max(len(true_indices), 1)


def hit_rate_at_k(true_indices, all_probabilities, k=3):
    """Fraction of datasets where the true best is in the top-k predictions."""
    hits = 0
    for true_idx, probs in zip(true_indices, all_probabilities):
        top_k = np.argsort(probs)[::-1][:k]
        if true_idx in top_k:
            hits += 1
    return hits / max(len(true_indices), 1)


def classification_accuracy_with_config(X, y, config_index, cv=EVAL_CV_FOLDS):
    """
    Evaluate a config on a dataset using cross-validation.

    Parameters
    ----------
    config_index : int  (index into CONFIG_REGISTRY)

    Returns
    -------
    accuracy : float
    std      : float
    """
    try:
        clf = build_classifier(config_index)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        return float(np.mean(scores)), float(np.std(scores))
    except Exception:
        return 0.0, 0.0


def hyperparameter_regret(achieved_accuracy, best_possible_accuracy):
    """Regret = best_possible - achieved. Lower is better."""
    return max(0.0, best_possible_accuracy - achieved_accuracy)


def algorithm_selection_accuracy(true_algos, pred_algos):
    """Fraction of datasets where the predicted algorithm matches the true best."""
    correct = sum(1 for t, p in zip(true_algos, pred_algos) if t == p)
    return correct / max(len(true_algos), 1)
