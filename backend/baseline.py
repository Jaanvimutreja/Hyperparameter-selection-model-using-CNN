"""
baseline.py
-----------
Baseline models for comparison:
  - Random configuration selection
  - MLP meta-learner baseline
  - Grid search (all configs) baseline
"""

import numpy as np
from sklearn.model_selection import cross_val_score

from backend.config import HP_CV_FOLDS, BASELINE_TRIALS, NUM_CONFIGS
from backend.algorithm_space import build_classifier, get_config_by_index, config_label
from backend.logger import get_logger

logger = get_logger(__name__)


def random_recommendation(seed=None):
    """Randomly select a configuration."""
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, NUM_CONFIGS)
    cfg = get_config_by_index(idx)
    return idx, cfg


def random_baseline_accuracy(X, y, n_trials=BASELINE_TRIALS, cv=HP_CV_FOLDS, seed=42):
    """
    Evaluate random config selection over multiple trials.

    Returns
    -------
    dict with mean_accuracy, std_accuracy, best_accuracy, worst_accuracy, all_accuracies
    """
    rng = np.random.RandomState(seed)
    accuracies = []

    for t in range(n_trials):
        idx = rng.randint(0, NUM_CONFIGS)
        try:
            clf = build_classifier(idx)
            scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
            acc = float(np.mean(scores))
        except Exception:
            acc = 0.0
        accuracies.append(acc)

    result = {
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "best_accuracy": float(np.max(accuracies)),
        "worst_accuracy": float(np.min(accuracies)),
        "all_accuracies": accuracies,
    }

    logger.info(f"Random baseline: mean={result['mean_accuracy']:.4f} "
                f"± {result['std_accuracy']:.4f} over {n_trials} trials")
    return result


def grid_baseline_accuracy(X, y, cv=HP_CV_FOLDS, max_configs=NUM_CONFIGS):
    """
    Evaluate all (or top-N) configs — simulates exhaustive grid search.
    Returns the best accuracy achievable.
    """
    best_acc = 0.0
    best_idx = 0
    all_scores = {}

    for idx in range(min(max_configs, NUM_CONFIGS)):
        try:
            clf = build_classifier(idx)
            scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
            acc = float(np.mean(scores))
        except Exception:
            acc = 0.0
        all_scores[idx] = acc
        if acc > best_acc:
            best_acc = acc
            best_idx = idx

    logger.info(f"Grid baseline: best={best_acc:.4f} [{config_label(best_idx)}]")
    return {
        "best_accuracy": best_acc,
        "best_index": best_idx,
        "all_scores": all_scores,
    }


def mlp_baseline_predict(X_train_features, y_train_labels, X_test_feature):
    """
    Simple MLP meta-learner baseline (uses sklearn MLPClassifier).

    Parameters
    ----------
    X_train_features : np.ndarray  (n_train, feature_dim) — meta-features
    y_train_labels   : np.ndarray  (n_train,) — best config indices
    X_test_feature   : np.ndarray  (feature_dim,) — test dataset meta-features

    Returns
    -------
    predicted_index : int
    probabilities   : np.ndarray
    """
    from sklearn.neural_network import MLPClassifier

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
    )
    mlp.fit(X_train_features, y_train_labels)
    pred = mlp.predict(X_test_feature.reshape(1, -1))[0]

    try:
        probs = mlp.predict_proba(X_test_feature.reshape(1, -1))[0]
    except Exception:
        probs = np.zeros(NUM_CONFIGS)
        probs[pred] = 1.0

    return int(pred), probs


def compute_regret(achieved_accuracy, best_possible_accuracy):
    """
    Compute hyperparameter regret.

    regret = best_possible - achieved (lower is better)
    """
    return max(0.0, best_possible_accuracy - achieved_accuracy)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler

    X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)

    r = random_baseline_accuracy(X, y, n_trials=10)
    print(f"Random: {r['mean_accuracy']:.4f} ± {r['std_accuracy']:.4f}")

    g = grid_baseline_accuracy(X, y, max_configs=NUM_CONFIGS)
    print(f"Grid:   {g['best_accuracy']:.4f} [{config_label(g['best_idx'])}]" if 'best_idx' in g else "")
