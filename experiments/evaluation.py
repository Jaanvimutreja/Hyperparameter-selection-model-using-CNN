"""
run_experiments.py
------------------
Full benchmark experiment runner.
Compares CNN meta-learner against multiple baselines.

Run with:  python -m experiments.run_experiments
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.config import TEST_DATASETS, RESULTS_DIR, MODEL_PATH
from backend.dataset_loader import load_all_datasets
from backend.feature_extraction import extract_and_reshape, extract_meta_features, normalize_features
from backend.hyperparameter_search import evaluate_all_configs
from backend.algorithm_space import config_label, get_config_by_index, NUM_CONFIGS
from backend.recommend import recommend_hyperparameters, load_model
from backend.baseline import random_baseline_accuracy, grid_baseline_accuracy, compute_regret
from experiments.metrics import (
    recommendation_accuracy, mean_reciprocal_rank, hit_rate_at_k,
    classification_accuracy_with_config, algorithm_selection_accuracy,
)


def run_full_benchmark():
    """Run a comprehensive benchmark and save results."""
    t0 = time.time()
    print("=" * 60)
    print("  HPSM — FULL BENCHMARK")
    print("=" * 60)

    model = load_model(MODEL_PATH)
    test_data = load_all_datasets(TEST_DATASETS)

    results = []
    true_indices, pred_indices = [], []
    true_algos, pred_algos = [], []
    all_probs = []

    for name, (X, y) in test_data.items():
        print(f"\n  Benchmarking: {name}")

        # Ground truth (exhaustive)
        best_idx, all_scores = evaluate_all_configs(X, y)
        best_cfg = get_config_by_index(best_idx)
        best_acc = all_scores[best_idx]

        # CNN prediction
        result = recommend_hyperparameters(X, y, model=model)
        cnn_idx = result["predicted_index"]
        cnn_acc, _ = classification_accuracy_with_config(X, y, cnn_idx)

        # Random baseline
        rand = random_baseline_accuracy(X, y)

        true_indices.append(best_idx)
        pred_indices.append(cnn_idx)
        true_algos.append(best_cfg["algo"])
        pred_algos.append(result["predicted_algo"])
        all_probs.append(result["all_probabilities"])

        results.append({
            "dataset": name,
            "best_config": config_label(best_idx),
            "best_accuracy": best_acc,
            "cnn_config": config_label(cnn_idx),
            "cnn_accuracy": cnn_acc,
            "cnn_confidence": result["confidence"],
            "random_accuracy": rand["mean_accuracy"],
            "cnn_regret": compute_regret(cnn_acc, best_acc),
            "random_regret": compute_regret(rand["mean_accuracy"], best_acc),
            "exact_match": best_idx == cnn_idx,
            "algo_match": best_cfg["algo"] == result["predicted_algo"],
        })

    # Aggregate
    probs_arr = np.array(all_probs)
    summary = {
        "recommendation_accuracy": recommendation_accuracy(true_indices, pred_indices),
        "algorithm_selection_accuracy": algorithm_selection_accuracy(true_algos, pred_algos),
        "mrr": mean_reciprocal_rank(true_indices, probs_arr),
        "hit_rate_at_1": hit_rate_at_k(true_indices, probs_arr, k=1),
        "hit_rate_at_3": hit_rate_at_k(true_indices, probs_arr, k=3),
        "mean_cnn_accuracy": float(np.mean([r["cnn_accuracy"] for r in results])),
        "mean_random_accuracy": float(np.mean([r["random_accuracy"] for r in results])),
        "mean_cnn_regret": float(np.mean([r["cnn_regret"] for r in results])),
        "mean_random_regret": float(np.mean([r["random_regret"] for r in results])),
    }

    # Print results table
    print("\n" + "=" * 90)
    print(f"{'Dataset':15s} {'Best Config':30s} {'CNN Config':30s} {'CNN':>6s} {'Rand':>6s} {'Match':>5s}")
    print("-" * 90)
    for r in results:
        match = "✅" if r["exact_match"] else "❌"
        print(f"{r['dataset']:15s} {r['best_config']:30s} {r['cnn_config']:30s} "
              f"{r['cnn_accuracy']:6.3f} {r['random_accuracy']:6.3f} {match:>5s}")
    print("=" * 90)
    print(f"\n  Recommendation Accuracy : {summary['recommendation_accuracy']:.4f}")
    print(f"  Algo Selection Accuracy : {summary['algorithm_selection_accuracy']:.4f}")
    print(f"  MRR                     : {summary['mrr']:.4f}")
    print(f"  Hit@3                   : {summary['hit_rate_at_3']:.4f}")
    print(f"  Mean CNN Accuracy       : {summary['mean_cnn_accuracy']:.4f}")
    print(f"  Mean Random Accuracy    : {summary['mean_random_accuracy']:.4f}")
    print(f"  Mean CNN Regret         : {summary['mean_cnn_regret']:.4f}")
    print(f"  Mean Random Regret      : {summary['mean_random_regret']:.4f}")

    # Save
    output = {"results": results, "summary": summary}
    path = os.path.join(RESULTS_DIR, "benchmark_results.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved: {path}")
    print(f"  Time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run_full_benchmark()
