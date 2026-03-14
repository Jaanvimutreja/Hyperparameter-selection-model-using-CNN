"""
pipeline.py
-----------
One-command automatic pipeline that runs the entire HPSM system:

    python pipeline.py

Steps:
  1. Load datasets (40 train, 10 test)
  2. Preprocess datasets
  3. Extract meta-features + multi-algorithm HP search
  4. Build knowledge base
  5. Train hybrid CNN meta-model
  6. Save model + metadata
  7. Evaluate on test datasets (CNN vs Random vs Grid)
  8. Compute aggregate metrics + regret
  9. Generate all plots
  10. Save results to CSV / JSON
"""

import os
import sys
import time
import json
import csv
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.config import (
    TRAIN_DATASETS, TEST_DATASETS, MODEL_PATH, MODEL_INFO_PATH,
    RESULTS_DIR, PLOTS_DIR, CNN_EPOCHS, CNN_BATCH_SIZE, CNN_LEARNING_RATE,
    CNN_WEIGHT_DECAY, NUM_AUGMENTED, NOISE_STD, NUM_CONFIGS,
)
from backend.logger import get_logger
from backend.dataset_loader import load_all_datasets
from backend.feature_extraction import extract_and_reshape, extract_meta_features, normalize_features
from backend.hyperparameter_search import evaluate_all_configs
from backend.algorithm_space import get_config_by_index, config_label, NUM_CONFIGS
from backend.cnn_model import MetaLearnerCNN, count_parameters
from backend.train_meta_model import (
    build_meta_dataset, augment_data, train_model, save_model,
)
from backend.recommend import recommend_hyperparameters, load_model
from backend.baseline import random_baseline_accuracy, compute_regret
from backend.knowledge_base import load_knowledge_base, save_knowledge_base, add_entry
from backend.dataset_similarity import find_nearest_datasets
from experiments.metrics import (
    recommendation_accuracy, mean_reciprocal_rank,
    hit_rate_at_k, classification_accuracy_with_config,
    hyperparameter_regret, algorithm_selection_accuracy,
)
from experiments.visualization import (
    plot_training_history, plot_accuracy_comparison,
    plot_metric_summary, plot_confidence_chart,
    plot_ablation_comparison, plot_algorithm_distribution,
    plot_regret_comparison,
)

logger = get_logger("pipeline")


def _save_csv(data, filename, fieldnames):
    """Save a list of dicts to CSV."""
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    logger.info(f"CSV saved: {path}")


def run_pipeline():
    """Execute the full training + evaluation + visualization pipeline."""

    t0 = time.time()
    separator = "=" * 65

    print(f"\n{separator}")
    print("  HPSM — FULL AUTOMATED PIPELINE (Advanced)")
    print(f"  {NUM_CONFIGS} algorithm+HP configs | 5 algorithms")
    print(separator)

    # ==================================================================
    # STEP 1 — Load training datasets
    # ==================================================================
    print(f"\n{'─' * 50}")
    print("  STEP 1/10 — Loading training datasets")
    print(f"{'─' * 50}")
    logger.info("Loading training datasets...")
    train_data = load_all_datasets(TRAIN_DATASETS)
    logger.info(f"Loaded {len(train_data)} training datasets")

    # ==================================================================
    # STEP 2 — Preprocess
    # ==================================================================
    print(f"\n{'─' * 50}")
    print("  STEP 2/10 — Preprocessing datasets")
    print(f"{'─' * 50}")
    for name, (X, y) in train_data.items():
        logger.info(f"  {name}: {X.shape[0]} × {X.shape[1]}, {len(np.unique(y))} classes")

    # ==================================================================
    # STEP 3 — Build meta-dataset (features + multi-algorithm HP search)
    # ==================================================================
    print(f"\n{'─' * 50}")
    print("  STEP 3/10 — Meta-features + Multi-algorithm HP search")
    print(f"{'─' * 50}")
    logger.info("Building meta-dataset...")
    matrices, labels, hp_info = build_meta_dataset(train_data)
    logger.info(f"Built {len(matrices)} meta-samples")

    if len(matrices) == 0:
        logger.error("No meta-samples built. Aborting.")
        return

    # Save HP search results to CSV
    hp_rows = []
    for name, info in hp_info.items():
        cfg = info["best_config"]
        hp_rows.append({
            "dataset": name,
            "best_index": info["best_index"],
            "algorithm": cfg["algo"],
            "params": str(cfg["params"]),
            "best_accuracy": info["best_accuracy"],
        })
    _save_csv(hp_rows, "hyperparameter_results.csv",
              ["dataset", "best_index", "algorithm", "params", "best_accuracy"])

    # Save meta-features to CSV
    mf_rows = []
    for name, mat in zip(hp_info.keys(), matrices):
        row = {"dataset": name}
        for i, val in enumerate(mat.flatten()):
            row[f"f_{i}"] = float(val)
        mf_rows.append(row)
    if mf_rows:
        _save_csv(mf_rows, "meta_features.csv", list(mf_rows[0].keys()))

    # ==================================================================
    # STEP 4 — Build knowledge base
    # ==================================================================
    print(f"\n{'─' * 50}")
    print("  STEP 4/10 — Building knowledge base")
    print(f"{'─' * 50}")
    kb = load_knowledge_base()
    for name, info in hp_info.items():
        idx = list(hp_info.keys()).index(name)
        cfg = info["best_config"]
        features = matrices[idx].flatten()
        add_entry(kb, name, features, info["best_index"],
                  cfg["algo"], cfg["params"], info["best_accuracy"])
    save_knowledge_base(kb)

    # ==================================================================
    # STEP 5 — Augment + Train CNN
    # ==================================================================
    print(f"\n{'─' * 50}")
    print("  STEP 5/10 — Training hybrid CNN meta-learner")
    print(f"{'─' * 50}")
    aug_matrices, aug_labels = augment_data(matrices, labels)
    logger.info(f"Augmented: {len(aug_matrices)} samples")

    model, history = train_model(
        aug_matrices, aug_labels,
        epochs=CNN_EPOCHS,
        lr=CNN_LEARNING_RATE,
        batch_size=CNN_BATCH_SIZE,
        weight_decay=CNN_WEIGHT_DECAY,
    )

    # Save training history CSV
    hist_rows = [{"epoch": i + 1, "loss": l, "accuracy": a}
                 for i, (l, a) in enumerate(zip(history["loss"], history["accuracy"]))]
    _save_csv(hist_rows, "cnn_training_history.csv", ["epoch", "loss", "accuracy"])

    # ==================================================================
    # STEP 6 — Save model + metadata
    # ==================================================================
    print(f"\n{'─' * 50}")
    print("  STEP 6/10 — Saving model")
    print(f"{'─' * 50}")
    save_model(model, MODEL_PATH)

    model_info = {
        "datasets_used": len(train_data),
        "dataset_names": list(train_data.keys()),
        "num_configs": NUM_CONFIGS,
        "algorithms": ["SVM", "RandomForest", "GradientBoosting", "LogisticRegression", "KNN"],
        "epochs": CNN_EPOCHS,
        "batch_size": CNN_BATCH_SIZE,
        "learning_rate": CNN_LEARNING_RATE,
        "augmented_samples": len(aug_matrices),
        "parameters": count_parameters(model),
        "final_loss": history["loss"][-1],
        "final_accuracy": history["accuracy"][-1],
    }
    with open(MODEL_INFO_PATH, "w") as f:
        json.dump(model_info, f, indent=2)
    logger.info(f"Model info saved: {MODEL_INFO_PATH}")

    # ==================================================================
    # STEP 7 — Evaluate on test datasets
    # ==================================================================
    print(f"\n{'─' * 50}")
    print("  STEP 7/10 — Evaluating on test datasets")
    print(f"{'─' * 50}")
    test_data = load_all_datasets(TEST_DATASETS)

    model.eval()
    true_indices, pred_indices = [], []
    true_algos, pred_algos = [], []
    all_probs = []
    per_dataset = {}

    for name, (X, y) in test_data.items():
        logger.info(f"Evaluating: {name}")

        # Ground truth
        best_idx, scores = evaluate_all_configs(X, y)
        best_cfg = get_config_by_index(best_idx)

        # CNN prediction
        result = recommend_hyperparameters(X, y, model=model)
        pred_idx = result["predicted_index"]
        pred_cfg = get_config_by_index(pred_idx)

        pred_acc, _ = classification_accuracy_with_config(X, y, pred_idx)

        # Random baseline
        random_result = random_baseline_accuracy(X, y)

        # Regret
        cnn_regret = compute_regret(pred_acc, scores[best_idx])
        rand_regret = compute_regret(random_result["mean_accuracy"], scores[best_idx])

        # Dataset similarity
        query_features = extract_meta_features(X, y)[0]
        query_features = normalize_features(query_features)
        nearest = find_nearest_datasets(query_features, kb, k=3)

        true_indices.append(best_idx)
        pred_indices.append(pred_idx)
        true_algos.append(best_cfg["algo"])
        pred_algos.append(pred_cfg["algo"])
        all_probs.append(result["all_probabilities"])

        per_dataset[name] = {
            "true_best_index": int(best_idx),
            "true_best_algo": best_cfg["algo"],
            "true_best_accuracy": float(scores[best_idx]),
            "pred_index": pred_idx,
            "pred_algo": pred_cfg["algo"],
            "pred_config": pred_cfg["params"],
            "pred_accuracy": float(pred_acc),
            "pred_confidence": result["confidence"],
            "random_mean_accuracy": random_result["mean_accuracy"],
            "cnn_regret": cnn_regret,
            "random_regret": rand_regret,
            "match": best_idx == pred_idx,
            "algo_match": best_cfg["algo"] == pred_cfg["algo"],
            "nearest_datasets": [n["name"] for n in nearest],
        }

        logger.info(f"  {name}: true=[{best_idx}]{best_cfg['algo']} "
                     f"pred=[{pred_idx}]{pred_cfg['algo']} "
                     f"acc={pred_acc:.4f} rand={random_result['mean_accuracy']:.4f}")

    # ==================================================================
    # STEP 8 — Aggregate metrics
    # ==================================================================
    print(f"\n{'─' * 50}")
    print("  STEP 8/10 — Computing aggregate metrics")
    print(f"{'─' * 50}")
    all_probs_arr = np.array(all_probs)

    aggregate = {
        "recommendation_accuracy": recommendation_accuracy(true_indices, pred_indices),
        "algorithm_selection_accuracy": algorithm_selection_accuracy(true_algos, pred_algos),
        "mrr": mean_reciprocal_rank(true_indices, all_probs_arr),
        "hit_rate_at_1": hit_rate_at_k(true_indices, all_probs_arr, k=1),
        "hit_rate_at_3": hit_rate_at_k(true_indices, all_probs_arr, k=3),
        "mean_cnn_regret": float(np.mean([d["cnn_regret"] for d in per_dataset.values()])),
        "mean_random_regret": float(np.mean([d["random_regret"] for d in per_dataset.values()])),
    }

    for k, v in aggregate.items():
        logger.info(f"  {k}: {v:.4f}")

    # Save evaluation results
    eval_results = {"per_dataset": {}, "aggregate": aggregate}
    eval_rows = []
    for name, info in per_dataset.items():
        eval_results["per_dataset"][name] = {
            k: v for k, v in info.items()
            if k not in ("pred_config",)
        }
        eval_rows.append({
            "dataset": name,
            "true_algo": info["true_best_algo"],
            "pred_algo": info["pred_algo"],
            "true_best_accuracy": info["true_best_accuracy"],
            "pred_accuracy": info["pred_accuracy"],
            "pred_confidence": info["pred_confidence"],
            "random_mean_accuracy": info["random_mean_accuracy"],
            "cnn_regret": info["cnn_regret"],
            "random_regret": info["random_regret"],
            "match": info["match"],
            "algo_match": info["algo_match"],
        })

    with open(os.path.join(RESULTS_DIR, "evaluation_results.json"), "w") as f:
        json.dump(eval_results, f, indent=2, default=str)

    _save_csv(eval_rows, "evaluation_metrics.csv", list(eval_rows[0].keys()))

    # ==================================================================
    # STEP 9 — Generate plots
    # ==================================================================
    print(f"\n{'─' * 50}")
    print("  STEP 9/10 — Generating plots")
    print(f"{'─' * 50}")

    plot_training_history(history)
    plot_accuracy_comparison(per_dataset)
    plot_metric_summary(aggregate)
    plot_confidence_chart(per_dataset)
    plot_algorithm_distribution(per_dataset)
    plot_regret_comparison(per_dataset)

    dataset_names = list(per_dataset.keys())
    cnn_accs = [per_dataset[n]["pred_accuracy"] for n in dataset_names]
    rand_accs = [per_dataset[n]["random_mean_accuracy"] for n in dataset_names]
    plot_ablation_comparison(cnn_accs, rand_accs, dataset_names)

    # ==================================================================
    # STEP 10 — Summary
    # ==================================================================
    elapsed = time.time() - t0

    print(f"\n{separator}")
    print("  PIPELINE COMPLETE")
    print(separator)
    print(f"  Total time             : {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  Train datasets         : {len(train_data)}")
    print(f"  Test datasets          : {len(test_data)}")
    print(f"  Algorithm configs      : {NUM_CONFIGS}")
    print(f"  Model params           : {count_parameters(model):,}")
    print(f"  Rec. Accuracy          : {aggregate['recommendation_accuracy']:.4f}")
    print(f"  Algo Selection Acc     : {aggregate['algorithm_selection_accuracy']:.4f}")
    print(f"  MRR                    : {aggregate['mrr']:.4f}")
    print(f"  Hit@3                  : {aggregate['hit_rate_at_3']:.4f}")
    print(f"  Mean CNN Regret        : {aggregate['mean_cnn_regret']:.4f}")
    print(f"  Mean Random Regret     : {aggregate['mean_random_regret']:.4f}")
    print(f"  Knowledge base entries : {len(kb)}")
    print(f"  Model saved            : {MODEL_PATH}")
    print(f"  Results dir            : {RESULTS_DIR}")
    print(f"  Plots dir              : {PLOTS_DIR}")
    print(separator)

    logger.info(f"Pipeline completed in {elapsed:.1f}s")

    return {
        "model": model,
        "history": history,
        "per_dataset": per_dataset,
        "aggregate": aggregate,
    }


if __name__ == "__main__":
    run_pipeline()
