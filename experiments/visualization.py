"""
visualization.py
----------------
Publication-quality plots for the HPSM system.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from backend.config import PLOTS_DIR
from backend.algorithm_space import config_label, ALGO_RANGES, ALGORITHM_NAMES

# Style
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.figsize": (10, 6),
})

# Color palette per algorithm
ALGO_COLORS = {
    "SVM": "#667eea",
    "RandomForest": "#2ecc71",
    "GradientBoosting": "#e74c3c",
    "LogisticRegression": "#f39c12",
    "KNN": "#9b59b6",
}


def _save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  📊 Plot saved: {name}")


def plot_training_history(history):
    """Training loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs = range(1, len(history["loss"]) + 1)
    ax1.plot(epochs, history["loss"], color="#667eea", linewidth=1.5)
    ax1.set_title("Training Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["accuracy"], color="#2ecc71", linewidth=1.5)
    ax2.set_title("Training Accuracy"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.05); ax2.grid(alpha=0.3)

    fig.suptitle("CNN Meta-Learner Training Curves", fontweight="bold")
    plt.tight_layout()
    _save(fig, "training_curves.png")


def plot_accuracy_comparison(per_dataset):
    """CNN vs Random vs True Best accuracy per dataset."""
    names = list(per_dataset.keys())
    cnn_acc = [per_dataset[n]["pred_accuracy"] for n in names]
    rand_acc = [per_dataset[n]["random_mean_accuracy"] for n in names]
    true_acc = [per_dataset[n]["true_best_accuracy"] for n in names]

    x = np.arange(len(names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.2), 5.5))
    ax.bar(x - w, true_acc, w, label="True Best", color="#2ecc71", alpha=0.9)
    ax.bar(x,     cnn_acc, w, label="CNN Predicted", color="#667eea", alpha=0.9)
    ax.bar(x + w, rand_acc, w, label="Random Baseline", color="#e74c3c", alpha=0.7)

    ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy"); ax.set_title("Accuracy Comparison per Dataset", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 1.1)
    plt.tight_layout()
    _save(fig, "accuracy_comparison.png")


def plot_metric_summary(aggregate):
    """Aggregate metrics bar chart."""
    labels = list(aggregate.keys())
    values = list(aggregate.values())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#667eea", "#2ecc71", "#f39c12", "#9b59b6", "#e74c3c"][:len(labels)]
    ax.bar(labels, values, color=colors, edgecolor="#333", linewidth=0.5)

    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold", fontsize=9)

    ax.set_ylim(0, 1.2); ax.set_title("Aggregate Evaluation Metrics", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "metric_summary.png")


def plot_confidence_chart(per_dataset):
    """CNN prediction confidence per dataset."""
    names = list(per_dataset.keys())
    confs = [per_dataset[n]["pred_confidence"] for n in names]
    match = [per_dataset[n]["match"] for n in names]
    colors = ["#2ecc71" if m else "#e74c3c" for m in match]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.9), 4.5))
    bars = ax.bar(names, confs, color=colors, edgecolor="#333", linewidth=0.5)
    ax.set_ylabel("Confidence"); ax.set_title("CNN Prediction Confidence (green=correct)", fontweight="bold")
    ax.set_ylim(0, 1.1); plt.xticks(rotation=40, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "confidence_chart.png")


def plot_ablation_comparison(cnn_accs, rand_accs, dataset_names, title="CNN vs Random Baseline"):
    """Ablation: CNN vs random accuracy per dataset."""
    x = np.arange(len(dataset_names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(dataset_names) * 1.2), 5.5))
    ax.bar(x - w/2, cnn_accs, w, label="CNN", color="#667eea")
    ax.bar(x + w/2, rand_accs, w, label="Random", color="#e74c3c", alpha=0.7)

    ax.set_xticks(x); ax.set_xticklabels(dataset_names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy"); ax.set_title(title, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 1.1)
    plt.tight_layout()
    _save(fig, "ablation_cnn_vs_random.png")


def plot_algorithm_distribution(per_dataset):
    """Pie chart showing which algorithms were recommended."""
    algo_counts = {}
    for info in per_dataset.values():
        algo = info.get("pred_algo", "unknown")
        algo_counts[algo] = algo_counts.get(algo, 0) + 1

    fig, ax = plt.subplots(figsize=(7, 7))
    colors = [ALGO_COLORS.get(a, "#999") for a in algo_counts.keys()]
    ax.pie(algo_counts.values(), labels=algo_counts.keys(), colors=colors,
           autopct="%1.0f%%", startangle=90, textprops={"fontsize": 10})
    ax.set_title("Algorithm Distribution (CNN Predictions)", fontweight="bold")
    plt.tight_layout()
    _save(fig, "algorithm_distribution.png")


def plot_regret_comparison(per_dataset):
    """Regret chart: CNN vs Random."""
    names = list(per_dataset.keys())
    cnn_regret = [per_dataset[n].get("cnn_regret", 0) for n in names]
    rand_regret = [per_dataset[n].get("random_regret", 0) for n in names]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.2), 5))
    ax.bar(x - w/2, cnn_regret, w, label="CNN Regret", color="#667eea")
    ax.bar(x + w/2, rand_regret, w, label="Random Regret", color="#e74c3c", alpha=0.7)

    ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Regret (lower is better)")
    ax.set_title("Hyperparameter Regret Comparison", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "regret_comparison.png")
