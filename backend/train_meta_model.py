"""
train_meta_model.py
-------------------
Training pipeline for the hybrid CNN meta-learner.
Supports multi-algorithm labels and advanced data augmentation:
  - Gaussian noise injection
  - Subsampling augmentation
  - Feature perturbation
"""

import os
import sys
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.config import (
    CNN_EPOCHS, CNN_BATCH_SIZE, CNN_LEARNING_RATE, CNN_DROPOUT,
    CNN_WEIGHT_DECAY, NUM_AUGMENTED, NOISE_STD,
    SUBSAMPLE_AUGMENT, FEATURE_PERTURB_AUGMENT,
    NUM_CONFIGS, MATRIX_SIZE, MODEL_PATH,
)
from backend.cnn_model import MetaLearnerCNN, count_parameters
from backend.feature_extraction import extract_and_reshape
from backend.hyperparameter_search import evaluate_all_configs
from backend.algorithm_space import get_config_by_index, config_label
from backend.logger import get_training_logger
from backend.knowledge_base import load_knowledge_base, save_knowledge_base, add_entry

logger = get_training_logger()


def build_meta_dataset(datasets_dict):
    """
    Build meta-dataset: extract meta-features and find best config per dataset.

    Parameters
    ----------
    datasets_dict : dict[str, (X, y)]

    Returns
    -------
    matrices : list[np.ndarray]  each (MATRIX_SIZE, MATRIX_SIZE)
    labels   : list[int]         best config index per dataset
    info     : dict              per-dataset HP search results
    """
    matrices = []
    labels = []
    info = {}

    for name, (X, y) in datasets_dict.items():
        try:
            print(f"\n  Processing '{name}'...")

            # Extract meta-features
            matrix, feat_names = extract_and_reshape(X, y)

            # Find best config across ALL algorithms
            best_idx, scores = evaluate_all_configs(X, y)
            best_cfg = get_config_by_index(best_idx)

            print(f"    Best config: [{best_idx}] {config_label(best_idx)}  "
                  f"accuracy={scores[best_idx]:.4f}")

            matrices.append(matrix)
            labels.append(best_idx)
            info[name] = {
                "best_index": best_idx,
                "best_config": best_cfg,
                "best_accuracy": scores[best_idx],
                "all_scores": {str(k): v for k, v in scores.items()},
            }

        except Exception as e:
            warnings.warn(f"Skipping '{name}': {e}")

    return matrices, labels, info


def augment_data(matrices, labels):
    """
    Augment meta-dataset with noise, subsampling, and perturbation.

    Returns
    -------
    aug_matrices : list[np.ndarray]
    aug_labels   : list[int]
    """
    aug_matrices = list(matrices)
    aug_labels = list(labels)

    for mat, lbl in zip(matrices, labels):
        # --- Gaussian noise ---
        for _ in range(NUM_AUGMENTED):
            noise = np.random.normal(0, NOISE_STD, mat.shape).astype(np.float32)
            noisy = np.clip(mat + noise, 0, 1)
            aug_matrices.append(noisy)
            aug_labels.append(lbl)

        # --- Subsampling (row shuffle + scale) ---
        for _ in range(SUBSAMPLE_AUGMENT):
            scale = np.random.uniform(0.85, 1.15, mat.shape).astype(np.float32)
            scaled = np.clip(mat * scale, 0, 1)
            aug_matrices.append(scaled)
            aug_labels.append(lbl)

        # --- Feature perturbation (swap rows/cols) ---
        for _ in range(FEATURE_PERTURB_AUGMENT):
            perturbed = mat.copy()
            # Swap 2 random rows
            i, j = np.random.choice(MATRIX_SIZE, 2, replace=False)
            perturbed[[i, j]] = perturbed[[j, i]]
            aug_matrices.append(perturbed)
            aug_labels.append(lbl)

    return aug_matrices, aug_labels


def train_model(matrices, labels, epochs=CNN_EPOCHS, lr=CNN_LEARNING_RATE,
                batch_size=CNN_BATCH_SIZE, weight_decay=CNN_WEIGHT_DECAY):
    """
    Train the CNN meta-learner.

    Returns
    -------
    model   : MetaLearnerCNN
    history : dict with 'loss' and 'accuracy' lists
    """
    # Prepare tensors
    X = np.array(matrices, dtype=np.float32)
    X = X[:, np.newaxis, :, :]  # add channel dim → (N, 1, 20, 20)
    y = np.array(labels, dtype=np.int64)

    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = MetaLearnerCNN(num_configs=NUM_CONFIGS, input_size=MATRIX_SIZE)
    n_params = count_parameters(model)
    print(f"\n  Model parameters: {n_params:,}")
    print(f"  Training samples: {len(X)}")
    print(f"  Epochs: {epochs}")

    # Optimizer + scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    criterion = nn.CrossEntropyLoss()

    history = {"loss": [], "accuracy": []}

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            preds = output.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

        scheduler.step()
        avg_loss = total_loss / total
        accuracy = correct / total
        history["loss"].append(avg_loss)
        history["accuracy"].append(accuracy)

        if (epoch + 1) % 50 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1:4d}/{epochs}  loss={avg_loss:.4f}  accuracy={accuracy:.4f}")
            logger.info(f"Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  acc={accuracy:.4f}")

    return model, history


def save_model(model, path=MODEL_PATH):
    """Save model weights."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"\n  Model saved to {path}")
    logger.info(f"Model saved: {path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from backend.dataset_loader import load_all_datasets
    from backend.config import TRAIN_DATASETS

    print("Loading training datasets...")
    data = load_all_datasets(TRAIN_DATASETS)

    print("\nBuilding meta-dataset...")
    matrices, labels, info = build_meta_dataset(data)

    print("\nUpdating knowledge base...")
    kb = load_knowledge_base()
    for i, (name, ds_info) in enumerate(info.items()):
        add_entry(
            kb=kb,
            dataset_name=name,
            features=matrices[i].flatten(),
            best_index=ds_info["best_index"],
            best_algo=ds_info["best_config"]["algo"],
            best_params=ds_info["best_config"]["params"],
            best_accuracy=ds_info["best_accuracy"],
            all_scores=ds_info["all_scores"]
        )
    save_knowledge_base(kb)

    print(f"\nAugmenting {len(matrices)} samples...")
    aug_m, aug_l = augment_data(matrices, labels)

    print(f"\nTraining on {len(aug_m)} augmented samples...")
    model, history = train_model(aug_m, aug_l)
    save_model(model)
