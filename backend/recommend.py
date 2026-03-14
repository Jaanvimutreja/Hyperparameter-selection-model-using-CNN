"""
recommend.py
------------
Inference module — predict best algorithm + hyperparameters for a new dataset.
"""

import os
import sys
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.config import MODEL_PATH, MATRIX_SIZE, NUM_CONFIGS
from backend.cnn_model import MetaLearnerCNN
from backend.feature_extraction import extract_and_reshape
from backend.algorithm_space import get_config_by_index, config_label, CONFIG_REGISTRY
from backend.knowledge_base import load_knowledge_base
from backend.dataset_similarity import similarity_based_recommendation
from backend.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL_PATH = MODEL_PATH


def load_model(path=DEFAULT_MODEL_PATH):
    """Load trained CNN model."""
    model = MetaLearnerCNN(num_configs=NUM_CONFIGS, input_size=MATRIX_SIZE)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    logger.info(f"Model loaded from {path}")
    return model


def recommend_hyperparameters(X, y, model=None, model_path=DEFAULT_MODEL_PATH):
    """
    Recommend the best algorithm + hyperparameters for a dataset.

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    model : MetaLearnerCNN or None

    Returns
    -------
    dict with keys:
      - predicted_index     : int
      - predicted_algo      : str
      - predicted_config    : dict
      - predicted_label     : str
      - confidence          : float
      - all_probabilities   : list[float]
      - top_configs         : list[dict]
    """
    if model is None:
        model = load_model(model_path)

    # Extract meta-features
    matrix, _ = extract_and_reshape(X, y)
    tensor = torch.tensor(
        matrix[np.newaxis, np.newaxis, :, :], dtype=torch.float32
    )

    # Predict
    preds, probs = model.predict(tensor)
    pred_idx = int(preds.item())
    probs_np = probs[0].numpy()

    cfg = get_config_by_index(pred_idx)
    confidence = float(probs_np[pred_idx])

    # Top 5 configs
    top_indices = np.argsort(probs_np)[::-1][:5]
    top_configs = []
    for rank, idx in enumerate(top_indices, 1):
        c = get_config_by_index(idx)
        top_configs.append({
            "rank": rank,
            "index": int(idx),
            "algo": c["algo"],
            "params": c["params"],
            "label": config_label(idx),
            "probability": float(probs_np[idx]),
        })

    # Similarity-based recommendation (Dataset Similarity / Knowledge Base)
    kb = load_knowledge_base()
    similarity_rec, nearest = similarity_based_recommendation(
        query_features=matrix.flatten(), 
        knowledge_base=kb
    )

    result = {
        "predicted_index": pred_idx,
        "predicted_algo": cfg["algo"],
        "predicted_config": cfg["params"],
        "predicted_label": config_label(pred_idx),
        "confidence": confidence,
        "all_probabilities": probs_np.tolist(),
        "top_configs": top_configs,
        "nearest_datasets": nearest,
        "similarity_recommended_index": similarity_rec,
    }

    logger.info(f"Recommendation: [{pred_idx}] {config_label(pred_idx)} "
                f"(conf={confidence:.4f})")

    return result


def recommend_top_k(X, y, k=3, model=None, model_path=DEFAULT_MODEL_PATH):
    """
    Get top-k recommended configurations.

    Returns
    -------
    top_k   : list[dict]
    result  : dict (full recommendation result)
    """
    result = recommend_hyperparameters(X, y, model=model, model_path=model_path)
    return result["top_configs"][:k], result


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler

    X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)

    result = recommend_hyperparameters(X, y)
    print(f"\nRecommendation: {result['predicted_label']}")
    print(f"Algorithm:      {result['predicted_algo']}")
    print(f"Params:         {result['predicted_config']}")
    print(f"Confidence:     {result['confidence']:.4f}")
    print(f"\nTop 5:")
    for t in result["top_configs"]:
        print(f"  [{t['rank']}] {t['label']:40s}  prob={t['probability']:.4f}")
