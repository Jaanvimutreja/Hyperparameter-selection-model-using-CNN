"""
test_model.py
--------------
Tests for CNN model architecture and hyperparameter search.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest

from backend.cnn_model import MetaLearnerCNN, count_parameters
from backend.config import MATRIX_SIZE
from backend.hyperparameter_search import (
    PARAM_GRID, get_config_by_index, get_index_by_config,
    evaluate_all_configs, NUM_CONFIGS,
)


class TestCNNModel:
    def test_output_shape(self):
        # can specify custom number of configs for testing
        model = MetaLearnerCNN(num_configs=NUM_CONFIGS)
        x = torch.randn(4, 1, MATRIX_SIZE, MATRIX_SIZE)
        out = model(x)
        assert out.shape == (4, NUM_CONFIGS)

    def test_single_sample(self):
        model = MetaLearnerCNN()
        x = torch.randn(1, 1, MATRIX_SIZE, MATRIX_SIZE)
        out = model(x)
        assert out.shape == (1, NUM_CONFIGS)

    def test_predict(self):
        model = MetaLearnerCNN()
        x = torch.randn(3, 1, MATRIX_SIZE, MATRIX_SIZE)
        pred, probs = model.predict(x)
        assert pred.shape == (3,)
        assert probs.shape == (3, NUM_CONFIGS)
        # Probabilities should sum to ~1
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(3), atol=1e-5)

    def test_parameter_count_reasonable(self):
        model = MetaLearnerCNN()
        n = count_parameters(model)
        # network has grown, upper bound relaxed
        assert 1000 < n < 1000000, f"Unexpected param count: {n}"

    def test_deterministic_eval(self):
        model = MetaLearnerCNN()
        model.eval()
        x = torch.randn(2, 1, 12, 12)
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2)


class TestHyperparameterSearch:
    def test_grid_size(self):
        assert len(PARAM_GRID) == NUM_CONFIGS
        assert NUM_CONFIGS == len(PARAM_GRID)

    def test_config_roundtrip(self):
        for i in range(NUM_CONFIGS):
            cfg = get_config_by_index(i)
            assert cfg["index"] == i
            # use new-style lookup with algorithm name and params dict
            assert get_index_by_config(cfg["algo"], cfg["params"]) == i

    def test_evaluate_iris(self):
        from sklearn.datasets import load_iris
        from sklearn.preprocessing import StandardScaler

        X, y = load_iris(return_X_y=True)
        X = StandardScaler().fit_transform(X)
        best_idx, scores = evaluate_all_configs(X, y, cv=3)

        assert 0 <= best_idx < NUM_CONFIGS
        assert len(scores) == NUM_CONFIGS
        assert all(0 <= s <= 1 for s in scores.values())
