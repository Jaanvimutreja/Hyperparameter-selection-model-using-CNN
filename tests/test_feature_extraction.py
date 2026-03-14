"""
test_feature_extraction.py
---------------------------
Tests for meta-feature extraction and reshaping.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from sklearn.datasets import load_iris

from backend.feature_extraction import (
    extract_meta_features, reshape_to_matrix,
    normalize_features, extract_and_reshape,
    MATRIX_SIZE, FEATURE_LENGTH,
)


@pytest.fixture
def iris_data():
    X, y = load_iris(return_X_y=True)
    return X, y


class TestConstants:
    def test_matrix_size(self):
        # configuration now uses a 20×20 meta-feature matrix
        assert MATRIX_SIZE == 20

    def test_feature_length(self):
        assert FEATURE_LENGTH == 400


class TestExtractMetaFeatures:
    def test_output_shape(self, iris_data):
        X, y = iris_data
        vec, names = extract_meta_features(X, y)
        assert vec.shape == (FEATURE_LENGTH,)
        assert isinstance(names, list)
        assert len(names) > 0

    def test_no_nan(self, iris_data):
        X, y = iris_data
        vec, _ = extract_meta_features(X, y)
        assert not np.isnan(vec).any()
        assert not np.isinf(vec).any()


class TestReshape:
    def test_reshape_shape(self):
        vec = np.random.rand(FEATURE_LENGTH).astype(np.float32)
        mat = reshape_to_matrix(vec)
        assert mat.shape == (MATRIX_SIZE, MATRIX_SIZE)


class TestNormalize:
    def test_range(self):
        vec = np.array([1, 5, 10, 2, 8], dtype=np.float32)
        norm = normalize_features(vec)
        assert norm.min() >= 0.0
        assert norm.max() <= 1.0

    def test_constant_vector(self):
        vec = np.ones(10, dtype=np.float32) * 5.0
        norm = normalize_features(vec)
        assert np.allclose(norm, 0.0)


class TestExtractAndReshape:
    def test_end_to_end(self, iris_data):
        X, y = iris_data
        matrix, names = extract_and_reshape(X, y)
        assert matrix.shape == (MATRIX_SIZE, MATRIX_SIZE)
        assert matrix.min() >= 0.0
        assert matrix.max() <= 1.0
