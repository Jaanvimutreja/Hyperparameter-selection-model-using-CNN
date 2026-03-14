"""
test_dataset_loader.py
-----------------------
Tests for dataset loading functionality.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from backend.dataset_loader import (
    load_dataset, load_all_datasets,
    TRAIN_DATASETS, TEST_DATASETS, ALL_DATASETS,
)


class TestDatasetLists:
    def test_train_count(self):
        # updated list now contains 40 training datasets
        assert len(TRAIN_DATASETS) == 40

    def test_test_count(self):
        # 10 datasets reserved for testing
        assert len(TEST_DATASETS) == 10

    def test_all_count(self):
        assert len(ALL_DATASETS) == 50

    def test_no_overlap(self):
        overlap = set(TRAIN_DATASETS) & set(TEST_DATASETS)
        assert len(overlap) == 0, f"Overlap: {overlap}"


class TestLoadDataset:
    @pytest.mark.parametrize("name", ["iris", "wine", "breast_cancer"])
    def test_sklearn_datasets(self, name):
        X, y = load_dataset(name)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]
        assert X.shape[0] > 10

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError):
            load_dataset("nonexistent_dataset_xyz")

    def test_no_nan_in_output(self):
        X, y = load_dataset("iris")
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()
