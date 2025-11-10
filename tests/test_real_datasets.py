"""Tests for real dataset loaders."""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import shutil

from gray_tunneled_hashing.data.real_datasets import (
    load_embeddings,
    load_queries_and_ground_truth,
    list_available_datasets,
    _get_data_dir,
)


@pytest.fixture
def temp_data_dir(monkeypatch):
    """Create temporary data directory for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    data_dir = temp_dir / "experiments" / "real" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Monkey patch _get_data_dir to return temp directory
    original_get_data_dir = _get_data_dir
    
    def mock_get_data_dir():
        return data_dir
    
    monkeypatch.setattr(
        "gray_tunneled_hashing.data.real_datasets._get_data_dir",
        mock_get_data_dir
    )
    
    yield data_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


def test_load_embeddings_base(temp_data_dir):
    """Test loading base embeddings."""
    # Create test embeddings
    embeddings = np.random.randn(100, 64).astype(np.float32)
    filepath = temp_data_dir / "test_base_embeddings.npy"
    np.save(filepath, embeddings)
    
    # Load
    loaded = load_embeddings("test", split="base")
    
    assert loaded.shape == (100, 64)
    assert loaded.dtype == np.float32
    assert np.allclose(loaded, embeddings)


def test_load_embeddings_queries(temp_data_dir):
    """Test loading query embeddings."""
    embeddings = np.random.randn(50, 64).astype(np.float32)
    filepath = temp_data_dir / "test_queries_embeddings.npy"
    np.save(filepath, embeddings)
    
    loaded = load_embeddings("test", split="queries")
    
    assert loaded.shape == (50, 64)
    assert np.allclose(loaded, embeddings)


def test_load_embeddings_invalid_split():
    """Test loading with invalid split."""
    with pytest.raises(ValueError, match="split must be"):
        load_embeddings("test", split="invalid")


def test_load_embeddings_not_found(temp_data_dir):
    """Test loading non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_embeddings("nonexistent", split="base")


def test_load_queries_and_ground_truth(temp_data_dir):
    """Test loading queries and ground truth."""
    # Create test files
    queries = np.random.randn(10, 64).astype(np.float32)
    gt_indices = np.random.randint(0, 100, size=(10, 20)).astype(np.int32)
    
    np.save(temp_data_dir / "test_queries_embeddings.npy", queries)
    np.save(temp_data_dir / "test_ground_truth_indices.npy", gt_indices)
    
    loaded_queries, loaded_gt = load_queries_and_ground_truth("test", k=20)
    
    assert loaded_queries.shape == (10, 64)
    assert loaded_gt.shape == (10, 20)
    assert loaded_gt.dtype == np.int32
    assert np.allclose(loaded_queries, queries)
    assert np.array_equal(loaded_gt, gt_indices)


def test_load_queries_and_ground_truth_with_k(temp_data_dir):
    """Test loading with k parameter."""
    queries = np.random.randn(10, 64).astype(np.float32)
    gt_indices = np.random.randint(0, 100, size=(10, 50)).astype(np.int32)
    
    np.save(temp_data_dir / "test_queries_embeddings.npy", queries)
    np.save(temp_data_dir / "test_ground_truth_indices.npy", gt_indices)
    
    loaded_queries, loaded_gt = load_queries_and_ground_truth("test", k=10)
    
    assert loaded_gt.shape == (10, 10)


def test_list_available_datasets(temp_data_dir):
    """Test listing available datasets."""
    # Create some test files
    np.save(temp_data_dir / "dataset1_base_embeddings.npy", np.random.randn(10, 64))
    np.save(temp_data_dir / "dataset2_base_embeddings.npy", np.random.randn(10, 64))
    
    datasets = list_available_datasets()
    
    assert "dataset1" in datasets
    assert "dataset2" in datasets

