"""Tests for Hamming index."""

import numpy as np
import pytest

from gray_tunneled_hashing.integrations.hamming_index import (
    HammingIndex,
    build_hamming_index,
    search_hamming_index,
)


def test_hamming_index_python_backend():
    """Test Hamming index with Python backend."""
    # Create simple test codes
    codes = np.array([
        [True, False, True, False],
        [True, True, False, False],
        [False, False, True, True],
    ])
    
    index = HammingIndex(codes, use_faiss=False, backend="python")
    
    assert index.backend == "python"
    assert index.n_samples == 3
    assert index.n_bits == 4
    
    # Test search
    query_codes = np.array([[True, False, True, False]])
    indices, distances = index.search(query_codes, k=2)
    
    assert indices.shape == (1, 2)
    assert distances.shape == (1, 2)
    assert indices[0, 0] == 0  # Exact match
    assert distances[0, 0] == 0


def test_hamming_index_bool_to_uint8():
    """Test that bool codes are converted to uint8."""
    codes = np.array([
        [True, False, True],
        [False, True, False],
    ])
    
    index = HammingIndex(codes, use_faiss=False, backend="python")
    
    assert index.codes.dtype == np.uint8
    assert index.n_bits == 3


def test_hamming_index_search_all_results():
    """Test search returns all results when k equals n_samples."""
    codes = np.random.rand(10, 8) > 0.5
    
    index = HammingIndex(codes, use_faiss=False, backend="python")
    
    query_codes = codes[:2]
    indices, distances = index.search(query_codes, k=10)
    
    assert indices.shape == (2, 10)
    assert distances.shape == (2, 10)


def test_hamming_index_search_k_too_large():
    """Test search with k > n_samples."""
    codes = np.random.rand(5, 8) > 0.5
    
    index = HammingIndex(codes, use_faiss=False, backend="python")
    
    query_codes = codes[:1]
    
    with pytest.raises(ValueError, match="cannot exceed"):
        index.search(query_codes, k=10)


def test_hamming_index_faiss_backend():
    """Test Hamming index with FAISS backend (if available)."""
    try:
        import faiss
        
        codes = np.random.rand(10, 8) > 0.5
        
        index = HammingIndex(codes, use_faiss=True, backend="faiss")
        
        assert index.backend == "faiss"
        
        # Test search
        query_codes = codes[:2]
        indices, distances = index.search(query_codes, k=3)
        
        assert indices.shape == (2, 3)
        assert distances.shape == (2, 3)
        
    except ImportError:
        pytest.skip("FAISS not available")


def test_hamming_index_faiss_fallback():
    """Test that FAISS fallback works when FAISS is not available."""
    codes = np.random.rand(5, 8) > 0.5
    
    # Try to use FAISS but should fallback to Python
    index = HammingIndex(codes, use_faiss=True, backend=None)
    
    # Should work with either backend
    assert index.backend in ["faiss", "python"]
    
    query_codes = codes[:1]
    indices, distances = index.search(query_codes, k=3)
    
    assert indices.shape == (1, 3)


def test_build_hamming_index():
    """Test build_hamming_index helper function."""
    codes = np.random.rand(10, 8) > 0.5
    
    index = build_hamming_index(codes, use_faiss=False)
    
    assert isinstance(index, HammingIndex)
    assert index.n_samples == 10


def test_search_hamming_index():
    """Test search_hamming_index helper function."""
    codes = np.random.rand(10, 8) > 0.5
    index = build_hamming_index(codes, use_faiss=False)
    
    query_codes = codes[:2]
    indices, distances = search_hamming_index(index, query_codes, k=3)
    
    assert indices.shape == (2, 3)
    assert distances.shape == (2, 3)


def test_hamming_distance_correctness():
    """Test that Hamming distances are computed correctly."""
    codes = np.array([
        [True, False, True, False],
        [True, True, False, False],
        [False, False, True, True],
    ])
    
    index = HammingIndex(codes, use_faiss=False, backend="python")
    
    # Query is same as first code
    query = np.array([[True, False, True, False]])
    indices, distances = index.search(query, k=3)
    
    # Distance to first (index 0) should be 0
    assert distances[0, 0] == 0
    assert indices[0, 0] == 0
    
    # Check that distances are sorted (increasing)
    assert distances[0, 0] <= distances[0, 1] <= distances[0, 2]
    
    # Verify actual distances match expected
    # codes[0] = [True, False, True, False] -> distance 0 (exact match)
    # codes[1] = [True, True, False, False] -> distance 2 (bits 1 and 2 differ: F vs T, T vs F)
    # codes[2] = [False, False, True, True] -> distance 2 (bits 0 and 3 differ: T vs F, F vs T)
    distances_list = distances[0].tolist()
    assert 0 in distances_list  # Exact match
    assert 2 in distances_list  # Both other codes have distance 2
    # All distances should be 0, 2, or 2 (no distance 4)

