"""Tests for block selection strategies."""

import numpy as np
import pytest
from gray_tunneled_hashing.algorithms.block_selection import (
    select_block_random,
    select_block_by_embedding_cluster,
    get_block_selection_fn,
)


def test_select_block_random():
    """Test random block selection."""
    N = 16
    block_size = 4
    
    block = select_block_random(N, block_size, random_state=42)
    
    assert len(block) == block_size
    assert np.all(block >= 0)
    assert np.all(block < N)
    assert len(np.unique(block)) == block_size  # All unique
    assert np.all(np.diff(block) > 0)  # Sorted


def test_select_block_random_deterministic():
    """Test that random block selection is deterministic with seed."""
    N = 16
    block_size = 4
    
    block1 = select_block_random(N, block_size, random_state=42)
    block2 = select_block_random(N, block_size, random_state=42)
    
    np.testing.assert_array_equal(block1, block2)


def test_select_block_random_errors():
    """Test error handling for random block selection."""
    with pytest.raises(ValueError, match="block_size.*cannot exceed"):
        select_block_random(10, block_size=11, random_state=42)
    
    with pytest.raises(ValueError, match="block_size must be positive"):
        select_block_random(10, block_size=0, random_state=42)


def test_select_block_by_embedding_cluster():
    """Test cluster-based block selection."""
    N = 16
    block_size = 4
    
    # Create permutation and cluster assignments
    pi = np.arange(N, dtype=np.int32)
    cluster_assignments = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    
    block = select_block_by_embedding_cluster(
        pi, cluster_assignments, block_size, random_state=42
    )
    
    assert len(block) == block_size
    assert np.all(block >= 0)
    assert np.all(block < N)
    assert len(np.unique(block)) == block_size


def test_select_block_by_embedding_cluster_small_cluster():
    """Test cluster-based selection when cluster is too small."""
    N = 16
    block_size = 4
    
    pi = np.arange(N, dtype=np.int32)
    # Single cluster with only 2 items
    cluster_assignments = np.array([0, 0] + [1] * 14)
    
    # Should fallback to random
    block = select_block_by_embedding_cluster(
        pi, cluster_assignments, block_size, random_state=42
    )
    
    assert len(block) == block_size


def test_select_block_by_embedding_cluster_errors():
    """Test error handling for cluster-based selection."""
    pi = np.array([0, 1, 2], dtype=np.int32)
    cluster_assignments = np.array([0, 1])  # Mismatch length
    
    with pytest.raises(ValueError, match="pi length.*!= cluster_assignments length"):
        select_block_by_embedding_cluster(pi, cluster_assignments, block_size=2)


def test_get_block_selection_fn_random():
    """Test getting random block selection function."""
    fn = get_block_selection_fn("random")
    
    block = fn(N=16, block_size=4, random_state=42)
    assert len(block) == 4
    assert np.all(block < 16)


def test_get_block_selection_fn_cluster():
    """Test getting cluster-based block selection function."""
    N = 16
    pi = np.arange(N, dtype=np.int32)
    cluster_assignments = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    
    fn = get_block_selection_fn("cluster", pi=pi, cluster_assignments=cluster_assignments)
    
    block = fn(N=N, block_size=4, random_state=42)
    assert len(block) == 4


def test_get_block_selection_fn_cluster_missing_args():
    """Test error when cluster strategy missing required args."""
    with pytest.raises(ValueError, match="pi and cluster_assignments are required"):
        get_block_selection_fn("cluster")


def test_get_block_selection_fn_invalid():
    """Test error for invalid strategy."""
    with pytest.raises(ValueError, match="Unknown block selection strategy"):
        get_block_selection_fn("invalid")

