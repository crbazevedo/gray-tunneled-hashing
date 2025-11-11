"""Tests for Sprint 9 J(φ)-aware tunneling."""

import numpy as np
import pytest
from gray_tunneled_hashing.distribution.j_phi_objective import tunneling_step_j_phi


def test_tunneling_step_j_phi_basic():
    """Test basic tunneling step execution."""
    K = 8
    n_bits = 4
    
    permutation = np.random.randint(0, 2, size=(K, n_bits), dtype=np.uint8)
    pi = np.ones(K) / K
    w = np.eye(K) * 0.5 + np.ones((K, K)) * 0.1
    
    queries = np.random.randn(10, 8)
    base_embeddings = np.random.randn(20, 8)
    ground_truth_neighbors = np.random.randint(0, 20, size=(10, 5))
    
    def mock_encoder(emb):
        return np.random.randint(0, 2, size=(len(emb), n_bits), dtype=np.uint8)
    
    code_to_bucket = {tuple(np.random.randint(0, 2, n_bits).tolist()): i for i in range(K)}
    
    perm_new, delta = tunneling_step_j_phi(
        permutation=permutation,
        pi=pi,
        w=w,
        queries=queries,
        base_embeddings=base_embeddings,
        ground_truth_neighbors=ground_truth_neighbors,
        encoder=mock_encoder,
        code_to_bucket=code_to_bucket,
        n_bits=n_bits,
        radii=None,
        radius_weights=None,
        sample_size_pairs=5,
        block_size=4,
        num_blocks=5,
        random_state=42,
    )
    
    assert perm_new.shape == permutation.shape
    assert isinstance(delta, float)
    # Delta should be <= 0 (improvement or no change)
    assert delta <= 1e-10


def test_tunneling_step_j_phi_multi_radius():
    """Test tunneling with multi-radius objective."""
    K = 8
    n_bits = 4
    
    permutation = np.random.randint(0, 2, size=(K, n_bits), dtype=np.uint8)
    pi = np.ones(K) / K
    w = np.eye(K) * 0.5 + np.ones((K, K)) * 0.1
    
    queries = np.random.randn(10, 8)
    base_embeddings = np.random.randn(20, 8)
    ground_truth_neighbors = np.random.randint(0, 20, size=(10, 5))
    
    def mock_encoder(emb):
        return np.random.randint(0, 2, size=(len(emb), n_bits), dtype=np.uint8)
    
    code_to_bucket = {tuple(np.random.randint(0, 2, n_bits).tolist()): i for i in range(K)}
    
    from gray_tunneled_hashing.distribution.j_phi_objective import validate_radius_weights
    
    radii = [1, 2, 3]
    radius_weights = validate_radius_weights(radii)
    
    perm_new, delta = tunneling_step_j_phi(
        permutation=permutation,
        pi=pi,
        w=w,
        queries=queries,
        base_embeddings=base_embeddings,
        ground_truth_neighbors=ground_truth_neighbors,
        encoder=mock_encoder,
        code_to_bucket=code_to_bucket,
        n_bits=n_bits,
        radii=radii,
        radius_weights=radius_weights,
        sample_size_pairs=5,
        block_size=4,
        num_blocks=5,
        random_state=42,
    )
    
    assert perm_new.shape == permutation.shape
    assert isinstance(delta, float)
    assert delta <= 1e-10


def test_tunneling_step_j_phi_no_improvement():
    """Test tunneling when no improvement is found."""
    K = 4
    n_bits = 4
    
    # Use a permutation that's already optimal (identity)
    permutation = np.random.randint(0, 2, size=(K, n_bits), dtype=np.uint8)
    pi = np.ones(K) / K
    w = np.eye(K) * 0.5
    
    queries = np.random.randn(5, 8)
    base_embeddings = np.random.randn(10, 8)
    ground_truth_neighbors = np.random.randint(0, 10, size=(5, 3))
    
    def mock_encoder(emb):
        return np.random.randint(0, 2, size=(len(emb), n_bits), dtype=np.uint8)
    
    code_to_bucket = {tuple(np.random.randint(0, 2, n_bits).tolist()): i for i in range(K)}
    
    perm_new, delta = tunneling_step_j_phi(
        permutation=permutation,
        pi=pi,
        w=w,
        queries=queries,
        base_embeddings=base_embeddings,
        ground_truth_neighbors=ground_truth_neighbors,
        encoder=mock_encoder,
        code_to_bucket=code_to_bucket,
        n_bits=n_bits,
        radii=None,
        radius_weights=None,
        sample_size_pairs=2,
        block_size=2,
        num_blocks=2,
        random_state=42,
    )
    
    # Should return original permutation or unchanged permutation
    assert perm_new.shape == permutation.shape
    assert delta >= -1e-10  # No improvement or very small


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

