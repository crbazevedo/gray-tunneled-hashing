"""Tests for Sprint 9 multi-radius objective."""

import numpy as np
import pytest
from gray_tunneled_hashing.distribution.j_phi_objective import (
    validate_radius_weights,
    compute_j_phi_cost_multi_radius,
    compute_j_phi_cost_delta_swap_buckets_multi_radius,
)


def test_validate_radius_weights_default():
    """Test default weight generation."""
    radii = [1, 2, 3]
    weights = validate_radius_weights(radii)
    
    assert len(weights) == 3
    assert weights[0] > weights[1] > weights[2]
    assert all(w > 0 for w in weights)
    assert weights[0] == 1.0  # w_1 = 1.0
    assert np.isclose(weights[1], 0.5)  # w_2 = 0.5
    assert np.isclose(weights[2], 0.25)  # w_3 = 0.25


def test_validate_radius_weights_custom():
    """Test custom weights validation."""
    radii = [1, 2, 3]
    custom_weights = np.array([1.0, 0.6, 0.3])
    
    weights = validate_radius_weights(radii, custom_weights)
    np.testing.assert_array_equal(weights, custom_weights)


def test_validate_radius_weights_constraint_violation():
    """Test that constraint violations are caught."""
    radii = [1, 2, 3]
    invalid_weights = np.array([1.0, 0.8, 0.9])  # w_3 > w_2
    
    with pytest.raises(ValueError, match="Weight constraint violated"):
        validate_radius_weights(radii, invalid_weights)


def test_validate_radius_weights_normalize():
    """Test weight normalization."""
    radii = [1, 2]
    weights = validate_radius_weights(radii, normalize=True)
    
    assert np.isclose(np.sum(weights), 1.0)


def test_compute_j_phi_cost_multi_radius_basic():
    """Test basic multi-radius cost computation."""
    K = 4
    n_bits = 4
    
    # Simple permutation: identity
    permutation = np.random.randint(0, 2, size=(K, n_bits), dtype=np.uint8)
    pi = np.ones(K) / K
    w = np.eye(K) * 0.5 + np.ones((K, K)) * 0.1
    
    # Mock encoder and data
    queries = np.random.randn(10, 8)
    base_embeddings = np.random.randn(20, 8)
    ground_truth_neighbors = np.random.randint(0, 20, size=(10, 5))
    
    def mock_encoder(emb):
        return np.random.randint(0, 2, size=(len(emb), n_bits), dtype=np.uint8)
    
    code_to_bucket = {tuple(np.random.randint(0, 2, n_bits).tolist()): i for i in range(K)}
    
    radii = [1, 2]
    radius_weights = validate_radius_weights(radii)
    
    cost = compute_j_phi_cost_multi_radius(
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
        sample_size=5,
    )
    
    assert isinstance(cost, float)
    assert cost >= 0


def test_compute_j_phi_cost_delta_swap_buckets_multi_radius():
    """Test multi-radius delta computation."""
    K = 4
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
    
    radii = [1, 2]
    radius_weights = validate_radius_weights(radii)
    
    delta = compute_j_phi_cost_delta_swap_buckets_multi_radius(
        permutation=permutation,
        pi=pi,
        w=w,
        queries=queries,
        base_embeddings=base_embeddings,
        ground_truth_neighbors=ground_truth_neighbors,
        encoder=mock_encoder,
        code_to_bucket=code_to_bucket,
        n_bits=n_bits,
        bucket_i=0,
        bucket_j=1,
        radii=radii,
        radius_weights=radius_weights,
        sample_size=5,
    )
    
    assert isinstance(delta, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

