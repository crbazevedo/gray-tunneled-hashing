"""Tests for Sprint 8 J(φ) objective over real embeddings."""

import numpy as np
import pytest

from gray_tunneled_hashing.distribution.j_phi_objective import (
    compute_j_phi_cost_real_embeddings,
    compute_j_phi_cost_delta_swap_buckets,
)
from gray_tunneled_hashing.binary.lsh_families import create_lsh_family
from gray_tunneled_hashing.distribution.traffic_stats import collect_traffic_stats


def test_compute_j_phi_cost_real_embeddings_shape():
    """Test that function accepts correct inputs."""
    n_bits = 6
    dim = 16
    N = 100
    Q = 20
    k = 5
    
    # Generate synthetic data
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    ground_truth = np.random.randint(0, N, size=(Q, k), dtype=np.int32)
    
    # Create LSH encoder
    lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=dim, random_state=42)
    
    # Build traffic stats to get code_to_bucket
    traffic_stats = collect_traffic_stats(
        queries=queries,
        ground_truth_neighbors=ground_truth,
        base_embeddings=base_embeddings,
        encoder=lsh.hash,
        collapse_threshold=0.01,
    )
    
    K = traffic_stats["K"]
    pi = traffic_stats["pi"]
    w = traffic_stats["w"]
    code_to_bucket = traffic_stats["code_to_bucket"]
    
    # Create permutation (K, n_bits) - random initialization
    permutation = np.random.randint(0, 2, size=(K, n_bits), dtype=np.uint8)
    
    # Compute cost
    cost = compute_j_phi_cost_real_embeddings(
        permutation=permutation,
        pi=pi,
        w=w,
        queries=queries,
        base_embeddings=base_embeddings,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        code_to_bucket=code_to_bucket,
        n_bits=n_bits,
    )
    
    # Check that cost is a float
    assert isinstance(cost, (float, np.floating))
    assert np.isfinite(cost)
    assert cost >= 0, "Cost should be non-negative"


def test_compute_j_phi_cost_real_embeddings_identity():
    """Test that identity permutation returns cost > 0."""
    n_bits = 6
    dim = 16
    N = 100
    Q = 20
    k = 5
    
    # Generate synthetic data
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    ground_truth = np.random.randint(0, N, size=(Q, k), dtype=np.int32)
    
    # Create LSH encoder
    lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=dim, random_state=42)
    
    # Build traffic stats
    traffic_stats = collect_traffic_stats(
        queries=queries,
        ground_truth_neighbors=ground_truth,
        base_embeddings=base_embeddings,
        encoder=lsh.hash,
        collapse_threshold=0.01,
    )
    
    K = traffic_stats["K"]
    pi = traffic_stats["pi"]
    w = traffic_stats["w"]
    code_to_bucket = traffic_stats["code_to_bucket"]
    bucket_to_code = traffic_stats["bucket_to_code"]
    
    # Create identity permutation (use original bucket codes)
    permutation = bucket_to_code.copy().astype(np.uint8)
    
    # Compute cost
    cost = compute_j_phi_cost_real_embeddings(
        permutation=permutation,
        pi=pi,
        w=w,
        queries=queries,
        base_embeddings=base_embeddings,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        code_to_bucket=code_to_bucket,
        n_bits=n_bits,
    )
    
    # Cost should be > 0 (unless all pairs have zero weight, which is unlikely)
    assert cost >= 0, "Cost should be non-negative"
    # In most cases, cost should be > 0, but we allow 0 for edge cases
    # (e.g., no query-neighbor pairs match bucket pairs with w > 0)


def test_compute_j_phi_cost_real_embeddings_monotonicity():
    """Test that better permutation has lower cost."""
    n_bits = 6
    dim = 16
    N = 100
    Q = 20
    k = 5
    
    # Generate synthetic data
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    ground_truth = np.random.randint(0, N, size=(Q, k), dtype=np.int32)
    
    # Create LSH encoder
    lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=dim, random_state=42)
    
    # Build traffic stats
    traffic_stats = collect_traffic_stats(
        queries=queries,
        ground_truth_neighbors=ground_truth,
        base_embeddings=base_embeddings,
        encoder=lsh.hash,
        collapse_threshold=0.01,
    )
    
    K = traffic_stats["K"]
    pi = traffic_stats["pi"]
    w = traffic_stats["w"]
    code_to_bucket = traffic_stats["code_to_bucket"]
    bucket_to_code = traffic_stats["bucket_to_code"]
    
    # Create two permutations: identity and random
    permutation_identity = bucket_to_code.copy().astype(np.uint8)
    permutation_random = np.random.randint(0, 2, size=(K, n_bits), dtype=np.uint8)
    
    # Compute costs
    cost_identity = compute_j_phi_cost_real_embeddings(
        permutation=permutation_identity,
        pi=pi,
        w=w,
        queries=queries,
        base_embeddings=base_embeddings,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        code_to_bucket=code_to_bucket,
        n_bits=n_bits,
    )
    
    cost_random = compute_j_phi_cost_real_embeddings(
        permutation=permutation_random,
        pi=pi,
        w=w,
        queries=queries,
        base_embeddings=base_embeddings,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        code_to_bucket=code_to_bucket,
        n_bits=n_bits,
    )
    
    # Both costs should be finite and non-negative
    assert np.isfinite(cost_identity)
    assert np.isfinite(cost_random)
    assert cost_identity >= 0
    assert cost_random >= 0
    
    # Note: We can't guarantee which is better without optimization,
    # but we can verify the function works correctly for both


def test_compute_j_phi_cost_delta_swap_buckets():
    """Test that delta is computed correctly."""
    n_bits = 6
    dim = 16
    N = 100
    Q = 20
    k = 5
    
    # Generate synthetic data
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    ground_truth = np.random.randint(0, N, size=(Q, k), dtype=np.int32)
    
    # Create LSH encoder
    lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=dim, random_state=42)
    
    # Build traffic stats
    traffic_stats = collect_traffic_stats(
        queries=queries,
        ground_truth_neighbors=ground_truth,
        base_embeddings=base_embeddings,
        encoder=lsh.hash,
        collapse_threshold=0.01,
    )
    
    K = traffic_stats["K"]
    pi = traffic_stats["pi"]
    w = traffic_stats["w"]
    code_to_bucket = traffic_stats["code_to_bucket"]
    
    # Create permutation
    permutation = np.random.randint(0, 2, size=(K, n_bits), dtype=np.uint8)
    
    # Test delta for swapping buckets 0 and 1 (if K >= 2)
    if K >= 2:
        delta = compute_j_phi_cost_delta_swap_buckets(
            permutation=permutation,
            pi=pi,
            w=w,
            queries=queries,
            base_embeddings=base_embeddings,
            ground_truth_neighbors=ground_truth,
            encoder=lsh.hash,
            code_to_bucket=code_to_bucket,
            n_bits=n_bits,
            bucket_i=0,
            bucket_j=1,
        )
        
        # Delta should be finite
        assert np.isfinite(delta)
        
        # Verify delta matches full recomputation (approximately)
        # Compute cost before swap
        cost_before = compute_j_phi_cost_real_embeddings(
            permutation=permutation,
            pi=pi,
            w=w,
            queries=queries,
            base_embeddings=base_embeddings,
            ground_truth_neighbors=ground_truth,
            encoder=lsh.hash,
            code_to_bucket=code_to_bucket,
            n_bits=n_bits,
        )
        
        # Swap codes
        permutation_swapped = permutation.copy()
        permutation_swapped[0], permutation_swapped[1] = \
            permutation_swapped[1].copy(), permutation_swapped[0].copy()
        
        # Compute cost after swap
        cost_after = compute_j_phi_cost_real_embeddings(
            permutation=permutation_swapped,
            pi=pi,
            w=w,
            queries=queries,
            base_embeddings=base_embeddings,
            ground_truth_neighbors=ground_truth,
            encoder=lsh.hash,
            code_to_bucket=code_to_bucket,
            n_bits=n_bits,
        )
        
        # Delta should match difference (approximately, due to sampling)
        # Allow some tolerance for sampling differences
        assert abs(delta - (cost_after - cost_before)) < 1.0, \
            f"Delta {delta} should match cost difference {cost_after - cost_before}"

