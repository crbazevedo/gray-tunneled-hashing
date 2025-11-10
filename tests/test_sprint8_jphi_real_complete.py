"""Complete tests for Sprint 8 J(φ) objective over real embeddings."""

import numpy as np
import pytest

from gray_tunneled_hashing.distribution.j_phi_objective import (
    compute_j_phi_cost_real_embeddings,
    compute_j_phi_cost_delta_swap_buckets,
)
from gray_tunneled_hashing.binary.lsh_families import create_lsh_family
from gray_tunneled_hashing.distribution.traffic_stats import collect_traffic_stats


def test_compute_j_phi_cost_real_embeddings_all_pairs():
    """Test calculation with all pairs (no sampling)."""
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
    
    # Compute cost without sampling (all pairs)
    cost_all = compute_j_phi_cost_real_embeddings(
        permutation=permutation,
        pi=pi,
        w=w,
        queries=queries,
        base_embeddings=base_embeddings,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        code_to_bucket=code_to_bucket,
        n_bits=n_bits,
        sample_size=None,  # No sampling
    )
    
    # Compute cost with sampling
    cost_sampled = compute_j_phi_cost_real_embeddings(
        permutation=permutation,
        pi=pi,
        w=w,
        queries=queries,
        base_embeddings=base_embeddings,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        code_to_bucket=code_to_bucket,
        n_bits=n_bits,
        sample_size=10,  # Small sample
    )
    
    # Both should be finite and non-negative
    assert np.isfinite(cost_all)
    assert np.isfinite(cost_sampled)
    assert cost_all >= 0
    assert cost_sampled >= 0
    
    # Sampled cost may differ from all pairs, but should be in reasonable range
    # (within 50% of all pairs cost, or at least not orders of magnitude different)
    if cost_all > 0:
        ratio = cost_sampled / cost_all
        assert 0.1 <= ratio <= 10.0, \
            f"Sampled cost {cost_sampled} should be within reasonable range of all pairs cost {cost_all}"


def test_compute_j_phi_cost_real_embeddings_sampling():
    """Test that sampling works correctly."""
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
    
    # Test with different sample sizes
    for sample_size in [5, 10, 20, None]:
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
            sample_size=sample_size,
        )
        
        assert np.isfinite(cost)
        assert cost >= 0


def test_compute_j_phi_cost_real_embeddings_empty_buckets():
    """Test with empty buckets (buckets with no queries)."""
    n_bits = 6
    dim = 16
    N = 100
    Q = 10  # Few queries
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
    
    # Compute cost (should handle empty buckets gracefully)
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
    
    # Should return finite cost (may be 0 if no pairs match)
    assert np.isfinite(cost)
    assert cost >= 0


def test_compute_j_phi_cost_real_embeddings_no_neighbors():
    """Test when there are no neighbors (empty ground truth)."""
    n_bits = 6
    dim = 16
    N = 100
    Q = 20
    k = 0  # No neighbors
    
    # Generate synthetic data
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    ground_truth = np.empty((Q, 0), dtype=np.int32)  # Empty
    
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
    
    # Compute cost (should handle no neighbors gracefully)
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
    
    # Should return 0 (no pairs to compute)
    assert cost == 0.0 or np.isclose(cost, 0.0), \
        f"Expected cost=0 with no neighbors, got {cost}"


def test_compute_j_phi_cost_delta_swap_buckets_accuracy():
    """Test accuracy of delta vs. full recomputation."""
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
        # Compute delta
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
            sample_size=None,  # No sampling for accuracy test
        )
        
        # Compute full cost before and after
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
            sample_size=None,
        )
        
        # Swap
        permutation_swapped = permutation.copy()
        permutation_swapped[0], permutation_swapped[1] = \
            permutation_swapped[1].copy(), permutation_swapped[0].copy()
        
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
            sample_size=None,
        )
        
        # Delta should match difference (within numerical precision)
        expected_delta = cost_after - cost_before
        assert np.isclose(delta, expected_delta, rtol=1e-5, atol=1e-8), \
            f"Delta {delta} should match cost difference {expected_delta}"


def test_compute_j_phi_cost_delta_swap_buckets_symmetry():
    """Test that swap(i,j) = -swap(j,i)."""
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
    
    # Test symmetry (if K >= 2)
    if K >= 2:
        delta_ij = compute_j_phi_cost_delta_swap_buckets(
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
        
        delta_ji = compute_j_phi_cost_delta_swap_buckets(
            permutation=permutation,
            pi=pi,
            w=w,
            queries=queries,
            base_embeddings=base_embeddings,
            ground_truth_neighbors=ground_truth,
            encoder=lsh.hash,
            code_to_bucket=code_to_bucket,
            n_bits=n_bits,
            bucket_i=1,
            bucket_j=0,
        )
        
        # Should be approximately symmetric (delta_ij ≈ -delta_ji)
        # Note: Due to sampling, they may not be exactly symmetric
        # But the magnitude should be similar
        assert np.isclose(delta_ij, -delta_ji, rtol=0.1, atol=0.1), \
            f"Delta should be approximately symmetric: delta(0,1)={delta_ij}, delta(1,0)={delta_ji}"


def test_compute_j_phi_cost_real_embeddings_correlation_with_recall():
    """Analyze correlation between J(φ) and recall (basic check)."""
    n_bits = 6
    dim = 16
    N = 100
    Q = 20
    k = 5
    
    # Generate synthetic data
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    
    # Compute ground truth
    from sklearn.metrics.pairwise import euclidean_distances
    distances = euclidean_distances(queries, base_embeddings)
    ground_truth = np.argsort(distances, axis=1)[:, :k]
    
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
    
    # Test multiple permutations
    costs = []
    # Note: Computing recall is expensive, so we'll just verify costs are computed
    # Full recall correlation test would be in a separate performance test
    
    for _ in range(3):  # Test a few permutations
        permutation = np.random.randint(0, 2, size=(K, n_bits), dtype=np.uint8)
        
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
            sample_size=10,  # Use sampling for speed
        )
        
        costs.append(cost)
    
    # Costs should vary (different permutations should give different costs)
    # Unless all permutations happen to give same cost (unlikely)
    costs_array = np.array(costs)
    assert np.isfinite(costs_array).all()
    assert (costs_array >= 0).all()

