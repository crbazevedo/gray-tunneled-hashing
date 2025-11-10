"""Tests for distribution-aware GTH components."""

import numpy as np
import pytest

from gray_tunneled_hashing.distribution.traffic_stats import (
    collect_traffic_stats,
    build_weighted_distance_matrix,
)
from gray_tunneled_hashing.distribution.pipeline import (
    build_distribution_aware_index,
    apply_permutation,
)
from gray_tunneled_hashing.algorithms.gray_tunneled_hasher import GrayTunneledHasher


def test_build_weighted_distance_matrix():
    """Test weighted distance matrix construction."""
    K = 5
    dim = 8
    
    pi = np.array([0.3, 0.2, 0.2, 0.2, 0.1], dtype=np.float64)
    w = np.eye(K, dtype=np.float64) * 0.5 + np.ones((K, K), dtype=np.float64) * 0.1
    # Normalize rows
    w = w / w.sum(axis=1, keepdims=True)
    
    bucket_embeddings = np.random.randn(K, dim).astype(np.float32)
    
    # Test with semantic distances
    D_weighted = build_weighted_distance_matrix(
        pi=pi,
        w=w,
        bucket_embeddings=bucket_embeddings,
        use_semantic_distances=True,
    )
    
    assert D_weighted.shape == (K, K)
    assert np.all(D_weighted >= 0)
    
    # Check that diagonal is zero (or very small due to numerical precision)
    assert np.allclose(np.diag(D_weighted), 0.0, atol=1e-6)
    
    # Test without semantic distances
    D_weighted_pure = build_weighted_distance_matrix(
        pi=pi,
        w=w,
        bucket_embeddings=bucket_embeddings,
        use_semantic_distances=False,
    )
    
    assert D_weighted_pure.shape == (K, K)
    assert np.all(D_weighted_pure >= 0)
    
    # Pure traffic should be different (no semantic distance multiplication)
    # Note: D_weighted_pure can be larger or smaller than D_weighted depending on
    # the relationship between pi*w and semantic distances
    assert not np.allclose(D_weighted_pure, D_weighted, atol=1e-6)


def test_collect_traffic_stats():
    """Test traffic statistics collection."""
    n_bits = 4
    dim = 8
    Q = 100
    k = 5
    
    # Generate synthetic data
    base_embeddings = np.random.randn(200, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    
    # Create ground truth neighbors
    ground_truth = np.random.randint(0, 200, size=(Q, k), dtype=np.int32)
    
    # Simple encoder: sign thresholding
    def encoder(emb):
        return (emb > 0).astype(bool)
    
    # Collect stats
    stats = collect_traffic_stats(
        queries=queries,
        ground_truth_neighbors=ground_truth,
        base_embeddings=base_embeddings,
        encoder=encoder,
        k=k,
    )
    
    assert "pi" in stats
    assert "w" in stats
    assert "bucket_to_code" in stats
    assert "code_to_bucket" in stats
    assert "bucket_counts" in stats
    assert "n_bits" in stats
    assert "K" in stats
    
    pi = stats["pi"]
    w = stats["w"]
    K = stats["K"]
    
    assert len(pi) == K
    assert w.shape == (K, K)
    assert np.allclose(pi.sum(), 1.0, atol=1e-6)
    # w rows may not sum to 1 if some buckets have no neighbors
    # Check that rows are either normalized or zero
    row_sums = w.sum(axis=1)
    assert np.all((row_sums == 0) | np.isclose(row_sums, 1.0, atol=1e-6))


def test_gray_tunneled_hasher_fit_with_traffic():
    """Test GrayTunneledHasher.fit_with_traffic()."""
    n_bits = 5
    K = 16  # K < 2**n_bits = 32
    dim = 8
    
    # Generate synthetic bucket embeddings
    bucket_embeddings = np.random.randn(K, dim).astype(np.float32)
    
    # Create traffic stats
    pi = np.ones(K, dtype=np.float64) / K  # Uniform
    w = np.eye(K, dtype=np.float64) * 0.5 + np.ones((K, K), dtype=np.float64) * 0.1
    w = w / w.sum(axis=1, keepdims=True)
    
    # Create hasher
    hasher = GrayTunneledHasher(
        n_bits=n_bits,
        block_size=4,
        max_two_swap_iters=10,
        num_tunneling_steps=2,
        mode="two_swap_only",  # Faster for testing
        random_state=42,
    )
    
    # Fit with traffic
    hasher.fit_with_traffic(
        bucket_embeddings=bucket_embeddings,
        pi=pi,
        w=w,
        use_semantic_distances=True,
    )
    
    assert hasher.is_fitted
    assert hasher.pi_ is not None
    assert hasher.cost_ is not None
    assert np.isfinite(hasher.cost_)
    
    # Check permutation shape
    assert hasher.pi_.shape == (2 ** n_bits,)


def test_build_distribution_aware_index_smoke():
    """Smoke test for build_distribution_aware_index."""
    n_bits = 4
    n_codes = 8
    dim = 8
    Q = 50
    k = 3
    
    # Generate synthetic data
    base_embeddings = np.random.randn(100, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    ground_truth = np.random.randint(0, 100, size=(Q, k), dtype=np.int32)
    
    # Simple encoder: sign thresholding, but limit to n_bits
    def encoder(emb):
        codes = (emb > 0).astype(bool)
        # Take only first n_bits
        return codes[:, :n_bits]
    
    # Build index
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=encoder,
        n_bits=n_bits,
        n_codes=n_codes,
        use_codebook=True,
        use_semantic_distances=True,
        block_size=4,
        max_two_swap_iters=5,
        num_tunneling_steps=1,
        mode="two_swap_only",  # Faster
        random_state=42,
    )
    
    assert index_obj.hasher is not None
    assert index_obj.bucket_to_code is not None
    assert index_obj.code_to_bucket is not None
    assert index_obj.bucket_embeddings is not None
    assert index_obj.pi is not None
    assert index_obj.w is not None
    assert index_obj.permutation is not None
    assert index_obj.n_bits == n_bits
    assert index_obj.K > 0


def test_apply_permutation():
    """Test apply_permutation function."""
    n_bits = 3
    N = 2 ** n_bits
    
    # Create some codes
    codes = np.random.randint(0, 2, size=(10, n_bits), dtype=bool)
    
    # Create bucket mappings
    unique_codes = np.unique(codes, axis=0)
    K = len(unique_codes)
    
    bucket_to_code = unique_codes
    code_to_bucket = {tuple(code.tolist()): i for i, code in enumerate(unique_codes)}
    
    # Create identity permutation
    permutation = np.arange(N, dtype=np.int32)
    
    # Apply permutation
    permuted_codes = apply_permutation(
        codes=codes,
        bucket_to_code=bucket_to_code,
        code_to_bucket=code_to_bucket,
        permutation=permutation,
        n_bits=n_bits,
    )
    
    assert permuted_codes.shape == codes.shape
    assert permuted_codes.dtype == codes.dtype

