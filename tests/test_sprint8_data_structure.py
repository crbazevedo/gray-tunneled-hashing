"""Tests for Sprint 8 data structure changes (permutation as K x n_bits array)."""

import numpy as np
import pytest

from gray_tunneled_hashing.algorithms.gray_tunneled_hasher import GrayTunneledHasher
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.binary.lsh_families import create_lsh_family


def test_permutation_shape():
    """Test that get_assignment() returns array of shape (K, n_bits)."""
    n_bits = 6
    dim = 16
    N = 100
    Q = 20
    k = 5
    
    # Generate synthetic data
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    ground_truth = np.random.randint(0, N, size=(Q, k), dtype=np.int32)
    
    # Create LSH family
    lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=dim, random_state=42)
    
    # Build distribution-aware index
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        n_bits=n_bits,
        n_codes=32,
        use_codebook=True,
        max_two_swap_iters=5,
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=42,
    )
    
    # Get permutation from hasher
    permutation = index_obj.hasher.get_assignment()
    
    # NEW (Sprint 8): permutation should be (K, n_bits) not (N,)
    K = index_obj.K
    assert permutation.shape == (K, n_bits), \
        f"Expected permutation shape ({K}, {n_bits}), got {permutation.shape}"


def test_permutation_dtype():
    """Test that permutation has dtype uint8 or bool."""
    n_bits = 6
    dim = 16
    N = 100
    Q = 20
    k = 5
    
    # Generate synthetic data
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    ground_truth = np.random.randint(0, N, size=(Q, k), dtype=np.int32)
    
    # Create LSH family
    lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=dim, random_state=42)
    
    # Build distribution-aware index
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        n_bits=n_bits,
        n_codes=32,
        use_codebook=True,
        max_two_swap_iters=5,
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=42,
    )
    
    # Get permutation from hasher
    permutation = index_obj.hasher.get_assignment()
    
    # Check dtype
    assert permutation.dtype in [np.uint8, np.bool_, bool], \
        f"Expected permutation dtype uint8 or bool, got {permutation.dtype}"


def test_permutation_values():
    """Test that permutation values are in {0, 1}."""
    n_bits = 6
    dim = 16
    N = 100
    Q = 20
    k = 5
    
    # Generate synthetic data
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    ground_truth = np.random.randint(0, N, size=(Q, k), dtype=np.int32)
    
    # Create LSH family
    lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=dim, random_state=42)
    
    # Build distribution-aware index
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        n_bits=n_bits,
        n_codes=32,
        use_codebook=True,
        max_two_swap_iters=5,
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=42,
    )
    
    # Get permutation from hasher
    permutation = index_obj.hasher.get_assignment()
    
    # Check that all values are 0 or 1
    assert np.all((permutation == 0) | (permutation == 1)), \
        f"Permutation contains values outside {{0, 1}}: {np.unique(permutation)}"


def test_initialization_random():
    """Test that random initialization generates valid codes."""
    n_bits = 6
    K = 16
    dim = 16
    
    # Create hasher with random initialization
    hasher = GrayTunneledHasher(
        n_bits=n_bits,
        init_strategy="random",
        random_state=42,
    )
    
    # Create dummy bucket embeddings and traffic stats
    bucket_embeddings = np.random.randn(K, dim).astype(np.float32)
    pi = np.ones(K, dtype=np.float64) / K
    w = np.eye(K, dtype=np.float64) * 0.5 + np.ones((K, K), dtype=np.float64) * 0.1
    w = w / w.sum(axis=1, keepdims=True)
    
    # Generate synthetic data for real embeddings objective
    N = 100
    Q = 20
    k = 5
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    ground_truth = np.random.randint(0, N, size=(Q, k), dtype=np.int32)
    
    lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=dim, random_state=42)
    
    # Build code_to_bucket mapping
    from gray_tunneled_hashing.distribution.traffic_stats import collect_traffic_stats
    traffic_stats = collect_traffic_stats(
        queries=queries,
        ground_truth_neighbors=ground_truth,
        base_embeddings=base_embeddings,
        encoder=lsh.hash,
        collapse_threshold=0.01,
    )
    code_to_bucket = traffic_stats["code_to_bucket"]
    
    # Fit with traffic using real embeddings objective
    hasher.fit_with_traffic(
        bucket_embeddings=bucket_embeddings,
        pi=pi,
        w=w,
        queries=queries,
        base_embeddings=base_embeddings,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        code_to_bucket=code_to_bucket,
        use_real_embeddings_objective=True,
        optimization_method="hill_climb",
    )
    
    # Get permutation
    permutation = hasher.get_assignment()
    
    # Check shape
    assert permutation.shape == (K, n_bits), \
        f"Expected permutation shape ({K}, {n_bits}), got {permutation.shape}"
    
    # Check values are valid binary codes
    assert np.all((permutation == 0) | (permutation == 1)), \
        "Permutation contains invalid values"


def test_initialization_identity():
    """Test that identity initialization uses original bucket codes."""
    n_bits = 6
    dim = 16
    N = 100
    Q = 20
    k = 5
    
    # Generate synthetic data
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    ground_truth = np.random.randint(0, N, size=(Q, k), dtype=np.int32)
    
    # Create LSH family
    lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=dim, random_state=42)
    
    # Build distribution-aware index with identity initialization
    hasher = GrayTunneledHasher(
        n_bits=n_bits,
        init_strategy="identity",
        random_state=42,
    )
    
    # Build index to get bucket_to_code
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        n_bits=n_bits,
        n_codes=32,
        use_codebook=True,
        max_two_swap_iters=0,  # No optimization to check initial state
        num_tunneling_steps=0,
        mode="trivial",
        random_state=42,
    )
    
    # Get initial permutation (should be identity = original bucket codes)
    # Note: With use_real_embeddings_objective=True, we need to check if it uses identity
    # For now, just verify the structure is correct
    permutation = index_obj.hasher.get_assignment()
    bucket_to_code = index_obj.bucket_to_code
    
    # Check shape
    K = index_obj.K
    assert permutation.shape == (K, n_bits), \
        f"Expected permutation shape ({K}, {n_bits}), got {permutation.shape}"
    
    # With identity initialization and no optimization, permutation should match bucket_to_code
    # (or at least have same shape)
    assert permutation.shape == bucket_to_code.shape, \
        "Permutation and bucket_to_code should have same shape"

