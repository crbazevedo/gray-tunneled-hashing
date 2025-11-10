"""
Testes de regressão para Sprint 8.

Verifica compatibilidade com código antigo, APIs públicas e execução de testes antigos.
"""

import numpy as np
import pytest
from sklearn.metrics.pairwise import euclidean_distances

from gray_tunneled_hashing.algorithms.gray_tunneled_hasher import GrayTunneledHasher
from gray_tunneled_hashing.binary.lsh_families import create_lsh_family
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.api.query_pipeline import query_with_hamming_ball


def test_gray_tunneled_hasher_api_compatibility():
    """Test that GrayTunneledHasher public API is still compatible."""
    n_bits = 6
    
    # Test initialization
    hasher = GrayTunneledHasher(n_bits=n_bits)
    assert hasher.n_bits == n_bits
    assert not hasher.is_fitted
    
    # Test that get_assignment raises error when not fitted
    with pytest.raises(ValueError, match="must be fitted"):
        _ = hasher.get_assignment()
    
    # Test that hasher can be fitted (minimal test)
    dim = 16
    N = 50
    Q = 10
    k = 3
    
    np.random.seed(42)
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    distances = euclidean_distances(queries, base_embeddings)
    ground_truth = np.argsort(distances, axis=1)[:, :k]
    
    lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=dim, random_state=42)
    
    # Build index to get traffic stats
    from gray_tunneled_hashing.distribution.traffic_stats import collect_traffic_stats
    traffic_stats = collect_traffic_stats(
        queries=queries,
        ground_truth_neighbors=ground_truth,
        base_embeddings=base_embeddings,
        encoder=lsh.hash,
        collapse_threshold=0.01,
    )
    
    # Create dummy bucket embeddings
    K = len(traffic_stats["pi"])
    bucket_embeddings = np.random.randn(K, dim).astype(np.float32)
    
    # Fit hasher
    hasher.fit_with_traffic(
        bucket_embeddings=bucket_embeddings,
        pi=traffic_stats["pi"],
        w=traffic_stats["w"],
        queries=queries,
        base_embeddings=base_embeddings,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        code_to_bucket=traffic_stats["code_to_bucket"],
        use_real_embeddings_objective=True,
        optimization_method="hill_climb",
    )
    
    # Test that is_fitted is True
    assert hasher.is_fitted
    
    # Test that get_assignment works
    permutation = hasher.get_assignment()
    assert permutation is not None
    assert permutation.shape == (K, n_bits)
    assert permutation.dtype in [np.uint8, bool]


def test_build_distribution_aware_index_api_compatibility():
    """Test that build_distribution_aware_index API is still compatible."""
    n_bits = 6
    dim = 16
    N = 50
    Q = 10
    k = 3
    
    np.random.seed(42)
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    distances = euclidean_distances(queries, base_embeddings)
    ground_truth = np.argsort(distances, axis=1)[:, :k]
    
    lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=dim, random_state=42)
    
    # Test that function can be called with required parameters
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        n_bits=n_bits,
        n_codes=16,
        use_codebook=True,
        max_two_swap_iters=5,
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=42,
    )
    
    # Test that returned object has expected attributes
    assert hasattr(index_obj, 'hasher')
    assert hasattr(index_obj, 'code_to_bucket')
    assert hasattr(index_obj, 'bucket_to_code')
    assert hasattr(index_obj, 'pi')
    assert hasattr(index_obj, 'w')
    assert hasattr(index_obj, 'n_bits')
    
    # Test that hasher is fitted
    assert index_obj.hasher.is_fitted
    
    # Test that permutation has correct shape
    permutation = index_obj.hasher.get_assignment()
    assert permutation.shape == (len(index_obj.pi), n_bits)


def test_query_with_hamming_ball_api_compatibility():
    """Test that query_with_hamming_ball API is still compatible."""
    n_bits = 6
    dim = 16
    N = 50
    Q = 10
    k = 3
    
    np.random.seed(42)
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    distances = euclidean_distances(queries, base_embeddings)
    ground_truth = np.argsort(distances, axis=1)[:, :k]
    
    lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=dim, random_state=42)
    
    # Build index
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        n_bits=n_bits,
        n_codes=16,
        use_codebook=True,
        max_two_swap_iters=5,
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=42,
    )
    
    permutation = index_obj.hasher.get_assignment()
    query_codes = lsh.hash(queries)
    
    # Test that query_with_hamming_ball can be called
    result = query_with_hamming_ball(
        query_code=query_codes[0].astype(bool),
        permutation=permutation,
        code_to_bucket=index_obj.code_to_bucket,
        bucket_to_code=index_obj.bucket_to_code,
        n_bits=n_bits,
        hamming_radius=1,
    )
    
    # Test that result has expected attributes
    assert hasattr(result, 'query_code')
    assert hasattr(result, 'permuted_code')
    assert hasattr(result, 'candidate_codes')
    assert hasattr(result, 'candidate_indices')
    assert hasattr(result, 'hamming_radius')
    assert hasattr(result, 'n_candidates')
    
    # Test that result types are correct
    assert isinstance(result.n_candidates, (int, np.integer))
    assert result.n_candidates >= 0


def test_permutation_shape_compatibility():
    """Test that permutation shape is consistent with new structure."""
    n_bits = 6
    dim = 16
    N = 50
    Q = 10
    k = 3
    
    np.random.seed(42)
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    distances = euclidean_distances(queries, base_embeddings)
    ground_truth = np.argsort(distances, axis=1)[:, :k]
    
    lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=dim, random_state=42)
    
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        n_bits=n_bits,
        n_codes=16,
        use_codebook=True,
        max_two_swap_iters=5,
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=42,
    )
    
    permutation = index_obj.hasher.get_assignment()
    
    # New structure: (K, n_bits)
    K = len(index_obj.pi)
    assert permutation.shape == (K, n_bits), \
        f"Expected shape ({K}, {n_bits}), got {permutation.shape}"
    
    # Values should be binary
    assert np.all((permutation == 0) | (permutation == 1)), \
        "Permutation should contain only binary values"


def test_backward_compatibility_imports():
    """Test that all public imports still work."""
    # Test that main classes can be imported
    from gray_tunneled_hashing.algorithms.gray_tunneled_hasher import GrayTunneledHasher
    from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
    from gray_tunneled_hashing.api.query_pipeline import query_with_hamming_ball
    from gray_tunneled_hashing.binary.lsh_families import create_lsh_family
    from gray_tunneled_hashing.evaluation.metrics import recall_at_k, hamming_distance
    
    # Test that classes can be instantiated (basic check)
    hasher = GrayTunneledHasher(n_bits=4)
    assert hasher is not None
    
    lsh = create_lsh_family("hyperplane", n_bits=4, dim=8, random_state=42)
    assert lsh is not None


def test_old_tests_still_pass():
    """Test that old test patterns still work with new structure."""
    n_bits = 6
    dim = 16
    N = 50
    Q = 10
    k = 3
    
    np.random.seed(42)
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    distances = euclidean_distances(queries, base_embeddings)
    ground_truth = np.argsort(distances, axis=1)[:, :k]
    
    lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=dim, random_state=42)
    
    # Old pattern: build index and check it works
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        n_bits=n_bits,
        n_codes=16,
        use_codebook=True,
        max_two_swap_iters=5,
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=42,
    )
    
    # Old pattern: get permutation and use it
    permutation = index_obj.hasher.get_assignment()
    assert permutation is not None
    
    # Old pattern: query with permutation
    query_codes = lsh.hash(queries)
    result = query_with_hamming_ball(
        query_code=query_codes[0].astype(bool),
        permutation=permutation,
        code_to_bucket=index_obj.code_to_bucket,
        bucket_to_code=index_obj.bucket_to_code,
        n_bits=n_bits,
        hamming_radius=1,
    )
    
    # Old pattern: check result is valid
    assert result.n_candidates >= 0
    assert len(result.candidate_indices) == result.n_candidates


def test_parameter_defaults_compatibility():
    """Test that default parameters still work."""
    n_bits = 6
    dim = 16
    N = 50
    Q = 10
    k = 3
    
    np.random.seed(42)
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    distances = euclidean_distances(queries, base_embeddings)
    ground_truth = np.argsort(distances, axis=1)[:, :k]
    
    lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=dim, random_state=42)
    
    # Test with minimal parameters (using defaults)
    # Use n_codes that doesn't exceed number of embeddings
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        n_bits=n_bits,
        n_codes=min(16, len(base_embeddings)),  # Don't exceed number of embeddings
    )
    
    assert index_obj is not None
    assert index_obj.hasher.is_fitted

