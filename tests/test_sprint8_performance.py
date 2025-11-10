"""
Testes de performance para Sprint 8.

Testa tempo de construção, tempo de query, escalabilidade e uso de memória.
"""

import numpy as np
import pytest
import time
import sys
from sklearn.metrics.pairwise import euclidean_distances

from gray_tunneled_hashing.binary.lsh_families import create_lsh_family
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.api.query_pipeline import query_with_hamming_ball


def test_build_time_scalability():
    """Test build time scalability with different dataset sizes."""
    n_bits = 6
    dim = 16
    Q = 20
    k = 5
    
    build_times = []
    dataset_sizes = [50, 100, 200]
    
    for N in dataset_sizes:
        # Generate synthetic data
        np.random.seed(42)
        base_embeddings = np.random.randn(N, dim).astype(np.float32)
        queries = np.random.randn(Q, dim).astype(np.float32)
        
        # Compute ground truth
        distances = euclidean_distances(queries, base_embeddings)
        ground_truth = np.argsort(distances, axis=1)[:, :k]
        
        # Create LSH encoder
        lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=dim, random_state=42)
        
        # Measure build time
        start_time = time.time()
        index_obj = build_distribution_aware_index(
            base_embeddings=base_embeddings,
            queries=queries,
            ground_truth_neighbors=ground_truth,
            encoder=lsh.hash,
            n_bits=n_bits,
            n_codes=32,
        use_codebook=True,
        max_two_swap_iters=5,  # Reduced for faster tests
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=42,
        )
        build_time = time.time() - start_time
        build_times.append(build_time)
        
        assert index_obj is not None
        assert index_obj.hasher.is_fitted
    
    # Build time should scale reasonably (not exponentially)
    # Check that time doesn't increase more than 4x when dataset doubles
    for i in range(1, len(build_times)):
        ratio = build_times[i] / build_times[i-1] if build_times[i-1] > 0 else float('inf')
        dataset_ratio = dataset_sizes[i] / dataset_sizes[i-1]
        # Time should not grow more than 2x faster than dataset size
        assert ratio <= 2 * dataset_ratio, \
            f"Build time scaling too fast: {build_times[i]:.3f}s vs {build_times[i-1]:.3f}s for {dataset_sizes[i]} vs {dataset_sizes[i-1]}"
    
    print(f"Build times: {dict(zip(dataset_sizes, build_times))}")


def test_query_time_scalability():
    """Test query time scalability with different numbers of queries."""
    n_bits = 6
    dim = 16
    N = 100
    k = 5
    hamming_radius = 1
    
    query_times = []
    num_queries_list = [10, 20, 50]
    
    # Generate base embeddings once
    np.random.seed(42)
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    
    # Create LSH encoder
    lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=dim, random_state=42)
    
    # Build index once
    queries_full = np.random.randn(50, dim).astype(np.float32)
    distances = euclidean_distances(queries_full, base_embeddings)
    ground_truth_full = np.argsort(distances, axis=1)[:, :k]
    
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries_full,
        ground_truth_neighbors=ground_truth_full,
        encoder=lsh.hash,
        n_bits=n_bits,
        n_codes=32,
        use_codebook=True,
        max_two_swap_iters=5,
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=42,
    )
    
    permutation = index_obj.hasher.get_assignment()
    base_codes = lsh.hash(base_embeddings)
    
    # Build bucket_to_dataset_indices mapping
    bucket_to_dataset_indices = {}
    for dataset_idx, code in enumerate(base_codes):
        code_tuple = tuple(code.astype(int).tolist())
        if code_tuple in index_obj.code_to_bucket:
            bucket_idx = index_obj.code_to_bucket[code_tuple]
            if bucket_idx not in bucket_to_dataset_indices:
                bucket_to_dataset_indices[bucket_idx] = []
            bucket_to_dataset_indices[bucket_idx].append(dataset_idx)
    
    for num_queries in num_queries_list:
        queries = queries_full[:num_queries]
        query_codes = lsh.hash(queries)
        
        # Measure query time
        start_time = time.time()
        for query_code in query_codes:
            result = query_with_hamming_ball(
                query_code=query_code.astype(bool),
                permutation=permutation,
                code_to_bucket=index_obj.code_to_bucket,
                bucket_to_code=index_obj.bucket_to_code,
                n_bits=n_bits,
                hamming_radius=hamming_radius,
            )
            # Simulate retrieving candidates
            _ = [bucket_to_dataset_indices.get(bucket_idx, []) for bucket_idx in result.candidate_indices]
        query_time = time.time() - start_time
        query_times.append(query_time / num_queries)  # Per-query time
    
    # Query time should be roughly constant per query
    # Check that per-query time doesn't vary too much
    mean_query_time = np.mean(query_times)
    std_query_time = np.std(query_times)
    assert std_query_time < mean_query_time * 0.5, \
        f"Query time too variable: {query_times} (mean={mean_query_time:.4f}, std={std_query_time:.4f})"
    
    print(f"Per-query times: {dict(zip(num_queries_list, query_times))}")


def test_build_time_vs_optimization_iterations():
    """Test that build time increases with optimization iterations."""
    n_bits = 6
    dim = 16
    N = 100
    Q = 20
    k = 5
    
    # Generate synthetic data
    np.random.seed(42)
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    
    # Compute ground truth
    distances = euclidean_distances(queries, base_embeddings)
    ground_truth = np.argsort(distances, axis=1)[:, :k]
    
    # Create LSH encoder
    lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=dim, random_state=42)
    
    build_times = []
    iterations_list = [0, 5, 10]
    
    for max_iter in iterations_list:
        start_time = time.time()
        index_obj = build_distribution_aware_index(
            base_embeddings=base_embeddings,
            queries=queries,
            ground_truth_neighbors=ground_truth,
            encoder=lsh.hash,
            n_bits=n_bits,
            n_codes=32,
            use_codebook=True,
            max_two_swap_iters=max_iter,
            num_tunneling_steps=0,
            mode="two_swap_only",
            random_state=42,
        )
        build_time = time.time() - start_time
        build_times.append(build_time)
    
    # More iterations should take more time (or at least not less)
    for i in range(1, len(build_times)):
        assert build_times[i] >= build_times[i-1] - 0.01, \
            f"More iterations should take more time: {build_times[i]:.3f}s vs {build_times[i-1]:.3f}s"
    
    print(f"Build times vs iterations: {dict(zip(iterations_list, build_times))}")


def test_query_time_vs_hamming_radius():
    """Test that query time increases with Hamming radius."""
    n_bits = 6
    dim = 16
    N = 100
    Q = 20
    k = 5
    
    # Generate synthetic data
    np.random.seed(42)
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    
    # Compute ground truth
    distances = euclidean_distances(queries, base_embeddings)
    ground_truth = np.argsort(distances, axis=1)[:, :k]
    
    # Create LSH encoder
    lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=dim, random_state=42)
    
    # Build index
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
    
    permutation = index_obj.hasher.get_assignment()
    query_codes = lsh.hash(queries)
    
    query_times = []
    radii = [1, 2, 3]
    
    for radius in radii:
        start_time = time.time()
        for query_code in query_codes:
            result = query_with_hamming_ball(
                query_code=query_code.astype(bool),
                permutation=permutation,
                code_to_bucket=index_obj.code_to_bucket,
                bucket_to_code=index_obj.bucket_to_code,
                n_bits=n_bits,
                hamming_radius=radius,
            )
            _ = len(result.candidate_indices)  # Use result
        query_time = time.time() - start_time
        query_times.append(query_time / len(query_codes))  # Per-query time
    
    # Larger radius should generally take more time (more candidates)
    # But allow some variance
    for i in range(1, len(query_times)):
        # Larger radius should not take significantly less time
        assert query_times[i] >= query_times[i-1] - 0.001, \
            f"Larger radius should not be faster: radius {radii[i]}={query_times[i]:.4f}s vs {radii[i-1]}={query_times[i-1]:.4f}s"
    
    print(f"Per-query times vs radius: {dict(zip(radii, query_times))}")


def test_scalability_with_n_bits():
    """Test scalability with different n_bits."""
    dim = 16
    N = 100
    Q = 20
    k = 5
    
    build_times = []
    n_bits_list = [4, 6, 8]
    
    for n_bits in n_bits_list:
        # Generate synthetic data
        np.random.seed(42)
        base_embeddings = np.random.randn(N, dim).astype(np.float32)
        queries = np.random.randn(Q, dim).astype(np.float32)
        
        # Compute ground truth
        distances = euclidean_distances(queries, base_embeddings)
        ground_truth = np.argsort(distances, axis=1)[:, :k]
        
        # Create LSH encoder
        lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=dim, random_state=42)
        
        # Measure build time
        start_time = time.time()
        index_obj = build_distribution_aware_index(
            base_embeddings=base_embeddings,
            queries=queries,
            ground_truth_neighbors=ground_truth,
            encoder=lsh.hash,
            n_bits=n_bits,
            n_codes=min(32, 2**n_bits),
            use_codebook=True,
            max_two_swap_iters=5,
            num_tunneling_steps=0,
            mode="two_swap_only",
            random_state=42,
        )
        build_time = time.time() - start_time
        build_times.append(build_time)
        
        assert index_obj is not None
    
    # Build time should scale reasonably with n_bits
    # (More bits means more computation, but not exponentially)
    print(f"Build times vs n_bits: {dict(zip(n_bits_list, build_times))}")


def test_memory_usage_basic():
    """Basic test for memory usage (qualitative check)."""
    n_bits = 6
    dim = 16
    N = 100
    Q = 20
    k = 5
    
    # Generate synthetic data
    np.random.seed(42)
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    
    # Compute ground truth
    distances = euclidean_distances(queries, base_embeddings)
    ground_truth = np.argsort(distances, axis=1)[:, :k]
    
    # Create LSH encoder
    lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=dim, random_state=42)
    
    # Build index
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
    
    # Check that key objects exist and have reasonable sizes
    assert index_obj.hasher is not None
    assert index_obj.hasher.pi_ is not None
    assert index_obj.hasher.pi_.shape == (len(index_obj.pi), n_bits)
    
    # Permutation should be relatively small (K * n_bits bytes)
    permutation_size = index_obj.hasher.pi_.nbytes
    expected_size = len(index_obj.pi) * n_bits * 1  # uint8 = 1 byte
    assert permutation_size <= expected_size * 2, \
        f"Permutation size ({permutation_size}) should be reasonable (expected ~{expected_size})"
    
    print(f"Permutation size: {permutation_size} bytes (K={len(index_obj.pi)}, n_bits={n_bits})")


@pytest.mark.slow
def test_large_scale_performance():
    """Test performance on larger dataset (marked as slow)."""
    n_bits = 8
    dim = 32
    N = 1000
    Q = 100
    k = 10
    
    # Generate synthetic data
    np.random.seed(42)
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    
    # Compute ground truth
    distances = euclidean_distances(queries, base_embeddings)
    ground_truth = np.argsort(distances, axis=1)[:, :k]
    
    # Create LSH encoder
    lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=dim, random_state=42)
    
    # Measure build time
    start_time = time.time()
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        n_bits=n_bits,
        n_codes=64,
        use_codebook=True,
        max_two_swap_iters=10,  # Reduced for speed
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=42,
    )
    build_time = time.time() - start_time
    
    # Build should complete in reasonable time (< 60 seconds for this size)
    assert build_time < 60.0, f"Build time ({build_time:.2f}s) too slow for N={N}"
    
    # Measure query time
    permutation = index_obj.hasher.get_assignment()
    query_codes = lsh.hash(queries)
    
    start_time = time.time()
    for query_code in query_codes[:10]:  # Sample first 10 queries
        result = query_with_hamming_ball(
            query_code=query_code.astype(bool),
            permutation=permutation,
            code_to_bucket=index_obj.code_to_bucket,
            bucket_to_code=index_obj.bucket_to_code,
            n_bits=n_bits,
            hamming_radius=2,
        )
        _ = len(result.candidate_indices)
    query_time = time.time() - start_time
    avg_query_time = query_time / 10
    
    # Query should be fast (< 0.1s per query)
    assert avg_query_time < 0.1, f"Query time ({avg_query_time:.4f}s) too slow"
    
    print(f"Large scale (N={N}, Q={Q}): Build={build_time:.2f}s, Query={avg_query_time:.4f}s/query")

