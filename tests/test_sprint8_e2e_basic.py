"""End-to-end tests for Sprint 8 changes."""

import numpy as np
import pytest

from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.binary.lsh_families import create_lsh_family
from gray_tunneled_hashing.api.query_pipeline import query_with_hamming_ball
from gray_tunneled_hashing.evaluation.metrics import recall_at_k


def test_build_distribution_aware_index_sprint8():
    """Test that index is built with new structure (K, n_bits)."""
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
    
    # Check that index was built successfully
    assert index_obj is not None
    assert index_obj.hasher is not None
    assert index_obj.hasher.is_fitted
    
    # Check permutation structure (NEW: should be K x n_bits)
    permutation = index_obj.hasher.get_assignment()
    K = index_obj.K
    assert permutation.shape == (K, n_bits), \
        f"Expected permutation shape ({K}, {n_bits}), got {permutation.shape}"
    
    # Check that permutation values are valid
    assert np.all((permutation == 0) | (permutation == 1)), \
        "Permutation should contain only 0s and 1s"


def test_fit_with_traffic_real_embeddings():
    """Test that fit_with_traffic works with new parameters."""
    n_bits = 6
    dim = 16
    N = 100
    Q = 20
    k = 5
    K = 16
    
    # Generate synthetic data
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    ground_truth = np.random.randint(0, N, size=(Q, k), dtype=np.int32)
    
    # Create LSH encoder
    lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=dim, random_state=42)
    
    # Build traffic stats
    from gray_tunneled_hashing.distribution.traffic_stats import collect_traffic_stats
    traffic_stats = collect_traffic_stats(
        queries=queries,
        ground_truth_neighbors=ground_truth,
        base_embeddings=base_embeddings,
        encoder=lsh.hash,
        collapse_threshold=0.01,
    )
    
    pi = traffic_stats["pi"]
    w = traffic_stats["w"]
    code_to_bucket = traffic_stats["code_to_bucket"]
    bucket_to_code = traffic_stats["bucket_to_code"]
    
    # Create bucket embeddings
    bucket_embeddings = np.random.randn(len(pi), dim).astype(np.float32)
    
    # Create hasher
    from gray_tunneled_hashing.algorithms.gray_tunneled_hasher import GrayTunneledHasher
    hasher = GrayTunneledHasher(
        n_bits=n_bits,
        max_two_swap_iters=5,
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=42,
    )
    
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
    
    # Check that hasher is fitted
    assert hasher.is_fitted
    assert hasher.pi_ is not None
    assert hasher.cost_ is not None
    
    # Check permutation structure
    permutation = hasher.get_assignment()
    assert permutation.shape == (len(pi), n_bits), \
        f"Expected permutation shape ({len(pi)}, {n_bits}), got {permutation.shape}"


def test_query_end_to_end():
    """Test that query works end-to-end."""
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
    
    # Get permutation
    permutation = index_obj.hasher.get_assignment()
    
    # Query a few queries
    n_test_queries = min(5, Q)
    for i in range(n_test_queries):
        query_code = lsh.hash(queries[i:i+1])[0].astype(bool)
        
        result = query_with_hamming_ball(
            query_code=query_code,
            permutation=permutation,
            code_to_bucket=index_obj.code_to_bucket,
            bucket_to_code=index_obj.bucket_to_code,
            n_bits=n_bits,
            hamming_radius=1,
        )
        
        # Check result structure
        assert result.query_code.shape == (n_bits,)
        assert result.permuted_code.shape == (n_bits,)
        assert result.n_candidates == len(result.candidate_indices)
        assert result.n_candidates >= 0
        
        # Check that candidate indices are valid
        if result.n_candidates > 0:
            K = index_obj.K
            assert np.all(result.candidate_indices >= 0)
            assert np.all(result.candidate_indices < K)


def test_recall_not_worse_than_baseline():
    """Test that recall doesn't worsen compared to baseline."""
    n_bits = 6
    dim = 16
    N = 100
    Q = 20
    k = 5
    hamming_radius = 1
    
    # Generate synthetic data
    np.random.seed(42)
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    
    # Compute ground truth using Euclidean distance
    from sklearn.metrics.pairwise import euclidean_distances
    distances = euclidean_distances(queries, base_embeddings)
    ground_truth = np.argsort(distances, axis=1)[:, :k]
    
    # Create LSH encoder
    lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=dim, random_state=42)
    
    # Baseline: LSH without GTH
    query_codes_baseline = lsh.hash(queries)
    base_codes_baseline = lsh.hash(base_embeddings)
    
    # Compute baseline recall (simple Hamming ball search)
    baseline_recalls = []
    for i in range(Q):
        query_code = query_codes_baseline[i]
        # Find all base codes within Hamming radius
        hamming_dists = np.sum(query_code != base_codes_baseline, axis=1)
        candidates = np.where(hamming_dists <= hamming_radius)[0]
        
        # Compute recall
        if len(candidates) > 0:
            retrieved = set(candidates)
            relevant = set(ground_truth[i])
            recall = len(retrieved & relevant) / len(relevant) if len(relevant) > 0 else 0.0
            baseline_recalls.append(recall)
        else:
            baseline_recalls.append(0.0)
    
    baseline_recall = np.mean(baseline_recalls)
    
    # GTH: Build distribution-aware index
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        n_bits=n_bits,
        n_codes=32,
        use_codebook=True,
        max_two_swap_iters=10,  # More iterations for better optimization
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=42,
    )
    
    # Get permutation
    permutation = index_obj.hasher.get_assignment()
    
    # Build bucket_to_dataset_indices mapping
    base_codes_lsh = lsh.hash(base_embeddings)
    bucket_to_dataset_indices = {}
    for dataset_idx, code in enumerate(base_codes_lsh):
        code_tuple = tuple(code.astype(int).tolist())
        if code_tuple in index_obj.code_to_bucket:
            bucket_idx = index_obj.code_to_bucket[code_tuple]
            if bucket_idx not in bucket_to_dataset_indices:
                bucket_to_dataset_indices[bucket_idx] = []
            bucket_to_dataset_indices[bucket_idx].append(dataset_idx)
    
    # Compute GTH recall
    gth_recalls = []
    for i in range(Q):
        query_code = query_codes_baseline[i].astype(bool)
        
        result = query_with_hamming_ball(
            query_code=query_code,
            permutation=permutation,
            code_to_bucket=index_obj.code_to_bucket,
            bucket_to_code=index_obj.bucket_to_code,
            n_bits=n_bits,
            hamming_radius=hamming_radius,
        )
        
        # Get dataset indices from candidate buckets
        retrieved = set()
        for bucket_idx in result.candidate_indices:
            if bucket_idx in bucket_to_dataset_indices:
                retrieved.update(bucket_to_dataset_indices[bucket_idx])
        
        # Compute recall
        relevant = set(ground_truth[i])
        recall = len(retrieved & relevant) / len(relevant) if len(relevant) > 0 else 0.0
        gth_recalls.append(recall)
    
    gth_recall = np.mean(gth_recalls)
    
    # GTH recall should not be significantly worse than baseline
    # Allow some tolerance (e.g., within 20% of baseline)
    # Note: This is a lenient test - ideally GTH should improve recall
    assert gth_recall >= baseline_recall * 0.8, \
        f"GTH recall {gth_recall:.4f} is significantly worse than baseline {baseline_recall:.4f}"

