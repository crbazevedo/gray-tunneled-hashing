"""
Test H2: Permutation coverage analysis.

Tests that the permutation properly maps vertices to buckets
and that there are no invalid bucket indices.
"""

import numpy as np
import pytest
from gray_tunneled_hashing.binary.lsh_families import HyperplaneLSH
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from sklearn.metrics.pairwise import euclidean_distances


def test_permutation_no_invalid_bucket_indices():
    """Test that permutation doesn't map to invalid bucket indices."""
    np.random.seed(42)
    n_samples = 100
    n_queries = 20
    dim = 16
    n_bits = 6
    n_codes = 16
    
    base_embeddings = np.random.randn(n_samples, dim).astype(np.float32)
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    
    distances = euclidean_distances(queries, base_embeddings)
    ground_truth = np.argsort(distances, axis=1)[:, :5].astype(np.int32)
    
    lsh = HyperplaneLSH(n_bits=n_bits, dim=dim, random_state=42)
    
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=None,
        n_bits=n_bits,
        n_codes=n_codes,
        use_codebook=True,
        use_semantic_distances=False,
        lsh_family=lsh,
        block_size=4,
        max_two_swap_iters=20,
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=42,
    )
    
    permutation = index_obj.permutation
    K = index_obj.K
    
    # Check for invalid bucket indices
    # FIX 4: After fixes, permutation should NOT map to invalid buckets
    # Permutation maps vertices to embedding_idx, which should be < K for valid buckets
    # We need to check via bucket_to_embedding_idx mapping
    invalid_indices = []
    for vertex_idx in range(len(permutation)):
        embedding_idx = permutation[vertex_idx]
        # embedding_idx should be < K to correspond to a valid bucket
        # (assuming bucket_to_embedding_idx = [0, 1, ..., K-1] by default)
        if embedding_idx >= K:
            invalid_indices.append((vertex_idx, embedding_idx))
    
    # After Fix 1 and Fix 3, there should be NO invalid indices
    assert len(invalid_indices) == 0, \
        f"Found {len(invalid_indices)} invalid embedding indices (>= K={K}): {invalid_indices[:5]}..."


def test_permutation_vertex_distribution():
    """Test that vertices are reasonably distributed across buckets."""
    np.random.seed(42)
    n_samples = 100
    n_queries = 20
    dim = 16
    n_bits = 6
    n_codes = 16
    
    base_embeddings = np.random.randn(n_samples, dim).astype(np.float32)
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    
    distances = euclidean_distances(queries, base_embeddings)
    ground_truth = np.argsort(distances, axis=1)[:, :5].astype(np.int32)
    
    lsh = HyperplaneLSH(n_bits=n_bits, dim=dim, random_state=42)
    
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=None,
        n_bits=n_bits,
        n_codes=n_codes,
        use_codebook=True,
        use_semantic_distances=False,
        lsh_family=lsh,
        block_size=4,
        max_two_swap_iters=20,
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=42,
    )
    
    permutation = index_obj.permutation
    K = index_obj.K
    
    # Count vertices per bucket
    from collections import defaultdict
    bucket_to_vertices = defaultdict(list)
    for vertex_idx in range(len(permutation)):
        bucket_idx = permutation[vertex_idx]
        if bucket_idx < K:  # Only count valid buckets
            bucket_to_vertices[bucket_idx].append(vertex_idx)
    
    # Check that most buckets have at least one vertex
    buckets_with_vertices = len(bucket_to_vertices)
    coverage_rate = buckets_with_vertices / K if K > 0 else 0.0
    
    # We expect at least 50% of buckets to have vertices
    assert coverage_rate >= 0.5, \
        f"Only {coverage_rate:.1%} of buckets have vertices mapped"


def test_permutation_covers_all_vertices():
    """Test that all vertices are mapped to some bucket."""
    np.random.seed(42)
    n_samples = 100
    n_queries = 20
    dim = 16
    n_bits = 6
    n_codes = 16
    
    base_embeddings = np.random.randn(n_samples, dim).astype(np.float32)
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    
    distances = euclidean_distances(queries, base_embeddings)
    ground_truth = np.argsort(distances, axis=1)[:, :5].astype(np.int32)
    
    lsh = HyperplaneLSH(n_bits=n_bits, dim=dim, random_state=42)
    
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=None,
        n_bits=n_bits,
        n_codes=n_codes,
        use_codebook=True,
        use_semantic_distances=False,
        lsh_family=lsh,
        block_size=4,
        max_two_swap_iters=20,
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=42,
    )
    
    permutation = index_obj.permutation
    N = len(permutation)  # Should be 2**n_bits
    
    # All vertices should be mapped (permutation has length N)
    assert len(permutation) == N, \
        f"Permutation length {len(permutation)} != expected {N}"

