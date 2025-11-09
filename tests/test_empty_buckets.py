"""
Test H4: Empty or sparsely populated buckets.

Tests that buckets are properly populated and that ground truth neighbors
are accessible through buckets.
"""

import numpy as np
import pytest
from gray_tunneled_hashing.binary.lsh_families import HyperplaneLSH
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from sklearn.metrics.pairwise import euclidean_distances


def test_ground_truth_neighbors_in_buckets():
    """Test that ground truth neighbors are in populated buckets."""
    np.random.seed(42)
    n_samples = 100
    n_queries = 20
    dim = 16
    n_bits = 6
    n_codes = 16
    k = 5
    
    base_embeddings = np.random.randn(n_samples, dim).astype(np.float32)
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    
    distances = euclidean_distances(queries, base_embeddings)
    ground_truth = np.argsort(distances, axis=1)[:, :k].astype(np.int32)
    
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
    
    # Build bucket → dataset mapping
    base_codes_lsh = lsh.hash(base_embeddings)
    bucket_to_dataset_indices = {}
    
    for dataset_idx, code in enumerate(base_codes_lsh):
        code_tuple = tuple(code.astype(int).tolist())
        if code_tuple in index_obj.code_to_bucket:
            bucket_idx = index_obj.code_to_bucket[code_tuple]
            if bucket_idx not in bucket_to_dataset_indices:
                bucket_to_dataset_indices[bucket_idx] = []
            bucket_to_dataset_indices[bucket_idx].append(dataset_idx)
    
    # Check that ground truth neighbors are in buckets
    gt_set = set(ground_truth.flatten())
    mapped_set = set()
    for indices in bucket_to_dataset_indices.values():
        mapped_set.update(indices)
    
    gt_in_buckets = len(gt_set & mapped_set)
    gt_coverage = gt_in_buckets / len(gt_set) if len(gt_set) > 0 else 0.0
    
    # We expect at least 90% of ground truth neighbors to be in buckets
    assert gt_coverage >= 0.9, \
        f"Only {gt_coverage:.1%} of ground truth neighbors are in buckets"


def test_bucket_population_distribution():
    """Test that buckets have reasonable population distribution."""
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
    
    # Build bucket → dataset mapping
    base_codes_lsh = lsh.hash(base_embeddings)
    bucket_to_dataset_indices = {}
    
    for dataset_idx, code in enumerate(base_codes_lsh):
        code_tuple = tuple(code.astype(int).tolist())
        if code_tuple in index_obj.code_to_bucket:
            bucket_idx = index_obj.code_to_bucket[code_tuple]
            if bucket_idx not in bucket_to_dataset_indices:
                bucket_to_dataset_indices[bucket_idx] = []
            bucket_to_dataset_indices[bucket_idx].append(dataset_idx)
    
    K = index_obj.K
    populated_buckets = len(bucket_to_dataset_indices)
    empty_buckets = K - populated_buckets
    
    # We expect at least 50% of buckets to be populated
    population_rate = populated_buckets / K if K > 0 else 0.0
    assert population_rate >= 0.5, \
        f"Only {population_rate:.1%} of buckets are populated ({empty_buckets} empty)"


def test_no_excessive_bucket_sizes():
    """Test that no single bucket has an excessive number of embeddings."""
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
    
    # Build bucket → dataset mapping
    base_codes_lsh = lsh.hash(base_embeddings)
    bucket_to_dataset_indices = {}
    
    for dataset_idx, code in enumerate(base_codes_lsh):
        code_tuple = tuple(code.astype(int).tolist())
        if code_tuple in index_obj.code_to_bucket:
            bucket_idx = index_obj.code_to_bucket[code_tuple]
            if bucket_idx not in bucket_to_dataset_indices:
                bucket_to_dataset_indices[bucket_idx] = []
            bucket_to_dataset_indices[bucket_idx].append(dataset_idx)
    
    # Check bucket sizes
    bucket_sizes = [len(indices) for indices in bucket_to_dataset_indices.values()]
    if bucket_sizes:
        max_size = max(bucket_sizes)
        avg_size = np.mean(bucket_sizes)
        
        # Max size should not be more than 3x the average
        assert max_size <= 3 * avg_size or max_size <= 20, \
            f"Excessive bucket size: max={max_size}, avg={avg_size:.1f}"

