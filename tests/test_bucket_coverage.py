"""
Test H1: Bucket → Dataset mapping coverage.

Tests that all base embedding LSH codes are properly mapped to buckets
and that code_to_bucket has sufficient coverage.
"""

import numpy as np
import pytest
from gray_tunneled_hashing.binary.lsh_families import HyperplaneLSH
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from sklearn.metrics.pairwise import euclidean_distances


def test_all_base_codes_in_code_to_bucket():
    """Test that all base embedding codes are in code_to_bucket."""
    np.random.seed(42)
    n_samples = 100
    n_queries = 20
    dim = 16
    n_bits = 6
    n_codes = 16
    
    base_embeddings = np.random.randn(n_samples, dim).astype(np.float32)
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    
    # Generate ground truth
    distances = euclidean_distances(queries, base_embeddings)
    ground_truth = np.argsort(distances, axis=1)[:, :5].astype(np.int32)
    
    # Build index
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
    
    # Encode base embeddings
    base_codes_lsh = lsh.hash(base_embeddings)
    
    # Check coverage
    mapped_count = 0
    for code in base_codes_lsh:
        code_tuple = tuple(code.astype(int).tolist())
        if code_tuple in index_obj.code_to_bucket:
            mapped_count += 1
    
    coverage_rate = mapped_count / len(base_embeddings)
    
    # FIX 4: After Fix 2, we expect 100% coverage (all base embeddings in code_to_bucket)
    assert coverage_rate == 1.0, \
        f"Expected 100% coverage after Fix 2, but only {coverage_rate:.1%} of base embeddings are in code_to_bucket"


def test_code_overlap_between_base_and_queries():
    """Test that there's reasonable overlap between base and query codes."""
    np.random.seed(42)
    n_samples = 100
    n_queries = 20
    dim = 16
    n_bits = 6
    
    base_embeddings = np.random.randn(n_samples, dim).astype(np.float32)
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    
    lsh = HyperplaneLSH(n_bits=n_bits, dim=dim, random_state=42)
    
    base_codes = lsh.hash(base_embeddings)
    query_codes = lsh.hash(queries)
    
    base_codes_set = set(tuple(code.astype(int).tolist()) for code in base_codes)
    query_codes_set = set(tuple(code.astype(int).tolist()) for code in query_codes)
    
    overlap = len(base_codes_set & query_codes_set) / len(base_codes_set) if len(base_codes_set) > 0 else 0.0
    
    # We expect at least 20% overlap (for small datasets)
    assert overlap >= 0.2, f"Low code overlap: only {overlap:.1%}"


def test_bucket_to_dataset_mapping_completeness():
    """Test that bucket_to_dataset mapping includes all mapped embeddings."""
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
    
    # Count total mapped embeddings
    total_mapped = sum(len(indices) for indices in bucket_to_dataset_indices.values())
    
    # Count embeddings in code_to_bucket
    mapped_in_code_to_bucket = sum(
        1 for code in base_codes_lsh
        if tuple(code.astype(int).tolist()) in index_obj.code_to_bucket
    )
    
    # They should match
    assert total_mapped == mapped_in_code_to_bucket, \
        f"Mapping mismatch: {total_mapped} in mapping vs {mapped_in_code_to_bucket} in code_to_bucket"

