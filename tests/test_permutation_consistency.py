"""
Test H5: Consistency between permutation and code_to_bucket.

Tests that all buckets returned by permutation exist in code_to_bucket
and vice versa.
"""

import numpy as np
import pytest
from gray_tunneled_hashing.binary.lsh_families import HyperplaneLSH
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from sklearn.metrics.pairwise import euclidean_distances


def test_permutation_buckets_in_code_to_bucket():
    """Test that all buckets in permutation exist in code_to_bucket."""
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
    code_to_bucket = index_obj.code_to_bucket
    K = index_obj.K
    
    # FIX 4: After fixes, permutation should map to embedding_idx < K
    # These embedding_idx correspond to buckets via bucket_to_embedding_idx
    # Since bucket_to_embedding_idx = [0, 1, ..., K-1] by default, embedding_idx == bucket_idx
    # So all valid embedding_idx should be in code_to_bucket
    
    # Get all embedding indices from permutation that are < K (valid)
    valid_embedding_indices = set(embedding_idx for embedding_idx in permutation if embedding_idx < K)
    code_to_bucket_buckets = set(code_to_bucket.values())
    
    # After Fix 1 and Fix 3, all valid embedding indices should be in code_to_bucket
    # (since Fix 2 ensures all base codes are included)
    buckets_not_in_code = valid_embedding_indices - code_to_bucket_buckets
    
    # We expect 100% consistency after all fixes
    assert len(buckets_not_in_code) == 0, \
        f"Found {len(buckets_not_in_code)} valid embedding indices not in code_to_bucket: {list(buckets_not_in_code)[:10]}..."


def test_code_to_bucket_buckets_in_permutation():
    """Test that all buckets in code_to_bucket can be reached via permutation."""
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
    code_to_bucket = index_obj.code_to_bucket
    K = index_obj.K
    
    # Get all buckets from code_to_bucket
    code_to_bucket_buckets = set(code_to_bucket.values())
    
    # Get all buckets from permutation (only valid ones)
    permutation_buckets = set(bucket_idx for bucket_idx in permutation if bucket_idx < K)
    
    # Check that code_to_bucket buckets are in permutation
    buckets_not_in_permutation = code_to_bucket_buckets - permutation_buckets
    
    # All code_to_bucket buckets should be reachable via permutation
    # (otherwise they can't be used for retrieval)
    assert len(buckets_not_in_permutation) == 0, \
        f"{len(buckets_not_in_permutation)} buckets in code_to_bucket are not in permutation: {list(buckets_not_in_permutation)[:10]}..."


def test_permutation_code_consistency_rate():
    """Test that permutation and code_to_bucket have reasonable consistency."""
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
    code_to_bucket = index_obj.code_to_bucket
    K = index_obj.K
    
    # FIX 4: Calculate consistency after fixes
    # Get valid embedding indices from permutation (< K)
    valid_embedding_indices = set(embedding_idx for embedding_idx in permutation if embedding_idx < K)
    code_to_bucket_buckets = set(code_to_bucket.values())
    
    intersection = valid_embedding_indices & code_to_bucket_buckets
    union = valid_embedding_indices | code_to_bucket_buckets
    consistency_rate = len(intersection) / len(union) if len(union) > 0 else 0.0
    
    # After all fixes, we expect 100% consistency
    assert consistency_rate == 1.0, \
        f"Expected 100% consistency after fixes, but got {consistency_rate:.1%} " \
        f"(intersection={len(intersection)}, union={len(union)})"

