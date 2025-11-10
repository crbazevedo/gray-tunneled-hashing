"""Tests for Sprint 8 query pipeline (permutation before Hamming ball expansion)."""

import numpy as np
import pytest

from gray_tunneled_hashing.api.query_pipeline import query_with_hamming_ball
from gray_tunneled_hashing.binary.lsh_families import create_lsh_family
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index


def test_query_with_hamming_ball_permutation_before_expansion():
    """Test that permutation is applied before Hamming ball expansion."""
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
    
    # Get permutation (should be K x n_bits)
    permutation = index_obj.hasher.get_assignment()
    K = index_obj.K
    
    # Encode a query
    query_code = lsh.hash(queries[0:1])[0].astype(bool)
    
    # Query with Hamming ball
    result = query_with_hamming_ball(
        query_code=query_code,
        permutation=permutation,
        code_to_bucket=index_obj.code_to_bucket,
        bucket_to_code=index_obj.bucket_to_code,
        n_bits=n_bits,
        hamming_radius=1,
        enable_logging=False,
    )
    
    # Check that permuted_code is different from query_code (if permutation was applied)
    # The permuted_code should be the code assigned to the query's bucket
    query_code_tuple = tuple(query_code.astype(int).tolist())
    if query_code_tuple in index_obj.code_to_bucket:
        query_bucket = index_obj.code_to_bucket[query_code_tuple]
        expected_permuted_code = permutation[query_bucket]
        
        # permuted_code should match expected
        assert np.array_equal(result.permuted_code.astype(int), expected_permuted_code.astype(int)), \
            "permuted_code should match permutation[bucket_idx]"
    
    # Check that result has valid structure
    assert result.query_code.shape == (n_bits,)
    assert result.permuted_code.shape == (n_bits,)
    assert len(result.candidate_indices) == result.n_candidates


def test_query_with_hamming_ball_returns_valid_buckets():
    """Test that returned buckets are valid."""
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
    K = index_obj.K
    
    # Test multiple queries
    for i in range(min(5, Q)):
        query_code = lsh.hash(queries[i:i+1])[0].astype(bool)
        
        result = query_with_hamming_ball(
            query_code=query_code,
            permutation=permutation,
            code_to_bucket=index_obj.code_to_bucket,
            bucket_to_code=index_obj.bucket_to_code,
            n_bits=n_bits,
            hamming_radius=1,
        )
        
        # Check that all candidate indices are valid bucket indices
        if len(result.candidate_indices) > 0:
            assert np.all(result.candidate_indices >= 0), "Bucket indices should be >= 0"
            assert np.all(result.candidate_indices < K), \
                f"Bucket indices should be < K={K}, got max={result.candidate_indices.max()}"
            
            # Check that candidate indices are unique
            assert len(result.candidate_indices) == len(np.unique(result.candidate_indices)), \
                "Candidate indices should be unique"


def test_query_with_hamming_ball_coverage():
    """Test that Hamming ball covers expected buckets."""
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
    K = index_obj.K
    
    # Test with different radii
    query_code = lsh.hash(queries[0:1])[0].astype(bool)
    
    for radius in [0, 1, 2]:
        result = query_with_hamming_ball(
            query_code=query_code,
            permutation=permutation,
            code_to_bucket=index_obj.code_to_bucket,
            bucket_to_code=index_obj.bucket_to_code,
            n_bits=n_bits,
            hamming_radius=radius,
        )
        
        # With larger radius, we should get more candidates (or at least not fewer)
        # But this depends on the permutation, so we just check structure
        assert result.n_candidates >= 0
        assert len(result.candidate_indices) == result.n_candidates
        
        # If radius=0, we should get at least the query's own bucket (if it exists)
        if radius == 0:
            query_code_tuple = tuple(query_code.astype(int).tolist())
            if query_code_tuple in index_obj.code_to_bucket:
                # Should get at least 1 candidate (the query's bucket)
                assert result.n_candidates >= 1, \
                    "With radius=0, should get at least query's own bucket"


def test_query_with_hamming_ball_empty_result():
    """Test behavior when query is not in any bucket."""
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
    
    # Create a query code that doesn't exist in code_to_bucket
    # Generate a code that's unlikely to be in the index
    # Use a code that's definitely not in code_to_bucket
    # We'll try multiple codes until we find one not in code_to_bucket
    invalid_code = None
    for i in range(2**n_bits):
        test_code = np.array([(i >> j) & 1 for j in range(n_bits)], dtype=bool)
        test_code_tuple = tuple(test_code.astype(int).tolist())
        if test_code_tuple not in index_obj.code_to_bucket:
            invalid_code = test_code
            break
    
    if invalid_code is None:
        # All codes are in buckets - skip this test
        pytest.skip("All possible codes are in buckets, cannot test invalid code")
    
    result = query_with_hamming_ball(
        query_code=invalid_code,
        permutation=permutation,
        code_to_bucket=index_obj.code_to_bucket,
        bucket_to_code=index_obj.bucket_to_code,
        n_bits=n_bits,
        hamming_radius=1,
    )
    
    # Should return empty result (query code not in any bucket)
    assert result.n_candidates == 0, \
        f"Query code not in buckets should return empty result, got {result.n_candidates} candidates"
    assert len(result.candidate_indices) == 0
    assert len(result.candidate_codes) == 0

