"""Tests for LSH + GTH integration and collision preservation."""

import numpy as np
import pytest

from gray_tunneled_hashing.binary.lsh_families import (
    HyperplaneLSH,
    PStableLSH,
    create_lsh_family,
)
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.distribution.pipeline import apply_permutation


def test_lsh_gth_integration_hyperplane():
    """Test integration of Hyperplane LSH with GTH."""
    n_bits = 6
    dim = 16
    N = 100
    Q = 20
    k = 5
    
    # Generate synthetic data
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    
    # Generate ground truth (random for test)
    ground_truth = np.random.randint(0, N, size=(Q, k), dtype=np.int32)
    
    # Create LSH family
    lsh = HyperplaneLSH(n_bits=n_bits, dim=dim, random_state=42)
    
    # Build distribution-aware index with LSH
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=None,  # Will use lsh_family
        n_bits=n_bits,
        n_codes=32,
        use_codebook=True,
        use_semantic_distances=False,
        lsh_family=lsh,
        block_size=4,
        max_two_swap_iters=10,
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=42,
    )
    
    # Check that index was built successfully
    assert index_obj is not None
    assert index_obj.permutation is not None
    assert index_obj.bucket_to_code is not None


def test_lsh_collision_preservation():
    """
    Test that GTH preserves LSH bucket membership (collision guarantees).
    
    Key property: If two vectors hash to the same bucket before GTH,
    they should hash to the same bucket after GTH (just relabeled).
    """
    n_bits = 6
    dim = 16
    N = 50
    Q = 10
    k = 3
    
    # Generate data
    np.random.seed(42)
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    ground_truth = np.random.randint(0, N, size=(Q, k), dtype=np.int32)
    
    # Create LSH
    lsh = HyperplaneLSH(n_bits=n_bits, dim=dim, random_state=42)
    
    # Build distribution-aware index
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=None,
        n_bits=n_bits,
        n_codes=32,
        use_codebook=True,
        use_semantic_distances=False,
        lsh_family=lsh,
        block_size=4,
        max_two_swap_iters=10,
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=42,
    )
    
    # Use proper validation function
    from gray_tunneled_hashing.experiments.collision_validation import validate_collision_preservation
    
    result = validate_collision_preservation(
        embeddings=base_embeddings,
        lsh=lsh,
        index_obj=index_obj,
    )
    
    # Should preserve 100% of collisions
    assert result.preservation_rate == 100.0, (
        f"Expected 100% preservation, got {result.preservation_rate}%"
    )
    assert len(result.violated_pairs) == 0, (
        f"Expected no violated pairs, got {len(result.violated_pairs)}"
    )


def test_lsh_vs_random_projection_comparison():
    """Test that LSH and random projection can both be used with GTH."""
    n_bits = 6
    dim = 16
    N = 50
    Q = 10
    k = 3
    
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    ground_truth = np.random.randint(0, N, size=(Q, k), dtype=np.int32)
    
    # Test with Hyperplane LSH
    lsh = HyperplaneLSH(n_bits=n_bits, dim=dim, random_state=42)
    
    index_lsh = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=None,
        n_bits=n_bits,
        n_codes=32,
        use_codebook=True,
        lsh_family=lsh,
        max_two_swap_iters=10,
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=42,
    )
    
    # Test with random projection (via encoder function)
    from gray_tunneled_hashing.binary.baselines import random_projection_binarize
    
    _, proj_matrix = random_projection_binarize(
        base_embeddings, n_bits=n_bits, random_state=42
    )
    
    def encoder_fn(emb):
        from gray_tunneled_hashing.binary.baselines import apply_random_projection
        proj = apply_random_projection(emb, proj_matrix)
        return (proj > 0).astype(bool)
    
    index_rp = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=encoder_fn,
        n_bits=n_bits,
        n_codes=32,
        use_codebook=True,
        max_two_swap_iters=10,
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=42,
    )
    
    # Both should build successfully
    assert index_lsh is not None
    assert index_rp is not None
    
    # Both should have valid permutations
    assert index_lsh.permutation is not None
    assert index_rp.permutation is not None


def test_p_stable_lsh_integration():
    """Test integration of p-stable LSH with GTH."""
    n_bits = 6
    dim = 16
    N = 50
    Q = 10
    k = 3
    
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    ground_truth = np.random.randint(0, N, size=(Q, k), dtype=np.int32)
    
    # Create p-stable LSH
    lsh = PStableLSH(n_bits=n_bits, dim=dim, w=1.0, random_state=42)
    
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=None,
        n_bits=n_bits,
        n_codes=32,
        use_codebook=True,
        lsh_family=lsh,
        max_two_swap_iters=10,
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=42,
    )
    
    assert index_obj is not None
    assert index_obj.permutation is not None

