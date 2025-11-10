"""Tests for collision preservation validation."""

import pytest
import numpy as np
from gray_tunneled_hashing.binary.lsh_families import HyperplaneLSH
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.experiments.collision_validation import (
    validate_collision_preservation,
)


def test_validate_collision_preservation_100_percent():
    """Test that collision preservation is 100%."""
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
    
    # Validate collision preservation
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


def test_validate_collision_preservation_edge_cases():
    """Test collision preservation with edge cases."""
    n_bits = 4
    dim = 8
    N = 10
    Q = 5
    k = 2
    
    # Generate data with many collisions (small n_bits, many samples)
    np.random.seed(42)
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    ground_truth = np.random.randint(0, N, size=(Q, k), dtype=np.int32)
    
    # Create LSH
    lsh = HyperplaneLSH(n_bits=n_bits, dim=dim, random_state=42)
    
    # Build index
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=None,
        n_bits=n_bits,
        n_codes=8,
        use_codebook=True,
        lsh_family=lsh,
        max_two_swap_iters=5,
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=42,
    )
    
    # Validate
    result = validate_collision_preservation(
        embeddings=base_embeddings,
        lsh=lsh,
        index_obj=index_obj,
    )
    
    # Should still preserve 100%
    assert result.preservation_rate == 100.0
    assert len(result.violated_pairs) == 0


def test_validate_collision_preservation_large_dataset():
    """Test collision preservation with larger dataset."""
    n_bits = 6
    dim = 16
    N = 100
    Q = 20
    k = 5
    
    np.random.seed(42)
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    ground_truth = np.random.randint(0, N, size=(Q, k), dtype=np.int32)
    
    lsh = HyperplaneLSH(n_bits=n_bits, dim=dim, random_state=42)
    
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
    
    result = validate_collision_preservation(
        embeddings=base_embeddings,
        lsh=lsh,
        index_obj=index_obj,
    )
    
    # Should preserve 100%
    assert result.preservation_rate == 100.0
    assert len(result.violated_pairs) == 0

