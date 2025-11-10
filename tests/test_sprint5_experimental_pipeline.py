"""End-to-end tests for Sprint 5 experimental pipeline."""

import pytest
import numpy as np
from gray_tunneled_hashing.experiments.config import LSHExperimentConfig
from gray_tunneled_hashing.experiments.setup import create_experimental_setup
from gray_tunneled_hashing.binary.lsh_families import HyperplaneLSH
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.experiments.collision_validation import validate_collision_preservation
from gray_tunneled_hashing.experiments.metrics import compute_recall_at_k


def test_experimental_pipeline_end_to_end():
    """Test complete pipeline: LSH → GTH → Query."""
    config = LSHExperimentConfig(
        n_samples=50,
        n_queries=10,
        n_bits=6,
        n_codes=16,
        k=3,
        hamming_radius=1,
        random_state=42,
    )
    
    # Create setup
    setup = create_experimental_setup(config)
    
    # Create LSH
    lsh = HyperplaneLSH(n_bits=config.n_bits, dim=config.dim, random_state=config.random_state)
    
    # Build index
    index_obj = build_distribution_aware_index(
        base_embeddings=setup.base_embeddings,
        queries=setup.queries,
        ground_truth_neighbors=setup.ground_truth,
        encoder=None,
        n_bits=config.n_bits,
        n_codes=config.n_codes,
        use_codebook=True,
        lsh_family=lsh,
        max_two_swap_iters=10,
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=config.random_state,
    )
    
    # Verify index was built
    assert index_obj is not None
    assert index_obj.permutation is not None
    assert index_obj.bucket_to_code is not None


def test_experimental_pipeline_collision_preservation():
    """Test collision preservation in pipeline."""
    config = LSHExperimentConfig(
        n_samples=50,
        n_queries=10,
        n_bits=6,
        n_codes=16,
        random_state=42,
    )
    
    setup = create_experimental_setup(config)
    lsh = HyperplaneLSH(n_bits=config.n_bits, dim=config.dim, random_state=config.random_state)
    
    index_obj = build_distribution_aware_index(
        base_embeddings=setup.base_embeddings,
        queries=setup.queries,
        ground_truth_neighbors=setup.ground_truth,
        encoder=None,
        n_bits=config.n_bits,
        n_codes=config.n_codes,
        use_codebook=True,
        lsh_family=lsh,
        max_two_swap_iters=10,
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=config.random_state,
    )
    
    # Validate collision preservation
    result = validate_collision_preservation(
        embeddings=setup.base_embeddings,
        lsh=lsh,
        index_obj=index_obj,
    )
    
    assert result.preservation_rate == 100.0


def test_experimental_pipeline_reproducibility():
    """Test that pipeline is reproducible with same random_state."""
    config = LSHExperimentConfig(
        n_samples=50,
        n_queries=10,
        n_bits=6,
        n_codes=16,
        random_state=42,
    )
    
    # Run twice with same seed
    setup1 = create_experimental_setup(config)
    setup2 = create_experimental_setup(config)
    
    # Should be identical
    np.testing.assert_array_equal(setup1.base_embeddings, setup2.base_embeddings)
    np.testing.assert_array_equal(setup1.queries, setup2.queries)
    np.testing.assert_array_equal(setup1.ground_truth, setup2.ground_truth)

