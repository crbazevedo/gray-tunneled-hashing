"""Tests for experimental setup."""

import pytest
import numpy as np
from gray_tunneled_hashing.experiments.config import LSHExperimentConfig
from gray_tunneled_hashing.experiments.setup import (
    create_experimental_setup,
    validate_setup,
    generate_synthetic_data,
)


def test_create_experimental_setup():
    """Test that setup is created correctly."""
    config = LSHExperimentConfig(
        n_samples=50,
        n_queries=10,
        dim=16,
        k=5,
        random_state=42,
    )
    
    setup = create_experimental_setup(config)
    
    assert setup.base_embeddings.shape == (50, 16)
    assert setup.queries.shape == (10, 16)
    assert setup.ground_truth.shape == (10, 5)
    assert setup.config == config


def test_validate_setup_constraints():
    """Test that setup validation checks constraints."""
    config = LSHExperimentConfig(
        n_samples=50,
        n_queries=10,
        dim=16,
        k=5,
        n_bits=6,
        n_codes=32,
        random_state=42,
    )
    
    setup = create_experimental_setup(config)
    result = validate_setup(setup)
    
    assert result.is_valid is True
    assert len(result.errors) == 0


def test_validate_setup_invalid():
    """Test that invalid setup is caught."""
    config = LSHExperimentConfig(
        n_samples=50,
        n_queries=10,
        dim=16,
        k=5,
        n_bits=4,
        n_codes=20,  # Invalid: 20 > 2**4 = 16
        random_state=42,
    )
    
    setup = create_experimental_setup(config)
    result = validate_setup(setup)
    
    # Should have errors from config validation
    assert result.is_valid is False
    assert len(result.errors) > 0


def test_generate_synthetic_data_reproducibility():
    """Test that synthetic data generation is reproducible."""
    config = LSHExperimentConfig(
        n_samples=50,
        n_queries=10,
        dim=16,
        k=5,
        random_state=42,
    )
    
    data1 = generate_synthetic_data(config)
    data2 = generate_synthetic_data(config)
    
    # Should be identical with same random_state
    np.testing.assert_array_equal(data1.base_embeddings, data2.base_embeddings)
    np.testing.assert_array_equal(data1.queries, data2.queries)
    np.testing.assert_array_equal(data1.ground_truth, data2.ground_truth)


def test_generate_synthetic_data_different_seeds():
    """Test that different seeds produce different data."""
    config1 = LSHExperimentConfig(
        n_samples=50,
        n_queries=10,
        dim=16,
        k=5,
        random_state=42,
    )
    
    config2 = LSHExperimentConfig(
        n_samples=50,
        n_queries=10,
        dim=16,
        k=5,
        random_state=123,
    )
    
    data1 = generate_synthetic_data(config1)
    data2 = generate_synthetic_data(config2)
    
    # Should be different with different random_state
    assert not np.array_equal(data1.base_embeddings, data2.base_embeddings)
    assert not np.array_equal(data1.queries, data2.queries)

