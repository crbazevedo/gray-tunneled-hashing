"""Tests for experimental configuration."""

import pytest
import json
from gray_tunneled_hashing.experiments.config import LSHExperimentConfig


def test_lsh_experiment_config_defaults():
    """Test that defaults are reasonable."""
    config = LSHExperimentConfig()
    
    assert config.n_samples == 100
    assert config.n_queries == 20
    assert config.dim == 16
    assert config.k == 5
    assert config.n_bits == 8
    assert config.n_codes == 32
    assert config.lsh_family == "hyperplane"
    assert config.use_gth is True
    assert config.hamming_radius == 1
    assert config.random_state == 42
    assert config.n_runs == 3


def test_lsh_experiment_config_validation():
    """Test parameter range validation."""
    # Valid config
    config = LSHExperimentConfig()
    errors = config.validate()
    assert len(errors) == 0, f"Valid config should have no errors, got: {errors}"
    
    # Invalid: n_samples <= 0
    config = LSHExperimentConfig(n_samples=0)
    errors = config.validate()
    assert len(errors) > 0
    assert any("n_samples" in e for e in errors)
    
    # Invalid: n_codes > 2**n_bits
    config = LSHExperimentConfig(n_bits=4, n_codes=20)  # 2**4 = 16
    errors = config.validate()
    assert len(errors) > 0
    assert any("n_codes" in e for e in errors)
    
    # Invalid: k > n_samples
    config = LSHExperimentConfig(n_samples=10, k=15)
    errors = config.validate()
    assert len(errors) > 0
    assert any("k" in e for e in errors)
    
    # Invalid: lsh_family
    config = LSHExperimentConfig(lsh_family="invalid")
    errors = config.validate()
    assert len(errors) > 0
    assert any("lsh_family" in e for e in errors)
    
    # Invalid: mode
    config = LSHExperimentConfig(mode="invalid")
    errors = config.validate()
    assert len(errors) > 0
    assert any("mode" in e for e in errors)
    
    # Invalid: hamming_radius > n_bits
    config = LSHExperimentConfig(n_bits=4, hamming_radius=10)
    errors = config.validate()
    assert len(errors) > 0
    assert any("hamming_radius" in e for e in errors)


def test_lsh_experiment_config_serialization():
    """Test JSON serialization/deserialization."""
    config = LSHExperimentConfig(
        n_samples=50,
        n_queries=10,
        n_bits=6,
        n_codes=16,
        lsh_family="p_stable",
        random_state=123,
    )
    
    # Serialize
    json_str = config.to_json()
    assert isinstance(json_str, str)
    
    # Deserialize
    config2 = LSHExperimentConfig.from_json(json_str)
    
    # Check equality
    assert config2.n_samples == config.n_samples
    assert config2.n_queries == config.n_queries
    assert config2.n_bits == config.n_bits
    assert config2.n_codes == config.n_codes
    assert config2.lsh_family == config.lsh_family
    assert config2.random_state == config.random_state
    
    # Validate deserialized config
    errors = config2.validate()
    assert len(errors) == 0


def test_lsh_experiment_config_dict():
    """Test dictionary conversion."""
    config = LSHExperimentConfig(n_samples=50, n_bits=6)
    d = config.to_dict()
    
    assert isinstance(d, dict)
    assert d["n_samples"] == 50
    assert d["n_bits"] == 6
    
    # Round-trip
    config2 = LSHExperimentConfig.from_dict(d)
    assert config2.n_samples == config.n_samples
    assert config2.n_bits == config.n_bits

