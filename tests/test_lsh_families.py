"""Tests for LSH families."""

import numpy as np
import pytest

from gray_tunneled_hashing.binary.lsh_families import (
    HyperplaneLSH,
    PStableLSH,
    create_lsh_family,
    validate_lsh_properties,
)


def test_hyperplane_lsh_basic():
    """Test basic hyperplane LSH functionality."""
    n_bits = 8
    dim = 16
    n_vectors = 10
    
    lsh = HyperplaneLSH(n_bits=n_bits, dim=dim, random_state=42)
    
    # Generate test vectors
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    
    # Hash vectors
    codes = lsh.hash(vectors)
    
    # Check output shape and dtype
    assert codes.shape == (n_vectors, n_bits)
    assert codes.dtype == bool
    
    # Check that hyperplanes are normalized
    norms = np.linalg.norm(lsh.hyperplanes, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_hyperplane_lsh_collision_probability():
    """Test collision probability calculation."""
    lsh = HyperplaneLSH(n_bits=8, dim=16, random_state=42)
    
    # High similarity should have high collision probability
    p_high = lsh.collision_probability(0.9)
    p_low = lsh.collision_probability(0.1)
    
    assert p_high > p_low, "Higher similarity should have higher collision probability"
    assert 0.0 <= p_high <= 1.0
    assert 0.0 <= p_low <= 1.0
    
    # Identical vectors (similarity = 1.0) should have high collision probability
    p_identical = lsh.collision_probability(1.0)
    assert p_identical > 0.5  # Should be quite high


def test_p_stable_lsh_basic():
    """Test basic p-stable LSH functionality."""
    n_bits = 8
    dim = 16
    n_vectors = 10
    
    lsh = PStableLSH(n_bits=n_bits, dim=dim, w=1.0, random_state=42)
    
    # Generate test vectors
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    
    # Hash vectors
    codes = lsh.hash(vectors)
    
    # Check output shape and dtype
    assert codes.shape == (n_vectors, n_bits)
    assert codes.dtype == bool


def test_p_stable_lsh_collision_probability():
    """Test collision probability calculation."""
    lsh = PStableLSH(n_bits=8, dim=16, w=1.0, random_state=42)
    
    # Small distance should have high collision probability
    p_small = lsh.collision_probability(0.1)
    p_large = lsh.collision_probability(10.0)
    
    assert p_small > p_large, "Smaller distance should have higher collision probability"
    assert 0.0 <= p_small <= 1.0
    assert 0.0 <= p_large <= 1.0
    
    # Zero distance should have high collision probability
    p_zero = lsh.collision_probability(0.0)
    assert p_zero > 0.5  # Should be quite high


def test_create_lsh_family():
    """Test factory function."""
    # Hyperplane
    lsh1 = create_lsh_family("hyperplane", n_bits=8, dim=16, random_state=42)
    assert isinstance(lsh1, HyperplaneLSH)
    
    # P-stable
    lsh2 = create_lsh_family("p_stable", n_bits=8, dim=16, w=2.0, random_state=42)
    assert isinstance(lsh2, PStableLSH)
    assert lsh2.w == 2.0
    
    # Invalid family
    with pytest.raises(ValueError, match="Unknown LSH family"):
        create_lsh_family("invalid", n_bits=8, dim=16)


def test_lsh_determinism():
    """Test that LSH is deterministic with same random_state."""
    n_bits = 8
    dim = 16
    vectors = np.random.randn(10, dim).astype(np.float32)
    
    lsh1 = HyperplaneLSH(n_bits=n_bits, dim=dim, random_state=42)
    lsh2 = HyperplaneLSH(n_bits=n_bits, dim=dim, random_state=42)
    
    codes1 = lsh1.hash(vectors)
    codes2 = lsh2.hash(vectors)
    
    assert np.array_equal(codes1, codes2), "LSH should be deterministic with same seed"


def test_lsh_different_seeds():
    """Test that different seeds produce different results."""
    n_bits = 8
    dim = 16
    vectors = np.random.randn(10, dim).astype(np.float32)
    
    lsh1 = HyperplaneLSH(n_bits=n_bits, dim=dim, random_state=42)
    lsh2 = HyperplaneLSH(n_bits=n_bits, dim=dim, random_state=43)
    
    codes1 = lsh1.hash(vectors)
    codes2 = lsh2.hash(vectors)
    
    # Should be different (very unlikely to be identical)
    assert not np.array_equal(codes1, codes2), "Different seeds should produce different codes"


def test_validate_lsh_properties_hyperplane():
    """Test LSH property validation for hyperplane LSH."""
    n_bits = 16
    dim = 32
    n_vectors = 100
    
    # Generate vectors with varying similarities
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    # Normalize for cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / (norms + 1e-10)
    
    lsh = HyperplaneLSH(n_bits=n_bits, dim=dim, random_state=42)
    
    def cosine_similarity(x, y):
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-10)
    
    results = validate_lsh_properties(
        lsh, vectors, cosine_similarity, n_samples=500
    )
    
    # Check that results are reasonable
    assert "collision_rates" in results
    assert "theoretical_probs" in results
    assert "correlation" in results
    
    # Correlation should be positive (empirical should follow theoretical trend)
    assert results["correlation"] > 0.0 or np.isnan(results["correlation"])


def test_lsh_shape_validation():
    """Test that LSH validates input shapes."""
    lsh = HyperplaneLSH(n_bits=8, dim=16, random_state=42)
    
    # Wrong dimension
    vectors_wrong = np.random.randn(10, 20).astype(np.float32)
    with pytest.raises(ValueError, match="dimension"):
        lsh.hash(vectors_wrong)


def test_p_stable_width_parameter():
    """Test that width parameter affects collision probability."""
    lsh1 = PStableLSH(n_bits=8, dim=16, w=0.5, random_state=42)
    lsh2 = PStableLSH(n_bits=8, dim=16, w=2.0, random_state=42)
    
    # Larger width should have higher collision probability for same distance
    p1 = lsh1.collision_probability(1.0)
    p2 = lsh2.collision_probability(1.0)
    
    assert p2 > p1, "Larger width should increase collision probability"

