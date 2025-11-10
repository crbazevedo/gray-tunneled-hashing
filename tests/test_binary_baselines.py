"""Tests for baseline binary encoding methods."""

import numpy as np
import pytest

from gray_tunneled_hashing.binary.baselines import (
    sign_binarize,
    random_projection_binarize,
    apply_random_projection,
)


def test_sign_binarize():
    """Test sign binarization."""
    embeddings = np.array([
        [1.0, -0.5, 0.0, 2.0],
        [-1.0, 0.5, -0.1, -2.0],
    ])
    
    codes = sign_binarize(embeddings)
    
    assert codes.shape == (2, 4)
    assert codes.dtype == bool
    assert np.array_equal(codes[0], [True, False, False, True])
    assert np.array_equal(codes[1], [False, True, False, False])


def test_sign_binarize_zero():
    """Test sign binarization with zeros."""
    embeddings = np.array([[0.0, 0.0]])
    codes = sign_binarize(embeddings)
    
    assert np.array_equal(codes, [[False, False]])


def test_random_projection_binarize():
    """Test random projection binarization."""
    embeddings = np.random.randn(10, 64).astype(np.float32)
    n_bits = 32
    random_state = 42
    
    codes, proj_matrix = random_projection_binarize(
        embeddings, n_bits=n_bits, random_state=random_state
    )
    
    assert codes.shape == (10, n_bits)
    assert codes.dtype == bool
    assert proj_matrix.shape == (n_bits, 64)
    
    # Test determinism
    codes2, proj_matrix2 = random_projection_binarize(
        embeddings, n_bits=n_bits, random_state=random_state
    )
    
    assert np.array_equal(codes, codes2)
    assert np.allclose(proj_matrix, proj_matrix2)


def test_random_projection_binarize_different_seeds():
    """Test that different seeds produce different results."""
    embeddings = np.random.randn(10, 64).astype(np.float32)
    n_bits = 32
    
    codes1, _ = random_projection_binarize(embeddings, n_bits=n_bits, random_state=42)
    codes2, _ = random_projection_binarize(embeddings, n_bits=n_bits, random_state=43)
    
    # Should be different (very unlikely to be identical)
    assert not np.array_equal(codes1, codes2)


def test_random_projection_binarize_invalid_n_bits():
    """Test with invalid n_bits."""
    embeddings = np.random.randn(10, 64).astype(np.float32)
    
    with pytest.raises(ValueError, match="must be positive"):
        random_projection_binarize(embeddings, n_bits=0)


def test_apply_random_projection():
    """Test applying pre-computed random projection."""
    embeddings = np.random.randn(5, 64).astype(np.float32)
    proj_matrix = np.random.randn(32, 64).astype(np.float32)
    
    codes = apply_random_projection(embeddings, proj_matrix)
    
    assert codes.shape == (5, 32)
    assert codes.dtype == bool
    
    # Verify it's the same as computing from scratch
    projections = embeddings @ proj_matrix.T
    expected_codes = (projections > 0).astype(bool)
    
    assert np.array_equal(codes, expected_codes)


def test_apply_random_projection_shape_mismatch():
    """Test with shape mismatch."""
    embeddings = np.random.randn(5, 64).astype(np.float32)
    proj_matrix = np.random.randn(32, 128).astype(np.float32)  # Wrong dimension
    
    with pytest.raises(ValueError):
        apply_random_projection(embeddings, proj_matrix)

