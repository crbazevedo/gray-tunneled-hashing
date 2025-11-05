"""Basic tests for GrayTunneledHasher."""

import numpy as np
import pytest

from gray_tunneled_hashing import GrayTunneledHasher, generate_synthetic_embeddings


def test_hasher_initialization():
    """Test that GrayTunneledHasher can be initialized."""
    hasher = GrayTunneledHasher(code_length=64)
    assert hasher.code_length == 64
    assert not hasher.is_fitted


def test_hasher_fit():
    """Test that hasher can be fitted to data."""
    hasher = GrayTunneledHasher(code_length=32)
    embeddings = generate_synthetic_embeddings(n_points=10, dim=64, seed=42)
    
    hasher.fit(embeddings)
    assert hasher.is_fitted
    assert hasher._mean is not None
    assert hasher._std is not None
    assert hasher._mean.shape == (64,)
    assert hasher._std.shape == (64,)


def test_hasher_fit_invalid_input():
    """Test that fit raises error for invalid input."""
    hasher = GrayTunneledHasher()
    # 1D array should raise error
    with pytest.raises(ValueError, match="must be 2D"):
        hasher.fit(np.array([1, 2, 3]))


def test_hasher_encode():
    """Test that hasher can encode embeddings after fitting."""
    hasher = GrayTunneledHasher(code_length=32)
    embeddings = generate_synthetic_embeddings(n_points=10, dim=64, seed=42)
    
    hasher.fit(embeddings)
    codes = hasher.encode(embeddings)
    
    assert codes.shape == (10, 32)
    assert codes.dtype == np.uint8
    assert np.all((codes == 0) | (codes == 1))


def test_hasher_encode_before_fit():
    """Test that encode raises error if not fitted."""
    hasher = GrayTunneledHasher()
    embeddings = generate_synthetic_embeddings(n_points=5, dim=32, seed=42)
    
    with pytest.raises(ValueError, match="must be fitted"):
        hasher.encode(embeddings)


def test_hasher_decode():
    """Test that hasher can decode codes (stub implementation)."""
    hasher = GrayTunneledHasher(code_length=32)
    embeddings = generate_synthetic_embeddings(n_points=10, dim=64, seed=42)
    
    hasher.fit(embeddings)
    codes = hasher.encode(embeddings)
    decoded = hasher.decode(codes)
    
    # Decode is a stub, so it returns zeros
    assert decoded.shape == (10, 64)
    assert decoded.dtype == np.float64 or decoded.dtype == np.float32


def test_hasher_evaluate():
    """Test that hasher can evaluate encoding quality."""
    hasher = GrayTunneledHasher(code_length=32)
    embeddings = generate_synthetic_embeddings(n_points=10, dim=64, seed=42)
    
    hasher.fit(embeddings)
    codes = hasher.encode(embeddings)
    metrics = hasher.evaluate(embeddings, codes)
    
    assert isinstance(metrics, dict)
    assert "n_samples" in metrics
    assert "embedding_dim" in metrics
    assert "code_length" in metrics
    assert metrics["n_samples"] == 10
    assert metrics["embedding_dim"] == 64
    assert metrics["code_length"] == 32


def test_end_to_end_flow():
    """Test complete flow: generate data, fit, encode, evaluate."""
    # Generate synthetic embeddings
    embeddings = generate_synthetic_embeddings(n_points=20, dim=128, seed=123)
    assert embeddings.shape == (20, 128)
    
    # Initialize and fit hasher
    hasher = GrayTunneledHasher(code_length=64)
    hasher.fit(embeddings)
    
    # Encode embeddings
    codes = hasher.encode(embeddings)
    assert codes.shape == (20, 64)
    
    # Evaluate
    metrics = hasher.evaluate(embeddings, codes)
    assert metrics["n_samples"] == 20
    assert metrics["embedding_dim"] == 128
    assert metrics["code_length"] == 64
    
    # Verify codes are binary
    assert np.all((codes == 0) | (codes == 1))

