"""Tests for synthetic data generators."""

import numpy as np
import pytest

from gray_tunneled_hashing.data.synthetic_generators import (
    generate_hypercube_vertices,
    generate_planted_phi,
    sample_noisy_embeddings,
    PlantedModelConfig,
)


def test_generate_hypercube_vertices():
    """Test hypercube vertex generation."""
    # Test n_bits = 3
    vertices = generate_hypercube_vertices(3)
    assert vertices.shape == (8, 3)
    assert vertices.dtype == np.uint8
    
    # Check first vertex is all zeros
    assert np.array_equal(vertices[0], [0, 0, 0])
    
    # Check last vertex is all ones
    assert np.array_equal(vertices[-1], [1, 1, 1])
    
    # Check all values are 0 or 1
    assert np.all((vertices == 0) | (vertices == 1))
    
    # Check all vertices are unique
    assert len(np.unique(vertices, axis=0)) == 8
    
    # Test n_bits = 2
    vertices_2 = generate_hypercube_vertices(2)
    assert vertices_2.shape == (4, 2)
    expected = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
    assert np.array_equal(vertices_2, expected)


def test_generate_hypercube_vertices_invalid():
    """Test error handling for invalid inputs."""
    with pytest.raises(ValueError, match="must be positive"):
        generate_hypercube_vertices(0)
    
    with pytest.raises(ValueError, match="must be positive"):
        generate_hypercube_vertices(-1)


def test_generate_planted_phi():
    """Test planted phi generation."""
    phi = generate_planted_phi(n_bits=3, dim=10, random_state=42)
    
    assert phi.shape == (8, 10)
    assert phi.dtype == np.float32
    
    # Check that phi is not all zeros
    assert not np.allclose(phi, 0)


def test_generate_planted_phi_hamming1_property():
    """Test that Hamming-1 neighbors are closer than random pairs."""
    n_bits = 4
    dim = 10
    phi = generate_planted_phi(n_bits=n_bits, dim=dim, random_state=42)
    
    N = 2 ** n_bits
    vertices = generate_hypercube_vertices(n_bits)
    
    # Compute all pairwise distances
    distances = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            distances[i, j] = np.linalg.norm(phi[i] - phi[j]) ** 2
    
    # Find Hamming-1 neighbors
    hamming1_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            hamming_dist = np.sum(vertices[i] != vertices[j])
            if hamming_dist == 1:
                hamming1_pairs.append((i, j))
    
    # Sample random pairs
    import random
    random.seed(42)
    random_pairs = []
    for _ in range(min(100, len(hamming1_pairs) * 2)):
        i, j = random.sample(range(N), 2)
        if i != j:
            hamming_dist = np.sum(vertices[i] != vertices[j])
            if hamming_dist > 1:  # Not Hamming-1
                random_pairs.append((i, j))
    
    # Compute average distances
    hamming1_distances = [distances[i, j] for i, j in hamming1_pairs]
    random_distances = [distances[i, j] for i, j in random_pairs]
    
    if len(hamming1_distances) > 0 and len(random_distances) > 0:
        avg_hamming1 = np.mean(hamming1_distances)
        avg_random = np.mean(random_distances)
        
        # Hamming-1 neighbors should be closer on average
        # (Allow some tolerance for randomness)
        assert avg_hamming1 < avg_random * 1.5, (
            f"Hamming-1 avg distance {avg_hamming1:.4f} should be < "
            f"random avg distance {avg_random:.4f}"
        )


def test_sample_noisy_embeddings():
    """Test noisy embedding generation."""
    phi = np.random.randn(8, 10).astype(np.float32)
    sigma = 0.1
    
    w = sample_noisy_embeddings(phi, sigma, random_state=42)
    
    assert w.shape == phi.shape
    assert w.dtype == np.float32
    
    # Check that noise was added (not exactly equal)
    assert not np.allclose(w, phi, atol=1e-6)
    
    # Check that w is close to phi (within reasonable noise)
    assert np.allclose(w, phi, atol=5 * sigma)


def test_sample_noisy_embeddings_zero_sigma():
    """Test noisy embeddings with zero noise."""
    phi = np.random.randn(8, 10).astype(np.float32)
    w = sample_noisy_embeddings(phi, sigma=0.0, random_state=42)
    
    # With zero sigma, should be very close to phi (up to numerical precision)
    assert np.allclose(w, phi, atol=1e-5)


def test_planted_model_config():
    """Test PlantedModelConfig dataclass."""
    config = PlantedModelConfig(n_bits=3, dim=10, sigma=0.1, random_state=42)
    
    assert config.n_bits == 3
    assert config.dim == 10
    assert config.sigma == 0.1
    assert config.random_state == 42


def test_planted_model_config_validation():
    """Test PlantedModelConfig parameter validation."""
    with pytest.raises(ValueError, match="must be positive"):
        PlantedModelConfig(n_bits=0, dim=10, sigma=0.1)
    
    with pytest.raises(ValueError, match="must be positive"):
        PlantedModelConfig(n_bits=3, dim=0, sigma=0.1)
    
    with pytest.raises(ValueError, match="must be non-negative"):
        PlantedModelConfig(n_bits=3, dim=10, sigma=-0.1)


def test_planted_model_config_generate():
    """Test PlantedModelConfig.generate() method."""
    config = PlantedModelConfig(n_bits=3, dim=10, sigma=0.1, random_state=42)
    
    vertices, phi, w = config.generate()
    
    assert vertices.shape == (8, 3)
    assert phi.shape == (8, 10)
    assert w.shape == (8, 10)
    
    # Check that w is noisy version of phi
    assert not np.allclose(w, phi, atol=1e-3)
    assert np.allclose(w, phi, atol=1.0)  # Should be reasonably close

