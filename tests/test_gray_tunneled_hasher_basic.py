"""Basic tests for GrayTunneledHasher."""

import numpy as np
import pytest

from gray_tunneled_hashing.algorithms.gray_tunneled_hasher import GrayTunneledHasher
from gray_tunneled_hashing.data.synthetic_generators import (
    PlantedModelConfig,
    generate_planted_phi,
    sample_noisy_embeddings,
)


def test_hasher_initialization():
    """Test that GrayTunneledHasher can be initialized."""
    hasher = GrayTunneledHasher(n_bits=4, random_state=42)
    assert hasher.n_bits == 4
    assert hasher.N == 16
    assert not hasher.is_fitted


def test_hasher_fit_synthetic():
    """Test that hasher can be fitted to synthetic planted embeddings."""
    n_bits = 4
    dim = 8
    N = 2 ** n_bits
    
    # Generate synthetic planted embeddings
    phi = generate_planted_phi(n_bits=n_bits, dim=dim, random_state=42)
    w = sample_noisy_embeddings(phi, sigma=0.1, random_state=43)
    
    hasher = GrayTunneledHasher(
        n_bits=n_bits,
        max_two_swap_iters=20,
        num_tunneling_steps=5,
        random_state=44,
    )
    
    hasher.fit(w)
    
    assert hasher.is_fitted
    assert hasher.pi_ is not None
    assert hasher.pi_.shape == (N,)
    assert hasher.cost_ is not None
    assert np.isfinite(hasher.cost_)
    assert hasher.cost_history_ is not None
    assert len(hasher.cost_history_) > 0


def test_hasher_fit_invalid_input():
    """Test that fit raises error for invalid input."""
    hasher = GrayTunneledHasher(n_bits=4)
    
    # Wrong shape
    with pytest.raises(ValueError, match="embeddings must have shape"):
        hasher.fit(np.array([[1, 2], [3, 4]]))  # Shape (2, 2) but N=16 expected
    
    # 1D array
    with pytest.raises(ValueError, match="must be 2D"):
        hasher.fit(np.array([1, 2, 3]))


def test_hasher_get_assignment():
    """Test that get_assignment returns the final permutation."""
    n_bits = 3
    dim = 6
    N = 2 ** n_bits
    
    # Generate synthetic embeddings
    phi = generate_planted_phi(n_bits=n_bits, dim=dim, random_state=42)
    w = sample_noisy_embeddings(phi, sigma=0.1, random_state=43)
    
    hasher = GrayTunneledHasher(
        n_bits=n_bits,
        max_two_swap_iters=10,
        num_tunneling_steps=3,
        random_state=44,
    )
    
    hasher.fit(w)
    pi = hasher.get_assignment()
    
    assert pi.shape == (N,)
    assert len(np.unique(pi)) == N  # Valid permutation
    assert np.all(pi >= 0)
    assert np.all(pi < N)
    
    # Should be the same as internal pi_
    assert np.array_equal(pi, hasher.pi_)


def test_hasher_get_assignment_before_fit():
    """Test that get_assignment raises error if not fitted."""
    hasher = GrayTunneledHasher(n_bits=4)
    
    with pytest.raises(ValueError, match="must be fitted"):
        hasher.get_assignment()


def test_hasher_cost_better_than_random():
    """Test that optimized cost is better than random assignment."""
    n_bits = 4
    dim = 8
    N = 2 ** n_bits
    
    # Generate synthetic embeddings
    phi = generate_planted_phi(n_bits=n_bits, dim=dim, random_state=42)
    w = sample_noisy_embeddings(phi, sigma=0.1, random_state=43)
    
    # Compute distance matrix
    from gray_tunneled_hashing.algorithms.qap_objective import (
        generate_hypercube_edges,
        qap_cost,
    )
    
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            D[i, j] = np.linalg.norm(w[i] - w[j]) ** 2
    
    edges = generate_hypercube_edges(n_bits)
    
    # Estimate average cost of random assignments
    random_costs = []
    for _ in range(10):
        pi_rand = np.random.permutation(N)
        cost_rand = qap_cost(pi_rand, D, edges)
        random_costs.append(cost_rand)
    avg_random_cost = np.mean(random_costs)
    
    # Run Gray-Tunneled optimization
    hasher = GrayTunneledHasher(
        n_bits=n_bits,
        max_two_swap_iters=30,
        num_tunneling_steps=5,
        random_state=44,
    )
    hasher.fit(w)
    
    # Optimized cost should be less than or equal to average random cost
    assert hasher.cost_ <= avg_random_cost * 1.1, (
        f"Optimized cost {hasher.cost_:.6f} should be <= average random cost "
        f"{avg_random_cost:.6f} (with small tolerance)"
    )


def test_end_to_end_synthetic_run():
    """Test complete end-to-end synthetic run."""
    n_bits = 3
    dim = 6
    N = 2 ** n_bits
    
    # Generate planted model instance
    config = PlantedModelConfig(
        n_bits=n_bits,
        dim=dim,
        sigma=0.1,
        random_state=42,
    )
    vertices, phi, w = config.generate()
    
    assert w.shape == (N, dim)
    
    # Initialize and fit hasher
    hasher = GrayTunneledHasher(
        n_bits=n_bits,
        block_size=4,
        max_two_swap_iters=20,
        num_tunneling_steps=5,
        random_state=43,
    )
    
    hasher.fit(w)
    
    # Verify results
    assert hasher.is_fitted
    assert hasher.pi_ is not None
    assert hasher.pi_.shape == (N,)
    assert hasher.cost_ is not None
    assert np.isfinite(hasher.cost_)
    
    # Get assignment
    pi = hasher.get_assignment()
    assert pi.shape == (N,)
    assert len(np.unique(pi)) == N
    
    # Test encode (minimal for Sprint 1)
    codes = hasher.encode(w)
    assert codes.shape == (N, n_bits)
    
    # Test evaluate
    metrics = hasher.evaluate(w, codes)
    assert isinstance(metrics, dict)
    assert "final_qap_cost" in metrics
    assert metrics["final_qap_cost"] == hasher.cost_
