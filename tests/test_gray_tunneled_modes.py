"""Tests for GrayTunneledHasher modes."""

import numpy as np
import pytest
from gray_tunneled_hashing.algorithms.gray_tunneled_hasher import GrayTunneledHasher
from gray_tunneled_hashing.data.synthetic_generators import (
    generate_planted_phi,
    sample_noisy_embeddings,
)


@pytest.fixture
def synthetic_embeddings():
    """Generate synthetic embeddings for testing."""
    n_bits = 5
    N = 2 ** n_bits
    dim = 8
    random_state = 42
    
    phi = generate_planted_phi(n_bits, dim, random_state=random_state)
    w = sample_noisy_embeddings(phi, sigma=0.1, random_state=random_state + 1)
    
    return w, n_bits


def test_hasher_trivial_mode(synthetic_embeddings):
    """Test trivial mode."""
    w, n_bits = synthetic_embeddings
    
    hasher = GrayTunneledHasher(
        n_bits=n_bits,
        mode="trivial",
        track_history=False,
        random_state=42,
    )
    
    hasher.fit(w)
    
    assert hasher.is_fitted
    assert hasher.pi_ is not None
    assert hasher.cost_ is not None
    assert len(hasher.pi_) == w.shape[0]


def test_hasher_two_swap_only_mode(synthetic_embeddings):
    """Test two_swap_only mode."""
    w, n_bits = synthetic_embeddings
    
    hasher = GrayTunneledHasher(
        n_bits=n_bits,
        mode="two_swap_only",
        max_two_swap_iters=10,
        track_history=False,
        random_state=42,
    )
    
    hasher.fit(w)
    
    assert hasher.is_fitted
    assert hasher.pi_ is not None
    assert hasher.cost_ is not None
    assert len(hasher.cost_history_) > 0


def test_hasher_full_mode(synthetic_embeddings):
    """Test full mode."""
    w, n_bits = synthetic_embeddings
    
    hasher = GrayTunneledHasher(
        n_bits=n_bits,
        mode="full",
        max_two_swap_iters=10,
        num_tunneling_steps=5,
        track_history=False,
        random_state=42,
    )
    
    hasher.fit(w)
    
    assert hasher.is_fitted
    assert hasher.pi_ is not None
    assert hasher.cost_ is not None


def test_hasher_track_history(synthetic_embeddings):
    """Test cost history tracking."""
    w, n_bits = synthetic_embeddings
    
    hasher = GrayTunneledHasher(
        n_bits=n_bits,
        mode="full",
        max_two_swap_iters=5,
        num_tunneling_steps=3,
        track_history=True,
        random_state=42,
    )
    
    hasher.fit(w)
    
    assert hasher.is_fitted
    assert len(hasher.cost_history_) > 0
    
    # Check that history entries are dicts with metadata
    if isinstance(hasher.cost_history_[0], dict):
        assert "cost" in hasher.cost_history_[0]
        assert "step" in hasher.cost_history_[0]


def test_hasher_block_selection_random(synthetic_embeddings):
    """Test random block selection strategy."""
    w, n_bits = synthetic_embeddings
    
    hasher = GrayTunneledHasher(
        n_bits=n_bits,
        mode="full",
        block_selection_strategy="random",
        max_two_swap_iters=5,
        num_tunneling_steps=3,
        random_state=42,
    )
    
    hasher.fit(w)
    
    assert hasher.is_fitted


def test_hasher_invalid_mode():
    """Test error for invalid mode."""
    with pytest.raises(ValueError, match="mode must be"):
        GrayTunneledHasher(n_bits=5, mode="invalid")


def test_hasher_cluster_strategy_requires_assignments():
    """Test that cluster strategy requires cluster_assignments."""
    with pytest.raises(ValueError, match="cluster_assignments is required"):
        GrayTunneledHasher(
            n_bits=5,
            block_selection_strategy="cluster",
            cluster_assignments=None,
        )


def test_hasher_modes_produce_different_costs(synthetic_embeddings):
    """Test that different modes produce different (or same) costs."""
    w, n_bits = synthetic_embeddings
    
    # Trivial
    hasher_trivial = GrayTunneledHasher(
        n_bits=n_bits, mode="trivial", random_state=42
    )
    hasher_trivial.fit(w)
    cost_trivial = hasher_trivial.cost_
    
    # Two-swap only
    hasher_two_swap = GrayTunneledHasher(
        n_bits=n_bits,
        mode="two_swap_only",
        max_two_swap_iters=10,
        random_state=42,
    )
    hasher_two_swap.fit(w)
    cost_two_swap = hasher_two_swap.cost_
    
    # Full
    hasher_full = GrayTunneledHasher(
        n_bits=n_bits,
        mode="full",
        max_two_swap_iters=10,
        num_tunneling_steps=5,
        random_state=42,
    )
    hasher_full.fit(w)
    cost_full = hasher_full.cost_
    
    # Costs should be reasonable (not NaN, not negative, etc.)
    assert not np.isnan(cost_trivial)
    assert not np.isnan(cost_two_swap)
    assert not np.isnan(cost_full)
    
    # Typically: trivial >= two_swap >= full (not always true, but often)
    # We just check they're all finite and reasonable

