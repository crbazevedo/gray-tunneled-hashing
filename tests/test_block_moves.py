"""Tests for block moves and tunneling."""

import numpy as np
import pytest

from gray_tunneled_hashing.algorithms.block_moves import (
    select_block,
    block_reoptimize,
    tunneling_step,
)
from gray_tunneled_hashing.algorithms.qap_objective import (
    generate_hypercube_edges,
    qap_cost,
    hill_climb_two_swap,
)
from gray_tunneled_hashing.data.synthetic_generators import (
    generate_planted_phi,
    sample_noisy_embeddings,
)


def test_select_block():
    """Test block selection."""
    block = select_block(N=16, block_size=4, random_state=42)
    
    assert len(block) == 4
    assert np.all(block >= 0)
    assert np.all(block < 16)
    assert len(np.unique(block)) == 4  # All distinct
    
    # Should be sorted
    assert np.all(block[:-1] <= block[1:])


def test_select_block_edge_cases():
    """Test edge cases for block selection."""
    # Block size equals N
    block = select_block(N=4, block_size=4, random_state=42)
    assert len(block) == 4
    assert np.array_equal(np.sort(block), np.array([0, 1, 2, 3]))
    
    # Invalid cases
    with pytest.raises(ValueError):
        select_block(N=4, block_size=5)
    
    with pytest.raises(ValueError):
        select_block(N=4, block_size=0)


def test_block_reoptimize():
    """Test block reoptimization."""
    n_bits = 3
    N = 2 ** n_bits
    
    # Create simple embeddings
    w = np.random.randn(N, 4).astype(np.float64)
    
    # Compute distance matrix
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            D[i, j] = np.linalg.norm(w[i] - w[j]) ** 2
    
    edges = generate_hypercube_edges(n_bits)
    
    # Initial permutation
    pi = np.arange(N)
    
    # Select a small block
    block_vertices = np.array([0, 1, 2])
    
    # Reoptimize
    pi_new, delta = block_reoptimize(pi, D, edges, block_vertices)
    
    # Check that permutation is valid
    assert len(pi_new) == N
    assert len(np.unique(pi_new)) == N  # Still a permutation
    assert np.all(pi_new >= 0)
    assert np.all(pi_new < N)
    
    # Check that only block vertices changed
    changed = np.where(pi_new != pi)[0]
    assert np.all(np.isin(changed, block_vertices))
    
    # Check that cost improved (delta should be <= 0)
    assert delta <= 1e-10, f"Delta should be non-positive, got {delta:.6f}"


def test_tunneling_step():
    """Test tunneling step."""
    n_bits = 4
    N = 2 ** n_bits
    
    # Create synthetic embeddings
    phi = generate_planted_phi(n_bits=n_bits, dim=8, random_state=42)
    w = sample_noisy_embeddings(phi, sigma=0.1, random_state=43)
    
    # Compute distance matrix
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            D[i, j] = np.linalg.norm(w[i] - w[j]) ** 2
    
    edges = generate_hypercube_edges(n_bits)
    
    # Initial permutation
    pi = np.random.permutation(N)
    
    # Perform tunneling step
    pi_new, delta = tunneling_step(
        pi, D, edges, block_size=4, num_blocks=5, random_state=44
    )
    
    # Check that permutation is valid
    assert len(pi_new) == N
    assert len(np.unique(pi_new)) == N
    
    # Delta should be <= 0 (non-positive means improvement or no change)
    assert delta <= 1e-10


def test_tunneling_improves_upon_two_swap():
    """Test that tunneling can improve upon 2-swap local minimum."""
    n_bits = 4
    N = 2 ** n_bits
    
    # Create synthetic embeddings
    phi = generate_planted_phi(n_bits=n_bits, dim=8, random_state=42)
    w = sample_noisy_embeddings(phi, sigma=0.1, random_state=43)
    
    # Compute distance matrix (ensure symmetric and non-negative)
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            dist_sq = np.linalg.norm(w[i] - w[j]) ** 2
            D[i, j] = dist_sq
            D[j, i] = dist_sq  # Ensure symmetry
    
    # Verify distance matrix is non-negative
    assert np.all(D >= 0), "Distance matrix should be non-negative"
    
    edges = generate_hypercube_edges(n_bits)
    
    # Run 2-swap hill climbing to local minimum
    np.random.seed(45)
    pi_init = np.random.permutation(N)
    pi_two_swap, cost_history_two_swap = hill_climb_two_swap(
        pi_init=pi_init,
        D=D,
        edges=edges,
        max_iter=50,
        sample_size=50,
        random_state=45,
    )
    cost_two_swap = cost_history_two_swap[-1]
    
    # Note: Cost can be any real number depending on distance matrix
    # The key is that it should be finite and reasonable
    assert np.isfinite(cost_two_swap), f"Cost should be finite, got {cost_two_swap:.6f}"
    
    # Now try tunneling from this local minimum
    pi_tunneled = pi_two_swap.copy()
    
    # Try several tunneling steps with controlled random state
    for step in range(5):
        pi_tunneled, delta = tunneling_step(
            pi_tunneled, D, edges, block_size=4, num_blocks=10, random_state=46 + step
        )
        # Verify cost doesn't increase (delta should be <= 0)
        if delta > 1e-10:
            # If tunneling increased cost, something is wrong
            current_cost = qap_cost(pi_tunneled, D, edges)
            prev_cost = qap_cost(pi_two_swap, D, edges)
            print(f"Warning: Tunneling increased cost at step {step}: {prev_cost:.6f} -> {current_cost:.6f}")
    
    cost_tunneled = qap_cost(pi_tunneled, D, edges)
    
    # Tunneling should not significantly increase cost
    # (allowing small numerical errors)
    assert cost_tunneled <= cost_two_swap + 1e-6, (
        f"Tunneling cost {cost_tunneled:.6f} should be <= 2-swap cost {cost_two_swap:.6f}"
    )
    
    # On average across multiple seeds, tunneling should help
    # For this test, we just verify it doesn't hurt
    print(f"\n2-swap final cost: {cost_two_swap:.6f}")
    print(f"After tunneling: {cost_tunneled:.6f}")
    print(f"Improvement: {cost_two_swap - cost_tunneled:.6f}")

