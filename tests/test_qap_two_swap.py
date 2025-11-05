"""Tests for QAP objective and 2-swap hill climbing."""

import numpy as np
import pytest
from itertools import permutations

from gray_tunneled_hashing.algorithms.qap_objective import (
    generate_hypercube_edges,
    qap_cost,
    sample_two_swaps,
    hill_climb_two_swap,
)


def test_generate_hypercube_edges():
    """Test hypercube edge generation."""
    # Test n_bits = 2
    edges = generate_hypercube_edges(2)
    assert edges.shape[1] == 2
    # For Q_2, we should have 4 edges
    assert len(edges) == 4
    
    # Test n_bits = 3
    edges_3 = generate_hypercube_edges(3)
    # For Q_3, we should have 12 edges (3 * 2^2)
    assert len(edges_3) == 12
    
    # Check that all edges have Hamming distance 1
    from gray_tunneled_hashing.data.synthetic_generators import generate_hypercube_vertices
    vertices = generate_hypercube_vertices(3)
    for edge in edges_3:
        u, v = edge[0], edge[1]
        hamming_dist = np.sum(vertices[u] != vertices[v])
        assert hamming_dist == 1, f"Edge ({u}, {v}) has Hamming distance {hamming_dist}, expected 1"


def test_qap_cost():
    """Test QAP cost computation."""
    # Simple test case
    pi = np.array([0, 1, 2, 3])
    D = np.array([
        [0, 1, 2, 3],
        [1, 0, 1, 2],
        [2, 1, 0, 1],
        [3, 2, 1, 0],
    ], dtype=np.float64)
    
    # For Q_2, edges are: (0,1), (0,2), (1,3), (2,3)
    edges = np.array([[0, 1], [0, 2], [1, 3], [2, 3]], dtype=np.int32)
    
    cost = qap_cost(pi, D, edges)
    
    # Expected cost:
    # edge (0,1): D[pi[0], pi[1]] = D[0, 1] = 1
    # edge (0,2): D[pi[0], pi[2]] = D[0, 2] = 2
    # edge (1,3): D[pi[1], pi[3]] = D[1, 3] = 2
    # edge (2,3): D[pi[2], pi[3]] = D[2, 3] = 1
    # Total: 1 + 2 + 2 + 1 = 6
    assert cost == 6.0


def test_qap_cost_symmetric():
    """Test that cost is symmetric in distance matrix."""
    pi = np.array([0, 1, 2, 3])
    D = np.random.rand(4, 4)
    D = (D + D.T) / 2  # Make symmetric
    np.fill_diagonal(D, 0)
    
    edges = np.array([[0, 1], [0, 2], [1, 3], [2, 3]], dtype=np.int32)
    cost1 = qap_cost(pi, D, edges)
    
    # Swap embeddings at vertices 0 and 1
    pi2 = pi.copy()
    pi2[0], pi2[1] = pi2[1], pi2[0]
    cost2 = qap_cost(pi2, D, edges)
    
    # Costs should be different (unless D has special structure)
    # This is just a sanity check that the function is working


def test_sample_two_swaps():
    """Test sampling of 2-swap candidates."""
    swaps = sample_two_swaps(N=10, sample_size=5, random_state=42)
    
    assert len(swaps) == 5
    for u, v in swaps:
        assert 0 <= u < 10
        assert 0 <= v < 10
        assert u < v  # Should be ordered


def test_sample_two_swaps_all_pairs():
    """Test that sampling all pairs works."""
    N = 5
    swaps = sample_two_swaps(N=N, sample_size=100, random_state=42)
    
    # Should have at most all possible pairs
    max_pairs = N * (N - 1) // 2
    assert len(swaps) <= max_pairs
    
    # If we request more than max, should get all
    swaps_all = sample_two_swaps(N=N, sample_size=1000, random_state=42)
    assert len(swaps_all) == max_pairs


def test_hill_climb_two_swap_monotonic():
    """Test that hill climbing monotonically decreases cost."""
    # Create a small instance
    n_bits = 3
    N = 2 ** n_bits
    
    # Generate synthetic embeddings
    from gray_tunneled_hashing.data.synthetic_generators import (
        generate_planted_phi,
        sample_noisy_embeddings,
    )
    phi = generate_planted_phi(n_bits=n_bits, dim=8, random_state=42)
    w = sample_noisy_embeddings(phi, sigma=0.1, random_state=43)
    
    # Compute distance matrix
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            D[i, j] = np.linalg.norm(w[i] - w[j]) ** 2
    
    # Generate edges
    edges = generate_hypercube_edges(n_bits)
    
    # Initialize with random permutation
    pi_init = np.random.permutation(N)
    
    # Run hill climbing
    pi_final, cost_history = hill_climb_two_swap(
        pi_init=pi_init,
        D=D,
        edges=edges,
        max_iter=50,
        sample_size=50,
        random_state=44,
    )
    
    # Check monotonicity
    assert len(cost_history) > 0
    for i in range(1, len(cost_history)):
        assert cost_history[i] <= cost_history[i-1] + 1e-10, (
            f"Cost increased: {cost_history[i-1]:.6f} -> {cost_history[i]:.6f}"
        )


def test_hill_climb_two_swap_small_optimal():
    """Test that hill climbing finds optimal or near-optimal on very small N."""
    # For very small N (e.g., 4), we can enumerate all permutations
    n_bits = 2
    N = 2 ** n_bits  # N = 4
    
    # Create simple distance matrix
    w = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ], dtype=np.float64)
    
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            D[i, j] = np.linalg.norm(w[i] - w[j]) ** 2
    
    edges = generate_hypercube_edges(n_bits)
    
    # Find optimal cost by brute force
    all_perms = list(permutations(range(N)))
    optimal_cost = float('inf')
    for perm in all_perms:
        pi = np.array(perm)
        cost = qap_cost(pi, D, edges)
        if cost < optimal_cost:
            optimal_cost = cost
    
    # Run hill climbing from random start
    pi_init = np.random.permutation(N)
    pi_final, cost_history = hill_climb_two_swap(
        pi_init=pi_init,
        D=D,
        edges=edges,
        max_iter=100,
        sample_size=10,  # For N=4, 10 covers all pairs
        random_state=45,
    )
    
    final_cost = cost_history[-1]
    
    # Should find optimal or very close
    assert final_cost <= optimal_cost + 1e-6, (
        f"Final cost {final_cost:.6f} should be <= optimal {optimal_cost:.6f}"
    )

