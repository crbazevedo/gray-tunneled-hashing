"""
Tests for J(φ) calculation consistency and monotonicity.
"""

import pytest
import numpy as np

from gray_tunneled_hashing.distribution.j_phi_objective import (
    compute_j_phi_cost,
    compute_j_phi_0,
    hill_climb_j_phi,
)
from gray_tunneled_hashing.data.synthetic_generators import generate_hypercube_vertices
from gray_tunneled_hashing.evaluation.metrics import hamming_distance


def test_j_phi_0_consistency():
    """Test that J(φ₀) computed via permutation equals direct calculation."""
    n_bits = 6
    K = 32
    dim = 16
    
    np.random.seed(42)
    
    # Generate synthetic data
    vertices = generate_hypercube_vertices(n_bits)
    N = len(vertices)
    
    # Use first K vertices as bucket codes
    bucket_to_code = vertices[:K]
    
    # Generate random pi and w
    pi = np.random.rand(K).astype(np.float64)
    pi = pi / pi.sum()
    
    w = np.random.rand(K, K).astype(np.float64)
    w = w / w.sum(axis=1, keepdims=True)
    
    # Identity permutation
    identity_perm = np.arange(N, dtype=np.int32)
    
    # Method 1: Direct calculation
    j_phi_0_direct = compute_j_phi_0(
        pi=pi,
        w=w,
        bucket_to_code=bucket_to_code,
        n_bits=n_bits,
        initial_permutation=None,
    )
    
    # Method 2: Via identity permutation
    j_phi_0_via_perm = compute_j_phi_0(
        pi=pi,
        w=w,
        bucket_to_code=bucket_to_code,
        n_bits=n_bits,
        initial_permutation=identity_perm,
    )
    
    # They should be equal (within numerical tolerance)
    assert abs(j_phi_0_direct - j_phi_0_via_perm) < 1e-6, \
        f"J(φ₀) mismatch: direct={j_phi_0_direct:.6f}, via_perm={j_phi_0_via_perm:.6f}"


def test_j_phi_monotonicity():
    """Test that hill_climb_j_phi guarantees cost never increases."""
    n_bits = 6
    K = 32
    dim = 16
    
    np.random.seed(42)
    
    # Generate synthetic data
    vertices = generate_hypercube_vertices(n_bits)
    N = len(vertices)
    
    # Use first K vertices as bucket codes
    bucket_to_code = vertices[:K]
    
    # Generate random pi and w
    pi = np.random.rand(K).astype(np.float64)
    pi = pi / pi.sum()
    
    w = np.random.rand(K, K).astype(np.float64)
    w = w / w.sum(axis=1, keepdims=True)
    
    # Identity permutation
    identity_perm = np.arange(N, dtype=np.int32)
    
    # Run hill climb
    pi_optimized, final_cost, initial_cost, cost_history = hill_climb_j_phi(
        pi_init=identity_perm,
        pi=pi,
        w=w,
        bucket_to_code=bucket_to_code,
        n_bits=n_bits,
        max_iter=20,
        sample_size=64,
        random_state=42,
    )
    
    # Validate monotonicity
    assert initial_cost == cost_history[0], \
        f"Initial cost mismatch: {initial_cost:.6f} != {cost_history[0]:.6f}"
    
    assert final_cost <= initial_cost + 1e-10, \
        f"Monotonicity violated: final={final_cost:.6f} > initial={initial_cost:.6f}"
    
    # Check that cost_history is non-increasing
    for i in range(1, len(cost_history)):
        assert cost_history[i] <= cost_history[i-1] + 1e-10, \
            f"Cost increased at iteration {i}: {cost_history[i]:.6f} > {cost_history[i-1]:.6f}"


def test_j_phi_0_equals_initial():
    """Test that J(φ₀) equals the cost of the initial permutation."""
    n_bits = 6
    K = 32
    
    np.random.seed(42)
    
    vertices = generate_hypercube_vertices(n_bits)
    N = len(vertices)
    bucket_to_code = vertices[:K]
    
    pi = np.random.rand(K).astype(np.float64)
    pi = pi / pi.sum()
    
    w = np.random.rand(K, K).astype(np.float64)
    w = w / w.sum(axis=1, keepdims=True)
    
    # Identity permutation
    identity_perm = np.arange(N, dtype=np.int32)
    
    # Compute J(φ₀) via initial permutation
    j_phi_0 = compute_j_phi_0(
        pi=pi,
        w=w,
        bucket_to_code=bucket_to_code,
        n_bits=n_bits,
        initial_permutation=identity_perm,
    )
    
    # Compute cost of initial permutation directly
    initial_cost = compute_j_phi_cost(
        permutation=identity_perm,
        pi=pi,
        w=w,
        bucket_to_code=bucket_to_code,
        n_bits=n_bits,
        bucket_to_embedding_idx=None,
    )
    
    assert abs(j_phi_0 - initial_cost) < 1e-6, \
        f"J(φ₀) mismatch: {j_phi_0:.6f} != {initial_cost:.6f}"


def test_identity_permutation_mapping():
    """Test that identity permutation correctly maps buckets to vertices."""
    n_bits = 6
    K = 32
    
    np.random.seed(42)
    
    vertices = generate_hypercube_vertices(n_bits)
    N = len(vertices)
    bucket_to_code = vertices[:K]
    
    pi = np.random.rand(K).astype(np.float64)
    pi = pi / pi.sum()
    
    w = np.random.rand(K, K).astype(np.float64)
    w = w / w.sum(axis=1, keepdims=True)
    
    # Identity permutation: bucket i should map to vertex i
    identity_perm = np.arange(N, dtype=np.int32)
    
    # Compute cost via identity permutation
    cost_via_perm = compute_j_phi_cost(
        permutation=identity_perm,
        pi=pi,
        w=w,
        bucket_to_code=bucket_to_code,
        n_bits=n_bits,
        bucket_to_embedding_idx=None,
    )
    
    # For identity permutation with bucket_to_code = vertices[:K],
    # bucket i maps to vertex i, so code should be vertices[i]
    # This should equal direct calculation
    cost_direct = 0.0
    for i in range(K):
        code_i = vertices[i]  # bucket i is at vertex i
        for j in range(K):
            code_j = vertices[j]  # bucket j is at vertex j
            d_h = hamming_distance(code_i[np.newaxis, :], code_j[np.newaxis, :])[0, 0]
            cost_direct += pi[i] * w[i, j] * d_h
    
    assert abs(cost_via_perm - cost_direct) < 1e-6, \
        f"Identity mapping mismatch: via_perm={cost_via_perm:.6f}, direct={cost_direct:.6f}"

