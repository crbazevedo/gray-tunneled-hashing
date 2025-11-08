"""Tests for query-time pipeline with Hamming ball expansion."""

import numpy as np
import pytest

from gray_tunneled_hashing.api.query_pipeline import (
    expand_hamming_ball,
    query_with_hamming_ball,
    get_candidate_set,
    batch_query_with_hamming_ball,
    analyze_hamming_ball_coverage,
)
from gray_tunneled_hashing.data.synthetic_generators import generate_hypercube_vertices


def test_expand_hamming_ball_radius_0():
    """Test Hamming ball expansion with radius 0 (exact match)."""
    n_bits = 4
    center_code = np.array([True, False, True, False], dtype=bool)
    
    candidates = expand_hamming_ball(center_code, radius=0, n_bits=n_bits)
    
    assert candidates.shape == (1, n_bits)
    assert np.array_equal(candidates[0], center_code)


def test_expand_hamming_ball_radius_1():
    """Test Hamming ball expansion with radius 1."""
    n_bits = 4
    center_code = np.array([True, False, True, False], dtype=bool)
    
    candidates = expand_hamming_ball(center_code, radius=1, n_bits=n_bits)
    
    # Should include center + all Hamming-1 neighbors
    # For n_bits=4, radius=1 should give 1 + 4 = 5 codes
    assert len(candidates) == 5
    assert candidates.shape[1] == n_bits
    
    # Center should be included
    assert any(np.array_equal(c, center_code) for c in candidates)
    
    # All candidates should be within distance 1
    vertices = generate_hypercube_vertices(n_bits)
    for candidate in candidates:
        distances = np.sum(candidate != vertices, axis=1)
        assert np.min(distances) <= 1


def test_expand_hamming_ball_max_codes():
    """Test Hamming ball expansion with max_codes limit."""
    n_bits = 6
    center_code = np.array([False] * n_bits, dtype=bool)
    
    # Without limit
    candidates_all = expand_hamming_ball(center_code, radius=2, n_bits=n_bits)
    
    # With limit
    max_codes = 10
    candidates_limited = expand_hamming_ball(
        center_code, radius=2, n_bits=n_bits, max_codes=max_codes
    )
    
    assert len(candidates_limited) <= max_codes
    assert len(candidates_limited) <= len(candidates_all)
    
    # Limited should be subset of all (closest codes)
    for code in candidates_limited:
        assert any(np.array_equal(code, c) for c in candidates_all)


def test_query_with_hamming_ball():
    """Test query with Hamming ball expansion."""
    n_bits = 4
    N = 2 ** n_bits
    
    # Create identity permutation (no change)
    permutation = np.arange(N, dtype=np.int32)
    
    # Create code_to_bucket mapping
    vertices = generate_hypercube_vertices(n_bits)
    code_to_bucket = {}
    for i, code in enumerate(vertices):
        code_tuple = tuple(code.astype(int).tolist())
        code_to_bucket[code_tuple] = i
    
    bucket_to_code = vertices
    
    # Query code
    query_code = vertices[0].astype(bool)
    
    result = query_with_hamming_ball(
        query_code=query_code,
        permutation=permutation,
        code_to_bucket=code_to_bucket,
        bucket_to_code=bucket_to_code,
        n_bits=n_bits,
        hamming_radius=1,
    )
    
    assert result.n_candidates > 0
    assert result.hamming_radius == 1
    assert result.candidate_codes.shape[1] == n_bits
    assert len(result.candidate_indices) == result.n_candidates


def test_get_candidate_set():
    """Test convenience function for getting candidate set."""
    n_bits = 4
    N = 2 ** n_bits
    
    permutation = np.arange(N, dtype=np.int32)
    vertices = generate_hypercube_vertices(n_bits)
    
    code_to_bucket = {}
    for i, code in enumerate(vertices):
        code_tuple = tuple(code.astype(int).tolist())
        code_to_bucket[code_tuple] = i
    
    query_code = vertices[0].astype(bool)
    
    indices, codes = get_candidate_set(
        query_code=query_code,
        permutation=permutation,
        code_to_bucket=code_to_bucket,
        bucket_to_code=vertices,
        n_bits=n_bits,
        hamming_radius=1,
    )
    
    assert len(indices) == len(codes)
    assert codes.shape[1] == n_bits


def test_batch_query_with_hamming_ball():
    """Test batch query processing."""
    n_bits = 4
    N = 2 ** n_bits
    
    permutation = np.arange(N, dtype=np.int32)
    vertices = generate_hypercube_vertices(n_bits)
    
    code_to_bucket = {}
    for i, code in enumerate(vertices):
        code_tuple = tuple(code.astype(int).tolist())
        code_to_bucket[code_tuple] = i
    
    # Multiple query codes
    query_codes = vertices[:3].astype(bool)
    
    results = batch_query_with_hamming_ball(
        query_codes=query_codes,
        permutation=permutation,
        code_to_bucket=code_to_bucket,
        bucket_to_code=vertices,
        n_bits=n_bits,
        hamming_radius=1,
    )
    
    assert len(results) == len(query_codes)
    for result in results:
        assert result.n_candidates > 0


def test_analyze_hamming_ball_coverage():
    """Test Hamming ball coverage analysis."""
    n_bits = 4
    N = 2 ** n_bits
    
    permutation = np.arange(N, dtype=np.int32)
    query_code = np.array([False] * n_bits, dtype=bool)
    
    coverage = analyze_hamming_ball_coverage(
        query_code=query_code,
        permutation=permutation,
        n_bits=n_bits,
        max_radius=3,
    )
    
    # Coverage should increase with radius
    assert coverage[0] < coverage[1] < coverage[2] < coverage[3]
    
    # Radius 0 should have exactly 1 code
    assert coverage[0] == 1
    
    # For n_bits=4, radius=1 should have 1 + 4 = 5 codes
    assert coverage[1] == 5


def test_expand_hamming_ball_validation():
    """Test input validation for expand_hamming_ball."""
    n_bits = 4
    center_code = np.array([True, False, True, False], dtype=bool)
    
    # Wrong shape
    wrong_code = np.array([True, False], dtype=bool)
    with pytest.raises(ValueError, match="shape"):
        expand_hamming_ball(wrong_code, radius=1, n_bits=n_bits)
    
    # Negative radius
    with pytest.raises(ValueError, match="radius"):
        expand_hamming_ball(center_code, radius=-1, n_bits=n_bits)

