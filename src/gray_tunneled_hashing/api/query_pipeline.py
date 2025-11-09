"""
Query-time pipeline with Hamming ball expansion.

This module implements the query-time pipeline as described in the theoretical paper:
1. Query → embedding
2. LSH: c_q = h(x_q)
3. GTH: c̃_q = σ(c_q)
4. Hamming ball: C_q(r) = {z : d_H(z, c̃_q) ≤ r}
5. Candidate set extraction

The Hamming ball expansion allows searching over multiple buckets efficiently
by exploiting the improved geometry from GTH.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Set
from dataclasses import dataclass

from gray_tunneled_hashing.data.synthetic_generators import generate_hypercube_vertices


def hamming_distance(codes1: np.ndarray, codes2: np.ndarray) -> np.ndarray:
    """
    Compute Hamming distance between two sets of binary codes.
    
    Args:
        codes1: Binary codes of shape (N1, n_bits)
        codes2: Binary codes of shape (N2, n_bits)
        
    Returns:
        Distance matrix of shape (N1, N2) with Hamming distances
    """
    if codes1.ndim != 2 or codes2.ndim != 2:
        raise ValueError("Both inputs must be 2D arrays")
    if codes1.shape[1] != codes2.shape[1]:
        raise ValueError("Codes must have same number of bits")
    
    # Convert to uint8 if bool
    if codes1.dtype == bool:
        codes1 = codes1.astype(np.uint8)
    if codes2.dtype == bool:
        codes2 = codes2.astype(np.uint8)
    
    # Compute XOR and count non-zero bits
    # codes1: (N1, n_bits), codes2: (N2, n_bits)
    # We want (N1, N2) output
    distances = np.zeros((codes1.shape[0], codes2.shape[0]), dtype=np.int32)
    
    for i in range(codes1.shape[0]):
        xor_result = np.bitwise_xor(codes1[i], codes2)
        distances[i] = np.count_nonzero(xor_result, axis=1)
    
    return distances


@dataclass
class QueryResult:
    """Result of a query with Hamming ball expansion."""
    query_code: np.ndarray  # Original query code (before GTH)
    permuted_code: np.ndarray  # Query code after GTH permutation
    candidate_codes: np.ndarray  # All codes in Hamming ball
    candidate_indices: np.ndarray  # Indices of candidates in dataset
    hamming_radius: int  # Radius used
    n_candidates: int  # Number of candidates found


def expand_hamming_ball(
    center_code: np.ndarray,
    radius: int,
    n_bits: int,
    max_codes: Optional[int] = None,
) -> np.ndarray:
    """
    Expand Hamming ball around a center code.
    
    Returns all codes within Hamming distance <= radius from center_code.
    
    Args:
        center_code: Center code of shape (n_bits,) with dtype bool
        radius: Hamming radius (0 = exact match, 1 = Hamming-1 neighbors, etc.)
        n_bits: Number of bits
        max_codes: Optional maximum number of codes to return (for efficiency)
                   If None, returns all codes in ball
        
    Returns:
        Array of codes of shape (M, n_bits) with dtype bool,
        where M is the number of codes in the ball (or max_codes if limited)
    """
    if center_code.shape != (n_bits,):
        raise ValueError(
            f"center_code must have shape ({n_bits},), got {center_code.shape}"
        )
    
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    
    if radius == 0:
        # Only exact match
        return center_code[np.newaxis, :].copy()
    
    # Generate all vertices of hypercube
    all_vertices = generate_hypercube_vertices(n_bits)
    
    # Convert center_code to same format for distance computation
    center_bool = center_code.astype(bool)
    
    # Compute Hamming distances from center to all vertices
    # center_bool: (n_bits,), all_vertices: (2**n_bits, n_bits)
    distances = hamming_distance(
        center_bool[np.newaxis, :],
        all_vertices
    )[0, :]  # Shape: (2**n_bits,)
    
    # Filter codes within radius
    mask = distances <= radius
    candidates = all_vertices[mask]
    
    # Limit if requested
    if max_codes is not None and len(candidates) > max_codes:
        # Sort by distance and take closest
        candidate_distances = distances[mask]
        sorted_indices = np.argsort(candidate_distances)[:max_codes]
        candidates = candidates[sorted_indices]
    
    return candidates


def query_with_hamming_ball(
    query_code: np.ndarray,
    permutation: np.ndarray,
    code_to_bucket: Dict[Tuple, int],
    bucket_to_code: np.ndarray,
    n_bits: int,
    hamming_radius: int = 1,
    max_candidates: Optional[int] = None,
    enable_logging: bool = False,
    bucket_to_embedding_idx: Optional[np.ndarray] = None,
) -> QueryResult:
    """
    Query with Hamming ball expansion after GTH permutation.
    
    Pipeline:
    1. Convert query_code to vertex index
    2. Expand Hamming ball around query_code in vertex space
    3. For each vertex in ball, apply permutation to get embedding_idx
    4. Map embedding_idx to bucket_idx using bucket_to_embedding_idx
    5. Return unique bucket indices
    
    Args:
        query_code: Query code of shape (n_bits,) with dtype bool (before GTH)
        permutation: GTH permutation array of shape (N,) where N = 2**n_bits
                    permutation[vertex_idx] = embedding_idx (0..N-1)
        code_to_bucket: Dictionary mapping code tuples to bucket indices
        bucket_to_code: Array of bucket codes of shape (K, n_bits)
        n_bits: Number of bits
        hamming_radius: Hamming radius for expansion (default: 1)
        max_candidates: Optional maximum number of candidates to return
        enable_logging: If True, print detailed logging
        bucket_to_embedding_idx: Optional mapping from bucket_idx to embedding_idx
                                If None, assumes bucket_idx == embedding_idx for first K buckets
        
    Returns:
        QueryResult with candidate codes and indices
    """
    # Step 1: Convert query_code to vertex index
    query_code_int = 0
    for i, bit in enumerate(query_code):
        if bit:
            query_code_int += 2 ** i
    
    if query_code_int >= len(permutation):
        raise ValueError(
            f"Query code maps to vertex {query_code_int}, "
            f"but permutation has length {len(permutation)}"
        )
    
    # Step 2: Expand Hamming ball around query_code in vertex space
    candidate_vertex_codes = expand_hamming_ball(
        query_code,
        radius=hamming_radius,
        n_bits=n_bits,
        max_codes=max_candidates,
    )
    
    # Step 3: For each vertex in ball, apply permutation to get bucket_idx
    # permutation[vertex_idx] = bucket_idx
    vertices = generate_hypercube_vertices(n_bits)
    candidate_buckets = set()
    valid_codes = []
    invalid_buckets = []
    buckets_not_in_code_to_bucket = []
    
    # Get valid bucket range from code_to_bucket
    K = len(bucket_to_code)
    valid_bucket_set = set(code_to_bucket.values())
    
    # Set up bucket_to_embedding_idx mapping
    if bucket_to_embedding_idx is None:
        # Default: bucket i maps to embedding i (for first K buckets)
        bucket_to_embedding_idx = np.arange(K, dtype=np.int32)
    
    for vertex_code in candidate_vertex_codes:
        # Convert vertex code to vertex index
        vertex_idx = 0
        for i, bit in enumerate(vertex_code):
            if bit:
                vertex_idx += 2 ** i
        
        if vertex_idx < len(permutation):
            # CRITICAL FIX: permutation[vertex_idx] = embedding_idx, not bucket_idx directly
            embedding_idx = permutation[vertex_idx]
            
            # Map embedding_idx to bucket_idx
            # Find which bucket this embedding corresponds to
            bucket_indices = np.where(bucket_to_embedding_idx == embedding_idx)[0]
            
            if len(bucket_indices) > 0:
                bucket_idx = bucket_indices[0]  # Take first match
                
                # Validate bucket_idx
                if bucket_idx >= K:
                    invalid_buckets.append(bucket_idx)
                    if enable_logging:
                        print(f"  ⚠️  Invalid bucket index: {bucket_idx} >= K={K} (from embedding_idx={embedding_idx})")
                elif bucket_idx not in valid_bucket_set:
                    buckets_not_in_code_to_bucket.append(bucket_idx)
                    if enable_logging:
                        print(f"  ⚠️  Bucket {bucket_idx} not in code_to_bucket (from embedding_idx={embedding_idx})")
                else:
                    # Valid bucket
                    candidate_buckets.add(bucket_idx)
                    valid_codes.append(vertex_code)
            else:
                # embedding_idx doesn't correspond to any bucket (embedding_idx >= K or not mapped)
                invalid_buckets.append(embedding_idx)  # Store embedding_idx for logging
                if enable_logging:
                    print(f"  ⚠️  Embedding index {embedding_idx} doesn't map to any bucket (K={K})")
    
    if enable_logging:
        print(f"  Query vertex: {query_code_int}")
        print(f"  Hamming ball size: {len(candidate_vertex_codes)}")
        print(f"  Valid buckets: {len(candidate_buckets)}")
        print(f"  Invalid buckets (>=K): {len(invalid_buckets)}")
        print(f"  Buckets not in code_to_bucket: {len(buckets_not_in_code_to_bucket)}")
    
    # Step 4: Convert bucket indices to codes for permuted_code representation
    # Use the first valid code as permuted_code (the query code itself after expansion)
    permuted_code = query_code.copy() if len(valid_codes) > 0 else np.zeros(n_bits, dtype=bool)
    
    # Convert bucket indices to array
    candidate_indices_array = np.array(sorted(candidate_buckets), dtype=np.int32)
    candidate_codes_array = np.array(valid_codes, dtype=bool) if len(valid_codes) > 0 else np.empty((0, n_bits), dtype=bool)
    
    return QueryResult(
        query_code=query_code,
        permuted_code=permuted_code,
        candidate_codes=candidate_codes_array,
        candidate_indices=candidate_indices_array,
        hamming_radius=hamming_radius,
        n_candidates=len(candidate_indices_array),
    )


def get_candidate_set(
    query_code: np.ndarray,
    permutation: np.ndarray,
    code_to_bucket: Dict[Tuple, int],
    bucket_to_code: np.ndarray,
    n_bits: int,
    hamming_radius: int = 1,
    max_candidates: Optional[int] = None,
    bucket_to_embedding_idx: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get candidate set from Hamming ball expansion.
    
    Convenience function that returns just the candidate indices and codes.
    
    Args:
        query_code: Query code of shape (n_bits,) with dtype bool
        permutation: GTH permutation array
        code_to_bucket: Dictionary mapping codes to bucket indices
        bucket_to_code: Array of bucket codes
        n_bits: Number of bits
        hamming_radius: Hamming radius for expansion
        max_candidates: Optional maximum number of candidates
        
    Returns:
        Tuple of (candidate_indices, candidate_codes)
    """
    result = query_with_hamming_ball(
        query_code=query_code,
        permutation=permutation,
        code_to_bucket=code_to_bucket,
        bucket_to_code=bucket_to_code,
        n_bits=n_bits,
        hamming_radius=hamming_radius,
        max_candidates=max_candidates,
        bucket_to_embedding_idx=bucket_to_embedding_idx,
    )
    
    return result.candidate_indices, result.candidate_codes


def batch_query_with_hamming_ball(
    query_codes: np.ndarray,
    permutation: np.ndarray,
    code_to_bucket: Dict[Tuple, int],
    bucket_to_code: np.ndarray,
    n_bits: int,
    hamming_radius: int = 1,
    max_candidates: Optional[int] = None,
    bucket_to_embedding_idx: Optional[np.ndarray] = None,
) -> list[QueryResult]:
    """
    Process multiple queries with Hamming ball expansion.
    
    Args:
        query_codes: Query codes of shape (Q, n_bits) with dtype bool
        permutation: GTH permutation array
        code_to_bucket: Dictionary mapping codes to bucket indices
        bucket_to_code: Array of bucket codes
        n_bits: Number of bits
        hamming_radius: Hamming radius for expansion
        max_candidates: Optional maximum number of candidates per query
        
    Returns:
        List of QueryResult objects, one per query
    """
    results = []
    for query_code in query_codes:
        result = query_with_hamming_ball(
            query_code=query_code,
            permutation=permutation,
            code_to_bucket=code_to_bucket,
            bucket_to_code=bucket_to_code,
            n_bits=n_bits,
            hamming_radius=hamming_radius,
            max_candidates=max_candidates,
            bucket_to_embedding_idx=bucket_to_embedding_idx,
        )
        results.append(result)
    
    return results


def analyze_hamming_ball_coverage(
    query_code: np.ndarray,
    permutation: np.ndarray,
    n_bits: int,
    max_radius: int = 5,
) -> Dict[int, int]:
    """
    Analyze how many codes are covered by Hamming balls of different radii.
    
    Useful for understanding the trade-off between recall and candidate set size.
    
    Args:
        query_code: Query code of shape (n_bits,)
        permutation: GTH permutation array
        n_bits: Number of bits
        max_radius: Maximum radius to analyze
        
    Returns:
        Dictionary mapping radius to number of codes in ball
    """
    coverage = {}
    
    for radius in range(max_radius + 1):
        candidates = expand_hamming_ball(
            query_code,
            radius=radius,
            n_bits=n_bits,
        )
        coverage[radius] = len(candidates)
    
    return coverage

