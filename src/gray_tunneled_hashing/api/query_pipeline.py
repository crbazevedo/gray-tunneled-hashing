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
    permutation: np.ndarray,  # Shape (K, n_bits) - NEW
    code_to_bucket: Dict[Tuple, int],
    bucket_to_code: np.ndarray,  # Shape (K, n_bits)
    n_bits: int,
    hamming_radius: int = 1,
    max_candidates: Optional[int] = None,
    enable_logging: bool = False,
) -> QueryResult:
    """
    Query with Hamming ball expansion after GTH permutation.
    
    Pipeline CORRIGIDO (Sprint 8):
    1. Map query_code to bucket_idx
    2. Apply permutation: query_code_permuted = permutation[bucket_idx]
    3. Expand Hamming ball around query_code_permuted
    4. For each code in ball, find which buckets have that code (inverse mapping)
    5. Return bucket indices
    
    Args:
        query_code: Query code of shape (n_bits,) with dtype bool (before GTH)
        permutation: GTH permutation array of shape (K, n_bits) where K = number of buckets
                    permutation[bucket_idx] = novo_código_binário
        code_to_bucket: Dictionary mapping code tuples to bucket indices
        bucket_to_code: Array of bucket codes of shape (K, n_bits) - original codes (not used in new pipeline)
        n_bits: Number of bits
        hamming_radius: Hamming radius for expansion (default: 1)
        max_candidates: Optional maximum number of candidates to return
        enable_logging: If True, print detailed logging
        
    Returns:
        QueryResult with candidate codes and indices
    """
    from collections import defaultdict
    
    # Step 1: Map query_code to bucket
    query_code_tuple = tuple(query_code.astype(int).tolist())
    if query_code_tuple not in code_to_bucket:
        # Query code not in any bucket - return empty result
        if enable_logging:
            print(f"  ⚠️  Query code not in any bucket: {query_code_tuple}")
        return QueryResult(
            query_code=query_code,
            permuted_code=np.zeros(n_bits, dtype=bool),
            candidate_codes=np.empty((0, n_bits), dtype=bool),
            candidate_indices=np.empty(0, dtype=np.int32),
            hamming_radius=hamming_radius,
            n_candidates=0,
        )
    
    query_bucket_idx = code_to_bucket[query_code_tuple]
    
    # Step 2: Apply permutation to get new code for query bucket
    query_code_permuted = permutation[query_bucket_idx]  # Shape (n_bits,)
    
    if enable_logging:
        print(f"  Query bucket: {query_bucket_idx}")
        print(f"  Query code (original): {query_code.astype(int).tolist()}")
        print(f"  Query code (permuted): {query_code_permuted.astype(int).tolist()}")
    
    # Step 3: Expand Hamming ball around permuted code
    candidate_codes = expand_hamming_ball(
        query_code_permuted.astype(bool),
        radius=hamming_radius,
        n_bits=n_bits,
        max_codes=max_candidates,
    )
    
    if enable_logging:
        print(f"  Hamming ball size: {len(candidate_codes)}")
    
    # Step 4: For each candidate code, find which buckets have that code
    # Need inverse mapping: code -> list of bucket indices
    # Build this from permutation: for each bucket, its permuted code
    code_to_buckets = defaultdict(list)
    for bucket_idx in range(len(permutation)):
        bucket_code = permutation[bucket_idx]
        code_tuple = tuple(bucket_code.astype(int).tolist())
        code_to_buckets[code_tuple].append(bucket_idx)
    
    candidate_buckets = set()
    for candidate_code in candidate_codes:
        candidate_tuple = tuple(candidate_code.astype(int).tolist())
        if candidate_tuple in code_to_buckets:
            candidate_buckets.update(code_to_buckets[candidate_tuple])
    
    if enable_logging:
        print(f"  Valid buckets: {len(candidate_buckets)}")
        print(f"  Candidate buckets: {sorted(candidate_buckets)[:10]}...")
    
    return QueryResult(
        query_code=query_code,
        permuted_code=query_code_permuted.astype(bool),
        candidate_codes=candidate_codes,
        candidate_indices=np.array(sorted(candidate_buckets), dtype=np.int32),
        hamming_radius=hamming_radius,
        n_candidates=len(candidate_buckets),
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

