"""
Direct optimization of J(φ) objective for distribution-aware GTH.

J(φ) = Σ_{i,j} π_i · w_ij · d_H(φ(c_i), φ(c_j))

This is different from QAP cost which only sums over hypercube edges.
We need to optimize J(φ) directly.

NEW (Sprint 8): J(φ) computed over real embeddings:
J(φ) = Σ_{i,j} π_i · w_ij · E[d_H(φ(h(q)), φ(h(x))) | q∈bucket_i, x∈bucket_j]

NEW (Sprint 9): Multi-radius objective:
J(φ) = Σ_r w_r · J_r(φ)
where J_r(φ) considers only pairs with d_H ≤ r
"""

import numpy as np
from typing import Optional, Tuple, Callable, Dict, List
from collections import defaultdict
import time

from gray_tunneled_hashing.data.synthetic_generators import generate_hypercube_vertices
from gray_tunneled_hashing.evaluation.metrics import hamming_distance


def validate_radius_weights(
    radii: List[int],
    weights: Optional[np.ndarray] = None,
    normalize: bool = False,
) -> np.ndarray:
    """
    Validate and optionally generate radius weights with constraints w_1 > w_2 > ... > 0.
    
    Args:
        radii: List of Hamming radii, e.g., [1, 2, 3]
        weights: Optional weights array of shape (len(radii),). If None, generates default weights.
        normalize: If True, normalize weights to sum to 1.0
        
    Returns:
        Validated weights array of shape (len(radii),)
        
    Raises:
        ValueError: If constraints are violated
    """
    if len(radii) == 0:
        raise ValueError("radii must be non-empty")
    
    # Ensure radii are sorted and positive
    sorted_radii = sorted(radii)
    if sorted_radii != radii:
        raise ValueError(f"radii must be sorted in ascending order, got {radii}")
    if any(r <= 0 for r in radii):
        raise ValueError(f"All radii must be positive, got {radii}")
    
    if weights is None:
        # Generate default weights: exponential decay w_r = 0.5^(r-1)
        weights = np.array([0.5 ** (r - 1) for r in radii], dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (len(radii),):
            raise ValueError(
                f"weights must have shape ({len(radii)},), got {weights.shape}"
            )
    
    # Validate constraints: w_1 > w_2 > ... > 0
    for i in range(len(radii) - 1):
        if weights[i] <= weights[i + 1]:
            raise ValueError(
                f"Weight constraint violated: w_{radii[i]}={weights[i]:.6f} <= "
                f"w_{radii[i+1]}={weights[i+1]:.6f}. Must have w_1 > w_2 > ... > 0"
            )
    
    if np.any(weights <= 0):
        raise ValueError("All weights must be positive")
    
    # Normalize if requested
    if normalize:
        weights = weights / np.sum(weights)
    
    return weights


def compute_j_phi_cost_multi_radius(
    permutation: np.ndarray,  # Shape (K, n_bits)
    pi: np.ndarray,  # Shape (K,)
    w: np.ndarray,  # Shape (K, K)
    queries: np.ndarray,  # Shape (Q, dim)
    base_embeddings: np.ndarray,  # Shape (N, dim)
    ground_truth_neighbors: np.ndarray,  # Shape (Q, k)
    encoder: Callable,  # LSH encoder
    code_to_bucket: Dict[Tuple, int],
    n_bits: int,
    radii: List[int],
    radius_weights: np.ndarray,
    sample_size: Optional[int] = None,
) -> float:
    """
    Compute multi-radius J(φ) objective.
    
    J(φ) = Σ_r w_r · J_r(φ)
    where J_r(φ) considers only pairs with d_H(φ(h(q)), φ(h(x))) ≤ r
    
    Args:
        permutation: Permutation array of shape (K, n_bits)
        pi: Query prior of shape (K,)
        w: Neighbor weights of shape (K, K)
        queries: Query embeddings of shape (Q, dim)
        base_embeddings: Base corpus embeddings of shape (N, dim)
        ground_truth_neighbors: Ground truth neighbor indices of shape (Q, k)
        encoder: LSH encoder function
        code_to_bucket: Dictionary mapping code tuples to bucket indices
        n_bits: Number of bits
        radii: List of Hamming radii, e.g., [1, 2, 3]
        radius_weights: Weights for each radius, shape (len(radii),)
        sample_size: Optional maximum number of pairs to sample per bucket pair
        
    Returns:
        Multi-radius J(φ) cost
    """
    if len(radii) != len(radius_weights):
        raise ValueError(
            f"radii and radius_weights must have same length, "
            f"got {len(radii)} and {len(radius_weights)}"
        )
    
    # Encode queries and base embeddings (cache these)
    query_codes = encoder(queries)  # Shape (Q, n_bits)
    base_codes = encoder(base_embeddings)  # Shape (N, n_bits)
    
    # Build query_bucket -> list of query indices
    query_bucket_to_indices = defaultdict(list)
    for q_idx, q_code in enumerate(query_codes):
        q_code_tuple = tuple(q_code.astype(int).tolist())
        if q_code_tuple in code_to_bucket:
            bucket_idx = code_to_bucket[q_code_tuple]
            query_bucket_to_indices[bucket_idx].append(q_idx)
    
    # Compute cost for each radius
    total_cost = 0.0
    
    for radius_idx, radius in enumerate(radii):
        radius_weight = radius_weights[radius_idx]
        radius_cost = 0.0
        
        # For each bucket pair (i, j), sample query-neighbor pairs
        for i in range(len(pi)):
            if i not in query_bucket_to_indices:
                continue
            
            queries_in_i = query_bucket_to_indices[i]
            
            for j in range(len(pi)):
                if w[i, j] == 0:
                    continue
                
                # Get neighbors for queries in bucket i that are in bucket j
                pairs = []
                for q_idx in queries_in_i:
                    for n_idx in ground_truth_neighbors[q_idx]:
                        if n_idx < len(base_embeddings):
                            n_code = base_codes[n_idx]
                            n_code_tuple = tuple(n_code.astype(int).tolist())
                            if n_code_tuple in code_to_bucket:
                                n_bucket = code_to_bucket[n_code_tuple]
                                if n_bucket == j:
                                    pairs.append((q_idx, n_idx))
                
                if len(pairs) == 0:
                    continue
                
                # Sample if needed
                if sample_size is not None and len(pairs) > sample_size:
                    sampled_indices = np.random.choice(len(pairs), size=sample_size, replace=False)
                    pairs = [pairs[k] for k in sampled_indices]
                
                # Compute average Hamming distance for pairs within radius
                hamming_dists = []
                for q_idx, n_idx in pairs:
                    q_code = query_codes[q_idx]
                    n_code = base_codes[n_idx]
                    
                    # Apply permutation: get new codes for buckets
                    q_bucket = code_to_bucket[tuple(q_code.astype(int).tolist())]
                    n_bucket = code_to_bucket[tuple(n_code.astype(int).tolist())]
                    
                    q_code_permuted = permutation[q_bucket]  # Shape (n_bits,)
                    n_code_permuted = permutation[n_bucket]  # Shape (n_bits,)
                    
                    d_h = hamming_distance(
                        q_code_permuted[np.newaxis, :],
                        n_code_permuted[np.newaxis, :]
                    )[0, 0]
                    
                    # Only include pairs within this radius
                    if d_h <= radius:
                        hamming_dists.append(d_h)
                
                # Average Hamming distance for pairs within radius
                if len(hamming_dists) > 0:
                    avg_hamming = np.mean(hamming_dists)
                    radius_cost += pi[i] * w[i, j] * avg_hamming
        
        # Weight by radius weight
        total_cost += radius_weight * radius_cost
    
    return float(total_cost)


def compute_j_phi_cost_real_embeddings(
    permutation: np.ndarray,  # Shape (K, n_bits)
    pi: np.ndarray,  # Shape (K,)
    w: np.ndarray,  # Shape (K, K)
    queries: np.ndarray,  # Shape (Q, dim)
    base_embeddings: np.ndarray,  # Shape (N, dim)
    ground_truth_neighbors: np.ndarray,  # Shape (Q, k)
    encoder: Callable,  # LSH encoder
    code_to_bucket: Dict[Tuple, int],
    n_bits: int,
    sample_size: Optional[int] = None,  # Se None, usa todos os pares
) -> float:
    """
    Compute J(φ) using real query-neighbor pairs.
    
    J(φ) = Σ_{i,j} π_i · w_ij · E[d_H(φ(h(q)), φ(h(x))) | q∈bucket_i, x∈bucket_j]
    
    Calcula a média de distâncias Hamming entre queries e neighbors reais,
    amostrados de cada par de buckets (i,j) conforme w_ij.
    
    Args:
        permutation: Permutation array of shape (K, n_bits) where permutation[bucket_idx] = novo_código_binário
        pi: Query prior of shape (K,)
        w: Neighbor weights of shape (K, K)
        queries: Query embeddings of shape (Q, dim)
        base_embeddings: Base corpus embeddings of shape (N, dim)
        ground_truth_neighbors: Ground truth neighbor indices of shape (Q, k)
        encoder: Function that maps embeddings to bucket codes (LSH encoder)
        code_to_bucket: Dictionary mapping code tuples to bucket indices
        n_bits: Number of bits
        sample_size: Optional maximum number of pairs to sample per bucket pair. If None, uses all pairs.
        
    Returns:
        J(φ) cost
    """
    K = len(pi)
    cost = 0.0
    
    # Encode queries and base embeddings
    query_codes = encoder(queries)  # Shape (Q, n_bits)
    base_codes = encoder(base_embeddings)  # Shape (N, n_bits)
    
    # Build query_bucket -> list of query indices
    query_bucket_to_indices = defaultdict(list)
    for q_idx, q_code in enumerate(query_codes):
        q_code_tuple = tuple(q_code.astype(int).tolist())
        if q_code_tuple in code_to_bucket:
            bucket_idx = code_to_bucket[q_code_tuple]
            query_bucket_to_indices[bucket_idx].append(q_idx)
    
    # For each bucket pair (i, j), sample query-neighbor pairs
    for i in range(K):
        if i not in query_bucket_to_indices:
            continue
        
        queries_in_i = query_bucket_to_indices[i]
        
        for j in range(K):
            if w[i, j] == 0:
                continue
            
            # Get neighbors for queries in bucket i that are in bucket j
            pairs = []
            for q_idx in queries_in_i:
                for n_idx in ground_truth_neighbors[q_idx]:
                    if n_idx < len(base_embeddings):
                        n_code = base_codes[n_idx]
                        n_code_tuple = tuple(n_code.astype(int).tolist())
                        if n_code_tuple in code_to_bucket:
                            n_bucket = code_to_bucket[n_code_tuple]
                            if n_bucket == j:
                                pairs.append((q_idx, n_idx))
            
            if len(pairs) == 0:
                continue
            
            # Sample if needed
            if sample_size is not None and len(pairs) > sample_size:
                sampled_indices = np.random.choice(len(pairs), size=sample_size, replace=False)
                pairs = [pairs[k] for k in sampled_indices]
            
            # Compute average Hamming distance for this bucket pair
            hamming_dists = []
            for q_idx, n_idx in pairs:
                q_code = query_codes[q_idx]
                n_code = base_codes[n_idx]
                
                # Apply permutation: get new codes for buckets
                q_bucket = code_to_bucket[tuple(q_code.astype(int).tolist())]
                n_bucket = code_to_bucket[tuple(n_code.astype(int).tolist())]
                
                q_code_permuted = permutation[q_bucket]  # Shape (n_bits,)
                n_code_permuted = permutation[n_bucket]  # Shape (n_bits,)
                
                d_h = hamming_distance(
                    q_code_permuted[np.newaxis, :],
                    n_code_permuted[np.newaxis, :]
                )[0, 0]
                hamming_dists.append(d_h)
            
            avg_hamming = np.mean(hamming_dists) if len(hamming_dists) > 0 else 0.0
            cost += pi[i] * w[i, j] * avg_hamming
    
    return float(cost)


def compute_j_phi_cost_delta_swap_buckets(
    permutation: np.ndarray,  # Shape (K, n_bits)
    pi: np.ndarray,
    w: np.ndarray,
    queries: np.ndarray,
    base_embeddings: np.ndarray,
    ground_truth_neighbors: np.ndarray,
    encoder: Callable,
    code_to_bucket: Dict[Tuple, int],
    n_bits: int,
    bucket_i: int,
    bucket_j: int,
    sample_size: Optional[int] = None,
) -> float:
    """
    Compute delta in J(φ) cost when swapping codes of buckets i and j.
    
    This is more efficient than recomputing the full cost by only computing
    the terms that change when swapping codes of two buckets.
    
    Args:
        permutation: Permutation array of shape (K, n_bits)
        pi: Query prior of shape (K,)
        w: Neighbor weights of shape (K, K)
        queries: Query embeddings of shape (Q, dim)
        base_embeddings: Base corpus embeddings of shape (N, dim)
        ground_truth_neighbors: Ground truth neighbor indices of shape (Q, k)
        encoder: LSH encoder function
        code_to_bucket: Dictionary mapping code tuples to bucket indices
        n_bits: Number of bits
        bucket_i: First bucket index to swap
        bucket_j: Second bucket index to swap
        sample_size: Optional maximum number of pairs to sample per bucket pair
        
    Returns:
        Delta = cost_after_swap - cost_before_swap
    """
    # For efficiency, we compute only the terms that change
    # Terms that change are those involving buckets i or j
    
    # Encode queries and base embeddings (cache these)
    query_codes = encoder(queries)  # Shape (Q, n_bits)
    base_codes = encoder(base_embeddings)  # Shape (N, n_bits)
    
    # Build query_bucket -> list of query indices
    query_bucket_to_indices = defaultdict(list)
    for q_idx, q_code in enumerate(query_codes):
        q_code_tuple = tuple(q_code.astype(int).tolist())
        if q_code_tuple in code_to_bucket:
            bucket_idx = code_to_bucket[q_code_tuple]
            query_bucket_to_indices[bucket_idx].append(q_idx)
    
    # Compute cost before swap (only terms involving i or j)
    cost_before = 0.0
    
    # Terms involving bucket i (with any bucket k)
    for k in range(len(pi)):
        if w[bucket_i, k] == 0 and w[k, bucket_i] == 0:
            continue
        
        # Get pairs for (i, k) or (k, i)
        pairs = []
        if bucket_i in query_bucket_to_indices and w[bucket_i, k] > 0:
            for q_idx in query_bucket_to_indices[bucket_i]:
                for n_idx in ground_truth_neighbors[q_idx]:
                    if n_idx < len(base_embeddings):
                        n_code = base_codes[n_idx]
                        n_code_tuple = tuple(n_code.astype(int).tolist())
                        if n_code_tuple in code_to_bucket:
                            n_bucket = code_to_bucket[n_code_tuple]
                            if n_bucket == k:
                                pairs.append((q_idx, n_idx))
        
        if k in query_bucket_to_indices and w[k, bucket_i] > 0:
            for q_idx in query_bucket_to_indices[k]:
                for n_idx in ground_truth_neighbors[q_idx]:
                    if n_idx < len(base_embeddings):
                        n_code = base_codes[n_idx]
                        n_code_tuple = tuple(n_code.astype(int).tolist())
                        if n_code_tuple in code_to_bucket:
                            n_bucket = code_to_bucket[n_code_tuple]
                            if n_bucket == bucket_i:
                                pairs.append((q_idx, n_idx))
        
        if len(pairs) == 0:
            continue
        
        # Sample if needed
        if sample_size is not None and len(pairs) > sample_size:
            sampled_indices = np.random.choice(len(pairs), size=sample_size, replace=False)
            pairs = [pairs[idx] for idx in sampled_indices]
        
        # Compute average Hamming distance
        hamming_dists = []
        for q_idx, n_idx in pairs:
            q_code = query_codes[q_idx]
            n_code = base_codes[n_idx]
            
            q_bucket = code_to_bucket[tuple(q_code.astype(int).tolist())]
            n_bucket = code_to_bucket[tuple(n_code.astype(int).tolist())]
            
            q_code_permuted = permutation[q_bucket]
            n_code_permuted = permutation[n_bucket]
            
            d_h = hamming_distance(
                q_code_permuted[np.newaxis, :],
                n_code_permuted[np.newaxis, :]
            )[0, 0]
            hamming_dists.append(d_h)
        
        avg_hamming = np.mean(hamming_dists) if len(hamming_dists) > 0 else 0.0
        
        # Add to cost (consider both directions)
        if w[bucket_i, k] > 0:
            cost_before += pi[bucket_i] * w[bucket_i, k] * avg_hamming
        if w[k, bucket_i] > 0 and k != bucket_i:
            cost_before += pi[k] * w[k, bucket_i] * avg_hamming
    
    # Terms involving bucket j (with any bucket k, but skip i since already counted)
    for k in range(len(pi)):
        if k == bucket_i:  # Already counted
            continue
        if w[bucket_j, k] == 0 and w[k, bucket_j] == 0:
            continue
        
        # Similar logic for (j, k) pairs
        pairs = []
        if bucket_j in query_bucket_to_indices and w[bucket_j, k] > 0:
            for q_idx in query_bucket_to_indices[bucket_j]:
                for n_idx in ground_truth_neighbors[q_idx]:
                    if n_idx < len(base_embeddings):
                        n_code = base_codes[n_idx]
                        n_code_tuple = tuple(n_code.astype(int).tolist())
                        if n_code_tuple in code_to_bucket:
                            n_bucket = code_to_bucket[n_code_tuple]
                            if n_bucket == k:
                                pairs.append((q_idx, n_idx))
        
        if k in query_bucket_to_indices and w[k, bucket_j] > 0:
            for q_idx in query_bucket_to_indices[k]:
                for n_idx in ground_truth_neighbors[q_idx]:
                    if n_idx < len(base_embeddings):
                        n_code = base_codes[n_idx]
                        n_code_tuple = tuple(n_code.astype(int).tolist())
                        if n_code_tuple in code_to_bucket:
                            n_bucket = code_to_bucket[n_code_tuple]
                            if n_bucket == bucket_j:
                                pairs.append((q_idx, n_idx))
        
        if len(pairs) == 0:
            continue
        
        # Sample if needed
        if sample_size is not None and len(pairs) > sample_size:
            sampled_indices = np.random.choice(len(pairs), size=sample_size, replace=False)
            pairs = [pairs[idx] for idx in sampled_indices]
        
        # Compute average Hamming distance
        hamming_dists = []
        for q_idx, n_idx in pairs:
            q_code = query_codes[q_idx]
            n_code = base_codes[n_idx]
            
            q_bucket = code_to_bucket[tuple(q_code.astype(int).tolist())]
            n_bucket = code_to_bucket[tuple(n_code.astype(int).tolist())]
            
            q_code_permuted = permutation[q_bucket]
            n_code_permuted = permutation[n_bucket]
            
            d_h = hamming_distance(
                q_code_permuted[np.newaxis, :],
                n_code_permuted[np.newaxis, :]
            )[0, 0]
            hamming_dists.append(d_h)
        
        avg_hamming = np.mean(hamming_dists) if len(hamming_dists) > 0 else 0.0
        
        # Add to cost
        if w[bucket_j, k] > 0:
            cost_before += pi[bucket_j] * w[bucket_j, k] * avg_hamming
        if w[k, bucket_j] > 0:
            cost_before += pi[k] * w[k, bucket_j] * avg_hamming
    
    # Swap codes
    permutation_swapped = permutation.copy()
    permutation_swapped[bucket_i], permutation_swapped[bucket_j] = \
        permutation_swapped[bucket_j].copy(), permutation_swapped[bucket_i].copy()
    
    # Compute cost after swap (same logic but with swapped permutation)
    cost_after = 0.0
    
    # Terms involving bucket i (with any bucket k)
    for k in range(len(pi)):
        if w[bucket_i, k] == 0 and w[k, bucket_i] == 0:
            continue
        
        pairs = []
        if bucket_i in query_bucket_to_indices and w[bucket_i, k] > 0:
            for q_idx in query_bucket_to_indices[bucket_i]:
                for n_idx in ground_truth_neighbors[q_idx]:
                    if n_idx < len(base_embeddings):
                        n_code = base_codes[n_idx]
                        n_code_tuple = tuple(n_code.astype(int).tolist())
                        if n_code_tuple in code_to_bucket:
                            n_bucket = code_to_bucket[n_code_tuple]
                            if n_bucket == k:
                                pairs.append((q_idx, n_idx))
        
        if k in query_bucket_to_indices and w[k, bucket_i] > 0:
            for q_idx in query_bucket_to_indices[k]:
                for n_idx in ground_truth_neighbors[q_idx]:
                    if n_idx < len(base_embeddings):
                        n_code = base_codes[n_idx]
                        n_code_tuple = tuple(n_code.astype(int).tolist())
                        if n_code_tuple in code_to_bucket:
                            n_bucket = code_to_bucket[n_code_tuple]
                            if n_bucket == bucket_i:
                                pairs.append((q_idx, n_idx))
        
        if len(pairs) == 0:
            continue
        
        if sample_size is not None and len(pairs) > sample_size:
            sampled_indices = np.random.choice(len(pairs), size=sample_size, replace=False)
            pairs = [pairs[idx] for idx in sampled_indices]
        
        hamming_dists = []
        for q_idx, n_idx in pairs:
            q_code = query_codes[q_idx]
            n_code = base_codes[n_idx]
            
            q_bucket = code_to_bucket[tuple(q_code.astype(int).tolist())]
            n_bucket = code_to_bucket[tuple(n_code.astype(int).tolist())]
            
            q_code_permuted = permutation_swapped[q_bucket]
            n_code_permuted = permutation_swapped[n_bucket]
            
            d_h = hamming_distance(
                q_code_permuted[np.newaxis, :],
                n_code_permuted[np.newaxis, :]
            )[0, 0]
            hamming_dists.append(d_h)
        
        avg_hamming = np.mean(hamming_dists) if len(hamming_dists) > 0 else 0.0
        
        if w[bucket_i, k] > 0:
            cost_after += pi[bucket_i] * w[bucket_i, k] * avg_hamming
        if w[k, bucket_i] > 0 and k != bucket_i:
            cost_after += pi[k] * w[k, bucket_i] * avg_hamming
    
    # Terms involving bucket j
    for k in range(len(pi)):
        if k == bucket_i:
            continue
        if w[bucket_j, k] == 0 and w[k, bucket_j] == 0:
            continue
        
        pairs = []
        if bucket_j in query_bucket_to_indices and w[bucket_j, k] > 0:
            for q_idx in query_bucket_to_indices[bucket_j]:
                for n_idx in ground_truth_neighbors[q_idx]:
                    if n_idx < len(base_embeddings):
                        n_code = base_codes[n_idx]
                        n_code_tuple = tuple(n_code.astype(int).tolist())
                        if n_code_tuple in code_to_bucket:
                            n_bucket = code_to_bucket[n_code_tuple]
                            if n_bucket == k:
                                pairs.append((q_idx, n_idx))
        
        if k in query_bucket_to_indices and w[k, bucket_j] > 0:
            for q_idx in query_bucket_to_indices[k]:
                for n_idx in ground_truth_neighbors[q_idx]:
                    if n_idx < len(base_embeddings):
                        n_code = base_codes[n_idx]
                        n_code_tuple = tuple(n_code.astype(int).tolist())
                        if n_code_tuple in code_to_bucket:
                            n_bucket = code_to_bucket[n_code_tuple]
                            if n_bucket == bucket_j:
                                pairs.append((q_idx, n_idx))
        
        if len(pairs) == 0:
            continue
        
        if sample_size is not None and len(pairs) > sample_size:
            sampled_indices = np.random.choice(len(pairs), size=sample_size, replace=False)
            pairs = [pairs[idx] for idx in sampled_indices]
        
        hamming_dists = []
        for q_idx, n_idx in pairs:
            q_code = query_codes[q_idx]
            n_code = base_codes[n_idx]
            
            q_bucket = code_to_bucket[tuple(q_code.astype(int).tolist())]
            n_bucket = code_to_bucket[tuple(n_code.astype(int).tolist())]
            
            q_code_permuted = permutation_swapped[q_bucket]
            n_code_permuted = permutation_swapped[n_bucket]
            
            d_h = hamming_distance(
                q_code_permuted[np.newaxis, :],
                n_code_permuted[np.newaxis, :]
            )[0, 0]
            hamming_dists.append(d_h)
        
        avg_hamming = np.mean(hamming_dists) if len(hamming_dists) > 0 else 0.0
        
        if w[bucket_j, k] > 0:
            cost_after += pi[bucket_j] * w[bucket_j, k] * avg_hamming
        if w[k, bucket_j] > 0:
            cost_after += pi[k] * w[k, bucket_j] * avg_hamming
    
    return float(cost_after - cost_before)


def compute_j_phi_cost_delta_swap_buckets_multi_radius(
    permutation: np.ndarray,  # Shape (K, n_bits)
    pi: np.ndarray,
    w: np.ndarray,
    queries: np.ndarray,
    base_embeddings: np.ndarray,
    ground_truth_neighbors: np.ndarray,
    encoder: Callable,
    code_to_bucket: Dict[Tuple, int],
    n_bits: int,
    bucket_i: int,
    bucket_j: int,
    radii: List[int],
    radius_weights: np.ndarray,
    sample_size: Optional[int] = None,
) -> float:
    """
    Compute delta in multi-radius J(φ) cost when swapping codes of buckets i and j.
    
    Delta = Σ_r w_r · delta_r where delta_r is the change for radius r.
    
    Args:
        permutation: Permutation array of shape (K, n_bits)
        pi: Query prior of shape (K,)
        w: Neighbor weights of shape (K, K)
        queries: Query embeddings of shape (Q, dim)
        base_embeddings: Base corpus embeddings of shape (N, dim)
        ground_truth_neighbors: Ground truth neighbor indices of shape (Q, k)
        encoder: LSH encoder function
        code_to_bucket: Dictionary mapping code tuples to bucket indices
        n_bits: Number of bits
        bucket_i: First bucket index to swap
        bucket_j: Second bucket index to swap
        radii: List of Hamming radii
        radius_weights: Weights for each radius
        sample_size: Optional maximum number of pairs to sample per bucket pair
        
    Returns:
        Delta = cost_after_swap - cost_before_swap
    """
    # Compute delta for each radius and combine
    total_delta = 0.0
    
    for radius_idx, radius in enumerate(radii):
        radius_weight = radius_weights[radius_idx]
        
        # For this radius, compute delta by only considering pairs with d_H <= radius
        # We can reuse the single-radius delta computation but filter by radius
        
        # Encode queries and base embeddings
        query_codes = encoder(queries)  # Shape (Q, n_bits)
        base_codes = encoder(base_embeddings)  # Shape (N, dim)
        
        # Build query_bucket -> list of query indices
        query_bucket_to_indices = defaultdict(list)
        for q_idx, q_code in enumerate(query_codes):
            q_code_tuple = tuple(q_code.astype(int).tolist())
            if q_code_tuple in code_to_bucket:
                bucket_idx = code_to_bucket[q_code_tuple]
                query_bucket_to_indices[bucket_idx].append(q_idx)
        
        # Compute cost before swap (only terms involving i or j, filtered by radius)
        cost_before = 0.0
        
        # Terms involving bucket i
        for k in range(len(pi)):
            if w[bucket_i, k] == 0 and w[k, bucket_i] == 0:
                continue
            
            pairs = []
            if bucket_i in query_bucket_to_indices and w[bucket_i, k] > 0:
                for q_idx in query_bucket_to_indices[bucket_i]:
                    for n_idx in ground_truth_neighbors[q_idx]:
                        if n_idx < len(base_embeddings):
                            n_code = base_codes[n_idx]
                            n_code_tuple = tuple(n_code.astype(int).tolist())
                            if n_code_tuple in code_to_bucket:
                                n_bucket = code_to_bucket[n_code_tuple]
                                if n_bucket == k:
                                    pairs.append((q_idx, n_idx))
            
            if k in query_bucket_to_indices and w[k, bucket_i] > 0:
                for q_idx in query_bucket_to_indices[k]:
                    for n_idx in ground_truth_neighbors[q_idx]:
                        if n_idx < len(base_embeddings):
                            n_code = base_codes[n_idx]
                            n_code_tuple = tuple(n_code.astype(int).tolist())
                            if n_code_tuple in code_to_bucket:
                                n_bucket = code_to_bucket[n_code_tuple]
                                if n_bucket == bucket_i:
                                    pairs.append((q_idx, n_idx))
            
            if len(pairs) == 0:
                continue
            
            if sample_size is not None and len(pairs) > sample_size:
                sampled_indices = np.random.choice(len(pairs), size=sample_size, replace=False)
                pairs = [pairs[idx] for idx in sampled_indices]
            
            # Compute Hamming distances and filter by radius
            hamming_dists = []
            for q_idx, n_idx in pairs:
                q_code = query_codes[q_idx]
                n_code = base_codes[n_idx]
                
                q_bucket = code_to_bucket[tuple(q_code.astype(int).tolist())]
                n_bucket = code_to_bucket[tuple(n_code.astype(int).tolist())]
                
                q_code_permuted = permutation[q_bucket]
                n_code_permuted = permutation[n_bucket]
                
                d_h = hamming_distance(
                    q_code_permuted[np.newaxis, :],
                    n_code_permuted[np.newaxis, :]
                )[0, 0]
                
                # Only include if within radius
                if d_h <= radius:
                    hamming_dists.append(d_h)
            
            avg_hamming = np.mean(hamming_dists) if len(hamming_dists) > 0 else 0.0
            
            if w[bucket_i, k] > 0:
                cost_before += pi[bucket_i] * w[bucket_i, k] * avg_hamming
            if w[k, bucket_i] > 0 and k != bucket_i:
                cost_before += pi[k] * w[k, bucket_i] * avg_hamming
        
        # Terms involving bucket j
        for k in range(len(pi)):
            if k == bucket_i:
                continue
            if w[bucket_j, k] == 0 and w[k, bucket_j] == 0:
                continue
            
            pairs = []
            if bucket_j in query_bucket_to_indices and w[bucket_j, k] > 0:
                for q_idx in query_bucket_to_indices[bucket_j]:
                    for n_idx in ground_truth_neighbors[q_idx]:
                        if n_idx < len(base_embeddings):
                            n_code = base_codes[n_idx]
                            n_code_tuple = tuple(n_code.astype(int).tolist())
                            if n_code_tuple in code_to_bucket:
                                n_bucket = code_to_bucket[n_code_tuple]
                                if n_bucket == k:
                                    pairs.append((q_idx, n_idx))
            
            if k in query_bucket_to_indices and w[k, bucket_j] > 0:
                for q_idx in query_bucket_to_indices[k]:
                    for n_idx in ground_truth_neighbors[q_idx]:
                        if n_idx < len(base_embeddings):
                            n_code = base_codes[n_idx]
                            n_code_tuple = tuple(n_code.astype(int).tolist())
                            if n_code_tuple in code_to_bucket:
                                n_bucket = code_to_bucket[n_code_tuple]
                                if n_bucket == bucket_j:
                                    pairs.append((q_idx, n_idx))
            
            if len(pairs) == 0:
                continue
            
            if sample_size is not None and len(pairs) > sample_size:
                sampled_indices = np.random.choice(len(pairs), size=sample_size, replace=False)
                pairs = [pairs[idx] for idx in sampled_indices]
            
            hamming_dists = []
            for q_idx, n_idx in pairs:
                q_code = query_codes[q_idx]
                n_code = base_codes[n_idx]
                
                q_bucket = code_to_bucket[tuple(q_code.astype(int).tolist())]
                n_bucket = code_to_bucket[tuple(n_code.astype(int).tolist())]
                
                q_code_permuted = permutation[q_bucket]
                n_code_permuted = permutation[n_bucket]
                
                d_h = hamming_distance(
                    q_code_permuted[np.newaxis, :],
                    n_code_permuted[np.newaxis, :]
                )[0, 0]
                
                if d_h <= radius:
                    hamming_dists.append(d_h)
            
            avg_hamming = np.mean(hamming_dists) if len(hamming_dists) > 0 else 0.0
            
            if w[bucket_j, k] > 0:
                cost_before += pi[bucket_j] * w[bucket_j, k] * avg_hamming
            if w[k, bucket_j] > 0:
                cost_before += pi[k] * w[k, bucket_j] * avg_hamming
        
        # Swap codes
        permutation_swapped = permutation.copy()
        permutation_swapped[bucket_i], permutation_swapped[bucket_j] = \
            permutation_swapped[bucket_j].copy(), permutation_swapped[bucket_i].copy()
        
        # Compute cost after swap (same logic but with swapped permutation)
        cost_after = 0.0
        
        # Terms involving bucket i
        for k in range(len(pi)):
            if w[bucket_i, k] == 0 and w[k, bucket_i] == 0:
                continue
            
            pairs = []
            if bucket_i in query_bucket_to_indices and w[bucket_i, k] > 0:
                for q_idx in query_bucket_to_indices[bucket_i]:
                    for n_idx in ground_truth_neighbors[q_idx]:
                        if n_idx < len(base_embeddings):
                            n_code = base_codes[n_idx]
                            n_code_tuple = tuple(n_code.astype(int).tolist())
                            if n_code_tuple in code_to_bucket:
                                n_bucket = code_to_bucket[n_code_tuple]
                                if n_bucket == k:
                                    pairs.append((q_idx, n_idx))
            
            if k in query_bucket_to_indices and w[k, bucket_i] > 0:
                for q_idx in query_bucket_to_indices[k]:
                    for n_idx in ground_truth_neighbors[q_idx]:
                        if n_idx < len(base_embeddings):
                            n_code = base_codes[n_idx]
                            n_code_tuple = tuple(n_code.astype(int).tolist())
                            if n_code_tuple in code_to_bucket:
                                n_bucket = code_to_bucket[n_code_tuple]
                                if n_bucket == bucket_i:
                                    pairs.append((q_idx, n_idx))
            
            if len(pairs) == 0:
                continue
            
            if sample_size is not None and len(pairs) > sample_size:
                sampled_indices = np.random.choice(len(pairs), size=sample_size, replace=False)
                pairs = [pairs[idx] for idx in sampled_indices]
            
            hamming_dists = []
            for q_idx, n_idx in pairs:
                q_code = query_codes[q_idx]
                n_code = base_codes[n_idx]
                
                q_bucket = code_to_bucket[tuple(q_code.astype(int).tolist())]
                n_bucket = code_to_bucket[tuple(n_code.astype(int).tolist())]
                
                q_code_permuted = permutation_swapped[q_bucket]
                n_code_permuted = permutation_swapped[n_bucket]
                
                d_h = hamming_distance(
                    q_code_permuted[np.newaxis, :],
                    n_code_permuted[np.newaxis, :]
                )[0, 0]
                
                if d_h <= radius:
                    hamming_dists.append(d_h)
            
            avg_hamming = np.mean(hamming_dists) if len(hamming_dists) > 0 else 0.0
            
            if w[bucket_i, k] > 0:
                cost_after += pi[bucket_i] * w[bucket_i, k] * avg_hamming
            if w[k, bucket_i] > 0 and k != bucket_i:
                cost_after += pi[k] * w[k, bucket_i] * avg_hamming
        
        # Terms involving bucket j
        for k in range(len(pi)):
            if k == bucket_i:
                continue
            if w[bucket_j, k] == 0 and w[k, bucket_j] == 0:
                continue
            
            pairs = []
            if bucket_j in query_bucket_to_indices and w[bucket_j, k] > 0:
                for q_idx in query_bucket_to_indices[bucket_j]:
                    for n_idx in ground_truth_neighbors[q_idx]:
                        if n_idx < len(base_embeddings):
                            n_code = base_codes[n_idx]
                            n_code_tuple = tuple(n_code.astype(int).tolist())
                            if n_code_tuple in code_to_bucket:
                                n_bucket = code_to_bucket[n_code_tuple]
                                if n_bucket == k:
                                    pairs.append((q_idx, n_idx))
            
            if k in query_bucket_to_indices and w[k, bucket_j] > 0:
                for q_idx in query_bucket_to_indices[k]:
                    for n_idx in ground_truth_neighbors[q_idx]:
                        if n_idx < len(base_embeddings):
                            n_code = base_codes[n_idx]
                            n_code_tuple = tuple(n_code.astype(int).tolist())
                            if n_code_tuple in code_to_bucket:
                                n_bucket = code_to_bucket[n_code_tuple]
                                if n_bucket == bucket_j:
                                    pairs.append((q_idx, n_idx))
            
            if len(pairs) == 0:
                continue
            
            if sample_size is not None and len(pairs) > sample_size:
                sampled_indices = np.random.choice(len(pairs), size=sample_size, replace=False)
                pairs = [pairs[idx] for idx in sampled_indices]
            
            hamming_dists = []
            for q_idx, n_idx in pairs:
                q_code = query_codes[q_idx]
                n_code = base_codes[n_idx]
                
                q_bucket = code_to_bucket[tuple(q_code.astype(int).tolist())]
                n_bucket = code_to_bucket[tuple(n_code.astype(int).tolist())]
                
                q_code_permuted = permutation_swapped[q_bucket]
                n_code_permuted = permutation_swapped[n_bucket]
                
                d_h = hamming_distance(
                    q_code_permuted[np.newaxis, :],
                    n_code_permuted[np.newaxis, :]
                )[0, 0]
                
                if d_h <= radius:
                    hamming_dists.append(d_h)
            
            avg_hamming = np.mean(hamming_dists) if len(hamming_dists) > 0 else 0.0
            
            if w[bucket_j, k] > 0:
                cost_after += pi[bucket_j] * w[bucket_j, k] * avg_hamming
            if w[k, bucket_j] > 0:
                cost_after += pi[k] * w[k, bucket_j] * avg_hamming
        
        # Delta for this radius
        radius_delta = cost_after - cost_before
        total_delta += radius_weight * radius_delta
    
    return float(total_delta)


def compute_j_phi_0(
    pi: np.ndarray,
    w: np.ndarray,
    bucket_to_code: np.ndarray,
    n_bits: int,
    initial_permutation: Optional[np.ndarray] = None,
    semantic_distances: Optional[np.ndarray] = None,
    semantic_weight: float = 0.0,
) -> float:
    """
    Compute J(φ₀) - the baseline cost.
    
    If initial_permutation is provided, use it to compute J(φ₀).
    Otherwise, compute directly from bucket_to_code.
    
    This ensures consistency: J(φ₀) should equal the cost of the initial permutation.
    
    Args:
        pi: Query prior of shape (K,)
        w: Neighbor weights of shape (K, K)
        bucket_to_code: Original bucket codes of shape (K, n_bits)
        n_bits: Number of bits
        initial_permutation: Optional initial permutation of shape (N,) where N = 2**n_bits
                           If provided, computes J(φ₀) via this permutation
                           If None, computes directly from bucket_to_code
        semantic_distances: Optional semantic distance matrix of shape (K, K)
        semantic_weight: Weight for semantic term in J(φ) (default: 0.0)
        
    Returns:
        J(φ₀) cost
    """
    if initial_permutation is not None:
        # Compute J(φ₀) via initial permutation
        # This ensures consistency with the optimization starting point
        return compute_j_phi_cost(
            permutation=initial_permutation,
            pi=pi,
            w=w,
            bucket_to_code=bucket_to_code,
            n_bits=n_bits,
            bucket_to_embedding_idx=None,
            semantic_distances=semantic_distances,
            semantic_weight=semantic_weight,
        )
    else:
        # Compute directly from bucket_to_code (legacy method)
        # This is the "true" baseline: each bucket keeps its original code
        K = len(pi)
        cost = 0.0
        for i in range(K):
            code_i = bucket_to_code[i]
            for j in range(K):
                code_j = bucket_to_code[j]
                d_h = hamming_distance(code_i[np.newaxis, :], code_j[np.newaxis, :])[0, 0]
                
                # Base term
                base_term = pi[i] * w[i, j] * d_h
                
                # Semantic term
                semantic_term = 0.0
                if semantic_distances is not None and semantic_weight > 0.0:
                    semantic_term = semantic_weight * pi[i] * w[i, j] * semantic_distances[i, j]
                
                cost += base_term + semantic_term
        return float(cost)


def compute_j_phi_cost(
    permutation: np.ndarray,
    pi: np.ndarray,
    w: np.ndarray,
    bucket_to_code: np.ndarray,
    n_bits: int,
    bucket_to_embedding_idx: Optional[np.ndarray] = None,
    semantic_distances: Optional[np.ndarray] = None,
    semantic_weight: float = 0.0,
) -> float:
    """
    Compute J(φ) cost directly.
    
    J(φ) = Σ_{i,j} π_i · w_ij · [d_H(φ(c_i), φ(c_j)) + α · d_semantic(i, j)]
    where α = semantic_weight
    
    If semantic_distances is None or semantic_weight == 0.0, uses original J(φ):
    J(φ) = Σ_{i,j} π_i · w_ij · d_H(φ(c_i), φ(c_j))
    
    Args:
        permutation: Permutation array of shape (N,) where permutation[vertex_idx] = embedding_idx
        pi: Query prior of shape (K,)
        w: Neighbor weights of shape (K, K)
        bucket_to_code: Original bucket codes of shape (K, n_bits)
        n_bits: Number of bits
        bucket_to_embedding_idx: Optional mapping from bucket_idx to embedding_idx
                                If None, assumes bucket_idx == embedding_idx for first K buckets
        semantic_distances: Optional semantic distance matrix of shape (K, K)
                           If provided, d_semantic[i, j] is the semantic distance between buckets i and j
        semantic_weight: Weight for semantic term (default: 0.0, i.e., no semantic term)
        
    Returns:
        J(φ) cost
    """
    vertices = generate_hypercube_vertices(n_bits)
    N = len(vertices)
    K = len(pi)
    
    if bucket_to_embedding_idx is None:
        bucket_to_embedding_idx = np.arange(K, dtype=np.int32)
    
    # Map: bucket_idx -> vertex_idx
    bucket_to_vertex = {}
    for vertex_idx in range(N):
        embedding_idx = permutation[vertex_idx]
        # Find which bucket this embedding corresponds to
        bucket_indices = np.where(bucket_to_embedding_idx == embedding_idx)[0]
        if len(bucket_indices) > 0:
            bucket_idx = bucket_indices[0]
            if bucket_idx < K:
                if bucket_idx not in bucket_to_vertex:
                    bucket_to_vertex[bucket_idx] = vertex_idx
    
    # Validate semantic_distances if provided
    if semantic_distances is not None:
        if semantic_distances.shape != (K, K):
            raise ValueError(
                f"semantic_distances must have shape (K, K) = ({K}, {K}), "
                f"got {semantic_distances.shape}"
            )
        if semantic_weight < 0.0:
            raise ValueError(f"semantic_weight must be >= 0, got {semantic_weight}")
    
    cost = 0.0
    for i in range(K):
        if i in bucket_to_vertex:
            code_i = vertices[bucket_to_vertex[i]]
        else:
            code_i = bucket_to_code[i]
        
        for j in range(K):
            if j in bucket_to_vertex:
                code_j = vertices[bucket_to_vertex[j]]
            else:
                code_j = bucket_to_code[j]
            
            d_h = hamming_distance(code_i[np.newaxis, :], code_j[np.newaxis, :])[0, 0]
            
            # Base term: π_i · w_ij · d_H
            base_term = pi[i] * w[i, j] * d_h
            
            # Semantic term: α · π_i · w_ij · d_semantic(i, j)
            semantic_term = 0.0
            if semantic_distances is not None and semantic_weight > 0.0:
                semantic_term = semantic_weight * pi[i] * w[i, j] * semantic_distances[i, j]
            
            cost += base_term + semantic_term
    
    return float(cost)


def compute_j_phi_cost_delta_swap(
    permutation: np.ndarray,
    pi: np.ndarray,
    w: np.ndarray,
    bucket_to_code: np.ndarray,
    n_bits: int,
    vertex_u: int,
    vertex_v: int,
    bucket_to_embedding_idx: Optional[np.ndarray] = None,
    semantic_distances: Optional[np.ndarray] = None,
    semantic_weight: float = 0.0,
) -> float:
    """
    Compute change in J(φ) cost after swapping vertices u and v.
    
    This is more efficient than recomputing the full cost by only computing
    the terms that change.
    """
    vertices = generate_hypercube_vertices(n_bits)
    N = len(vertices)
    K = len(pi)
    
    if bucket_to_embedding_idx is None:
        bucket_to_embedding_idx = np.arange(K, dtype=np.int32)
    
    # Get current codes for vertices u and v
    def get_bucket_code(vertex_idx):
        embedding_idx = permutation[vertex_idx]
        bucket_indices = np.where(bucket_to_embedding_idx == embedding_idx)[0]
        if len(bucket_indices) > 0 and bucket_indices[0] < K:
            return vertices[vertex_idx]
        else:
            # Not a real bucket, return None
            return None
    
    code_u_old = get_bucket_code(vertex_u)
    code_v_old = get_bucket_code(vertex_v)
    
    # After swap: vertex_u gets embedding from vertex_v, vertex_v gets embedding from vertex_u
    embedding_u_new = permutation[vertex_v]
    embedding_v_new = permutation[vertex_u]
    
    def get_bucket_code_after_swap(vertex_idx, new_embedding_idx):
        bucket_indices = np.where(bucket_to_embedding_idx == new_embedding_idx)[0]
        if len(bucket_indices) > 0 and bucket_indices[0] < K:
            return vertices[vertex_idx]
        else:
            return None
    
    code_u_new = get_bucket_code_after_swap(vertex_u, embedding_u_new)
    code_v_new = get_bucket_code_after_swap(vertex_v, embedding_v_new)
    
    # Compute delta: only terms involving buckets at vertices u or v change
    delta = 0.0
    
    # Find which buckets are at u and v
    bucket_u_old = None
    bucket_v_old = None
    bucket_u_new = None
    bucket_v_new = None
    
    for i in range(K):
        if code_u_old is not None and np.array_equal(vertices[vertex_u], code_u_old):
            bucket_u_old = i
        if code_v_old is not None and np.array_equal(vertices[vertex_v], code_v_old):
            bucket_v_old = i
        if code_u_new is not None and np.array_equal(vertices[vertex_u], code_u_new):
            bucket_u_new = i
        if code_v_new is not None and np.array_equal(vertices[vertex_v], code_v_new):
            bucket_v_new = i
    
    # For efficiency, we'll compute the full cost difference
    # This is still O(K²) but avoids recomputing the entire cost
    # In practice, we can optimize further by only computing affected terms
    
    # Simplified: compute full cost difference
    cost_old = compute_j_phi_cost(
        permutation, pi, w, bucket_to_code, n_bits, bucket_to_embedding_idx,
        semantic_distances, semantic_weight
    )
    perm_new = permutation.copy()
    perm_new[vertex_u], perm_new[vertex_v] = perm_new[vertex_v], perm_new[vertex_u]
    cost_new = compute_j_phi_cost(
        perm_new, pi, w, bucket_to_code, n_bits, bucket_to_embedding_idx,
        semantic_distances, semantic_weight
    )
    
    return cost_new - cost_old


def hill_climb_j_phi(
    pi_init: np.ndarray,
    pi: np.ndarray,
    w: np.ndarray,
    bucket_to_code: np.ndarray,
    n_bits: int,
    max_iter: int = 100,
    sample_size: int = 256,
    random_state: Optional[int] = None,
    bucket_to_embedding_idx: Optional[np.ndarray] = None,
    semantic_distances: Optional[np.ndarray] = None,
    semantic_weight: float = 0.0,
    verbose: bool = False,
    progress_callback: Optional[Callable[[int, float, float], None]] = None,
) -> Tuple[np.ndarray, float, float, list]:
    """
    Hill climb to minimize J(φ) directly using 2-swap moves.
    
    This guarantees that the final cost is <= initial cost (monotonic improvement).
    Since we start from identity (or random), and identity corresponds to J(φ₀),
    we guarantee J(φ*) ≤ J(φ₀).
    
    Args:
        pi_init: Initial permutation of shape (N,) where N = 2**n_bits
        pi: Query prior of shape (K,) where K <= N
        w: Neighbor weights of shape (K, K)
        bucket_to_code: Original bucket codes of shape (K, n_bits) or (N, n_bits) if padded
        n_bits: Number of bits
        max_iter: Maximum iterations
        sample_size: Number of swaps to sample per iteration
        random_state: Random seed
        bucket_to_embedding_idx: Optional bucket to embedding mapping (default: bucket i -> embedding i)
        semantic_distances: Optional semantic distance matrix of shape (K, K)
        semantic_weight: Weight for semantic term in J(φ) (default: 0.0)
        
    Returns:
        (best_permutation, best_cost, initial_cost, cost_history)
        where initial_cost is J(φ₀) computed from the initial permutation
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    N = len(pi_init)
    perm = pi_init.copy()
    initial_cost = compute_j_phi_cost(
        perm, pi, w, bucket_to_code, n_bits, bucket_to_embedding_idx,
        semantic_distances, semantic_weight
    )
    cost = initial_cost
    cost_history = [initial_cost]
    
    # FIX 3: Get K to validate swaps maintain embedding_idx < K constraint
    K = len(pi)
    if bucket_to_embedding_idx is None:
        # Default: bucket i maps to embedding i, so embedding_idx must be < K
        max_valid_embedding_idx = K - 1
    else:
        # If bucket_to_embedding_idx is provided, find max valid embedding_idx
        max_valid_embedding_idx = bucket_to_embedding_idx.max() if len(bucket_to_embedding_idx) > 0 else K - 1
    
    last_print_time = time.time()
    print_interval = 10.0  # Print every 10 seconds
    
    for iteration in range(max_iter):
        iter_start_time = time.time()
        
        # Sample random 2-swaps
        candidates = []
        for _ in range(sample_size):
            u, v = np.random.choice(N, size=2, replace=False)
            candidates.append((u, v))
        
        # Evaluate all candidates
        best_delta = 0.0
        best_swap = None
        
        for u, v in candidates:
            # FIX 3: Check if swap maintains validity constraint (embedding_idx < K)
            # After swap: perm[u] gets perm[v], perm[v] gets perm[u]
            # Both resulting values must be <= max_valid_embedding_idx
            # Since we initialize with all values < K, swaps should maintain this
            # But we verify to be safe
            new_u_val = perm[v]
            new_v_val = perm[u]
            
            # Skip swap if it would create invalid values
            if new_u_val > max_valid_embedding_idx or new_v_val > max_valid_embedding_idx:
                continue
            
            # Compute delta efficiently
            delta = compute_j_phi_cost_delta_swap(
                perm, pi, w, bucket_to_code, n_bits, u, v, bucket_to_embedding_idx,
                semantic_distances, semantic_weight
            )
            if delta < best_delta:
                best_delta = delta
                best_swap = (u, v)
        
        # Apply best improving swap
        if best_swap is not None:
            u, v = best_swap
            # FIX 3: Validate swap maintains constraint before applying
            # After swap, both values should be <= max_valid_embedding_idx
            # Since we check this in the candidate evaluation, this should always pass
            # But we verify to be safe
            temp_u, temp_v = perm[v], perm[u]
            if temp_u <= max_valid_embedding_idx and temp_v <= max_valid_embedding_idx:
                perm[u], perm[v] = temp_u, temp_v
                cost += best_delta
                
                # Validate monotonicity
                if cost > initial_cost + 1e-10:
                    raise ValueError(
                        f"Monotonicity violated: cost {cost:.6f} > initial_cost {initial_cost:.6f}"
                    )
                
                cost_history.append(cost)
            else:
                # This shouldn't happen since we check in candidate evaluation
                # But if it does, skip the swap
                continue
        else:
            # No improvement found
            if verbose:
                print(f"  No improvement found at iteration {iteration}, stopping.")
            break
        
        # Print progress periodically
        current_time = time.time()
        if verbose or (current_time - last_print_time >= print_interval):
            elapsed = current_time - last_print_time
            improvement = ((initial_cost - cost) / initial_cost * 100) if initial_cost > 0 else 0
            delta_str = f"{best_delta:.4f}" if best_swap is not None else "0.0000"
            print(f"  [Iter {iteration:3d}] cost={cost:.4f} ({improvement:.1f}% improvement), "
                  f"delta={delta_str}, time={elapsed:.1f}s")
            last_print_time = current_time
        
        # Call progress callback if provided
        if progress_callback is not None:
            progress_callback(iteration, cost, best_delta if best_swap is not None else 0.0)
    
    return perm, cost, initial_cost, cost_history


def detect_stagnation(
    cost_history: list,
    window: int = 10,
    threshold: float = 0.001,
) -> bool:
    """
    Detect stagnation in optimization based on cost history.
    
    Args:
        cost_history: List of costs (most recent last)
        window: Number of recent iterations to consider
        threshold: Relative improvement threshold (default: 0.001 = 0.1%)
        
    Returns:
        True if stagnation detected, False otherwise
    """
    if len(cost_history) < window + 1:
        return False
    
    # Get costs at window start and end
    cost_start = cost_history[-window - 1]
    cost_end = cost_history[-1]
    
    # Compute relative improvement
    if cost_start <= 0:
        return False
    
    relative_improvement = (cost_start - cost_end) / cost_start
    
    # Stagnation if improvement < threshold
    return relative_improvement < threshold


def hill_climb_j_phi_real_embeddings(
    pi_init: np.ndarray,  # Shape (K, n_bits) - NEW
    pi: np.ndarray,  # Shape (K,)
    w: np.ndarray,  # Shape (K, K)
    queries: np.ndarray,  # Shape (Q, dim)
    base_embeddings: np.ndarray,  # Shape (N, dim)
    ground_truth_neighbors: np.ndarray,  # Shape (Q, k)
    encoder: Callable,
    code_to_bucket: Dict[Tuple, int],
    n_bits: int,
    max_iter: int = 100,
    sample_size: int = 256,
    random_state: Optional[int] = None,
    sample_size_pairs: Optional[int] = None,  # Sample size for pairs in cost computation
    verbose: bool = False,
    progress_callback: Optional[Callable[[int, float, float], None]] = None,
    # NEW (Sprint 9): Multi-radius support
    radii: Optional[List[int]] = None,
    radius_weights: Optional[np.ndarray] = None,
    # NEW (Sprint 9): Tunneling support
    tunneling_on_stagnation: bool = False,
    tunneling_probability: float = 0.0,
    stagnation_window: int = 10,
    stagnation_threshold: float = 0.001,
    tunneling_step_fn: Optional[Callable] = None,
) -> Tuple[np.ndarray, float, float, list]:
    """
    Hill climb to minimize J(φ) using real embeddings and 2-swap moves on bucket codes.
    
    NEW (Sprint 8): Works with permutation as (K, n_bits) array where
    permutation[bucket_idx] = novo_código_binário.
    
    NEW (Sprint 9): Supports multi-radius objective and adaptive tunneling.
    
    This guarantees that the final cost is <= initial cost (monotonic improvement).
    
    Args:
        pi_init: Initial permutation of shape (K, n_bits) - codes for each bucket
        pi: Query prior of shape (K,)
        w: Neighbor weights of shape (K, K)
        queries: Query embeddings of shape (Q, dim)
        base_embeddings: Base corpus embeddings of shape (N, dim)
        ground_truth_neighbors: Ground truth neighbor indices of shape (Q, k)
        encoder: LSH encoder function
        code_to_bucket: Dictionary mapping code tuples to bucket indices
        n_bits: Number of bits
        max_iter: Maximum iterations
        sample_size: Number of bucket pairs to sample for swaps per iteration
        random_state: Random seed
        sample_size_pairs: Optional maximum number of query-neighbor pairs to sample per bucket pair
        verbose: If True, print progress
        progress_callback: Optional callback function(iteration, cost, delta)
        radii: Optional list of Hamming radii for multi-radius objective. If None, uses single radius.
        radius_weights: Optional weights for each radius. If None and radii provided, generates default weights.
        tunneling_on_stagnation: If True, apply tunneling when stagnation detected
        tunneling_probability: Base probability for probabilistic tunneling (0.0 to 1.0)
        stagnation_window: Number of iterations for stagnation detection
        stagnation_threshold: Relative improvement threshold for stagnation
        tunneling_step_fn: Optional function to perform tunneling step. Signature: (perm, ...) -> (perm_new, delta)
        
    Returns:
        (best_permutation, best_cost, initial_cost, cost_history)
        where permutation is shape (K, n_bits)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    K = len(pi)
    perm = pi_init.copy()  # Shape (K, n_bits)
    
    # Determine if using multi-radius objective
    use_multi_radius = radii is not None and len(radii) > 1
    if use_multi_radius:
        if radius_weights is None:
            radius_weights = validate_radius_weights(radii)
        else:
            radius_weights = validate_radius_weights(radii, radius_weights)
    
    # Compute initial cost
    if use_multi_radius:
        initial_cost = compute_j_phi_cost_multi_radius(
            perm, pi, w, queries, base_embeddings, ground_truth_neighbors,
            encoder, code_to_bucket, n_bits, radii, radius_weights,
            sample_size=sample_size_pairs
        )
    else:
        initial_cost = compute_j_phi_cost_real_embeddings(
            perm, pi, w, queries, base_embeddings, ground_truth_neighbors,
            encoder, code_to_bucket, n_bits, sample_size=sample_size_pairs
        )
    cost = initial_cost
    cost_history = [initial_cost]
    
    last_print_time = time.time()
    print_interval = 10.0  # Print every 10 seconds
    stagnation_count = 0  # Track consecutive stagnation detections
    
    for iteration in range(max_iter):
        iter_start_time = time.time()
        
        # Sample random 2-swaps of bucket codes
        candidates = []
        for _ in range(sample_size):
            i, j = np.random.choice(K, size=2, replace=False)
            candidates.append((i, j))
        
        # Evaluate all candidates
        best_delta = 0.0
        best_swap = None
        
        for bucket_i, bucket_j in candidates:
            # Compute delta efficiently
            if use_multi_radius:
                delta = compute_j_phi_cost_delta_swap_buckets_multi_radius(
                    perm, pi, w, queries, base_embeddings, ground_truth_neighbors,
                    encoder, code_to_bucket, n_bits, bucket_i, bucket_j,
                    radii, radius_weights, sample_size=sample_size_pairs
                )
            else:
                delta = compute_j_phi_cost_delta_swap_buckets(
                    perm, pi, w, queries, base_embeddings, ground_truth_neighbors,
                    encoder, code_to_bucket, n_bits, bucket_i, bucket_j,
                    sample_size=sample_size_pairs
                )
            if delta < best_delta:
                best_delta = delta
                best_swap = (bucket_i, bucket_j)
        
        # Apply best improving swap
        improvement_found = False
        if best_swap is not None:
            bucket_i, bucket_j = best_swap
            # Swap codes of buckets i and j
            perm[bucket_i], perm[bucket_j] = perm[bucket_j].copy(), perm[bucket_i].copy()
            cost += best_delta
            
            # Validate monotonicity
            if cost > initial_cost + 1e-10:
                raise ValueError(
                    f"Monotonicity violated: cost {cost:.6f} > initial_cost {initial_cost:.6f}"
                )
            
            cost_history.append(cost)
            improvement_found = True
            stagnation_count = 0  # Reset stagnation counter
        else:
            # No improvement found from 2-swap
            stagnation_count += 1
        
        # Check for stagnation
        is_stagnant = detect_stagnation(cost_history, stagnation_window, stagnation_threshold)
        
        # Apply tunneling if needed
        tunneling_applied = False
        tunneling_delta = 0.0
        
        if tunneling_step_fn is not None:
            # Probabilistic tunneling
            if tunneling_probability > 0.0 and np.random.random() < tunneling_probability:
                perm_new, tunneling_delta = tunneling_step_fn(
                    perm, pi, w, queries, base_embeddings, ground_truth_neighbors,
                    encoder, code_to_bucket, n_bits, radii, radius_weights,
                    sample_size_pairs
                )
                if tunneling_delta < -1e-10:  # Improvement
                    perm = perm_new
                    cost += tunneling_delta
                    cost_history.append(cost)
                    tunneling_applied = True
                    stagnation_count = 0
            
            # Stagnation-based tunneling
            elif tunneling_on_stagnation and is_stagnant:
                perm_new, tunneling_delta = tunneling_step_fn(
                    perm, pi, w, queries, base_embeddings, ground_truth_neighbors,
                    encoder, code_to_bucket, n_bits, radii, radius_weights,
                    sample_size_pairs
                )
                if tunneling_delta < -1e-10:  # Improvement
                    perm = perm_new
                    cost += tunneling_delta
                    cost_history.append(cost)
                    tunneling_applied = True
                    stagnation_count = 0
        
        # Stop if no improvement and not using tunneling
        if not improvement_found and not tunneling_applied:
            if tunneling_step_fn is None or (not tunneling_on_stagnation and tunneling_probability == 0.0):
                if verbose:
                    print(f"  No improvement found at iteration {iteration}, stopping.")
                break
        
        # Print progress periodically
        current_time = time.time()
        if verbose or (current_time - last_print_time >= print_interval):
            elapsed = current_time - last_print_time
            improvement = ((initial_cost - cost) / initial_cost * 100) if initial_cost > 0 else 0
            delta_str = f"{best_delta:.4f}" if best_swap is not None else "0.0000"
            tunnel_str = f", tunnel={tunneling_delta:.4f}" if tunneling_applied else ""
            stagnant_str = " [STAGNANT]" if is_stagnant else ""
            print(f"  [Iter {iteration:3d}] cost={cost:.4f} ({improvement:.1f}% improvement), "
                  f"delta={delta_str}{tunnel_str}{stagnant_str}, time={elapsed:.1f}s")
            last_print_time = current_time
        
        # Call progress callback if provided
        if progress_callback is not None:
            progress_callback(iteration, cost, best_delta if best_swap is not None else tunneling_delta)
    
    return perm, cost, initial_cost, cost_history


def tunneling_step_j_phi(
    permutation: np.ndarray,  # Shape (K, n_bits)
    pi: np.ndarray,
    w: np.ndarray,
    queries: np.ndarray,
    base_embeddings: np.ndarray,
    ground_truth_neighbors: np.ndarray,
    encoder: Callable,
    code_to_bucket: Dict[Tuple, int],
    n_bits: int,
    radii: Optional[List[int]] = None,
    radius_weights: Optional[np.ndarray] = None,
    sample_size_pairs: Optional[int] = None,
    block_size: int = 4,
    num_blocks: int = 10,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """
    Perform one J(φ)-aware tunneling step: try multiple blocks of buckets and apply best improving move.
    
    Similar to QAP tunneling but uses J(φ) cost instead. Reoptimizes code assignments
    within small blocks of buckets using brute force.
    
    Args:
        permutation: Current permutation of shape (K, n_bits)
        pi: Query prior of shape (K,)
        w: Neighbor weights of shape (K, K)
        queries: Query embeddings of shape (Q, dim)
        base_embeddings: Base corpus embeddings of shape (N, dim)
        ground_truth_neighbors: Ground truth neighbor indices of shape (Q, k)
        encoder: LSH encoder function
        code_to_bucket: Dictionary mapping code tuples to bucket indices
        n_bits: Number of bits
        radii: Optional list of Hamming radii for multi-radius objective
        radius_weights: Optional weights for each radius
        sample_size_pairs: Optional maximum number of pairs to sample per bucket pair
        block_size: Size of blocks to try (default: 4)
        num_blocks: Number of candidate blocks to sample (default: 10)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (perm_new, best_delta) where:
        - perm_new: Updated permutation (may be unchanged if no improvement)
        - best_delta: Best cost delta found (negative means improvement)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    K = len(pi)
    if block_size > K:
        block_size = K
    
    best_perm = permutation.copy()
    best_delta = 0.0
    
    # Determine if using multi-radius
    use_multi_radius = radii is not None and len(radii) > 1
    
    # Compute current cost for the block
    def compute_block_cost(perm_block: np.ndarray, block_buckets: np.ndarray) -> float:
        """Compute cost contribution from a block of buckets."""
        # Create full permutation with block codes
        perm_full = permutation.copy()
        for idx, bucket_idx in enumerate(block_buckets):
            perm_full[bucket_idx] = perm_block[idx].copy()
        
        if use_multi_radius:
            return compute_j_phi_cost_multi_radius(
                perm_full, pi, w, queries, base_embeddings, ground_truth_neighbors,
                encoder, code_to_bucket, n_bits, radii, radius_weights,
                sample_size=sample_size_pairs
            )
        else:
            return compute_j_phi_cost_real_embeddings(
                perm_full, pi, w, queries, base_embeddings, ground_truth_neighbors,
                encoder, code_to_bucket, n_bits, sample_size=sample_size_pairs
            )
    
    # Try multiple random blocks
    for _ in range(num_blocks):
        # Select random block of buckets
        if block_size >= K:
            block_buckets = np.arange(K)
        else:
            block_buckets = np.sort(np.random.choice(K, size=block_size, replace=False))
        
        # Get current codes for this block
        block_codes = permutation[block_buckets].copy()  # Shape (block_size, n_bits)
        
        # For small blocks, try all permutations
        if block_size <= 6:  # Reasonable limit for brute force
            from itertools import permutations
            
            current_cost = compute_block_cost(block_codes, block_buckets)
            best_block_codes = block_codes.copy()
            best_block_delta = 0.0
            
            # Try all permutations of codes within block
            for perm_indices in permutations(range(block_size)):
                permuted_codes = block_codes[list(perm_indices)]
                new_cost = compute_block_cost(permuted_codes, block_buckets)
                delta = new_cost - current_cost
                
                if delta < best_block_delta:
                    best_block_delta = delta
                    best_block_codes = permuted_codes.copy()
            
            # If improvement found, update best overall
            if best_block_delta < best_delta:
                best_delta = best_block_delta
                best_perm = permutation.copy()
                for idx, bucket_idx in enumerate(block_buckets):
                    best_perm[bucket_idx] = best_block_codes[idx].copy()
        else:
            # For larger blocks, use greedy 2-swap within block
            # This is a simplified approach - could be improved
            block_perm = block_codes.copy()
            block_current_cost = compute_block_cost(block_perm, block_buckets)
            
            # Try a few 2-swaps within the block
            for _ in range(min(block_size * 2, 20)):
                i, j = np.random.choice(block_size, size=2, replace=False)
                
                # Swap codes
                block_perm[i], block_perm[j] = block_perm[j].copy(), block_perm[i].copy()
                block_new_cost = compute_block_cost(block_perm, block_buckets)
                block_delta = block_new_cost - block_current_cost
                
                if block_delta < 0:
                    # Improvement - keep swap
                    block_current_cost = block_new_cost
                    if block_delta < best_delta:
                        best_delta = block_delta
                        best_perm = permutation.copy()
                        for idx, bucket_idx in enumerate(block_buckets):
                            best_perm[bucket_idx] = block_perm[idx].copy()
                else:
                    # No improvement - revert swap
                    block_perm[i], block_perm[j] = block_perm[j].copy(), block_perm[i].copy()
    
    # Only return improved permutation if we found an improvement
    if best_delta < -1e-10:  # Small tolerance for numerical errors
        return best_perm, best_delta
    else:
        return permutation.copy(), 0.0

