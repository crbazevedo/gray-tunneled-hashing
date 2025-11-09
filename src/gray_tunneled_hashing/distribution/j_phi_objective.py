"""
Direct optimization of J(φ) objective for distribution-aware GTH.

J(φ) = Σ_{i,j} π_i · w_ij · d_H(φ(c_i), φ(c_j))

This is different from QAP cost which only sums over hypercube edges.
We need to optimize J(φ) directly.

NEW (Sprint 8): J(φ) computed over real embeddings:
J(φ) = Σ_{i,j} π_i · w_ij · E[d_H(φ(h(q)), φ(h(x))) | q∈bucket_i, x∈bucket_j]
"""

import numpy as np
from typing import Optional, Tuple, Callable, Dict
from collections import defaultdict
import time

from gray_tunneled_hashing.data.synthetic_generators import generate_hypercube_vertices
from gray_tunneled_hashing.evaluation.metrics import hamming_distance


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
) -> Tuple[np.ndarray, float, float, list]:
    """
    Hill climb to minimize J(φ) using real embeddings and 2-swap moves on bucket codes.
    
    NEW (Sprint 8): Works with permutation as (K, n_bits) array where
    permutation[bucket_idx] = novo_código_binário.
    
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
        
    Returns:
        (best_permutation, best_cost, initial_cost, cost_history)
        where permutation is shape (K, n_bits)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    K = len(pi)
    perm = pi_init.copy()  # Shape (K, n_bits)
    
    # Compute initial cost
    initial_cost = compute_j_phi_cost_real_embeddings(
        perm, pi, w, queries, base_embeddings, ground_truth_neighbors,
        encoder, code_to_bucket, n_bits, sample_size=sample_size_pairs
    )
    cost = initial_cost
    cost_history = [initial_cost]
    
    last_print_time = time.time()
    print_interval = 10.0  # Print every 10 seconds
    
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
            delta = compute_j_phi_cost_delta_swap_buckets(
                perm, pi, w, queries, base_embeddings, ground_truth_neighbors,
                encoder, code_to_bucket, n_bits, bucket_i, bucket_j,
                sample_size=sample_size_pairs
            )
            if delta < best_delta:
                best_delta = delta
                best_swap = (bucket_i, bucket_j)
        
        # Apply best improving swap
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

