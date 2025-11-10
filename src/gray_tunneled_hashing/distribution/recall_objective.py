"""
Recall-based objective functions for direct recall optimization.

This module implements surrogate objectives that directly optimize for recall@k
instead of using J(φ) as a proxy.
"""

import numpy as np
from typing import Optional, Tuple, Dict
from collections import defaultdict

from gray_tunneled_hashing.data.synthetic_generators import generate_hypercube_vertices
from gray_tunneled_hashing.evaluation.metrics import hamming_distance


def compute_recall_surrogate_cost(
    permutation: np.ndarray,
    pi: np.ndarray,
    w: np.ndarray,
    bucket_to_code: np.ndarray,
    n_bits: int,
    hamming_radius: int = 1,
    bucket_to_embedding_idx: Optional[np.ndarray] = None,
) -> float:
    """
    Compute recall surrogate cost.
    
    This objective penalizes large Hamming distances for query-neighbor pairs
    that should be within the Hamming ball.
    
    Cost = Σ_{i,j} π_i · w_ij · max(0, d_H(φ(c_i), φ(c_j)) - R)²
    
    where R is the Hamming radius. This encourages neighbors to be within
    the Hamming ball.
    
    Args:
        permutation: Permutation array of shape (N,)
        pi: Query prior of shape (K,)
        w: Neighbor weights of shape (K, K)
        bucket_to_code: Original bucket codes of shape (K, n_bits)
        n_bits: Number of bits
        hamming_radius: Hamming ball radius
        bucket_to_embedding_idx: Optional bucket to embedding mapping
        
    Returns:
        Recall surrogate cost
    """
    vertices = generate_hypercube_vertices(n_bits)
    N = len(vertices)
    K = len(pi)
    
    if bucket_to_embedding_idx is None:
        bucket_to_embedding_idx = np.arange(K, dtype=np.int32)
    
    # Map bucket to vertex
    bucket_to_vertex = {}
    for vertex_idx in range(N):
        embedding_idx = permutation[vertex_idx]
        bucket_indices = np.where(bucket_to_embedding_idx == embedding_idx)[0]
        if len(bucket_indices) > 0:
            bucket_idx = bucket_indices[0]
            if bucket_idx < K:
                if bucket_idx not in bucket_to_vertex:
                    bucket_to_vertex[bucket_idx] = vertex_idx
    
    cost = 0.0
    for i in range(K):
        if i not in bucket_to_vertex:
            continue
        
        code_i = vertices[bucket_to_vertex[i]]
        
        for j in range(K):
            if j not in bucket_to_vertex:
                continue
            
            code_j = vertices[bucket_to_vertex[j]]
            d_h = hamming_distance(code_i[np.newaxis, :], code_j[np.newaxis, :])[0, 0]
            
            # Penalize if distance > radius
            if d_h > hamming_radius:
                penalty = (d_h - hamming_radius) ** 2
                cost += pi[i] * w[i, j] * penalty
    
    return float(cost)


def compute_recall_surrogate_cost_delta_swap(
    permutation: np.ndarray,
    pi: np.ndarray,
    w: np.ndarray,
    bucket_to_code: np.ndarray,
    n_bits: int,
    vertex_u: int,
    vertex_v: int,
    hamming_radius: int = 1,
    bucket_to_embedding_idx: Optional[np.ndarray] = None,
) -> float:
    """
    Compute change in recall surrogate cost after swapping vertices u and v.
    """
    # For efficiency, compute full cost difference
    cost_old = compute_recall_surrogate_cost(
        permutation, pi, w, bucket_to_code, n_bits, hamming_radius, bucket_to_embedding_idx
    )
    
    perm_new = permutation.copy()
    perm_new[vertex_u], perm_new[vertex_v] = perm_new[vertex_v], perm_new[vertex_u]
    
    cost_new = compute_recall_surrogate_cost(
        perm_new, pi, w, bucket_to_code, n_bits, hamming_radius, bucket_to_embedding_idx
    )
    
    return cost_new - cost_old


def compute_recall_weighted_j_phi_cost(
    permutation: np.ndarray,
    pi: np.ndarray,
    w: np.ndarray,
    bucket_to_code: np.ndarray,
    n_bits: int,
    hamming_radius: int = 1,
    bucket_to_embedding_idx: Optional[np.ndarray] = None,
    alpha: float = 0.5,
) -> float:
    """
    Compute weighted combination of J(φ) and recall surrogate.
    
    Cost = α · J(φ) + (1 - α) · Recall_Surrogate
    
    where α balances between the two objectives.
    
    Args:
        permutation: Permutation array
        pi: Query prior
        w: Neighbor weights
        bucket_to_code: Bucket codes
        n_bits: Number of bits
        hamming_radius: Hamming ball radius
        bucket_to_embedding_idx: Optional mapping
        alpha: Weight for J(φ) (0 = only recall, 1 = only J(φ))
        
    Returns:
        Combined cost
    """
    from gray_tunneled_hashing.distribution.j_phi_objective import compute_j_phi_cost
    
    j_phi = compute_j_phi_cost(
        permutation, pi, w, bucket_to_code, n_bits, bucket_to_embedding_idx, None, 0.0
    )
    
    recall_surrogate = compute_recall_surrogate_cost(
        permutation, pi, w, bucket_to_code, n_bits, hamming_radius, bucket_to_embedding_idx
    )
    
    return alpha * j_phi + (1 - alpha) * recall_surrogate

