"""
Cosine distance-based objective functions for GTH optimization.

This module explores different objective functions that better approximate
the relationship between binary Hamming distances and cosine distances in
the original embedding space.
"""

import numpy as np
from typing import Optional, Tuple
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

from gray_tunneled_hashing.data.synthetic_generators import generate_hypercube_vertices
from gray_tunneled_hashing.evaluation.metrics import hamming_distance


def compute_cosine_distance_matrix(
    embeddings: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute cosine distance matrix between embeddings.
    
    Cosine distance = 1 - cosine_similarity
    Range: [0, 2] where 0 = identical, 2 = opposite
    
    Args:
        embeddings: Array of shape (K, dim)
        normalize: If True, normalize embeddings to unit length (default: True)
        
    Returns:
        Cosine distance matrix of shape (K, K)
    """
    if normalize:
        # Normalize embeddings to unit length
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid division by zero
        embeddings = embeddings / norms
    
    # Compute cosine distances
    cosine_dist = cosine_distances(embeddings)
    
    return cosine_dist


def compute_j_phi_cosine_cost(
    permutation: np.ndarray,
    pi: np.ndarray,
    w: np.ndarray,
    bucket_to_code: np.ndarray,
    bucket_embeddings: np.ndarray,
    n_bits: int,
    bucket_to_embedding_idx: Optional[np.ndarray] = None,
    cosine_weight: float = 1.0,
    hamming_weight: float = 1.0,
    distance_metric: str = "cosine",
) -> float:
    """
    Compute J(φ) cost using cosine distance in original space.
    
    This objective function aims to better approximate the relationship
    between binary Hamming distances and cosine distances in the original
    embedding space.
    
    J(φ) = Σ_{i,j} π_i · w_ij · [α · d_H(φ(c_i), φ(c_j)) + β · d_cosine(emb_i, emb_j)]
    
    where:
    - d_H is Hamming distance in binary space
    - d_cosine is cosine distance in original embedding space
    - α = hamming_weight, β = cosine_weight
    
    Alternative formulations:
    1. Pure cosine: J(φ) = Σ π_i · w_ij · d_cosine(emb_i, emb_j) · d_H(φ(c_i), φ(c_j))
       (penalize large Hamming distances for semantically close embeddings)
    
    2. Weighted combination: J(φ) = Σ π_i · w_ij · [α·d_H + β·d_cosine]
       (balance between binary and semantic distances)
    
    3. Correlation-based: J(φ) = Σ π_i · w_ij · |d_H(φ(c_i), φ(c_j)) - f(d_cosine(emb_i, emb_j))|
       (minimize mismatch between Hamming and cosine distances)
    
    Args:
        permutation: Permutation array of shape (N,) where permutation[vertex_idx] = embedding_idx
        pi: Query prior of shape (K,)
        w: Neighbor weights of shape (K, K)
        bucket_to_code: Original bucket codes of shape (K, n_bits)
        bucket_embeddings: Bucket embeddings of shape (K, dim)
        n_bits: Number of bits
        bucket_to_embedding_idx: Optional mapping from bucket_idx to embedding_idx
        cosine_weight: Weight for cosine distance term (default: 1.0)
        hamming_weight: Weight for Hamming distance term (default: 1.0)
        distance_metric: Which distance metric to use ("cosine", "l2", "dot_product")
        
    Returns:
        J(φ) cost
    """
    vertices = generate_hypercube_vertices(n_bits)
    N = len(vertices)
    K = len(pi)
    
    if bucket_to_embedding_idx is None:
        bucket_to_embedding_idx = np.arange(K, dtype=np.int32)
    
    # Compute cosine distance matrix
    if distance_metric == "cosine":
        cosine_dist_matrix = compute_cosine_distance_matrix(bucket_embeddings, normalize=True)
    elif distance_metric == "l2":
        # Use L2 distance instead
        from sklearn.metrics.pairwise import euclidean_distances
        cosine_dist_matrix = euclidean_distances(bucket_embeddings)
        # Normalize to [0, 2] range similar to cosine
        if cosine_dist_matrix.max() > 0:
            cosine_dist_matrix = cosine_dist_matrix / cosine_dist_matrix.max() * 2.0
    elif distance_metric == "dot_product":
        # Use negative dot product (since we want distance, not similarity)
        # Normalize embeddings first
        norms = np.linalg.norm(bucket_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_emb = bucket_embeddings / norms
        dot_products = np.dot(normalized_emb, normalized_emb.T)
        # Convert to distance: distance = 1 - similarity (for normalized vectors)
        cosine_dist_matrix = 1.0 - dot_products
    else:
        raise ValueError(f"Unknown distance_metric: {distance_metric}")
    
    # Map: bucket_idx -> vertex_idx
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
        if i in bucket_to_vertex:
            code_i = vertices[bucket_to_vertex[i]]
        else:
            code_i = bucket_to_code[i]
        
        for j in range(K):
            if j in bucket_to_vertex:
                code_j = vertices[bucket_to_vertex[j]]
            else:
                code_j = bucket_to_code[j]
            
            # Hamming distance in binary space
            d_h = hamming_distance(code_i[np.newaxis, :], code_j[np.newaxis, :])[0, 0]
            
            # Cosine distance in original embedding space
            d_cosine = cosine_dist_matrix[i, j]
            
            # Combined objective
            if distance_metric == "correlation":
                # Minimize mismatch: |d_H - f(d_cosine)|
                # f(d_cosine) = scale cosine distance to Hamming range [0, n_bits]
                f_cosine = d_cosine / 2.0 * n_bits  # Scale from [0, 2] to [0, n_bits]
                mismatch = abs(d_h - f_cosine)
                term = pi[i] * w[i, j] * mismatch
            else:
                # Weighted combination
                term = pi[i] * w[i, j] * (
                    hamming_weight * d_h + cosine_weight * d_cosine
                )
            
            cost += term
    
    return float(cost)


def compute_j_phi_cosine_cost_delta_swap(
    permutation: np.ndarray,
    pi: np.ndarray,
    w: np.ndarray,
    bucket_to_code: np.ndarray,
    bucket_embeddings: np.ndarray,
    n_bits: int,
    vertex_u: int,
    vertex_v: int,
    bucket_to_embedding_idx: Optional[np.ndarray] = None,
    cosine_weight: float = 1.0,
    hamming_weight: float = 1.0,
    distance_metric: str = "cosine",
) -> float:
    """
    Compute change in J(φ) cosine cost after swapping vertices u and v.
    
    This is more efficient than recomputing the full cost.
    """
    # For efficiency, compute full cost difference
    cost_old = compute_j_phi_cosine_cost(
        permutation, pi, w, bucket_to_code, bucket_embeddings, n_bits,
        bucket_to_embedding_idx, cosine_weight, hamming_weight, distance_metric
    )
    
    perm_new = permutation.copy()
    perm_new[vertex_u], perm_new[vertex_v] = perm_new[vertex_v], perm_new[vertex_u]
    
    cost_new = compute_j_phi_cosine_cost(
        perm_new, pi, w, bucket_to_code, bucket_embeddings, n_bits,
        bucket_to_embedding_idx, cosine_weight, hamming_weight, distance_metric
    )
    
    return cost_new - cost_old


def analyze_cosine_hamming_correlation(
    bucket_embeddings: np.ndarray,
    bucket_to_code: np.ndarray,
    n_bits: int,
) -> dict:
    """
    Analyze correlation between cosine distances and Hamming distances.
    
    This helps determine the best objective function formulation.
    
    Args:
        bucket_embeddings: Bucket embeddings of shape (K, dim)
        bucket_to_code: Bucket codes of shape (K, n_bits)
        n_bits: Number of bits
        
    Returns:
        Dictionary with correlation metrics and statistics
    """
    K = len(bucket_embeddings)
    
    # Compute cosine distances
    cosine_dist_matrix = compute_cosine_distance_matrix(bucket_embeddings, normalize=True)
    
    # Compute Hamming distances
    hamming_dist_matrix = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            d_h = hamming_distance(
                bucket_to_code[i:i+1], bucket_to_code[j:j+1]
            )[0, 0]
            hamming_dist_matrix[i, j] = d_h
    
    # Extract upper triangular (avoid duplicates)
    triu_indices = np.triu_indices(K, k=1)
    cosine_distances_flat = cosine_dist_matrix[triu_indices]
    hamming_distances_flat = hamming_dist_matrix[triu_indices]
    
    # Compute correlation
    correlation = np.corrcoef(cosine_distances_flat, hamming_distances_flat)[0, 1]
    
    # Compute statistics
    cosine_mean = np.mean(cosine_distances_flat)
    cosine_std = np.std(cosine_distances_flat)
    hamming_mean = np.mean(hamming_distances_flat)
    hamming_std = np.std(hamming_distances_flat)
    
    # Compute scaling factor to map cosine to Hamming range
    if cosine_std > 0:
        scale_factor = hamming_std / cosine_std
    else:
        scale_factor = 1.0
    
    return {
        "correlation": correlation,
        "cosine_mean": cosine_mean,
        "cosine_std": cosine_std,
        "hamming_mean": hamming_mean,
        "hamming_std": hamming_std,
        "scale_factor": scale_factor,
        "cosine_distances": cosine_distances_flat,
        "hamming_distances": hamming_distances_flat,
    }

