"""Codebook construction and encoding for Gray-Tunneled Hashing."""

import numpy as np
from typing import Optional, Tuple, Dict
from sklearn.cluster import KMeans


def build_codebook_kmeans(
    embeddings: np.ndarray,
    n_codes: int,
    random_state: Optional[int] = None,
    n_init: int = 10,
    max_iter: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build codebook using k-means clustering.
    
    Args:
        embeddings: Input embeddings of shape (N, dim)
        n_codes: Number of codebook vectors (centroids)
        random_state: Random seed for reproducibility
        n_init: Number of k-means initializations (default: 10)
        max_iter: Maximum iterations for k-means (default: 300)
        
    Returns:
        Tuple of (centroids, assignments) where:
        - centroids: Array of shape (n_codes, dim) with cluster centroids
        - assignments: Array of shape (N,) where assignments[i] is the cluster index
                      of embedding i
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
    
    if n_codes <= 0:
        raise ValueError(f"n_codes must be positive, got {n_codes}")
    
    if n_codes > embeddings.shape[0]:
        raise ValueError(
            f"n_codes={n_codes} cannot exceed number of embeddings={embeddings.shape[0]}"
        )
    
    # Run k-means
    kmeans = KMeans(
        n_clusters=n_codes,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        init="k-means++",
    )
    
    assignments = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_
    
    return centroids.astype(np.float32), assignments.astype(np.int32)


def find_nearest_centroids(
    embeddings: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """
    Find nearest centroid for each embedding.
    
    Args:
        embeddings: Query embeddings of shape (Q, dim)
        centroids: Codebook centroids of shape (n_codes, dim)
        
    Returns:
        Array of shape (Q,) with centroid indices for each embedding
    """
    if embeddings.ndim != 2 or centroids.ndim != 2:
        raise ValueError("Both embeddings and centroids must be 2D")
    
    if embeddings.shape[1] != centroids.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: {embeddings.shape[1]} vs {centroids.shape[1]}"
        )
    
    # Compute squared L2 distances
    # distances[i, j] = ||embeddings[i] - centroids[j]||^2
    distances = np.zeros((embeddings.shape[0], centroids.shape[0]))
    
    for i, emb in enumerate(embeddings):
        dists = np.sum((centroids - emb) ** 2, axis=1)
        distances[i] = dists
    
    # Find nearest centroid for each embedding
    nearest_indices = np.argmin(distances, axis=1)
    
    return nearest_indices.astype(np.int32)


def encode_with_codebook(
    embeddings: np.ndarray,
    centroids: np.ndarray,
    centroid_to_code: Dict[int, np.ndarray],
    assignments: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Encode embeddings to binary codes via codebook.
    
    For each embedding:
    1. Find nearest centroid (or use precomputed assignment)
    2. Look up binary code for that centroid
    3. Return the binary code
    
    Args:
        embeddings: Embeddings to encode of shape (N, dim)
        centroids: Codebook centroids of shape (n_codes, dim)
        centroid_to_code: Dictionary mapping centroid index to binary code (bool array)
        assignments: Optional precomputed assignments (if None, computes nearest centroids)
        
    Returns:
        Binary codes of shape (N, n_bits) with dtype bool
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
    
    if centroids.ndim != 2:
        raise ValueError(f"Expected 2D centroids, got shape {centroids.shape}")
    
    N = embeddings.shape[0]
    
    # Find nearest centroids
    if assignments is not None:
        if len(assignments) != N:
            raise ValueError(
                f"Assignments length {len(assignments)} != number of embeddings {N}"
            )
        centroid_indices = assignments
    else:
        centroid_indices = find_nearest_centroids(embeddings, centroids)
    
    # Get code length from first code
    if len(centroid_to_code) == 0:
        raise ValueError("centroid_to_code dictionary is empty")
    
    first_code = next(iter(centroid_to_code.values()))
    n_bits = len(first_code)
    
    # Encode each embedding
    codes = np.zeros((N, n_bits), dtype=bool)
    
    for i, centroid_idx in enumerate(centroid_indices):
        if centroid_idx not in centroid_to_code:
            raise ValueError(
                f"Centroid index {centroid_idx} not found in centroid_to_code mapping"
            )
        codes[i] = centroid_to_code[centroid_idx]
    
    return codes

