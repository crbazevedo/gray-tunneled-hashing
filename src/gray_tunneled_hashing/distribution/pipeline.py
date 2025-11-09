"""Distribution-aware GTH integration pipeline."""

import numpy as np
from typing import Callable, Dict, Optional, Tuple, Any
from dataclasses import dataclass

from gray_tunneled_hashing.distribution.traffic_stats import (
    collect_traffic_stats,
    build_weighted_distance_matrix,
)
from gray_tunneled_hashing.algorithms.gray_tunneled_hasher import GrayTunneledHasher
from gray_tunneled_hashing.binary.lsh_families import LSHFamily


@dataclass
class DistributionAwareIndex:
    """Container for distribution-aware index components."""
    
    hasher: GrayTunneledHasher
    bucket_to_code: np.ndarray  # Original bucket codes (K, n_bits)
    code_to_bucket: Dict  # Mapping from code (tuple) to bucket_idx
    bucket_embeddings: np.ndarray  # Bucket representative embeddings (K, dim)
    pi: np.ndarray  # Query prior (K,)
    w: np.ndarray  # Neighbor weights (K, K)
    permutation: np.ndarray  # Final permutation from hasher
    n_bits: int
    K: int  # Number of buckets


def apply_permutation(
    codes: np.ndarray,
    bucket_to_code: np.ndarray,
    code_to_bucket: Dict,
    permutation: np.ndarray,
    n_bits: int,
) -> np.ndarray:
    """
    Apply the learned permutation to bucket codes.
    
    Args:
        codes: Original bucket codes of shape (N, n_bits)
        bucket_to_code: Mapping from bucket_idx to original code (K, n_bits)
        code_to_bucket: Mapping from code (tuple) to bucket_idx
        permutation: Learned permutation from hasher (N,)
        n_bits: Number of bits
        
    Returns:
        Permuted codes of shape (N, n_bits)
    """
    # Generate hypercube vertices (all possible codes)
    from gray_tunneled_hashing.data.synthetic_generators import generate_hypercube_vertices
    
    vertices = generate_hypercube_vertices(n_bits)  # (2**n_bits, n_bits)
    N = len(vertices)
    
    # Map each code to its bucket, then to its permuted vertex
    permuted_codes = np.zeros_like(codes)
    
    for i, code in enumerate(codes):
        code_tuple = tuple(code.astype(int).tolist())
        
        if code_tuple in code_to_bucket:
            bucket_idx = code_to_bucket[code_tuple]
            # Find which vertex this bucket is assigned to after permutation
            # permutation[vertex_idx] = bucket_idx means vertex vertex_idx is assigned to bucket bucket_idx
            # We need inverse: which vertex is assigned to this bucket?
            # Actually, pi[u] = bucket_idx means vertex u is assigned to bucket bucket_idx
            # So we need to find u such that permutation[u] == bucket_idx
            vertex_idx = np.where(permutation == bucket_idx)[0]
            if len(vertex_idx) > 0:
                vertex_idx = vertex_idx[0]
                permuted_codes[i] = vertices[vertex_idx]
            else:
                # Fallback: use original code
                permuted_codes[i] = code
        else:
            # Code not in buckets, keep original
            permuted_codes[i] = code
    
    return permuted_codes


def build_distribution_aware_index(
    base_embeddings: np.ndarray,
    queries: np.ndarray,
    ground_truth_neighbors: np.ndarray,
    encoder: Callable,
    n_bits: int,
    n_codes: Optional[int] = None,
    use_codebook: bool = True,
    use_semantic_distances: bool = True,
    block_size: int = 8,
    max_two_swap_iters: int = 100,
    num_tunneling_steps: int = 10,
    mode: str = "full",
    random_state: Optional[int] = None,
    collapse_threshold: float = 0.01,
    lsh_family: Optional[Any] = None,
) -> DistributionAwareIndex:
    """
    Build a distribution-aware Gray-Tunneled Hashing index.
    
    Pipeline:
    1. Apply base encoder to get bucket codes
    2. Collect traffic stats (pi, w) from queries + neighbors
    3. Get bucket representatives (centroids or embeddings)
    4. Build weighted distance matrix D_weighted
    5. Run GrayTunneledHasher with D_weighted
    6. Store permutation and metadata
    
    Args:
        base_embeddings: Base corpus embeddings of shape (N, dim)
        queries: Query embeddings of shape (Q, dim)
        ground_truth_neighbors: Ground truth neighbor indices of shape (Q, k)
        encoder: Function that maps embeddings to bucket codes
        n_bits: Number of bits for binary codes
        n_codes: Number of codebook centroids (if use_codebook=True). If None, uses 2**n_bits
        use_codebook: If True, use k-means codebook; if False, use all embeddings
        use_semantic_distances: If True, include semantic distances in D_weighted
        block_size: Block size for tunneling
        max_two_swap_iters: Max iterations for 2-swap hill climbing
        num_tunneling_steps: Number of tunneling steps
        mode: Optimization mode ("trivial", "two_swap_only", "full")
        random_state: Random seed
        collapse_threshold: Threshold for collapsing low-traffic buckets
        
    Returns:
        DistributionAwareIndex with all components
    """
    # Step 1: Collect traffic statistics
    # If lsh_family is provided, use it as encoder
    actual_encoder = encoder
    if lsh_family is not None:
        if not isinstance(lsh_family, LSHFamily):
            raise TypeError(
                f"lsh_family must be an LSHFamily instance, got {type(lsh_family)}"
            )
        # Create encoder function from LSH family
        actual_encoder = lambda emb: lsh_family.hash(emb)
    
    traffic_stats = collect_traffic_stats(
        queries=queries,
        ground_truth_neighbors=ground_truth_neighbors,
        base_embeddings=base_embeddings,
        encoder=actual_encoder,
        collapse_threshold=collapse_threshold,
    )
    
    pi = traffic_stats["pi"]
    w = traffic_stats["w"]
    bucket_to_code = traffic_stats["bucket_to_code"]
    code_to_bucket = traffic_stats["code_to_bucket"]
    K = traffic_stats["K"]
    n_bits_actual = traffic_stats["n_bits"]
    
    if n_bits_actual != n_bits:
        raise ValueError(
            f"Encoder produced {n_bits_actual}-bit codes, but n_bits={n_bits} was requested"
        )
    
    # Step 2: Get bucket representative embeddings
    if use_codebook:
        # Use codebook centroids
        from gray_tunneled_hashing.binary.codebooks import build_codebook_kmeans
        
        if n_codes is None:
            n_codes = 2 ** n_bits
        
        # Build codebook from base embeddings
        codebook, assignments = build_codebook_kmeans(
            embeddings=base_embeddings,
            n_codes=n_codes,
            random_state=random_state,
        )
        
        # Map buckets to codebook centroids
        # For each bucket, find which codebook centroid it corresponds to
        # We'll use the centroid that's closest to the average embedding in that bucket
        bucket_embeddings = np.zeros((K, base_embeddings.shape[1]), dtype=base_embeddings.dtype)
        
        # Encode base embeddings to get their bucket codes
        base_codes = actual_encoder(base_embeddings)
        
        for bucket_idx in range(K):
            bucket_code = bucket_to_code[bucket_idx]
            bucket_code_tuple = tuple(bucket_code.astype(int).tolist())
            
            # Find all base embeddings in this bucket
            bucket_mask = np.array([
                tuple(code.astype(int).tolist()) == bucket_code_tuple
                for code in base_codes
            ])
            
            if bucket_mask.sum() > 0:
                # Use mean embedding of points in this bucket
                bucket_embeddings[bucket_idx] = base_embeddings[bucket_mask].mean(axis=0)
            else:
                # No points in bucket, use closest centroid
                bucket_embeddings[bucket_idx] = codebook[0]
    else:
        # Use all embeddings (not recommended for large N)
        # For each bucket, use mean embedding
        base_codes = actual_encoder(base_embeddings)
        bucket_embeddings = np.zeros((K, base_embeddings.shape[1]), dtype=base_embeddings.dtype)
        
        for bucket_idx in range(K):
            bucket_code = bucket_to_code[bucket_idx]
            bucket_code_tuple = tuple(bucket_code.astype(int).tolist())
            
            bucket_mask = np.array([
                tuple(code.astype(int).tolist()) == bucket_code_tuple
                for code in base_codes
            ])
            
            if bucket_mask.sum() > 0:
                bucket_embeddings[bucket_idx] = base_embeddings[bucket_mask].mean(axis=0)
            else:
                # Fallback: use random embedding
                bucket_embeddings[bucket_idx] = base_embeddings[0]
    
    # Step 3: Build weighted distance matrix
    D_weighted = build_weighted_distance_matrix(
        pi=pi,
        w=w,
        bucket_embeddings=bucket_embeddings,
        use_semantic_distances=use_semantic_distances,
    )
    
    # Step 4: Run GrayTunneledHasher
    hasher = GrayTunneledHasher(
        n_bits=n_bits,
        block_size=block_size,
        max_two_swap_iters=max_two_swap_iters,
        num_tunneling_steps=num_tunneling_steps,
        mode=mode,
        random_state=random_state,
    )
    
    # Store bucket_to_code in hasher before fitting (needed for J(φ) optimization)
    hasher.bucket_to_code_ = bucket_to_code
    
    # Fit with weighted distance matrix (use direct J(φ) optimization for guarantee)
    # NEW (Sprint 8): Pass queries, base_embeddings, ground_truth_neighbors, encoder, code_to_bucket
    # for real embeddings objective
    hasher.fit_with_traffic(
        bucket_embeddings=bucket_embeddings,
        pi=pi,
        w=w,
        queries=queries,
        base_embeddings=base_embeddings,
        ground_truth_neighbors=ground_truth_neighbors,
        encoder=actual_encoder,
        code_to_bucket=code_to_bucket,
        use_semantic_distances=use_semantic_distances,
        optimize_j_phi_directly=True,  # Use direct optimization to guarantee J(φ*) ≤ J(φ₀)
        use_real_embeddings_objective=True,  # NEW: Use real embeddings objective by default
    )
    
    # Store traffic stats for later use
    hasher.pi_traffic_ = pi
    hasher.w_traffic_ = w
    
    permutation = hasher.get_assignment()
    
    return DistributionAwareIndex(
        hasher=hasher,
        bucket_to_code=bucket_to_code,
        code_to_bucket=code_to_bucket,
        bucket_embeddings=bucket_embeddings,
        pi=pi,
        w=w,
        permutation=permutation,
        n_bits=n_bits,
        K=K,
    )


