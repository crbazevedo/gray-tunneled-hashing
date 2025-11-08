"""Traffic statistics collection for distribution-aware GTH."""

import numpy as np
from typing import Callable, Dict, Tuple, Optional
from collections import Counter, defaultdict


def collect_query_neighbor_pairs(
    queries: np.ndarray,
    ground_truth_neighbors: np.ndarray,
    encoder: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect query-neighbor bucket pairs from traffic.
    
    Args:
        queries: Query embeddings of shape (Q, dim)
        ground_truth_neighbors: Ground truth neighbor indices of shape (Q, k)
        encoder: Function that maps embeddings to bucket codes (shape (N, dim) -> (N, n_bits))
        
    Returns:
        Tuple of (query_buckets, neighbor_buckets) where:
        - query_buckets: Array of shape (Q, n_bits) with query bucket codes
        - neighbor_buckets: Array of shape (Q, k, n_bits) with neighbor bucket codes
    """
    if queries.ndim != 2:
        raise ValueError(f"Expected 2D queries, got shape {queries.shape}")
    if ground_truth_neighbors.ndim != 2:
        raise ValueError(
            f"Expected 2D ground_truth_neighbors, got shape {ground_truth_neighbors.shape}"
        )
    if queries.shape[0] != ground_truth_neighbors.shape[0]:
        raise ValueError(
            f"Mismatch: {queries.shape[0]} queries vs {ground_truth_neighbors.shape[0]} neighbor rows"
        )
    
    # Encode queries
    query_buckets = encoder(queries)
    
    # For neighbors, we need the actual embeddings, not just indices
    # This function assumes neighbors are provided as indices into a base corpus
    # We'll need to handle this differently - for now, return query buckets only
    # and note that neighbor buckets need to be computed separately
    
    return query_buckets, None  # Will be handled by collect_traffic_stats


def collect_traffic_stats(
    queries: np.ndarray,
    ground_truth_neighbors: np.ndarray,
    base_embeddings: np.ndarray,
    encoder: Callable,
    k: Optional[int] = None,
    collapse_threshold: float = 0.01,
) -> Dict[str, np.ndarray]:
    """
    Collect traffic statistics from query-neighbor pairs.
    
    This is the main function that estimates:
    - π_i: Query prior (fraction of queries in bucket i)
    - w_ij: Neighbor weight (fraction of query-neighbor pairs where query in i, neighbor in j)
    
    Args:
        queries: Query embeddings of shape (Q, dim)
        ground_truth_neighbors: Ground truth neighbor indices of shape (Q, k_max)
        base_embeddings: Base corpus embeddings of shape (N, dim)
        encoder: Function that maps embeddings to bucket codes
        k: Number of neighbors to use (if None, uses all available)
        collapse_threshold: Threshold for collapsing low-traffic buckets (default: 0.01)
        
    Returns:
        Dictionary with:
        - 'pi': Query prior array of shape (K,) where K is number of non-empty buckets
        - 'w': Neighbor weights array of shape (K, K)
        - 'bucket_to_code': Mapping from bucket_idx to original code (K, n_bits)
        - 'code_to_bucket': Mapping from code (as tuple) to bucket_idx
        - 'bucket_counts': Count of queries per bucket
    """
    if queries.ndim != 2:
        raise ValueError(f"Expected 2D queries, got shape {queries.shape}")
    if ground_truth_neighbors.ndim != 2:
        raise ValueError(
            f"Expected 2D ground_truth_neighbors, got shape {ground_truth_neighbors.shape}"
        )
    if base_embeddings.ndim != 2:
        raise ValueError(f"Expected 2D base_embeddings, got shape {base_embeddings.shape}")
    
    # Limit to k neighbors if specified
    if k is not None:
        ground_truth_neighbors = ground_truth_neighbors[:, :k]
    
    Q = queries.shape[0]
    k_actual = ground_truth_neighbors.shape[1]
    
    # Encode queries
    query_codes = encoder(queries)
    n_bits = query_codes.shape[1]
    
    # Encode base embeddings (for neighbors)
    base_codes = encoder(base_embeddings)
    
    # Convert codes to hashable format (tuples) for counting
    def code_to_tuple(code):
        if isinstance(code, np.ndarray):
            return tuple(code.astype(int).tolist())
        return tuple(code)
    
    # Count query buckets
    query_bucket_counts = Counter()
    for q_code in query_codes:
        bucket_key = code_to_tuple(q_code)
        query_bucket_counts[bucket_key] += 1
    
    # Count neighbor buckets for each query
    neighbor_pair_counts = defaultdict(int)  # (query_bucket, neighbor_bucket) -> count
    
    for q_idx in range(Q):
        q_code = query_codes[q_idx]
        q_bucket = code_to_tuple(q_code)
        
        # Get neighbor embeddings
        neighbor_indices = ground_truth_neighbors[q_idx]
        for n_idx in neighbor_indices:
            if n_idx < base_embeddings.shape[0]:
                n_code = base_codes[n_idx]
                n_bucket = code_to_tuple(n_code)
                neighbor_pair_counts[(q_bucket, n_bucket)] += 1
    
    # Get all unique buckets
    all_buckets = set(query_bucket_counts.keys())
    for (q_b, n_b) in neighbor_pair_counts.keys():
        all_buckets.add(q_b)
        all_buckets.add(n_b)
    
    # Filter by query traffic (collapse low-traffic buckets)
    total_queries = sum(query_bucket_counts.values())
    significant_buckets = {
        bucket
        for bucket, count in query_bucket_counts.items()
        if count / total_queries >= collapse_threshold
    }
    
    # Add buckets that appear as neighbors even if low query traffic
    for (q_b, n_b) in neighbor_pair_counts.keys():
        if q_b in significant_buckets:
            significant_buckets.add(n_b)
    
    # Create mapping: bucket (tuple) -> bucket_idx
    bucket_list = sorted(significant_buckets)
    code_to_bucket = {bucket: idx for idx, bucket in enumerate(bucket_list)}
    K = len(bucket_list)
    
    # Estimate π_i (query prior)
    pi = np.zeros(K, dtype=np.float64)
    bucket_counts = np.zeros(K, dtype=np.int32)
    
    for bucket, count in query_bucket_counts.items():
        if bucket in code_to_bucket:
            bucket_idx = code_to_bucket[bucket]
            pi[bucket_idx] = count / total_queries
            bucket_counts[bucket_idx] = count
    
    # Normalize pi (should sum to 1, but may be < 1 due to collapsed buckets)
    pi_sum = pi.sum()
    if pi_sum > 0:
        pi = pi / pi_sum
    
    # Estimate w_ij (neighbor weights)
    w = np.zeros((K, K), dtype=np.float64)
    total_pairs = sum(neighbor_pair_counts.values())
    
    if total_pairs > 0:
        for (q_bucket, n_bucket), count in neighbor_pair_counts.items():
            if q_bucket in code_to_bucket and n_bucket in code_to_bucket:
                i = code_to_bucket[q_bucket]
                j = code_to_bucket[n_bucket]
                w[i, j] = count / total_pairs
    
    # Normalize w (each row should sum to probability distribution)
    row_sums = w.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    w = w / row_sums
    
    # Add small uniform prior to avoid exact zeros (helps optimization)
    # This ensures D_weighted has gradient information everywhere
    uniform_prior = 1e-6 / K
    w = w * (1 - uniform_prior * K) + uniform_prior
    # Renormalize
    w = w / w.sum(axis=1, keepdims=True)
    
    # Create bucket_to_code mapping
    bucket_to_code = np.array([list(bucket) for bucket in bucket_list], dtype=np.uint8)
    
    return {
        "pi": pi,
        "w": w,
        "bucket_to_code": bucket_to_code,
        "code_to_bucket": code_to_bucket,
        "bucket_counts": bucket_counts,
        "n_bits": n_bits,
        "K": K,
    }


def estimate_bucket_mass(query_buckets: np.ndarray) -> np.ndarray:
    """
    Estimate query prior π_i from query buckets.
    
    Args:
        query_buckets: Query bucket codes of shape (Q, n_bits)
        
    Returns:
        Array of shape (K,) with query prior, where K is number of unique buckets
    """
    if query_buckets.ndim != 2:
        raise ValueError(f"Expected 2D query_buckets, got shape {query_buckets.shape}")
    
    Q = query_buckets.shape[0]
    
    # Count buckets
    bucket_counts = Counter()
    for q_code in query_buckets:
        bucket_key = tuple(q_code.astype(int).tolist())
        bucket_counts[bucket_key] += 1
    
    # Convert to array
    unique_buckets = sorted(bucket_counts.keys())
    K = len(unique_buckets)
    pi = np.zeros(K, dtype=np.float64)
    
    for idx, bucket in enumerate(unique_buckets):
        pi[idx] = bucket_counts[bucket] / Q
    
    return pi


def estimate_neighbor_weights(
    query_buckets: np.ndarray,
    neighbor_buckets: np.ndarray,
) -> np.ndarray:
    """
    Estimate neighbor weights w_ij from query-neighbor bucket pairs.
    
    Args:
        query_buckets: Query bucket codes of shape (Q, n_bits)
        neighbor_buckets: Neighbor bucket codes of shape (Q, k, n_bits)
        
    Returns:
        Array of shape (K, K) with neighbor weights, where K is number of unique buckets
    """
    if query_buckets.ndim != 2:
        raise ValueError(f"Expected 2D query_buckets, got shape {query_buckets.shape}")
    if neighbor_buckets.ndim != 3:
        raise ValueError(
            f"Expected 3D neighbor_buckets, got shape {neighbor_buckets.shape}"
        )
    
    Q, k = neighbor_buckets.shape[:2]
    
    # Count pairs
    pair_counts = defaultdict(int)
    
    for q_idx in range(Q):
        q_bucket = tuple(query_buckets[q_idx].astype(int).tolist())
        for n_idx in range(k):
            n_bucket = tuple(neighbor_buckets[q_idx, n_idx].astype(int).tolist())
            pair_counts[(q_bucket, n_bucket)] += 1
    
    # Get unique buckets
    all_buckets = set()
    for (q_b, n_b) in pair_counts.keys():
        all_buckets.add(q_b)
        all_buckets.add(n_b)
    
    unique_buckets = sorted(all_buckets)
    K = len(unique_buckets)
    bucket_to_idx = {bucket: idx for idx, bucket in enumerate(unique_buckets)}
    
    # Build w matrix
    w = np.zeros((K, K), dtype=np.float64)
    total_pairs = sum(pair_counts.values())
    
    if total_pairs > 0:
        for (q_bucket, n_bucket), count in pair_counts.items():
            i = bucket_to_idx[q_bucket]
            j = bucket_to_idx[n_bucket]
            w[i, j] = count / total_pairs
    
    # Normalize rows
    row_sums = w.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    w = w / row_sums
    
    return w


def build_weighted_distance_matrix(
    pi: np.ndarray,
    w: np.ndarray,
    bucket_embeddings: np.ndarray,
    use_semantic_distances: bool = True,
) -> np.ndarray:
    """
    Build weighted distance matrix for hypercube QAP.
    
    Creates D_weighted where:
    - If use_semantic_distances=True: D_weighted[i,j] = π_i · w_ij · ||emb_i - emb_j||²
    - If use_semantic_distances=False: D_weighted[i,j] = π_i · w_ij
    
    This matrix is used in the standard QAP objective:
    f(π) = Σ_{(u,v) ∈ edges} D_weighted[π(u), π(v)]
    
    Args:
        pi: Query prior of shape (K,)
        w: Neighbor weights of shape (K, K)
        bucket_embeddings: Bucket representative embeddings of shape (K, dim)
        use_semantic_distances: If True, multiply by semantic distance (default: True)
        
    Returns:
        Weighted distance matrix of shape (K, K)
    """
    if pi.ndim != 1:
        raise ValueError(f"Expected 1D pi, got shape {pi.shape}")
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError(f"Expected square w, got shape {w.shape}")
    if pi.shape[0] != w.shape[0]:
        raise ValueError(f"Shape mismatch: pi={pi.shape}, w={w.shape}")
    if bucket_embeddings.ndim != 2:
        raise ValueError(f"Expected 2D bucket_embeddings, got shape {bucket_embeddings.shape}")
    if bucket_embeddings.shape[0] != pi.shape[0]:
        raise ValueError(
            f"Shape mismatch: bucket_embeddings={bucket_embeddings.shape}, pi={pi.shape}"
        )
    
    K = len(pi)
    D_weighted = np.zeros((K, K), dtype=np.float64)
    
    if use_semantic_distances:
        # Compute semantic distances
        for i in range(K):
            for j in range(K):
                # Squared L2 distance
                d_semantic = np.linalg.norm(bucket_embeddings[i] - bucket_embeddings[j]) ** 2
                # Weighted: π_i · w_ij · d_semantic
                # Add small epsilon to avoid exact zeros (helps optimization)
                D_weighted[i, j] = pi[i] * w[i, j] * d_semantic + 1e-10
    else:
        # Pure traffic-based weights
        # Add small epsilon to avoid exact zeros
        for i in range(K):
            for j in range(K):
                D_weighted[i, j] = pi[i] * w[i, j] + 1e-10
    
    return D_weighted

