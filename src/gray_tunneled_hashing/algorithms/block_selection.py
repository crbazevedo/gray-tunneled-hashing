"""Block selection strategies for tunneling moves."""

import numpy as np
from typing import Optional, Literal, Callable


def select_block_random(
    N: int,
    block_size: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Select a random block of vertices.
    
    This is a uniform random selection of block_size distinct vertices.
    
    Args:
        N: Total number of vertices
        block_size: Size of the block to select
        random_state: Random seed for reproducibility
        
    Returns:
        Sorted array of block_size distinct vertex indices
        
    Examples:
        >>> block = select_block_random(16, 4, random_state=42)
        >>> len(block)
        4
        >>> np.all(block < 16)
        True
    """
    if block_size > N:
        raise ValueError(f"block_size {block_size} cannot exceed N {N}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    block = np.sort(np.random.choice(N, size=block_size, replace=False))
    return block.astype(np.int32)


def select_block_by_embedding_cluster(
    pi: np.ndarray,
    cluster_assignments: np.ndarray,
    block_size: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Select a block based on embedding clusters.
    
    This strategy:
    1. Selects a random cluster (weighted by cluster size)
    2. Within that cluster, selects block_size items
    3. Maps those items to vertex indices via permutation pi
    
    Args:
        pi: Permutation array of shape (N,) where pi[u] is embedding index at vertex u
        cluster_assignments: Array of shape (M,) where M is number of embeddings,
                            cluster_assignments[i] is cluster ID of embedding i
        block_size: Size of the block to select
        random_state: Random seed for reproducibility
        
    Returns:
        Sorted array of block_size distinct vertex indices
        
    Raises:
        ValueError: If cluster_assignments doesn't match pi, or if cluster is too small
    """
    if len(pi) != len(cluster_assignments):
        raise ValueError(
            f"pi length {len(pi)} != cluster_assignments length {len(cluster_assignments)}"
        )
    
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # Get unique clusters and their sizes
    unique_clusters = np.unique(cluster_assignments)
    cluster_sizes = {
        cluster_id: np.sum(cluster_assignments == cluster_id)
        for cluster_id in unique_clusters
    }
    
    # Filter clusters that are large enough
    valid_clusters = [
        cid for cid, size in cluster_sizes.items() if size >= block_size
    ]
    
    if len(valid_clusters) == 0:
        # Fallback to random if no cluster is large enough
        return select_block_random(len(pi), block_size, random_state=random_state)
    
    # Select a random cluster (uniform over valid clusters)
    selected_cluster = np.random.choice(valid_clusters)
    
    # Find embeddings in this cluster
    cluster_embedding_indices = np.where(cluster_assignments == selected_cluster)[0]
    
    # Randomly select block_size embeddings from cluster
    if len(cluster_embedding_indices) > block_size:
        selected_embeddings = np.random.choice(
            cluster_embedding_indices, size=block_size, replace=False
        )
    else:
        selected_embeddings = cluster_embedding_indices
    
    # Map embeddings to vertices via pi
    # pi[u] = embedding index, so we need inverse: pi_inv[embedding_idx] = vertex_idx
    # For each selected embedding, find vertex where pi[vertex] == embedding
    vertex_indices = []
    for emb_idx in selected_embeddings:
        vertices_with_emb = np.where(pi == emb_idx)[0]
        if len(vertices_with_emb) > 0:
            vertex_indices.append(vertices_with_emb[0])
    
    if len(vertex_indices) < block_size:
        # If we couldn't find enough vertices, pad with random
        all_vertices = np.arange(len(pi))
        remaining = block_size - len(vertex_indices)
        additional = np.random.choice(
            all_vertices, size=remaining, replace=False
        )
        vertex_indices.extend(additional.tolist())
    
    return np.sort(np.array(vertex_indices[:block_size], dtype=np.int32))


def get_block_selection_fn(
    strategy: Literal["random", "cluster"],
    pi: Optional[np.ndarray] = None,
    cluster_assignments: Optional[np.ndarray] = None,
) -> Callable:
    """
    Get a block selection function based on strategy.
    
    Args:
        strategy: Block selection strategy ("random" or "cluster")
        pi: Permutation array (required for "cluster" strategy)
        cluster_assignments: Cluster assignments (required for "cluster" strategy)
        
    Returns:
        Function that takes (N, block_size, random_state) and returns block indices
        
    Raises:
        ValueError: If strategy is invalid or required args missing
    """
    if strategy == "random":
        return lambda N, block_size, random_state=None: select_block_random(
            N, block_size, random_state
        )
    elif strategy == "cluster":
        if pi is None or cluster_assignments is None:
            raise ValueError(
                "pi and cluster_assignments are required for cluster strategy"
            )
        
        return lambda N, block_size, random_state=None: select_block_by_embedding_cluster(
            pi, cluster_assignments, block_size, random_state
        )
    else:
        raise ValueError(f"Unknown block selection strategy: {strategy}")

