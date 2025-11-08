"""Unified API for building and searching binary indices."""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from gray_tunneled_hashing.binary.baselines import (
    sign_binarize,
    random_projection_binarize,
    apply_random_projection,
)
from gray_tunneled_hashing.binary.codebooks import (
    build_codebook_kmeans,
    encode_with_codebook,
    find_nearest_centroids,
)
from gray_tunneled_hashing.data.synthetic_generators import generate_hypercube_vertices
from gray_tunneled_hashing.algorithms.gray_tunneled_hasher import GrayTunneledHasher
from gray_tunneled_hashing.integrations.hamming_index import (
    HammingIndex,
    build_hamming_index,
)


@dataclass
class BinaryIndex:
    """
    Binary index container.
    
    This is a simple container that holds the index and metadata.
    In a full implementation, this could be serializable and have more features.
    """
    index: HammingIndex
    method: str
    metadata: Dict[str, Any]
    
    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the index.
        
        Args:
            queries: Query embeddings or binary codes (shape depends on method)
            k: Number of neighbors to retrieve
            
        Returns:
            Tuple of (indices, distances)
        """
        # This is a placeholder - in full implementation, would handle
        # encoding queries if needed
        if self.method in ["sign", "random_proj"]:
            # Queries should already be encoded
            return self.index.search(queries, k)
        elif self.method == "gray_tunneled":
            # Would need to encode queries via codebook
            # For now, assume queries are already encoded
            return self.index.search(queries, k)
        else:
            raise ValueError(f"Unknown method: {self.method}")


def build_binary_index(
    embeddings: np.ndarray,
    method: str = "baseline",
    n_bits: int = 64,
    n_codes: Optional[int] = None,
    random_state: Optional[int] = None,
    use_faiss: bool = True,
    **kwargs,
) -> BinaryIndex:
    """
    Build a binary index from embeddings.
    
    This is a unified API that supports multiple methods:
    - "baseline" / "sign": Sign thresholding binarization
    - "baseline" / "random_proj": Random projection binarization  
    - "gray_tunneled": Gray-Tunneled Hashing with codebook
    
    Default parameters are based on Sprint 3 empirical analysis.
    See docs/DEFAULTS_AND_TUNING.md for tuning guidelines.
    
    Args:
        embeddings: Base embeddings of shape (N, dim)
        method: Method to use ("baseline", "sign", "random_proj", or "gray_tunneled")
        n_bits: Number of bits for binary codes (default: 64, recommended: 64-128)
        n_codes: Number of codebook vectors (required for "gray_tunneled", default: 512)
        random_state: Random seed for reproducibility
        use_faiss: Whether to try FAISS (fallback to Python if not available)
        **kwargs: Additional method-specific parameters:
            - For "random_proj": no additional params
            - For "gray_tunneled":
                - block_size: Block size for tunneling (default: 8, recommended: 8)
                - max_two_swap_iters: Max iterations for 2-swap (default: 50)
                - num_tunneling_steps: Number of tunneling steps (default: 10, recommended: 10)
                - mode: Optimization mode ("trivial", "two_swap_only", "full", default: "full")
                - block_selection_strategy: Block selection ("random" or "cluster", default: "random")
    
    Returns:
        BinaryIndex instance
        
    Examples:
        >>> embeddings = np.random.randn(1000, 128)
        >>> # Baseline with random projection
        >>> index = build_binary_index(
        ...     embeddings, method="random_proj", n_bits=64, random_state=42
        ... )
        >>> # Gray-Tunneled (defaults)
        >>> index = build_binary_index(
        ...     embeddings, method="gray_tunneled", n_bits=64, n_codes=512
        ... )
        >>> # Gray-Tunneled (custom)
        >>> index = build_binary_index(
        ...     embeddings, method="gray_tunneled", n_bits=128, n_codes=1024,
        ...     num_tunneling_steps=15, mode="full"
        ... )
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
    
    # Normalize method name
    if method == "baseline":
        # Default baseline to random_proj
        method = kwargs.pop("baseline_method", "random_proj")
    
    metadata = {
        "method": method,
        "n_bits": n_bits,
        "n_samples": embeddings.shape[0],
        "dim": embeddings.shape[1],
    }
    
    if method in ["sign", "random_proj"]:
        # Baseline methods
        if method == "sign":
            codes = sign_binarize(embeddings)
            projection_matrix = None
        else:  # random_proj
            codes, projection_matrix = random_projection_binarize(
                embeddings, n_bits=n_bits, random_state=random_state
            )
        
        metadata["projection_matrix"] = projection_matrix
        
        # Build Hamming index
        index = build_hamming_index(codes, use_faiss=use_faiss)
        
        return BinaryIndex(
            index=index,
            method=method,
            metadata=metadata,
        )
    
    elif method == "gray_tunneled":
        if n_codes is None:
            # Default based on Sprint 3 recommendations
            n_codes = 512
        
        if n_codes > 2 ** n_bits:
            raise ValueError(
                f"n_codes={n_codes} cannot exceed 2**n_bits={2**n_bits}"
            )
        
        # Extract Gray-Tunneled specific params (defaults from Sprint 3)
        block_size = kwargs.pop("block_size", 8)
        max_two_swap_iters = kwargs.pop("max_two_swap_iters", 50)
        num_tunneling_steps = kwargs.pop("num_tunneling_steps", 10)
        mode = kwargs.pop("mode", "full")
        block_selection_strategy = kwargs.pop("block_selection_strategy", "random")
        
        metadata.update({
            "n_codes": n_codes,
            "block_size": block_size,
            "max_two_swap_iters": max_two_swap_iters,
            "num_tunneling_steps": num_tunneling_steps,
            "mode": mode,
            "block_selection_strategy": block_selection_strategy,
        })
        
        # Build codebook
        centroids, assignments = build_codebook_kmeans(
            embeddings, n_codes=n_codes, random_state=random_state
        )
        
        # Prepare centroids for hasher (need exactly 2**n_bits)
        max_codes = 2 ** n_bits
        if n_codes < max_codes:
            n_pad = max_codes - n_codes
            padding = np.tile(centroids[-1:], (n_pad, 1))
            centroids_for_hasher = np.vstack([centroids, padding])
        else:
            centroids_for_hasher = centroids[:max_codes]
        
        # Run Gray-Tunneled optimization
        cluster_assignments = None
        if block_selection_strategy == "cluster":
            cluster_assignments = assignments
        
        hasher = GrayTunneledHasher(
            n_bits=n_bits,
            block_size=block_size,
            max_two_swap_iters=max_two_swap_iters,
            num_tunneling_steps=num_tunneling_steps if mode == "full" else 0,
            two_swap_sample_size=min(256, max_codes * (max_codes - 1) // 2),
            init_strategy="random",
            random_state=random_state + 100 if random_state is not None else None,
            mode=mode,
            track_history=False,
            block_selection_strategy=block_selection_strategy if mode == "full" else "random",
            cluster_assignments=cluster_assignments,
        )
        
        hasher.fit(centroids_for_hasher)
        
        # Create centroid-to-code mapping
        pi = hasher.get_assignment()
        vertices = generate_hypercube_vertices(n_bits)
        
        # Map centroids to codes
        centroid_to_code_map = np.zeros(n_codes, dtype=np.int32)
        for i in range(n_codes):
            vertices_with_centroid = np.where(pi == i)[0]
            if len(vertices_with_centroid) > 0:
                centroid_to_code_map[i] = vertices_with_centroid[0]
            else:
                # Fallback for padded centroids
                centroid_to_code_map[i] = i % max_codes
        
        # Convert to binary codes
        centroid_to_code = {}
        for i in range(n_codes):
            vertex_idx = centroid_to_code_map[i]
            binary_code = vertices[vertex_idx]
            centroid_to_code[i] = binary_code
        
        # Encode embeddings
        codes = encode_with_codebook(
            embeddings, centroids, centroid_to_code, assignments=assignments
        )
        
        # Store metadata for query encoding
        metadata["centroids"] = centroids
        metadata["centroid_to_code"] = centroid_to_code
        
        # Build Hamming index
        index = build_hamming_index(codes, use_faiss=use_faiss)
        
        return BinaryIndex(
            index=index,
            method=method,
            metadata=metadata,
        )
    
    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Must be one of: 'sign', 'random_proj', 'gray_tunneled'"
        )


def search_binary_index(
    index: BinaryIndex,
    queries: np.ndarray,
    k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search a binary index.
    
    Args:
        index: BinaryIndex instance
        queries: Query embeddings or binary codes
                  - For baseline methods: should be binary codes (shape Q, n_bits)
                  - For gray_tunneled: should be embeddings (shape Q, dim)
        k: Number of neighbors to retrieve
        
    Returns:
        Tuple of (indices, distances) where:
        - indices: Array of shape (Q, k) with indices of nearest neighbors
        - distances: Array of shape (Q, k) with Hamming distances
        
    Note:
        This is a simplified API. In a full implementation, queries would
        be automatically encoded based on the index method.
    """
    # For now, assume queries are already in the correct format
    # (binary codes for baseline, embeddings for gray_tunneled)
    # In a full implementation, we'd encode them here
    
    if index.method == "gray_tunneled":
        # Encode queries via codebook
        centroids = index.metadata["centroids"]
        centroid_to_code = index.metadata["centroid_to_code"]
        
        query_assignments = find_nearest_centroids(queries, centroids)
        query_codes = encode_with_codebook(
            queries, centroids, centroid_to_code, assignments=query_assignments
        )
    else:
        # Assume queries are already binary codes
        query_codes = queries
    
    indices, distances = index.index.search(query_codes, k)
    
    return indices, distances

