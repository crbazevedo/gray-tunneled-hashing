"""
Fix the J(φ) calculation bug.

Root cause identified:
1. When K < 2**n_bits, we pad D_weighted and embeddings
2. Permutation maps vertices (0..N-1) to embedding indices (0..N-1, including padded)
3. But J(φ) needs to map buckets (0..K-1) to codes
4. The mapping bucket_idx -> vertex_idx is lost after padding

Solution:
- Store bucket-to-vertex mapping explicitly
- Use this mapping when computing J(φ)
"""

import sys
from pathlib import Path
from typing import Optional
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gray_tunneled_hashing.binary.baselines import random_projection_binarize
from gray_tunneled_hashing.binary.codebooks import build_codebook_kmeans
from gray_tunneled_hashing.algorithms.gray_tunneled_hasher import GrayTunneledHasher
from gray_tunneled_hashing.distribution.traffic_stats import (
    collect_traffic_stats,
    build_weighted_distance_matrix,
)
from gray_tunneled_hashing.data.synthetic_generators import generate_hypercube_vertices
from gray_tunneled_hashing.evaluation.metrics import hamming_distance


class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def compute_j_phi_fixed(
    pi: np.ndarray,
    w: np.ndarray,
    bucket_to_code: np.ndarray,
    permutation: np.ndarray,
    n_bits: int,
    bucket_to_embedding_idx: Optional[np.ndarray] = None,
) -> float:
    """
    Compute J(φ) with correct bucket-to-vertex mapping.
    
    Args:
        bucket_to_embedding_idx: Optional mapping from bucket_idx to embedding_idx in padded space
                                If None, assumes bucket_idx == embedding_idx for first K buckets
    """
    vertices = generate_hypercube_vertices(n_bits)
    N = len(vertices)
    K = len(pi)
    
    # Map: bucket_idx -> vertex_idx
    # permutation[vertex_idx] = embedding_idx means vertex vertex_idx is assigned to embedding embedding_idx
    # We need: for bucket i, which vertex is it assigned to?
    
    if bucket_to_embedding_idx is None:
        # Assume bucket i maps to embedding i (for first K buckets)
        bucket_to_embedding_idx = np.arange(K, dtype=np.int32)
    
    bucket_to_vertex = {}
    for vertex_idx in range(N):
        embedding_idx = permutation[vertex_idx]
        # Find which bucket this embedding corresponds to
        bucket_idx = np.where(bucket_to_embedding_idx == embedding_idx)[0]
        if len(bucket_idx) > 0:
            bucket_idx = bucket_idx[0]
            if bucket_idx < K:
                if bucket_idx not in bucket_to_vertex:
                    bucket_to_vertex[bucket_idx] = vertex_idx
    
    cost = 0.0
    for i in range(K):
        if i in bucket_to_vertex:
            vertex_i = bucket_to_vertex[i]
            code_i = vertices[vertex_i]
        else:
            # Fallback: use original code
            code_i = bucket_to_code[i]
        
        for j in range(K):
            if j in bucket_to_vertex:
                vertex_j = bucket_to_vertex[j]
                code_j = vertices[vertex_j]
            else:
                code_j = bucket_to_code[j]
            
            d_h = hamming_distance(code_i[np.newaxis, :], code_j[np.newaxis, :])[0, 0]
            cost += pi[i] * w[i, j] * d_h
    
    return cost


def test_fix():
    """Test the fix with a simple case."""
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}TESTING FIX FOR J(φ) CALCULATION{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    
    n_bits = 6
    K = 32  # K < 2**6 = 64
    dim = 16
    
    np.random.seed(42)
    
    # Generate synthetic data
    N = 200
    Q = 50
    k = 5
    
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    ground_truth = np.random.randint(0, N, size=(Q, k), dtype=np.int32)
    
    # Create encoder
    _, proj_matrix = random_projection_binarize(
        base_embeddings,
        n_bits=n_bits,
        random_state=42,
    )
    
    def encoder_fn(emb):
        from gray_tunneled_hashing.binary.baselines import apply_random_projection
        proj = apply_random_projection(emb, proj_matrix)
        return (proj > 0).astype(bool)
    
    # Collect traffic stats
    print(f"{Colors.BLUE}Collecting traffic statistics...{Colors.END}")
    traffic_stats = collect_traffic_stats(
        queries=queries,
        ground_truth_neighbors=ground_truth,
        base_embeddings=base_embeddings,
        encoder=encoder_fn,
    )
    
    pi = traffic_stats["pi"]
    w = traffic_stats["w"]
    bucket_to_code = traffic_stats["bucket_to_code"]
    K_actual = traffic_stats["K"]
    
    print(f"  K (buckets): {K_actual}")
    
    # Build codebook
    centroids, _ = build_codebook_kmeans(
        embeddings=base_embeddings,
        n_codes=K_actual,
        random_state=42,
    )
    
    bucket_embeddings = centroids
    
    # Build D_weighted
    D_weighted = build_weighted_distance_matrix(
        pi=pi,
        w=w,
        bucket_embeddings=bucket_embeddings,
        use_semantic_distances=True,
    )
    
    # Compute J(φ₀) - direct method
    j_phi_0_direct = 0.0
    for i in range(K_actual):
        code_i = bucket_to_code[i]
        for j in range(K_actual):
            code_j = bucket_to_code[j]
            d_h = hamming_distance(code_i[np.newaxis, :], code_j[np.newaxis, :])[0, 0]
            j_phi_0_direct += pi[i] * w[i, j] * d_h
    
    print(f"\n{Colors.BLUE}J(φ₀) (direct):{Colors.END} {j_phi_0_direct:.6f}")
    
    # Fit hasher with padding
    N_hypercube = 2 ** n_bits
    if K_actual < N_hypercube:
        D_padded = np.zeros((N_hypercube, N_hypercube), dtype=np.float64)
        D_padded[:K_actual, :K_actual] = D_weighted
        dummy_weight = np.mean(D_weighted[D_weighted > 0]) * 0.01 if np.any(D_weighted > 0) else 1e-6
        D_padded[K_actual:, :] = dummy_weight
        D_padded[:, K_actual:] = dummy_weight
        D_padded[K_actual:, K_actual:] = 0
        
        emb_padded = np.zeros((N_hypercube, dim), dtype=bucket_embeddings.dtype)
        emb_padded[:K_actual] = bucket_embeddings
        emb_padded[K_actual:] = bucket_embeddings[-1:]
    else:
        D_padded = D_weighted
        emb_padded = bucket_embeddings
    
    hasher = GrayTunneledHasher(
        n_bits=n_bits,
        block_size=8,
        max_two_swap_iters=30,
        num_tunneling_steps=3,
        mode="two_swap_only",
        random_state=42,
    )
    
    hasher.fit(emb_padded, D=D_padded)
    permutation = hasher.get_assignment()
    
    # Compute J(φ*) with correct mapping
    # bucket i maps to embedding i (for first K_actual buckets)
    bucket_to_embedding_idx = np.arange(K_actual, dtype=np.int32)
    j_phi_star = compute_j_phi_fixed(
        pi=pi,
        w=w,
        bucket_to_code=bucket_to_code,
        permutation=permutation,
        n_bits=n_bits,
        bucket_to_embedding_idx=bucket_to_embedding_idx,
    )
    
    print(f"{Colors.BLUE}J(φ*) (fixed):{Colors.END} {j_phi_star:.6f}")
    print(f"{Colors.BLUE}Difference:{Colors.END} {j_phi_star - j_phi_0_direct:.6f}")
    
    if j_phi_star <= j_phi_0_direct + 1e-6:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ GUARANTEE HOLDS: J(φ*) ≤ J(φ₀){Colors.END}")
        return True
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ GUARANTEE VIOLATED: J(φ*) > J(φ₀){Colors.END}")
        return False


if __name__ == "__main__":
    success = test_fix()
    sys.exit(0 if success else 1)

