"""
Validate relationship between QAP cost and J(φ).

Key insight: 
- QAP cost: f(π) = Σ_{(u,v) ∈ edges} D_weighted[π(u), π(v)]
- J(φ): J(φ) = Σ_{i,j} π_i · w_ij · d_H(φ(c_i), φ(c_j))

These are NOT the same! QAP only sums over hypercube edges, while J(φ) sums over all bucket pairs.

We need to verify if minimizing QAP cost actually minimizes J(φ).
"""

import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

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
from gray_tunneled_hashing.algorithms.qap_objective import qap_cost, generate_hypercube_edges


class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def compute_j_phi_from_permutation(pi, w, bucket_to_code, permutation, n_bits):
    """Compute J(φ) from permutation."""
    vertices = generate_hypercube_vertices(n_bits)
    N = len(vertices)
    K = len(pi)
    
    bucket_to_vertex = {}
    for vertex_idx in range(N):
        embedding_idx = permutation[vertex_idx]
        if embedding_idx < K:
            bucket_idx = embedding_idx
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
            
            d_h = hamming_distance(code_i[np.newaxis, :], code_j[np.newaxis, :])[0, 0]
            cost += pi[i] * w[i, j] * d_h
    
    return cost


def compute_qap_cost_from_permutation(permutation, D_weighted, edges):
    """Compute QAP cost from permutation."""
    return qap_cost(permutation, D_weighted, edges)


def main():
    print(f"{Colors.BOLD}{Colors.MAGENTA}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}VALIDATING QAP COST vs J(φ) RELATIONSHIP{Colors.END}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}{'='*70}{Colors.END}")
    
    n_bits = 6
    n_codes = 32
    dim = 16
    
    np.random.seed(42)
    
    # Generate data
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
    K = traffic_stats["K"]
    
    print(f"  K: {K}")
    
    # Build codebook
    centroids, _ = build_codebook_kmeans(
        embeddings=base_embeddings,
        n_codes=min(n_codes, K),
        random_state=42,
    )
    
    bucket_embeddings = centroids[:K]
    if len(bucket_embeddings) < K:
        padding = np.tile(bucket_embeddings[-1:], (K - len(bucket_embeddings), 1))
        bucket_embeddings = np.vstack([bucket_embeddings, padding])
    
    # Build D_weighted
    D_weighted = build_weighted_distance_matrix(
        pi=pi,
        w=w,
        bucket_embeddings=bucket_embeddings,
        use_semantic_distances=True,
    )
    
    # Pad if needed
    N_hypercube = 2 ** n_bits
    if K < N_hypercube:
        D_padded = np.zeros((N_hypercube, N_hypercube), dtype=np.float64)
        D_padded[:K, :K] = D_weighted
        dummy_weight = np.mean(D_weighted[D_weighted > 0]) * 0.01 if np.any(D_weighted > 0) else 1e-6
        D_padded[K:, :] = dummy_weight
        D_padded[:, K:] = dummy_weight
        D_padded[K:, K:] = 0
        D_weighted = D_padded
        
        emb_padded = np.zeros((N_hypercube, dim), dtype=bucket_embeddings.dtype)
        emb_padded[:K] = bucket_embeddings
        emb_padded[K:] = bucket_embeddings[-1:]
        bucket_embeddings = emb_padded
    
    edges = generate_hypercube_edges(n_bits)
    
    # Compute J(φ₀)
    j_phi_0 = 0.0
    for i in range(K):
        code_i = bucket_to_code[i]
        for j in range(K):
            code_j = bucket_to_code[j]
            d_h = hamming_distance(code_i[np.newaxis, :], code_j[np.newaxis, :])[0, 0]
            j_phi_0 += pi[i] * w[i, j] * d_h
    
    print(f"\n{Colors.BLUE}J(φ₀):{Colors.END} {j_phi_0:.6f}")
    
    # Test multiple permutations
    print(f"\n{Colors.BLUE}Testing multiple permutations...{Colors.END}")
    results = []
    
    # Identity permutation
    identity_perm = np.arange(N_hypercube, dtype=np.int32)
    qap_identity = compute_qap_cost_from_permutation(identity_perm, D_weighted, edges)
    j_phi_identity = compute_j_phi_from_permutation(pi, w, bucket_to_code, identity_perm, n_bits)
    results.append(("identity", qap_identity, j_phi_identity))
    
    # Random permutations
    for i in tqdm(range(20), desc="Random perms", leave=False):
        random_perm = np.random.permutation(N_hypercube).astype(np.int32)
        qap_cost_val = compute_qap_cost_from_permutation(random_perm, D_weighted, edges)
        j_phi_val = compute_j_phi_from_permutation(pi, w, bucket_to_code, random_perm, n_bits)
        results.append((f"random_{i}", qap_cost_val, j_phi_val))
    
    # Optimized permutation
    print(f"{Colors.BLUE}Optimizing with GrayTunneledHasher...{Colors.END}")
    hasher = GrayTunneledHasher(
        n_bits=n_bits,
        block_size=8,
        max_two_swap_iters=50,
        num_tunneling_steps=5,
        mode="two_swap_only",
        random_state=42,
    )
    hasher.fit(bucket_embeddings, D=D_weighted)
    learned_perm = hasher.get_assignment()
    qap_learned = compute_qap_cost_from_permutation(learned_perm, D_weighted, edges)
    j_phi_learned = compute_j_phi_from_permutation(pi, w, bucket_to_code, learned_perm, n_bits)
    results.append(("learned", qap_learned, j_phi_learned))
    
    # Print results
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}RESULTS{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{'Permutation':<20} {'QAP Cost':<15} {'J(φ)':<15} {'J(φ) vs J(φ₀)':<20}")
    print("-" * 70)
    
    for name, qap, j_phi in results:
        vs_baseline = "✓" if j_phi <= j_phi_0 + 1e-6 else "✗"
        color = Colors.GREEN if j_phi <= j_phi_0 + 1e-6 else Colors.RED
        print(f"{name:<20} {qap:<15.6f} {j_phi:<15.6f} {color}{vs_baseline}{Colors.END}")
    
    # Check correlation
    qap_costs = [r[1] for r in results]
    j_phi_costs = [r[2] for r in results]
    correlation = np.corrcoef(qap_costs, j_phi_costs)[0, 1]
    
    print(f"\n{Colors.BLUE}Correlation (QAP cost vs J(φ)):{Colors.END} {correlation:.4f}")
    
    if correlation > 0.8:
        print(f"{Colors.GREEN}✓ Strong correlation - QAP minimization should minimize J(φ){Colors.END}")
    elif correlation > 0.5:
        print(f"{Colors.YELLOW}⚠ Moderate correlation - may need different objective{Colors.END}")
    else:
        print(f"{Colors.RED}✗ Weak correlation - QAP and J(φ) are different objectives!{Colors.END}")
        print(f"{Colors.YELLOW}  This explains why minimizing QAP doesn't guarantee J(φ*) ≤ J(φ₀){Colors.END}")


if __name__ == "__main__":
    main()

