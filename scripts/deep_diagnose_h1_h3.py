"""
Deep diagnosis of H1 and H3 - the most promising hypotheses.

H1: J(φ₀) calculation incorrect
H3: Permutation not minimizing J(φ)

Tests:
- Verify identity permutation mapping
- Check if D_weighted has correct structure
- Validate that permutation actually minimizes QAP cost
- Compare different ways to compute J(φ₀)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Add project root to path
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


def compute_j_phi_direct(pi, w, bucket_to_code):
    """Compute J(φ) directly using bucket codes."""
    K = len(pi)
    cost = 0.0
    for i in range(K):
        code_i = bucket_to_code[i]
        for j in range(K):
            code_j = bucket_to_code[j]
            d_h = hamming_distance(code_i[np.newaxis, :], code_j[np.newaxis, :])[0, 0]
            cost += pi[i] * w[i, j] * d_h
    return cost


def compute_j_phi_via_permutation(pi, w, bucket_to_code, permutation, n_bits):
    """Compute J(φ) via permutation mapping."""
    vertices = generate_hypercube_vertices(n_bits)
    N = len(vertices)
    K = len(pi)
    
    # Map: bucket_idx -> vertex_idx
    bucket_to_vertex = {}
    for vertex_idx in range(N):
        bucket_idx = permutation[vertex_idx]
        if bucket_idx < K:
            if bucket_idx not in bucket_to_vertex:
                bucket_to_vertex[bucket_idx] = vertex_idx
    
    cost = 0.0
    for i in range(K):
        if i in bucket_to_vertex:
            vertex_i = bucket_to_vertex[i]
            code_i = vertices[vertex_i]
        else:
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


def find_permutation_for_original_codes(bucket_to_code, n_bits):
    """
    Find permutation that maps buckets to their original codes.
    
    This is the true φ₀: each bucket should map to its original code.
    """
    vertices = generate_hypercube_vertices(n_bits)
    N = len(vertices)
    K = len(bucket_to_code)
    
    # Find which vertex corresponds to each bucket's original code
    permutation = np.zeros(N, dtype=np.int32)
    used_vertices = set()
    
    for bucket_idx in range(K):
        code = bucket_to_code[bucket_idx]
        # Find vertex with this code
        for vertex_idx in range(N):
            if vertex_idx not in used_vertices:
                if np.array_equal(vertices[vertex_idx], code):
                    permutation[vertex_idx] = bucket_idx
                    used_vertices.add(vertex_idx)
                    break
        else:
            # Code not found in vertices - assign to first unused vertex
            for vertex_idx in range(N):
                if vertex_idx not in used_vertices:
                    permutation[vertex_idx] = bucket_idx
                    used_vertices.add(vertex_idx)
                    break
    
    # Fill remaining vertices with bucket indices (cyclic)
    for vertex_idx in range(N):
        if vertex_idx not in used_vertices:
            permutation[vertex_idx] = vertex_idx % K
    
    return permutation


def test_h1_deep(pi, w, bucket_to_code, n_bits):
    """Deep test of H1: J(φ₀) calculation."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}Testing H1: J(φ₀) calculation{Colors.END}")
    print(f"{Colors.CYAN}{'='*70}{Colors.END}")
    
    # Method 1: Direct computation (using original codes)
    j_phi_0_direct = compute_j_phi_direct(pi, w, bucket_to_code)
    print(f"{Colors.BLUE}Method 1 (direct):{Colors.END} J(φ₀) = {j_phi_0_direct:.6f}")
    
    # Method 2: Identity permutation (wrong assumption)
    identity_perm = np.arange(2 ** n_bits, dtype=np.int32)
    j_phi_0_identity = compute_j_phi_via_permutation(pi, w, bucket_to_code, identity_perm, n_bits)
    print(f"{Colors.BLUE}Method 2 (identity perm):{Colors.END} J(φ₀) = {j_phi_0_identity:.6f}")
    
    # Method 3: Find correct permutation for original codes
    correct_perm = find_permutation_for_original_codes(bucket_to_code, n_bits)
    j_phi_0_correct = compute_j_phi_via_permutation(pi, w, bucket_to_code, correct_perm, n_bits)
    print(f"{Colors.BLUE}Method 3 (correct perm):{Colors.END} J(φ₀) = {j_phi_0_correct:.6f}")
    
    # Compare
    diff_identity = abs(j_phi_0_direct - j_phi_0_identity)
    diff_correct = abs(j_phi_0_direct - j_phi_0_correct)
    
    print(f"\n{Colors.YELLOW}Differences:{Colors.END}")
    print(f"  Direct vs Identity: {diff_identity:.6f}")
    print(f"  Direct vs Correct: {diff_correct:.6f}")
    
    if diff_correct < 1e-6:
        print(f"{Colors.GREEN}✓ Method 3 matches Method 1{Colors.END}")
        return correct_perm, j_phi_0_correct
    else:
        print(f"{Colors.RED}✗ Method 3 does not match Method 1{Colors.END}")
        return None, j_phi_0_direct


def test_h3_deep(pi, w, bucket_to_code, bucket_embeddings, n_bits, n_codes):
    """Deep test of H3: Permutation not minimizing."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}Testing H3: Permutation minimization{Colors.END}")
    print(f"{Colors.CYAN}{'='*70}{Colors.END}")
    
    K = len(pi)
    N = 2 ** n_bits
    
    # Build D_weighted
    D_weighted = build_weighted_distance_matrix(
        pi=pi,
        w=w,
        bucket_embeddings=bucket_embeddings[:K],
        use_semantic_distances=True,
    )
    
    # Pad if needed
    if K < N:
        D_padded = np.zeros((N, N), dtype=np.float64)
        D_padded[:K, :K] = D_weighted
        D_weighted = D_padded
    
    edges = generate_hypercube_edges(n_bits)
    
    # Test 1: Check if D_weighted has zeros (problematic for optimization)
    zero_ratio = np.sum(D_weighted == 0) / (N * N)
    print(f"{Colors.BLUE}D_weighted zero ratio:{Colors.END} {zero_ratio:.4f}")
    
    if zero_ratio > 0.5:
        print(f"{Colors.YELLOW}⚠ Warning: Many zeros in D_weighted (may cause optimization issues){Colors.END}")
    
    # Test 2: Try identity permutation
    identity_perm = np.arange(N, dtype=np.int32)
    cost_identity = qap_cost(identity_perm, D_weighted, edges)
    print(f"{Colors.BLUE}QAP cost (identity):{Colors.END} {cost_identity:.6f}")
    
    # Test 3: Try random permutations
    print(f"{Colors.BLUE}Trying random permutations...{Colors.END}")
    random_costs = []
    for _ in tqdm(range(100), desc="Random perms", leave=False):
        random_perm = np.random.permutation(N).astype(np.int32)
        cost = qap_cost(random_perm, D_weighted, edges)
        random_costs.append(cost)
    
    min_random = min(random_costs)
    mean_random = np.mean(random_costs)
    print(f"  Min random: {min_random:.6f}")
    print(f"  Mean random: {mean_random:.6f}")
    
    # Test 4: Optimize with hasher
    print(f"{Colors.BLUE}Optimizing with GrayTunneledHasher...{Colors.END}")
    hasher = GrayTunneledHasher(
        n_bits=n_bits,
        block_size=8,
        max_two_swap_iters=50,
        num_tunneling_steps=5,
        mode="two_swap_only",
        random_state=42,
    )
    
    # Pad bucket embeddings
    if K < N:
        bucket_emb_padded = np.zeros((N, bucket_embeddings.shape[1]), dtype=bucket_embeddings.dtype)
        bucket_emb_padded[:K] = bucket_embeddings[:K]
        bucket_emb_padded[K:] = bucket_embeddings[-1:]
        bucket_embeddings_fit = bucket_emb_padded
    else:
        bucket_embeddings_fit = bucket_embeddings
    
    hasher.fit(bucket_embeddings_fit, D=D_weighted)
    learned_perm = hasher.get_assignment()
    cost_learned = qap_cost(learned_perm, D_weighted, edges)
    
    print(f"{Colors.BLUE}QAP cost (learned):{Colors.END} {cost_learned:.6f}")
    
    # Compare
    print(f"\n{Colors.YELLOW}Comparison:{Colors.END}")
    print(f"  Identity: {cost_identity:.6f}")
    print(f"  Learned: {cost_learned:.6f}")
    print(f"  Min random: {min_random:.6f}")
    
    if cost_learned < cost_identity:
        print(f"{Colors.GREEN}✓ Learned is better than identity{Colors.END}")
    else:
        print(f"{Colors.RED}✗ Learned is NOT better than identity{Colors.END}")
    
    if cost_learned <= min_random + 1e-6:
        print(f"{Colors.GREEN}✓ Learned is at least as good as best random{Colors.END}")
    else:
        print(f"{Colors.RED}✗ Learned is worse than best random{Colors.END}")
        print(f"{Colors.YELLOW}  This suggests optimization is not working correctly{Colors.END}")
    
    return learned_perm, cost_learned, D_weighted


def main():
    parser = argparse.ArgumentParser(description="Deep diagnosis of H1 and H3")
    parser.add_argument("--n-bits", type=int, default=8, help="Number of bits")
    parser.add_argument("--n-codes", type=int, default=16, help="Number of codebook vectors")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print(f"{Colors.BOLD}{Colors.MAGENTA}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}DEEP DIAGNOSIS: H1 & H3{Colors.END}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}Configuration:{Colors.END} n_bits={args.n_bits}, n_codes={args.n_codes}")
    
    np.random.seed(args.random_state)
    
    # Generate data
    N = 200
    Q = 50
    dim = 16
    k = 5
    
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    ground_truth = np.random.randint(0, N, size=(Q, k), dtype=np.int32)
    
    # Create encoder
    _, proj_matrix = random_projection_binarize(
        base_embeddings,
        n_bits=args.n_bits,
        random_state=args.random_state,
    )
    
    def encoder_fn(emb):
        from gray_tunneled_hashing.binary.baselines import apply_random_projection
        proj = apply_random_projection(emb, proj_matrix)
        return (proj > 0).astype(bool)
    
    # Collect traffic stats
    print(f"\n{Colors.BLUE}Collecting traffic statistics...{Colors.END}")
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
    
    print(f"  K (buckets): {K}")
    print(f"  pi sum: {pi.sum():.6f}")
    print(f"  w shape: {w.shape}")
    
    # Build codebook
    centroids, _ = build_codebook_kmeans(
        embeddings=base_embeddings,
        n_codes=args.n_codes,
        random_state=args.random_state,
    )
    
    bucket_embeddings = centroids[:min(K, args.n_codes)]
    if K > args.n_codes:
        padding = np.tile(bucket_embeddings[-1:], (K - args.n_codes, 1))
        bucket_embeddings = np.vstack([bucket_embeddings, padding])
    
    # Test H1
    correct_perm_phi0, j_phi_0_correct = test_h1_deep(pi, w, bucket_to_code, args.n_bits)
    
    # Test H3
    learned_perm, cost_learned, D_weighted = test_h3_deep(
        pi, w, bucket_to_code, bucket_embeddings, args.n_bits, args.n_codes
    )
    
    # Final validation: Compute J(φ*) and J(φ₀) with correct methods
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}FINAL VALIDATION{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    
    if correct_perm_phi0 is not None:
        j_phi_0_final = compute_j_phi_via_permutation(pi, w, bucket_to_code, correct_perm_phi0, args.n_bits)
        j_phi_star_final = compute_j_phi_via_permutation(pi, w, bucket_to_code, learned_perm, args.n_bits)
        
        print(f"\n{Colors.BLUE}J(φ₀) (correct method):{Colors.END} {j_phi_0_final:.6f}")
        print(f"{Colors.BLUE}J(φ*) (learned perm):{Colors.END} {j_phi_star_final:.6f}")
        print(f"{Colors.BLUE}Difference:{Colors.END} {j_phi_star_final - j_phi_0_final:.6f}")
        
        if j_phi_star_final <= j_phi_0_final + 1e-6:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✓ GUARANTEE HOLDS: J(φ*) ≤ J(φ₀){Colors.END}")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}✗ GUARANTEE VIOLATED: J(φ*) > J(φ₀){Colors.END}")
            print(f"{Colors.YELLOW}  Root cause likely in:{Colors.END}")
            print(f"    1. Permutation not minimizing QAP cost correctly")
            print(f"    2. D_weighted structure causing optimization issues")
            print(f"    3. Mismatch between QAP objective and J(φ) objective")


if __name__ == "__main__":
    main()

