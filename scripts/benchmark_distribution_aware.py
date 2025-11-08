"""Benchmark script comparing canonical vs distribution-aware GTH."""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gray_tunneled_hashing.data.real_datasets import (
    load_embeddings,
    load_queries_and_ground_truth,
)
from gray_tunneled_hashing.binary.baselines import (
    sign_binarize,
    random_projection_binarize,
)
from gray_tunneled_hashing.binary.codebooks import (
    build_codebook_kmeans,
    encode_with_codebook,
    find_nearest_centroids,
)
from gray_tunneled_hashing.algorithms.gray_tunneled_hasher import GrayTunneledHasher
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.distribution.traffic_stats import (
    collect_traffic_stats,
    build_weighted_distance_matrix,
)
from gray_tunneled_hashing.integrations.hamming_index import build_hamming_index
from gray_tunneled_hashing.data.synthetic_generators import generate_hypercube_vertices
from gray_tunneled_hashing.evaluation.metrics import recall_at_k


def compute_distribution_aware_cost(
    pi: np.ndarray,
    w: np.ndarray,
    bucket_to_code: np.ndarray,
    permutation: np.ndarray,
    n_bits: int,
) -> float:
    """
    Compute distribution-aware cost J(φ).
    
    J(φ) = Σ_{i,j} π_i · w_ij · d_H(φ(c_i), φ(c_j))
    
    Args:
        pi: Query prior (K,)
        w: Neighbor weights (K, K)
        bucket_to_code: Original bucket codes (K, n_bits)
        permutation: Learned permutation (N,)
        n_bits: Number of bits
        
    Returns:
        Distribution-aware cost
    """
    from gray_tunneled_hashing.evaluation.metrics import hamming_distance
    
    vertices = generate_hypercube_vertices(n_bits)  # (N, n_bits)
    N = len(vertices)
    K = len(pi)
    
    # Map buckets to permuted codes
    # permutation[vertex_idx] = bucket_idx means vertex vertex_idx is assigned to bucket bucket_idx
    # We need: for bucket i, which vertex is it assigned to?
    bucket_to_vertex = {}
    for vertex_idx in range(N):
        bucket_idx = permutation[vertex_idx]
        if bucket_idx < K:
            if bucket_idx not in bucket_to_vertex:
                bucket_to_vertex[bucket_idx] = vertex_idx
    
    # Compute cost
    cost = 0.0
    for i in range(K):
        if i not in bucket_to_vertex:
            continue
        vertex_i = bucket_to_vertex[i]
        code_i = vertices[vertex_i]
        
        for j in range(K):
            if j not in bucket_to_vertex:
                continue
            vertex_j = bucket_to_vertex[j]
            code_j = vertices[vertex_j]
            
            # hamming_distance returns array, extract scalar
            d_h = hamming_distance(code_i[np.newaxis, :], code_j[np.newaxis, :])[0, 0]
            cost += pi[i] * w[i, j] * d_h
    
    return cost


def run_canonical_gth(
    base_embeddings: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    encoder_fn,
    n_bits: int,
    n_codes: int,
    block_size: int,
    max_two_swap_iters: int,
    num_tunneling_steps: int,
    mode: str,
    random_state: Optional[int],
    k: int = 10,
) -> Dict[str, Any]:
    """
    Run canonical GTH (semantic distances only, no traffic weights).
    
    Returns:
        Dictionary with metrics and metadata
    """
    print(f"\n{'='*60}")
    print("Running Canonical GTH")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Build codebook
    centroids, assignments = build_codebook_kmeans(
        embeddings=base_embeddings,
        n_codes=n_codes,
        random_state=random_state,
    )
    
    # Pad centroids if needed
    N = 2 ** n_bits
    if n_codes < N:
        # Pad with duplicates
        centroids_padded = np.zeros((N, centroids.shape[1]), dtype=centroids.dtype)
        centroids_padded[:n_codes] = centroids
        centroids_padded[n_codes:] = centroids[-1:]
        centroids = centroids_padded
    
    # Fit hasher
    hasher = GrayTunneledHasher(
        n_bits=n_bits,
        block_size=block_size,
        max_two_swap_iters=max_two_swap_iters,
        num_tunneling_steps=num_tunneling_steps if mode == "full" else 0,
        mode=mode,
        random_state=random_state,
    )
    
    hasher.fit(centroids)
    permutation = hasher.get_assignment()
    
    # Map centroids to codes
    vertices = generate_hypercube_vertices(n_bits)
    centroid_to_code = {}
    for i in range(min(n_codes, N)):
        vertex_indices = np.where(permutation == i)[0]
        if len(vertex_indices) > 0:
            vertex_idx = vertex_indices[0]
            centroid_to_code[i] = vertices[vertex_idx].astype(bool)
        else:
            # Fallback
            centroid_to_code[i] = vertices[i % N].astype(bool)
    
    # Encode
    base_codes = encode_with_codebook(
        base_embeddings,
        centroids[:n_codes],
        centroid_to_code,
        assignments=assignments,
    )
    
    query_assignments = find_nearest_centroids(queries, centroids[:n_codes])
    query_codes = encode_with_codebook(
        queries,
        centroids[:n_codes],
        centroid_to_code,
        assignments=query_assignments,
    )
    
    # Build index and search
    index = build_hamming_index(base_codes, use_faiss=True)
    search_start = time.time()
    retrieved_indices, distances = index.search(query_codes, k)
    search_time = time.time() - search_start
    
    # Compute recall
    recall = recall_at_k(ground_truth, retrieved_indices, k)
    
    build_time = time.time() - start_time
    
    # Compute baseline cost (identity permutation)
    # For canonical, we use semantic distances only
    D_semantic = np.zeros((n_codes, n_codes), dtype=np.float64)
    for i in range(n_codes):
        for j in range(n_codes):
            D_semantic[i, j] = np.linalg.norm(centroids[i] - centroids[j]) ** 2
    
    # Compute QAP cost (simplified, using semantic distances)
    from gray_tunneled_hashing.algorithms.qap_objective import qap_cost, generate_hypercube_edges
    
    edges = generate_hypercube_edges(n_bits)
    # Pad D if needed
    if n_codes < N:
        D_padded = np.zeros((N, N), dtype=np.float64)
        D_padded[:n_codes, :n_codes] = D_semantic
        D_semantic = D_padded
    
    qap_cost_final = qap_cost(permutation, D_semantic, edges)
    
    return {
        "method": "canonical_gth",
        "recall@k": float(recall),
        "build_time": build_time,
        "search_time": search_time,
        "qap_cost": float(qap_cost_final),
        "n_bits": n_bits,
        "n_codes": n_codes,
        "mode": mode,
    }


def run_distribution_aware_gth(
    base_embeddings: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    encoder_fn,
    n_bits: int,
    n_codes: int,
    block_size: int,
    max_two_swap_iters: int,
    num_tunneling_steps: int,
    mode: str,
    random_state: Optional[int],
    use_semantic_distances: bool,
    k: int = 10,
) -> Dict[str, Any]:
    """
    Run distribution-aware GTH (with traffic weights).
    
    Returns:
        Dictionary with metrics and metadata
    """
    print(f"\n{'='*60}")
    print(f"Running Distribution-Aware GTH (semantic={use_semantic_distances})")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Build distribution-aware index
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=encoder_fn,
        n_bits=n_bits,
        n_codes=n_codes,
        use_codebook=True,
        use_semantic_distances=use_semantic_distances,
        block_size=block_size,
        max_two_swap_iters=max_two_swap_iters,
        num_tunneling_steps=num_tunneling_steps,
        mode=mode,
        random_state=random_state,
    )
    
    # Encode queries
    query_assignments = find_nearest_centroids(queries, index_obj.bucket_embeddings)
    
    # Map buckets to codes via permutation
    vertices = generate_hypercube_vertices(n_bits)
    bucket_to_code_permuted = {}
    permutation = index_obj.permutation
    
    for bucket_idx in range(index_obj.K):
        # Find vertex assigned to this bucket
        vertex_indices = np.where(permutation == bucket_idx)[0]
        if len(vertex_indices) > 0:
            vertex_idx = vertex_indices[0]
            bucket_to_code_permuted[bucket_idx] = vertices[vertex_idx].astype(bool)
        else:
            # Fallback
            bucket_to_code_permuted[bucket_idx] = index_obj.bucket_to_code[bucket_idx].astype(bool)
    
    # Encode base embeddings
    base_codes = np.zeros((len(base_embeddings), n_bits), dtype=bool)
    base_codes_original = encoder_fn(base_embeddings)
    
    for i, code in enumerate(base_codes_original):
        code_tuple = tuple(code.astype(int).tolist())
        if code_tuple in index_obj.code_to_bucket:
            bucket_idx = index_obj.code_to_bucket[code_tuple]
            if bucket_idx in bucket_to_code_permuted:
                base_codes[i] = bucket_to_code_permuted[bucket_idx]
            else:
                base_codes[i] = code
        else:
            base_codes[i] = code
    
    # Encode queries
    query_codes = np.zeros((len(queries), n_bits), dtype=bool)
    query_codes_original = encoder_fn(queries)
    
    for i, code in enumerate(query_codes_original):
        code_tuple = tuple(code.astype(int).tolist())
        if code_tuple in index_obj.code_to_bucket:
            bucket_idx = index_obj.code_to_bucket[code_tuple]
            if bucket_idx in bucket_to_code_permuted:
                query_codes[i] = bucket_to_code_permuted[bucket_idx]
            else:
                query_codes[i] = code
        else:
            query_codes[i] = code
    
    # Build index and search
    index = build_hamming_index(base_codes, use_faiss=True)
    search_start = time.time()
    retrieved_indices, distances = index.search(query_codes, k)
    search_time = time.time() - search_start
    
    # Compute recall
    recall = recall_at_k(ground_truth, retrieved_indices, k)
    
    build_time = time.time() - start_time
    
    # Compute distribution-aware cost J(φ)
    j_phi = compute_distribution_aware_cost(
        pi=index_obj.pi,
        w=index_obj.w,
        bucket_to_code=index_obj.bucket_to_code,
        permutation=permutation,
        n_bits=n_bits,
    )
    
    # Compute baseline cost (identity permutation)
    # For baseline, we use identity: each bucket maps to its original code
    vertices = generate_hypercube_vertices(n_bits)
    identity_permutation = np.arange(2 ** n_bits, dtype=np.int32)
    j_phi_0 = compute_distribution_aware_cost(
        pi=index_obj.pi,
        w=index_obj.w,
        bucket_to_code=index_obj.bucket_to_code,
        permutation=identity_permutation,
        n_bits=n_bits,
    )
    
    return {
        "method": f"distribution_aware_gth_semantic_{use_semantic_distances}",
        "recall@k": float(recall),
        "build_time": build_time,
        "search_time": search_time,
        "j_phi": float(j_phi),
        "j_phi_0": float(j_phi_0),
        "j_phi_improvement": float((j_phi_0 - j_phi) / j_phi_0 * 100) if j_phi_0 > 0 else 0.0,
        "n_bits": n_bits,
        "n_codes": n_codes,
        "mode": mode,
        "use_semantic_distances": use_semantic_distances,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark canonical vs distribution-aware GTH"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., 'quora')",
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        default=64,
        help="Number of bits (default: 64)",
    )
    parser.add_argument(
        "--n-codes",
        type=int,
        default=512,
        help="Number of codebook vectors (default: 512)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of neighbors to retrieve (default: 10)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=8,
        help="Block size for tunneling (default: 8)",
    )
    parser.add_argument(
        "--max-two-swap-iters",
        type=int,
        default=100,
        help="Max iterations for 2-swap (default: 100)",
    )
    parser.add_argument(
        "--num-tunneling-steps",
        type=int,
        default=10,
        help="Number of tunneling steps (default: 10)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["trivial", "two_swap_only", "full"],
        help="Optimization mode (default: full)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="random_proj",
        choices=["sign", "random_proj"],
        help="Base encoder (default: random_proj)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/real/results_distribution_aware_benchmark.json",
        help="Output JSON file path",
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading dataset: {args.dataset}")
    base_embeddings = load_embeddings(args.dataset)
    queries, ground_truth = load_queries_and_ground_truth(args.dataset, k=args.k)
    
    print(f"Base embeddings: {base_embeddings.shape}")
    print(f"Queries: {queries.shape}")
    print(f"Ground truth: {ground_truth.shape}")
    
    # Create encoder
    if args.encoder == "sign":
        def encoder_fn(emb):
            return sign_binarize(emb)
    elif args.encoder == "random_proj":
        _, proj_matrix = random_projection_binarize(
            base_embeddings,
            n_bits=args.n_bits,
            random_state=args.random_state,
        )
        def encoder_fn(emb):
            from gray_tunneled_hashing.binary.baselines import apply_random_projection
            proj = apply_random_projection(emb, proj_matrix)
            return (proj > 0).astype(bool)
    else:
        raise ValueError(f"Unknown encoder: {args.encoder}")
    
    results = []
    
    # Run canonical GTH
    result_canonical = run_canonical_gth(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth=ground_truth,
        encoder_fn=encoder_fn,
        n_bits=args.n_bits,
        n_codes=args.n_codes,
        block_size=args.block_size,
        max_two_swap_iters=args.max_two_swap_iters,
        num_tunneling_steps=args.num_tunneling_steps,
        mode=args.mode,
        random_state=args.random_state,
        k=args.k,
    )
    results.append(result_canonical)
    
    # Run distribution-aware GTH (with semantic)
    result_da_semantic = run_distribution_aware_gth(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth=ground_truth,
        encoder_fn=encoder_fn,
        n_bits=args.n_bits,
        n_codes=args.n_codes,
        block_size=args.block_size,
        max_two_swap_iters=args.max_two_swap_iters,
        num_tunneling_steps=args.num_tunneling_steps,
        mode=args.mode,
        random_state=args.random_state,
        use_semantic_distances=True,
        k=args.k,
    )
    results.append(result_da_semantic)
    
    # Run distribution-aware GTH (without semantic)
    result_da_pure = run_distribution_aware_gth(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth=ground_truth,
        encoder_fn=encoder_fn,
        n_bits=args.n_bits,
        n_codes=args.n_codes,
        block_size=args.block_size,
        max_two_swap_iters=args.max_two_swap_iters,
        num_tunneling_steps=args.num_tunneling_steps,
        mode=args.mode,
        random_state=args.random_state,
        use_semantic_distances=False,
        k=args.k,
    )
    results.append(result_da_pure)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for result in results:
        method = result["method"]
        recall = result["recall@k"]
        build_time = result["build_time"]
        print(f"{method:40s} | Recall@k: {recall:.4f} | Build: {build_time:.2f}s")
        if "j_phi" in result:
            print(f"  J(φ): {result['j_phi']:.4f}, J(φ₀): {result['j_phi_0']:.4f}, "
                  f"Improvement: {result['j_phi_improvement']:.2f}%")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump({
            "args": vars(args),
            "results": results,
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Validate guarantee: J(φ*) ≤ J(φ₀)
    for result in results:
        if "j_phi" in result and "j_phi_0" in result:
            j_phi = result["j_phi"]
            j_phi_0 = result["j_phi_0"]
            if j_phi > j_phi_0 + 1e-6:  # Small tolerance for numerical errors
                print(f"\nWARNING: J(φ*) > J(φ₀) for {result['method']}: "
                      f"{j_phi:.6f} > {j_phi_0:.6f}")
            else:
                print(f"\n✓ Guarantee verified for {result['method']}: "
                      f"J(φ*) = {j_phi:.6f} ≤ J(φ₀) = {j_phi_0:.6f}")


if __name__ == "__main__":
    main()

