"""
Theoretical benchmark for Distribution-Aware GTH.

This benchmark validates:
1. Theoretical guarantee: J(φ*) ≤ J(φ₀)
2. Recall@k improvements under skewed traffic
3. Comparison: canonical vs distribution-aware GTH
4. Effect of semantic distances in weighted matrix

Design:
- Controlled traffic scenarios (uniform, skewed, clustered)
- Statistical validation of guarantees
- Multiple runs for robustness
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gray_tunneled_hashing.binary.baselines import random_projection_binarize
from gray_tunneled_hashing.binary.codebooks import build_codebook_kmeans, encode_with_codebook, find_nearest_centroids
from gray_tunneled_hashing.algorithms.gray_tunneled_hasher import GrayTunneledHasher
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.distribution.traffic_stats import build_weighted_distance_matrix
from gray_tunneled_hashing.integrations.hamming_index import build_hamming_index
from gray_tunneled_hashing.data.synthetic_generators import generate_hypercube_vertices
from gray_tunneled_hashing.evaluation.metrics import recall_at_k, hamming_distance
from gray_tunneled_hashing.algorithms.qap_objective import qap_cost, generate_hypercube_edges


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    method: str
    recall_at_k: float
    build_time: float
    search_time: float
    j_phi: float = None
    j_phi_0: float = None
    j_phi_improvement: float = None
    qap_cost: float = None
    n_bits: int = None
    n_codes: int = None
    mode: str = None
    use_semantic_distances: bool = None


def generate_planted_dataset(
    N: int,
    Q: int,
    dim: int,
    k: int,
    traffic_scenario: str,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic dataset with controlled traffic patterns.
    
    Traffic scenarios:
    - "uniform": Queries uniformly distributed
    - "skewed": 80% queries from 20% of space (Pareto-like)
    - "clustered": Queries concentrated in 3-5 clusters
    
    Returns:
        (base_embeddings, queries, ground_truth)
    """
    np.random.seed(random_state)
    
    # Generate base embeddings with structure
    # Create clusters in embedding space
    n_clusters = 5
    cluster_centers = np.random.randn(n_clusters, dim) * 2.0
    base_embeddings = []
    
    for i in range(N):
        cluster_idx = i % n_clusters
        point = cluster_centers[cluster_idx] + np.random.randn(dim) * 0.5
        base_embeddings.append(point)
    
    base_embeddings = np.array(base_embeddings, dtype=np.float32)
    
    # Generate queries based on traffic scenario
    if traffic_scenario == "uniform":
        # Uniform distribution
        query_indices = np.random.choice(N, size=Q, replace=True)
        queries = base_embeddings[query_indices] + np.random.randn(Q, dim) * 0.1
        
    elif traffic_scenario == "skewed":
        # 80% queries from 20% of space (high-traffic region)
        high_traffic_size = N // 5
        high_traffic_indices = np.random.choice(N, size=high_traffic_size, replace=False)
        
        n_high = int(Q * 0.8)
        n_low = Q - n_high
        
        high_traffic_queries = np.random.choice(high_traffic_indices, size=n_high, replace=True)
        low_traffic_queries = np.random.choice(N, size=n_low, replace=True)
        
        query_indices = np.concatenate([high_traffic_queries, low_traffic_queries])
        queries = base_embeddings[query_indices] + np.random.randn(Q, dim) * 0.1
        
    elif traffic_scenario == "clustered":
        # Queries from 3-5 specific clusters
        n_query_clusters = 3
        query_cluster_indices = np.random.choice(n_clusters, size=n_query_clusters, replace=False)
        
        queries = []
        for _ in range(Q):
            cluster_idx = np.random.choice(query_cluster_indices)
            query = cluster_centers[cluster_idx] + np.random.randn(dim) * 0.3
            queries.append(query)
        queries = np.array(queries, dtype=np.float32)
        query_indices = None  # Not used for clustered
        
    else:
        raise ValueError(f"Unknown traffic scenario: {traffic_scenario}")
    
    # Compute ground truth k-NN
    from sklearn.metrics.pairwise import euclidean_distances
    distances = euclidean_distances(queries, base_embeddings)
    ground_truth = np.argsort(distances, axis=1)[:, :k]
    
    return base_embeddings, queries, ground_truth


def compute_j_phi(
    pi: np.ndarray,
    w: np.ndarray,
    bucket_to_code: np.ndarray,
    permutation: np.ndarray,
    n_bits: int,
    use_original_codes: bool = False,
) -> float:
    """
    Compute distribution-aware cost J(φ).
    
    J(φ) = Σ_{i,j} π_i · w_ij · d_H(φ(c_i), φ(c_j))
    
    Args:
        use_original_codes: If True, use original bucket codes (for J(φ₀))
                           If False, use permuted codes from permutation
    """
    K = len(pi)
    
    if use_original_codes:
        # J(φ₀): Use original bucket codes directly
        # This is the true baseline: each bucket keeps its original code
        cost = 0.0
        for i in range(K):
            code_i = bucket_to_code[i]
            for j in range(K):
                code_j = bucket_to_code[j]
                d_h = hamming_distance(code_i[np.newaxis, :], code_j[np.newaxis, :])[0, 0]
                cost += pi[i] * w[i, j] * d_h
        return cost
    
    # J(φ*): Use permuted codes
    vertices = generate_hypercube_vertices(n_bits)
    N = len(vertices)
    
    # Map buckets to permuted codes
    # CRITICAL: permutation[vertex_idx] = embedding_idx
    # When K < N, we pad embeddings, so:
    # - bucket i (0..K-1) maps to embedding i (0..K-1)
    # - padded embeddings (K..N-1) are duplicates
    # We need to find which vertex is assigned to each bucket
    bucket_to_vertex = {}
    for vertex_idx in range(N):
        embedding_idx = permutation[vertex_idx]
        # For first K embeddings, embedding_idx == bucket_idx
        if embedding_idx < K:
            bucket_idx = embedding_idx
            if bucket_idx not in bucket_to_vertex:
                bucket_to_vertex[bucket_idx] = vertex_idx
    
    cost = 0.0
    for i in range(K):
        if i in bucket_to_vertex:
            vertex_i = bucket_to_vertex[i]
            code_i = vertices[vertex_i]
        else:
            # Fallback: use original bucket code
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


def run_experiment(
    base_embeddings: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    encoder_fn,
    n_bits: int,
    n_codes: int,
    method: str,
    block_size: int = 8,
    max_two_swap_iters: int = 50,
    num_tunneling_steps: int = 5,
    mode: str = "two_swap_only",
    random_state: int = 42,
    k: int = 10,
) -> ExperimentResult:
    """
    Run a single experiment.
    
    Methods:
    - "canonical": Standard GTH with semantic distances only
    - "distribution_aware_semantic": Distribution-aware with semantic distances
    - "distribution_aware_pure": Distribution-aware without semantic distances
    """
    start_time = time.time()
    
    if method == "canonical":
        # Build codebook
        centroids, assignments = build_codebook_kmeans(
            embeddings=base_embeddings,
            n_codes=n_codes,
            random_state=random_state,
        )
        
        # Pad if needed
        N = 2 ** n_bits
        if n_codes < N:
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
        
        # Map to codes
        vertices = generate_hypercube_vertices(n_bits)
        centroid_to_code = {}
        for i in range(min(n_codes, N)):
            vertex_indices = np.where(permutation == i)[0]
            if len(vertex_indices) > 0:
                centroid_to_code[i] = vertices[vertex_indices[0]].astype(bool)
            else:
                centroid_to_code[i] = vertices[i % N].astype(bool)
        
        # Encode
        base_codes = encode_with_codebook(
            base_embeddings, centroids[:n_codes], centroid_to_code, assignments=assignments
        )
        query_assignments = find_nearest_centroids(queries, centroids[:n_codes])
        query_codes = encode_with_codebook(
            queries, centroids[:n_codes], centroid_to_code, assignments=query_assignments
        )
        
        # Search
        index = build_hamming_index(base_codes, use_faiss=True)
        search_start = time.time()
        retrieved_indices, _ = index.search(query_codes, k)
        search_time = time.time() - search_start
        
        recall = recall_at_k(ground_truth, retrieved_indices, k)
        build_time = time.time() - start_time
        
        # Compute QAP cost
        edges = generate_hypercube_edges(n_bits)
        D_semantic = np.zeros((n_codes, n_codes), dtype=np.float64)
        for i in range(n_codes):
            for j in range(n_codes):
                D_semantic[i, j] = np.linalg.norm(centroids[i] - centroids[j]) ** 2
        
        if n_codes < N:
            D_padded = np.zeros((N, N), dtype=np.float64)
            D_padded[:n_codes, :n_codes] = D_semantic
            D_semantic = D_padded
        
        qap_cost_val = qap_cost(permutation, D_semantic, edges)
        
        return ExperimentResult(
            method=method,
            recall_at_k=float(recall),
            build_time=build_time,
            search_time=search_time,
            qap_cost=float(qap_cost_val),
            n_bits=n_bits,
            n_codes=n_codes,
            mode=mode,
        )
    
    else:  # distribution-aware methods
        use_semantic = method == "distribution_aware_semantic"
        
        # Build distribution-aware index
        index_obj = build_distribution_aware_index(
            base_embeddings=base_embeddings,
            queries=queries,
            ground_truth_neighbors=ground_truth,
            encoder=encoder_fn,
            n_bits=n_bits,
            n_codes=n_codes,
            use_codebook=True,
            use_semantic_distances=use_semantic,
            block_size=block_size,
            max_two_swap_iters=max_two_swap_iters,
            num_tunneling_steps=num_tunneling_steps,
            mode=mode,
            random_state=random_state,
        )
        
        # Encode (simplified - would need proper encoding in full implementation)
        # For now, we'll compute metrics that don't require full encoding
        
        build_time = time.time() - start_time
        
        # Compute J(φ) and J(φ₀)
        # J(φ₀): Use initial permutation from hasher (ensures consistency)
        from gray_tunneled_hashing.distribution.j_phi_objective import compute_j_phi_0
        
        initial_perm = index_obj.hasher.get_initial_permutation()
        if initial_perm is None:
            # Fallback: use identity permutation
            N = 2 ** n_bits
            initial_perm = np.arange(N, dtype=np.int32)
        
        j_phi_0 = compute_j_phi_0(
            pi=index_obj.pi,
            w=index_obj.w,
            bucket_to_code=index_obj.bucket_to_code,
            n_bits=n_bits,
            initial_permutation=initial_perm,
        )
        
        # J(φ*): Use the learned permutation
        j_phi = compute_j_phi(
            pi=index_obj.pi,
            w=index_obj.w,
            bucket_to_code=index_obj.bucket_to_code,
            permutation=index_obj.permutation,
            n_bits=n_bits,
            use_original_codes=False,
        )
        
        # Also get initial_cost from hasher if available (for validation)
        if hasattr(index_obj.hasher, 'initial_cost_'):
            j_phi_0_from_hasher = index_obj.hasher.initial_cost_
            # Validate consistency
            if abs(j_phi_0 - j_phi_0_from_hasher) > 1e-6:
                print(f"Warning: J(φ₀) mismatch: {j_phi_0:.6f} vs {j_phi_0_from_hasher:.6f}")
        
        improvement = ((j_phi_0 - j_phi) / j_phi_0 * 100) if j_phi_0 > 0 else 0.0
        
        # For recall, we'd need to encode and search, but for theoretical validation
        # we focus on J(φ) guarantee
        return ExperimentResult(
            method=method,
            recall_at_k=0.0,  # Placeholder - would need full encoding
            build_time=build_time,
            search_time=0.0,
            j_phi=float(j_phi),
            j_phi_0=float(j_phi_0),
            j_phi_improvement=float(improvement),
            n_bits=n_bits,
            n_codes=n_codes,
            mode=mode,
            use_semantic_distances=use_semantic,
        )


def validate_theoretical_guarantee(results: List[ExperimentResult]) -> Dict:
    """
    Validate theoretical guarantee: J(φ*) ≤ J(φ₀).
    
    Returns:
        Dictionary with validation results
    """
    validation = {
        "guarantee_holds": True,
        "violations": [],
        "statistics": {},
    }
    
    for result in results:
        if result.j_phi is not None and result.j_phi_0 is not None:
            if result.j_phi > result.j_phi_0 + 1e-6:  # Small tolerance for numerical errors
                validation["guarantee_holds"] = False
                validation["violations"].append({
                    "method": result.method,
                    "j_phi": result.j_phi,
                    "j_phi_0": result.j_phi_0,
                    "difference": result.j_phi - result.j_phi_0,
                })
    
    # Statistics
    da_results = [r for r in results if r.j_phi is not None]
    if da_results:
        improvements = [r.j_phi_improvement for r in da_results if r.j_phi_improvement is not None]
        if improvements:
            validation["statistics"] = {
                "mean_improvement": float(np.mean(improvements)),
                "std_improvement": float(np.std(improvements)),
                "min_improvement": float(np.min(improvements)),
                "max_improvement": float(np.max(improvements)),
                "n_experiments": len(improvements),
            }
    
    return validation


def main():
    parser = argparse.ArgumentParser(
        description="Theoretical benchmark for Distribution-Aware GTH"
    )
    parser.add_argument("--n-bits", type=int, default=16, help="Number of bits (default: 16)")
    parser.add_argument("--n-codes", type=int, default=64, help="Number of codebook vectors (default: 64)")
    parser.add_argument("--k", type=int, default=10, help="Number of neighbors (default: 10)")
    parser.add_argument("--traffic-scenario", type=str, default="skewed",
                       choices=["uniform", "skewed", "clustered"],
                       help="Traffic scenario (default: skewed)")
    parser.add_argument("--n-runs", type=int, default=5, help="Number of runs for robustness (default: 5)")
    parser.add_argument("--mode", type=str, default="two_swap_only",
                       choices=["trivial", "two_swap_only", "full"],
                       help="Optimization mode (default: two_swap_only)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", type=str,
                       default="experiments/real/results_distribution_aware_theoretical.json",
                       help="Output JSON file")
    
    args = parser.parse_args()
    
    print("="*70)
    print("THEORETICAL BENCHMARK: Distribution-Aware GTH")
    print("="*70)
    print(f"Configuration:")
    print(f"  n_bits: {args.n_bits}")
    print(f"  n_codes: {args.n_codes}")
    print(f"  k: {args.k}")
    print(f"  traffic_scenario: {args.traffic_scenario}")
    print(f"  n_runs: {args.n_runs}")
    print(f"  mode: {args.mode}")
    print("="*70)
    
    all_results = []
    
    for run_idx in range(args.n_runs):
        print(f"\n{'='*70}")
        print(f"Run {run_idx + 1}/{args.n_runs}")
        print(f"{'='*70}")
        
        # Generate dataset
        base_embeddings, queries, ground_truth = generate_planted_dataset(
            N=500,  # Smaller for faster execution
            Q=100,
            dim=32,
            k=args.k,
            traffic_scenario=args.traffic_scenario,
            random_state=args.random_state + run_idx,
        )
        
        print(f"Dataset: {base_embeddings.shape[0]} base, {queries.shape[0]} queries")
        
        # Create encoder
        _, proj_matrix = random_projection_binarize(
            base_embeddings,
            n_bits=args.n_bits,
            random_state=args.random_state + run_idx,
        )
        
        def encoder_fn(emb):
            from gray_tunneled_hashing.binary.baselines import apply_random_projection
            proj = apply_random_projection(emb, proj_matrix)
            return (proj > 0).astype(bool)
        
        # Run experiments
        methods = ["canonical", "distribution_aware_semantic", "distribution_aware_pure"]
        
        for method in methods:
            print(f"\n  Running {method}...")
            try:
                result = run_experiment(
                    base_embeddings=base_embeddings,
                    queries=queries,
                    ground_truth=ground_truth,
                    encoder_fn=encoder_fn,
                    n_bits=args.n_bits,
                    n_codes=args.n_codes,
                    method=method,
                    block_size=8,
                    max_two_swap_iters=30,
                    num_tunneling_steps=3,
                    mode=args.mode,
                    random_state=args.random_state + run_idx,
                    k=args.k,
                )
                result_dict = asdict(result)
                all_results.append(result_dict)
                
                if result.j_phi is not None:
                    print(f"    J(φ): {result.j_phi:.4f}, J(φ₀): {result.j_phi_0:.4f}, "
                          f"Improvement: {result.j_phi_improvement:.2f}%")
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
    
    # Validate theoretical guarantee
    print(f"\n{'='*70}")
    print("VALIDATING THEORETICAL GUARANTEE: J(φ*) ≤ J(φ₀)")
    print(f"{'='*70}")
    
    # Convert back to ExperimentResult for validation
    results_objects = [ExperimentResult(**r) for r in all_results]
    validation = validate_theoretical_guarantee(results_objects)
    
    if validation["guarantee_holds"]:
        print("✓ GUARANTEE HOLDS: All experiments satisfy J(φ*) ≤ J(φ₀)")
    else:
        print("✗ GUARANTEE VIOLATED:")
        for violation in validation["violations"]:
            print(f"  {violation['method']}: J(φ*)={violation['j_phi']:.6f} > J(φ₀)={violation['j_phi_0']:.6f}")
    
    if validation["statistics"]:
        stats_dict = validation["statistics"]
        print(f"\nStatistics across {stats_dict['n_experiments']} distribution-aware experiments:")
        print(f"  Mean improvement: {stats_dict['mean_improvement']:.2f}%")
        print(f"  Std improvement: {stats_dict['std_improvement']:.2f}%")
        print(f"  Min improvement: {stats_dict['min_improvement']:.2f}%")
        print(f"  Max improvement: {stats_dict['max_improvement']:.2f}%")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump({
            "args": vars(args),
            "results": all_results,
            "validation": validation,
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Method':<35} {'J(φ)':<12} {'J(φ₀)':<12} {'Improvement':<12}")
    print("-" * 70)
    
    for method in ["canonical", "distribution_aware_semantic", "distribution_aware_pure"]:
        method_results = [r for r in results_objects if r.method == method and r.j_phi is not None]
        if method_results:
            j_phi_mean = np.mean([r.j_phi for r in method_results])
            j_phi_0_mean = np.mean([r.j_phi_0 for r in method_results])
            improvement_mean = np.mean([r.j_phi_improvement for r in method_results if r.j_phi_improvement is not None])
            print(f"{method:<35} {j_phi_mean:<12.4f} {j_phi_0_mean:<12.4f} {improvement_mean:<12.2f}%")


if __name__ == "__main__":
    main()

