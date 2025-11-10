"""
Lightweight benchmark for Distribution-Aware GTH validation.

Reduced configuration for quick validation:
- n_bits: 6 (vs 8-10 no benchmark completo)
- n_codes: 16 (vs 32-64)
- n_runs: 2 (vs 5-10)
- max_two_swap_iters: 10 (vs 30-100)
- sample_size: 32 (vs 256)
- num_tunneling_steps: 0 (desabilitado para velocidade)

Tempo esperado: ~1-2 minutos (vs 10-20 minutos do benchmark completo)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gray_tunneled_hashing.binary.baselines import random_projection_binarize
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.distribution.j_phi_objective import compute_j_phi_0, compute_j_phi_cost


# Colors for output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    run_id: int
    n_bits: int
    n_codes: int
    traffic_scenario: str
    method: str
    j_phi_star: float
    j_phi_0: float
    guarantee_holds: bool
    improvement: float
    build_time: float


def run_single_experiment(args_tuple) -> Dict:
    """Run a single experiment (for parallel execution)."""
    (run_id, n_bits, n_codes, k, traffic_scenario, method, random_state) = args_tuple
    
    np.random.seed(random_state)
    
    # Generate synthetic data (smaller for speed)
    N = 200  # Reduced from 500
    Q = 50   # Reduced from 100
    dim = 16  # Reduced from 32
    
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    
    # Generate ground truth based on traffic scenario
    if traffic_scenario == "uniform":
        ground_truth = np.random.randint(0, N, size=(Q, k), dtype=np.int32)
    elif traffic_scenario == "skewed":
        # 80% of queries from 20% of space
        hot_indices = np.random.choice(N, size=N // 5, replace=False)
        ground_truth = np.zeros((Q, k), dtype=np.int32)
        for i in range(Q):
            if np.random.rand() < 0.8:
                neighbors = np.random.choice(hot_indices, size=k, replace=True)
            else:
                neighbors = np.random.randint(0, N, size=k)
            ground_truth[i] = neighbors
    else:  # clustered
        n_clusters = np.random.randint(3, 5)
        cluster_centers = np.random.randn(n_clusters, dim).astype(np.float32)
        cluster_assignments = np.random.randint(0, n_clusters, size=N)
        ground_truth = np.zeros((Q, k), dtype=np.int32)
        for i in range(Q):
            q_cluster = np.random.randint(0, n_clusters)
            same_cluster = np.where(cluster_assignments == q_cluster)[0]
            if len(same_cluster) >= k:
                neighbors = np.random.choice(same_cluster, size=k, replace=False)
            else:
                neighbors = np.concatenate([
                    same_cluster,
                    np.random.randint(0, N, size=k - len(same_cluster))
                ])
            ground_truth[i] = neighbors
    
    # Create encoder
    _, proj_matrix = random_projection_binarize(
        base_embeddings,
        n_bits=n_bits,
        random_state=random_state,
    )
    
    def encoder_fn(emb):
        from gray_tunneled_hashing.binary.baselines import apply_random_projection
        proj = apply_random_projection(emb, proj_matrix)
        return (proj > 0).astype(bool)
    
    start_time = time.time()
    
    # Build distribution-aware index
    use_semantic = method == "distribution_aware_semantic"
    
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=encoder_fn,
        n_bits=n_bits,
        n_codes=n_codes,
        use_codebook=True,
        use_semantic_distances=use_semantic,
        block_size=4,  # Reduced from 8
        max_two_swap_iters=10,  # Reduced from 30-100
        num_tunneling_steps=0,  # Disabled for speed
        mode="two_swap_only",
        random_state=random_state,
    )
    
    build_time = time.time() - start_time
    
    # Get J(φ₀) and J(φ*)
    initial_perm = index_obj.hasher.get_initial_permutation()
    if initial_perm is None:
        N = 2 ** n_bits
        initial_perm = np.arange(N, dtype=np.int32)
    
    # Calculate semantic distances if used
    semantic_distances = None
    semantic_weight = 0.0
    if use_semantic and hasattr(index_obj.hasher, 'semantic_distances_'):
        semantic_distances = index_obj.hasher.semantic_distances_
        semantic_weight = index_obj.hasher.semantic_weight_
    
    j_phi_0 = compute_j_phi_0(
        pi=index_obj.pi,
        w=index_obj.w,
        bucket_to_code=index_obj.bucket_to_code,
        n_bits=n_bits,
        initial_permutation=initial_perm,
        semantic_distances=semantic_distances,
        semantic_weight=semantic_weight,
    )
    
    # J(φ*): Use learned permutation
    j_phi_star = compute_j_phi_cost(
        permutation=index_obj.permutation,
        pi=index_obj.pi,
        w=index_obj.w,
        bucket_to_code=index_obj.bucket_to_code,
        n_bits=n_bits,
        bucket_to_embedding_idx=None,
        semantic_distances=semantic_distances,
        semantic_weight=semantic_weight,
    )
    
    guarantee_holds = j_phi_star <= j_phi_0 + 1e-6
    improvement = ((j_phi_0 - j_phi_star) / j_phi_0 * 100) if j_phi_0 > 0 else 0.0
    
    return {
        "run_id": run_id,
        "n_bits": n_bits,
        "n_codes": n_codes,
        "traffic_scenario": traffic_scenario,
        "method": method,
        "j_phi_star": float(j_phi_star),
        "j_phi_0": float(j_phi_0),
        "guarantee_holds": bool(guarantee_holds),
        "improvement": float(improvement),
        "build_time": float(build_time),
    }


def main():
    parser = argparse.ArgumentParser(description="Lightweight benchmark for Distribution-Aware GTH")
    parser.add_argument("--n-bits", type=int, default=6, help="Number of bits (reduced for speed)")
    parser.add_argument("--n-codes", type=int, default=16, help="Number of codebook vectors (reduced)")
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors")
    parser.add_argument("--traffic-scenario", type=str, default="skewed",
                        choices=["uniform", "skewed", "clustered"],
                        help="Traffic scenario")
    parser.add_argument("--n-runs", type=int, default=2, help="Number of runs (reduced for speed)")
    parser.add_argument("--n-workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--output", type=str,
                        default="experiments/real/results_benchmark_lightweight.json")
    
    args = parser.parse_args()
    
    if args.n_workers is None:
        args.n_workers = min(args.n_runs * 2, mp.cpu_count())  # 2 methods per run
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}LIGHTWEIGHT BENCHMARK: Distribution-Aware GTH{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}Configuration:{Colors.END}")
    print(f"  n_bits: {args.n_bits} (reduced)")
    print(f"  n_codes: {args.n_codes} (reduced)")
    print(f"  k: {args.k}")
    print(f"  traffic_scenario: {args.traffic_scenario}")
    print(f"  n_runs: {args.n_runs} (reduced)")
    print(f"  n_workers: {args.n_workers}")
    print(f"{Colors.YELLOW}Expected time: ~1-2 minutes{Colors.END}")
    print()
    
    # Prepare arguments for parallel execution
    methods = ["distribution_aware_semantic", "distribution_aware_pure"]
    configs = []
    for method in methods:
        for run_id in range(args.n_runs):
            configs.append((
                run_id,
                args.n_bits,
                args.n_codes,
                args.k,
                args.traffic_scenario,
                method,
                42 + run_id,
            ))
    
    # Run experiments in parallel
    print(f"{Colors.YELLOW}Running {len(configs)} experiments in parallel...{Colors.END}")
    all_results = []
    
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {executor.submit(run_single_experiment, config): config for config in configs}
        
        with tqdm(total=len(configs), desc="Benchmarking", colour="green") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    all_results.append(result)
                    pbar.update(1)
                except Exception as e:
                    config = futures[future]
                    print(f"{Colors.RED}Error in config {config}: {e}{Colors.END}")
                    import traceback
                    traceback.print_exc()
    
    # Analyze results
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}RESULTS{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    
    # Count violations
    violations = [r for r in all_results if not r["guarantee_holds"]]
    n_violations = len(violations)
    n_total = len(all_results)
    
    if n_violations == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ GUARANTEE HOLDS: {n_total}/{n_total} experiments (100%){Colors.END}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ GUARANTEE VIOLATED: {n_violations}/{n_total} experiments{Colors.END}")
    
    # Statistics by method
    print(f"\n{Colors.BOLD}Statistics by method:{Colors.END}\n")
    for method in methods:
        method_results = [r for r in all_results if r["method"] == method]
        method_violations = [r for r in method_results if not r["guarantee_holds"]]
        
        if len(method_results) > 0:
            improvements = [r["improvement"] for r in method_results]
            avg_improvement = np.mean(improvements)
            std_improvement = np.std(improvements)
            
            color = Colors.GREEN if len(method_violations) == 0 else Colors.RED
            print(f"{Colors.BOLD}{method}:{Colors.END}")
            print(f"  {color}Guarantee: {len(method_results) - len(method_violations)}/{len(method_results)} pass{Colors.END}")
            print(f"  {Colors.BLUE}Avg improvement: {avg_improvement:.2f}% (std: {std_improvement:.2f}%){Colors.END}")
            print()
    
    # Check if methods produce different results
    semantic_results = [r for r in all_results if r["method"] == "distribution_aware_semantic"]
    pure_results = [r for r in all_results if r["method"] == "distribution_aware_pure"]
    
    if semantic_results and pure_results:
        semantic_improvements = [r["improvement"] for r in semantic_results]
        pure_improvements = [r["improvement"] for r in pure_results]
        
        mean_semantic = np.mean(semantic_improvements)
        mean_pure = np.mean(pure_improvements)
        diff = abs(mean_semantic - mean_pure)
        
        print(f"{Colors.BOLD}Method Comparison:{Colors.END}")
        print(f"  Semantic: {mean_semantic:.2f}% (mean)")
        print(f"  Pure: {mean_pure:.2f}% (mean)")
        print(f"  Difference: {diff:.2f}%")
        
        if diff > 0.1:
            print(f"  {Colors.GREEN}✓ Methods produce different results (fix validated!){Colors.END}")
        else:
            print(f"  {Colors.YELLOW}⚠ Methods still produce similar results{Colors.END}")
        print()
    
    # Overall statistics
    improvements = [r["improvement"] for r in all_results]
    print(f"{Colors.BOLD}Overall statistics:{Colors.END}")
    print(f"  Mean improvement: {np.mean(improvements):.2f}%")
    print(f"  Std improvement: {np.std(improvements):.2f}%")
    print(f"  Min improvement: {np.min(improvements):.2f}%")
    print(f"  Max improvement: {np.max(improvements):.2f}%")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump({
            "args": vars(args),
            "summary": {
                "n_total": n_total,
                "n_violations": n_violations,
                "violation_rate": n_violations / n_total if n_total > 0 else 0,
                "mean_improvement": float(np.mean(improvements)),
                "std_improvement": float(np.std(improvements)),
            },
            "results": all_results,
        }, f, indent=2)
    
    print(f"\n{Colors.GREEN}Results saved to: {output_path}{Colors.END}")


if __name__ == "__main__":
    main()

