#!/usr/bin/env python3
"""
Compare different optimization methods and objective functions.

This script tests:
1. Hill climbing vs Simulated Annealing vs Memetic Algorithm
2. Standard J(φ) vs Cosine-based objective
3. Impact on recall@k
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import argparse
import json
import time
from typing import Dict, List, Any
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from gray_tunneled_hashing.binary.lsh_families import HyperplaneLSH
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.algorithms.gray_tunneled_hasher import GrayTunneledHasher
from gray_tunneled_hashing.evaluation.metrics import recall_at_k
from gray_tunneled_hashing.api.query_pipeline import query_with_hamming_ball


def evaluate_method(
    method_name: str,
    index_obj: any,
    base_embeddings: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
    hamming_radius: int = 1,
    optimization_method: str = "hill_climb",
    use_cosine_objective: bool = False,
    verbose: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Evaluate a specific optimization method."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating {method_name}...")
        print(f"{'='*60}")
    
    # Get hasher from index_obj if available, otherwise create new one
    if hasattr(index_obj, 'hasher') and index_obj.hasher is not None:
        hasher = index_obj.hasher
    else:
        hasher = GrayTunneledHasher(
            n_bits=index_obj.n_bits,
            max_two_swap_iters=kwargs.get('max_iter', 100),
            two_swap_sample_size=kwargs.get('sample_size', 256),
            num_tunneling_steps=kwargs.get('num_tunneling_steps', 10),
            block_size=kwargs.get('block_size', 8),
            random_state=kwargs.get('random_state', 42),
            mode=kwargs.get('mode', 'full'),
        )
    
    # Fit with traffic
    if verbose:
        print(f"  Starting optimization ({optimization_method})...")
        print(f"  Max iterations: {kwargs.get('max_iter', 100)}")
        print(f"  Sample size: {kwargs.get('sample_size', 256)}")
    
    start_time = time.time()
    last_print_time = start_time
    print_interval = 10.0  # Print every 10 seconds
    
    # Monkey-patch to add progress printing
    original_fit = hasher.fit_with_traffic
    iteration_count = [0]
    
    def fit_with_progress(*args, **fit_kwargs):
        result = original_fit(*args, **fit_kwargs)
        iteration_count[0] += 1
        current_time = time.time()
        if verbose and (current_time - last_print_time >= print_interval):
            elapsed = current_time - start_time
            if hasattr(hasher, 'cost_') and hasattr(hasher, 'initial_cost_'):
                improvement = ((hasher.initial_cost_ - hasher.cost_) / hasher.initial_cost_ * 100) if hasher.initial_cost_ > 0 else 0
                print(f"  [Progress] Iterations: {iteration_count[0]}, "
                      f"Cost: {hasher.cost_:.4f} ({improvement:.1f}% improvement), "
                      f"Time: {elapsed:.1f}s")
        return result
    
    hasher.fit_with_traffic = fit_with_progress
    
    try:
        hasher.fit_with_traffic(
            bucket_embeddings=index_obj.bucket_embeddings,
            pi=index_obj.pi,
            w=index_obj.w,
            use_semantic_distances=False,
            optimize_j_phi_directly=True,
            optimization_method=optimization_method,
            use_cosine_objective=use_cosine_objective,
            cosine_weight=kwargs.get('cosine_weight', 1.0),
            hamming_weight=kwargs.get('hamming_weight', 1.0),
            distance_metric=kwargs.get('distance_metric', 'cosine'),
        )
    finally:
        hasher.fit_with_traffic = original_fit
    
    optimization_time = time.time() - start_time
    
    if verbose:
        print(f"  Optimization completed in {optimization_time:.2f}s")
        if hasattr(hasher, 'cost_') and hasattr(hasher, 'initial_cost_'):
            improvement = ((hasher.initial_cost_ - hasher.cost_) / hasher.initial_cost_ * 100) if hasher.initial_cost_ > 0 else 0
            print(f"  Final cost: {hasher.cost_:.4f} (improvement: {improvement:.1f}%)")
    
    # Get permutation
    permutation = hasher.get_assignment()
    
    # Compute J(φ) cost
    from gray_tunneled_hashing.distribution.j_phi_objective import compute_j_phi_cost
    j_phi_cost = compute_j_phi_cost(
        permutation=permutation,
        pi=index_obj.pi,
        w=index_obj.w,
        bucket_to_code=index_obj.bucket_to_code,
        n_bits=index_obj.n_bits,
    )
    
    # Compute recall@k - need to recreate LSH object
    from gray_tunneled_hashing.binary.lsh_families import create_lsh_family
    lsh = create_lsh_family("hyperplane", n_bits=index_obj.n_bits, dim=base_embeddings.shape[1], random_state=kwargs.get('random_state', 42))
    query_codes = lsh.hash(queries)
    base_codes_lsh = lsh.hash(base_embeddings)
    
    # Build bucket to dataset mapping
    bucket_to_dataset_indices = {}
    for dataset_idx, code in enumerate(base_codes_lsh):
        code_tuple = tuple(code.astype(int).tolist())
        if code_tuple in index_obj.code_to_bucket:
            bucket_idx = index_obj.code_to_bucket[code_tuple]
            if bucket_idx not in bucket_to_dataset_indices:
                bucket_to_dataset_indices[bucket_idx] = []
            bucket_to_dataset_indices[bucket_idx].append(dataset_idx)
    
    # Query with Hamming ball
    recalls = []
    for query_idx, query_code in enumerate(query_codes):
        result = query_with_hamming_ball(
            query_code=query_code,
            permutation=permutation,
            code_to_bucket=index_obj.code_to_bucket,
            bucket_to_code=index_obj.bucket_to_code,
            n_bits=index_obj.n_bits,
            hamming_radius=hamming_radius,
        )
        
        # Get dataset indices from bucket indices
        retrieved_indices = []
        for bucket_idx in result.candidate_indices:
            if bucket_idx in bucket_to_dataset_indices:
                retrieved_indices.extend(bucket_to_dataset_indices[bucket_idx])
        
        # Compute recall
        retrieved_set = set(retrieved_indices[:k])
        gt_set = set(ground_truth[query_idx][:k])
        if len(gt_set) > 0:
            recall = len(retrieved_set & gt_set) / len(gt_set)
            recalls.append(recall)
    
    avg_recall = np.mean(recalls) if recalls else 0.0
    
    return {
        "method_name": method_name,
        "optimization_method": optimization_method,
        "use_cosine_objective": use_cosine_objective,
        "j_phi_cost": float(j_phi_cost),
        "recall": float(avg_recall),
        "optimization_time": float(optimization_time),
        "n_queries": len(queries),
        "avg_candidates_per_query": float(np.mean([len(r.candidate_indices) for r in [query_with_hamming_ball(
            query_code=qc,
            permutation=permutation,
            code_to_bucket=index_obj.code_to_bucket,
            bucket_to_code=index_obj.bucket_to_code,
            n_bits=index_obj.n_bits,
            hamming_radius=hamming_radius,
        ) for qc in query_codes[:10]]])),  # Sample first 10 queries for efficiency
    }


def main():
    parser = argparse.ArgumentParser(description="Compare optimization methods")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of base samples")
    parser.add_argument("--n-queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--n-bits", type=int, default=8, help="Number of bits")
    parser.add_argument("--k", type=int, default=10, help="k for recall@k")
    parser.add_argument("--hamming-radius", type=int, default=1, help="Hamming ball radius")
    parser.add_argument("--max-iter", type=int, default=100, help="Max iterations")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="optimization_methods_comparison.json", help="Output file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--n-workers", type=int, default=1, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Generate synthetic data
    np.random.seed(args.random_state)
    base_embeddings = np.random.randn(args.n_samples, args.dim).astype(np.float32)
    queries = np.random.randn(args.n_queries, args.dim).astype(np.float32)
    
    # Compute ground truth
    from sklearn.metrics.pairwise import euclidean_distances
    distances = euclidean_distances(queries, base_embeddings)
    ground_truth = np.argsort(distances, axis=1)[:, :args.k]
    
    # Build distribution-aware index
    print("Building distribution-aware index...")
    from gray_tunneled_hashing.binary.lsh_families import create_lsh_family
    lsh = create_lsh_family("hyperplane", n_bits=args.n_bits, dim=args.dim, random_state=args.random_state)
    
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        n_bits=args.n_bits,
        lsh_family=lsh,
    )
    
    # Test different methods
    methods = [
        {
            "name": "Hill Climb (J(φ))",
            "optimization_method": "hill_climb",
            "use_cosine_objective": False,
        },
        {
            "name": "Simulated Annealing (J(φ))",
            "optimization_method": "simulated_annealing",
            "use_cosine_objective": False,
        },
        {
            "name": "Memetic Algorithm (J(φ))",
            "optimization_method": "memetic",
            "use_cosine_objective": False,
        },
        {
            "name": "Hill Climb (Cosine)",
            "optimization_method": "hill_climb",
            "use_cosine_objective": True,
            "cosine_weight": 1.0,
            "hamming_weight": 1.0,
        },
        {
            "name": "Simulated Annealing (Cosine)",
            "optimization_method": "simulated_annealing",
            "use_cosine_objective": True,
            "cosine_weight": 1.0,
            "hamming_weight": 1.0,
        },
    ]
    
    results = []
    
    if args.n_workers > 1:
        # Parallel execution
        def run_method(method_config):
            try:
                return evaluate_method(
                    method_name=method_config["name"],
                    index_obj=index_obj,
                    base_embeddings=base_embeddings,
                    queries=queries,
                    ground_truth=ground_truth,
                    k=args.k,
                    hamming_radius=args.hamming_radius,
                    optimization_method=method_config["optimization_method"],
                    use_cosine_objective=method_config.get("use_cosine_objective", False),
                    max_iter=args.max_iter,
                    random_state=args.random_state,
                    **{k: v for k, v in method_config.items() if k not in ["name", "optimization_method", "use_cosine_objective"]},
                )
            except Exception as e:
                return {"error": str(e), "method_name": method_config["name"]}
        
        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            futures = {executor.submit(run_method, m): m for m in methods}
            with tqdm(total=len(methods), desc="Comparing methods", unit="method") as pbar:
                for future in as_completed(futures):
                    method_config = futures[future]
                    try:
                        result = future.result()
                        if "error" in result:
                            if args.verbose:
                                print(f"  ERROR in {result['method_name']}: {result['error']}")
                        else:
                            results.append(result)
                            if args.verbose:
                                print(f"  {result['method_name']}: Recall={result['recall']:.4f}, "
                                      f"J(φ)={result['j_phi_cost']:.2f}, Time={result['optimization_time']:.2f}s")
                        pbar.update(1)
                        if "error" not in result:
                            pbar.set_postfix({"last_recall": f"{result['recall']:.4f}"})
                    except Exception as e:
                        if args.verbose:
                            print(f"  ERROR in {method_config['name']}: {e}")
                        pbar.update(1)
    else:
        # Sequential execution
        for method_config in tqdm(methods, desc="Comparing methods", unit="method"):
            try:
                if args.verbose:
                    print(f"  Testing {method_config['name']}...")
                result = evaluate_method(
                    method_name=method_config["name"],
                    index_obj=index_obj,
                    base_embeddings=base_embeddings,
                    queries=queries,
                    ground_truth=ground_truth,
                    k=args.k,
                    hamming_radius=args.hamming_radius,
                    optimization_method=method_config["optimization_method"],
                    use_cosine_objective=method_config.get("use_cosine_objective", False),
                    max_iter=args.max_iter,
                    random_state=args.random_state,
                    **{k: v for k, v in method_config.items() if k not in ["name", "optimization_method", "use_cosine_objective"]},
                )
                results.append(result)
                if args.verbose:
                    print(f"    Recall: {result['recall']:.4f}, J(φ): {result['j_phi_cost']:.2f}, "
                          f"Time: {result['optimization_time']:.2f}s")
            except Exception as e:
                if args.verbose:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()
    
    # Summary
    summary = {
        "configuration": {
            "n_samples": args.n_samples,
            "n_queries": args.n_queries,
            "dim": args.dim,
            "n_bits": args.n_bits,
            "k": args.k,
            "hamming_radius": args.hamming_radius,
            "max_iter": args.max_iter,
        },
        "results": results,
        "best_recall": max([r["recall"] for r in results]) if results else 0.0,
        "best_method": max(results, key=lambda r: r["recall"])["method_name"] if results else None,
    }
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print(f"\nBest method: {summary['best_method']} (recall: {summary['best_recall']:.4f})")


if __name__ == "__main__":
    main()

