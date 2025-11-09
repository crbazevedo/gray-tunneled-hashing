#!/usr/bin/env python3
"""
Analyze impact of block tunneling on recall.

This script tests hypothesis H3: Block tunneling is not being used effectively.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import argparse
import json
from typing import Dict, List
from tqdm import tqdm
import time

from gray_tunneled_hashing.binary.lsh_families import HyperplaneLSH
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.algorithms.gray_tunneled_hasher import GrayTunneledHasher
from gray_tunneled_hashing.evaluation.metrics import recall_at_k
from gray_tunneled_hashing.api.query_pipeline import query_with_hamming_ball


def evaluate_with_tunneling(
    index_obj: any,
    base_embeddings: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
    hamming_radius: int,
    num_tunneling_steps: int,
    block_size: int,
    random_state: int = 42,
    verbose: bool = False,
) -> Dict[str, any]:
    """Evaluate GTH with different tunneling configurations."""
    if verbose:
        print(f"    Config: tunneling_steps={num_tunneling_steps}, block_size={block_size}")
        print(f"    Starting optimization...")
    
    hasher = GrayTunneledHasher(
        n_bits=index_obj.n_bits,
        max_two_swap_iters=50,
        num_tunneling_steps=num_tunneling_steps,
        block_size=block_size,
        random_state=random_state,
        mode="full" if num_tunneling_steps > 0 else "two_swap_only",
    )
    
    start_time = time.time()
    last_print_time = start_time
    print_interval = 10.0
    
    # Add progress monitoring
    original_fit = hasher.fit_with_traffic
    def fit_with_progress(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        current_time = time.time()
        if verbose and (current_time - last_print_time >= print_interval):
            elapsed = current_time - start_time
            if hasattr(hasher, 'cost_') and hasattr(hasher, 'initial_cost_'):
                improvement = ((hasher.initial_cost_ - hasher.cost_) / hasher.initial_cost_ * 100) if hasher.initial_cost_ > 0 else 0
                print(f"      [Progress] Cost: {hasher.cost_:.4f} ({improvement:.1f}% improvement), Time: {elapsed:.1f}s")
        return result
    
    hasher.fit_with_traffic = fit_with_progress
    
    try:
        hasher.fit_with_traffic(
            bucket_embeddings=index_obj.bucket_embeddings,
            pi=index_obj.pi,
            w=index_obj.w,
            use_semantic_distances=False,
            optimize_j_phi_directly=True,
        )
    finally:
        hasher.fit_with_traffic = original_fit
    
    if verbose:
        opt_time = time.time() - start_time
        print(f"    Optimization completed in {opt_time:.2f}s")
    
    permutation = hasher.get_assignment()
    
    # Compute recall
    query_codes = index_obj.lsh.hash(queries)
    base_codes_lsh = index_obj.lsh.hash(base_embeddings)
    
    bucket_to_dataset_indices = {}
    for dataset_idx, code in enumerate(base_codes_lsh):
        code_tuple = tuple(code.astype(int).tolist())
        if code_tuple in index_obj.code_to_bucket:
            bucket_idx = index_obj.code_to_bucket[code_tuple]
            if bucket_idx not in bucket_to_dataset_indices:
                bucket_to_dataset_indices[bucket_idx] = []
            bucket_to_dataset_indices[bucket_idx].append(dataset_idx)
    
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
        
        retrieved_indices = []
        for bucket_idx in result.candidate_indices:
            if bucket_idx in bucket_to_dataset_indices:
                retrieved_indices.extend(bucket_to_dataset_indices[bucket_idx])
        
        retrieved_set = set(retrieved_indices[:k])
        gt_set = set(ground_truth[query_idx][:k])
        if len(gt_set) > 0:
            recalls.append(len(retrieved_set & gt_set) / len(gt_set))
    
    avg_recall = np.mean(recalls) if recalls else 0.0
    
    return {
        "num_tunneling_steps": num_tunneling_steps,
        "block_size": block_size,
        "recall": float(avg_recall),
        "j_phi_cost": float(hasher.cost_),
        "initial_cost": float(hasher.initial_cost_),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze block tunneling impact")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of base samples")
    parser.add_argument("--n-queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--n-bits", type=int, default=8, help="Number of bits")
    parser.add_argument("--k", type=int, default=10, help="k for recall@k")
    parser.add_argument("--hamming-radius", type=int, default=1, help="Hamming ball radius")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="block_tunneling_analysis.json", help="Output file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
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
    
    # Test different tunneling configurations
    print("Testing different tunneling configurations...")
    results = []
    
    configs = [
        (0, 0),  # No tunneling
        (5, 4),  # 5 steps, block_size=4
        (10, 4),  # 10 steps, block_size=4
        (5, 8),  # 5 steps, block_size=8
        (10, 8),  # 10 steps, block_size=8
    ]
    
    for num_tunneling_steps, block_size in configs:
        print(f"  Testing: tunneling_steps={num_tunneling_steps}, block_size={block_size}...")
        result = evaluate_with_tunneling(
            index_obj=index_obj,
            base_embeddings=base_embeddings,
            queries=queries,
            ground_truth=ground_truth,
            hamming_radius=args.hamming_radius,
            num_tunneling_steps=num_tunneling_steps,
            block_size=block_size,
            random_state=args.random_state,
            verbose=args.verbose,
        )
        results.append(result)
        print(f"    Recall: {result['recall']:.4f}, J(φ): {result['j_phi_cost']:.2f}")
    
    # Find best configuration
    best_result = max(results, key=lambda r: r["recall"])
    
    # Save results
    output_dict = {
        "configuration": {
            "n_samples": args.n_samples,
            "n_queries": args.n_queries,
            "dim": args.dim,
            "n_bits": args.n_bits,
            "k": args.k,
            "hamming_radius": args.hamming_radius,
        },
        "results": results,
        "best_configuration": best_result,
        "recommendations": {
            "use_tunneling": best_result["num_tunneling_steps"] > 0,
            "recommended_tunneling_steps": best_result["num_tunneling_steps"],
            "recommended_block_size": best_result["block_size"],
        },
    }
    
    with open(args.output, "w") as f:
        json.dump(output_dict, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print(f"Best configuration: tunneling_steps={best_result['num_tunneling_steps']}, block_size={best_result['block_size']}")
    print(f"Best recall: {best_result['recall']:.4f}")


if __name__ == "__main__":
    main()

