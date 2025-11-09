#!/usr/bin/env python3
"""
Compare different initialization strategies.

This script tests different initialization approaches for GTH optimization.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import argparse
import json
import time
from typing import Dict, List
from sklearn.metrics.pairwise import cosine_similarity

from gray_tunneled_hashing.binary.lsh_families import HyperplaneLSH
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.algorithms.gray_tunneled_hasher import GrayTunneledHasher
from gray_tunneled_hashing.evaluation.metrics import recall_at_k
from gray_tunneled_hashing.api.query_pipeline import query_with_hamming_ball


def create_initialization(
    strategy: str,
    N: int,
    K: int,
    bucket_embeddings: np.ndarray = None,
    random_state: int = 42,
) -> np.ndarray:
    """Create initial permutation using different strategies."""
    np.random.seed(random_state)
    
    if strategy == "identity":
        # Identity: vertex i -> embedding (i % K)
        return (np.arange(N, dtype=np.int32) % K).astype(np.int32)
    
    elif strategy == "random":
        # Random valid permutation
        base_perm = np.random.permutation(K).astype(np.int32)
        perm = np.array([base_perm[i % K] for i in range(N)], dtype=np.int32)
        np.random.shuffle(perm)
        return perm
    
    elif strategy == "gray_code":
        # Gray-code based: use Gray sequence
        from gray_tunneled_hashing.algorithms.gray_tunneled_hasher import GrayTunneledHasher
        hasher = GrayTunneledHasher(n_bits=int(np.log2(N)), init_strategy="identity")
        gray_seq = hasher._generate_gray_code_sequence(int(np.log2(N)))
        return (gray_seq % K).astype(np.int32)
    
    elif strategy == "semantic":
        # Semantic-based: sort buckets by embedding similarity
        if bucket_embeddings is None:
            return (np.arange(N, dtype=np.int32) % K).astype(np.int32)
        
        # Compute similarity matrix
        similarities = cosine_similarity(bucket_embeddings)
        
        # Use first bucket as anchor
        sorted_indices = [0]
        remaining = set(range(1, K))
        
        while remaining:
            last_idx = sorted_indices[-1]
            # Find most similar remaining bucket
            best_idx = max(remaining, key=lambda i: similarities[last_idx, i])
            sorted_indices.append(best_idx)
            remaining.remove(best_idx)
        
        # Map to vertices
        perm = np.zeros(N, dtype=np.int32)
        for vertex_idx in range(N):
            perm[vertex_idx] = sorted_indices[vertex_idx % K]
        
        return perm
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def evaluate_initialization(
    strategy: str,
    index_obj: any,
    base_embeddings: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
    hamming_radius: int,
    random_state: int = 42,
    verbose: bool = False,
) -> Dict[str, any]:
    """Evaluate GTH with specific initialization strategy."""
    if verbose:
        print(f"      Strategy: {strategy}")
        print(f"      Creating initialization...")
    
    N = 2 ** index_obj.n_bits
    K = index_obj.K
    
    # Create initialization
    pi_init = create_initialization(
        strategy=strategy,
        N=N,
        K=K,
        bucket_embeddings=index_obj.bucket_embeddings,
        random_state=random_state,
    )
    
    if verbose:
        print(f"      Starting optimization...")
    
    # Run optimization
    hasher = GrayTunneledHasher(
        n_bits=index_obj.n_bits,
        max_two_swap_iters=100,
        num_tunneling_steps=10,
        block_size=8,
        random_state=random_state,
        mode="full",
    )
    
    # Manually set initial permutation
    hasher.pi_init_ = pi_init
    
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
                print(f"        [Progress] Cost: {hasher.cost_:.4f} ({improvement:.1f}% improvement), Time: {elapsed:.1f}s")
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
        print(f"      Optimization completed in {opt_time:.2f}s")
    
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
        "strategy": strategy,
        "recall": float(avg_recall),
        "j_phi_cost": float(hasher.cost_),
        "initial_cost": float(hasher.initial_cost_),
        "cost_improvement": float(hasher.initial_cost_ - hasher.cost_),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare initialization strategies")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of base samples")
    parser.add_argument("--n-queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--n-bits", type=int, default=8, help="Number of bits")
    parser.add_argument("--k", type=int, default=10, help="k for recall@k")
    parser.add_argument("--hamming-radius", type=int, default=1, help="Hamming ball radius")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="initialization_strategies_analysis.json", help="Output file")
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
    
    # Test different initialization strategies
    print("Testing different initialization strategies...")
    strategies = ["identity", "random", "gray_code", "semantic"]
    results = []
    
    for strategy in strategies:
        print(f"  Testing {strategy}...")
        result = evaluate_initialization(
            strategy=strategy,
            index_obj=index_obj,
            base_embeddings=base_embeddings,
            queries=queries,
            ground_truth=ground_truth,
                k=args.k,
                hamming_radius=args.hamming_radius,
                random_state=args.random_state,
                verbose=args.verbose,
            )
        results.append(result)
        print(f"    Recall: {result['recall']:.4f}, Cost improvement: {result['cost_improvement']:.2f}")
    
    # Find best strategy
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
        "best_strategy": best_result["strategy"],
        "recommendations": {
            "recommended_strategy": best_result["strategy"],
            "recall_improvement_vs_identity": (
                best_result["recall"] - next(r["recall"] for r in results if r["strategy"] == "identity")
            ),
        },
    }
    
    with open(args.output, "w") as f:
        json.dump(output_dict, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print(f"Best strategy: {best_result['strategy']}")
    print(f"Best recall: {best_result['recall']:.4f}")


if __name__ == "__main__":
    main()

