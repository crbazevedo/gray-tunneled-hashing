#!/usr/bin/env python3
"""
Instrument hill climbing to measure optimization dynamics.

This script tests hypothesis H2: Hill climbing is converging to bad local minima.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import argparse
import json
from typing import Dict, List, Tuple, Optional
import time

from gray_tunneled_hashing.binary.lsh_families import HyperplaneLSH
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.distribution.j_phi_objective import hill_climb_j_phi
from gray_tunneled_hashing.evaluation.metrics import recall_at_k
from gray_tunneled_hashing.api.query_pipeline import query_with_hamming_ball


def instrumented_hill_climb(
    pi_init: np.ndarray,
    pi: np.ndarray,
    w: np.ndarray,
    bucket_to_code: np.ndarray,
    n_bits: int,
    max_iter: int = 100,
    sample_size: int = 256,
    random_state: Optional[int] = None,
    index_obj: any = None,
    base_embeddings: np.ndarray = None,
    queries: np.ndarray = None,
    ground_truth: np.ndarray = None,
    k: int = 10,
    hamming_radius: int = 1,
) -> Tuple[np.ndarray, float, float, List[Dict]]:
    """
    Instrumented version of hill_climb_j_phi that tracks recall at each iteration.
    """
    from gray_tunneled_hashing.distribution.j_phi_objective import (
        compute_j_phi_cost,
        compute_j_phi_cost_delta_swap,
    )
    
    if random_state is not None:
        np.random.seed(random_state)
    
    N = len(pi_init)
    perm = pi_init.copy()
    initial_cost = compute_j_phi_cost(perm, pi, w, bucket_to_code, n_bits)
    cost = initial_cost
    
    K = len(pi)
    max_valid_embedding_idx = K - 1
    
    history = []
    n_swaps_tried = 0
    n_swaps_accepted = 0
    delta_distribution = []
    
    for iteration in range(max_iter):
        iteration_start = time.time()
        
        # Sample random 2-swaps
        candidates = []
        for _ in range(sample_size):
            u, v = np.random.choice(N, size=2, replace=False)
            candidates.append((u, v))
        
        # Evaluate all candidates
        best_delta = 0.0
        best_swap = None
        deltas = []
        
        for u, v in candidates:
            new_u_val = perm[v]
            new_v_val = perm[u]
            
            if new_u_val > max_valid_embedding_idx or new_v_val > max_valid_embedding_idx:
                continue
            
            delta = compute_j_phi_cost_delta_swap(
                perm, pi, w, bucket_to_code, n_bits, u, v, None, None, 0.0
            )
            deltas.append(delta)
            n_swaps_tried += 1
            
            if delta < best_delta:
                best_delta = delta
                best_swap = (u, v)
        
        # Record delta distribution
        if deltas:
            delta_distribution.extend(deltas)
        
        # Apply best improving swap
        accepted = False
        if best_swap is not None:
            u, v = best_swap
            temp_u, temp_v = perm[v], perm[u]
            if temp_u <= max_valid_embedding_idx and temp_v <= max_valid_embedding_idx:
                perm[u], perm[v] = temp_u, temp_v
                cost += best_delta
                n_swaps_accepted += 1
                accepted = True
        
        # Compute recall if index_obj provided
        recall = None
        if index_obj is not None and base_embeddings is not None and queries is not None:
            try:
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
                for query_idx, query_code in enumerate(query_codes[:10]):  # Sample first 10 for speed
                    result = query_with_hamming_ball(
                        query_code=query_code,
                        permutation=perm,
                        code_to_bucket=index_obj.code_to_bucket,
                        bucket_to_code=index_obj.bucket_to_code,
                        n_bits=n_bits,
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
                
                recall = np.mean(recalls) if recalls else 0.0
            except Exception as e:
                pass  # Skip recall computation if it fails
        
        history.append({
            "iteration": iteration,
            "cost": float(cost),
            "delta": float(best_delta) if best_swap is not None else 0.0,
            "accepted": accepted,
            "recall": float(recall) if recall is not None else None,
            "time": time.time() - iteration_start,
        })
        
        if not accepted:
            break
    
    return perm, cost, initial_cost, history


def main():
    parser = argparse.ArgumentParser(description="Instrument hill climbing")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of base samples")
    parser.add_argument("--n-queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--n-bits", type=int, default=8, help="Number of bits")
    parser.add_argument("--k", type=int, default=10, help="k for recall@k")
    parser.add_argument("--hamming-radius", type=int, default=1, help="Hamming ball radius")
    parser.add_argument("--max-iter", type=int, default=100, help="Max iterations")
    parser.add_argument("--sample-size", type=int, default=256, help="Sample size per iteration")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="hill_climbing_instrumentation.json", help="Output file")
    
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
    
    # Initialize permutation
    N = 2 ** args.n_bits
    K = index_obj.K
    pi_init = (np.arange(N, dtype=np.int32) % K).astype(np.int32)
    
    # Run instrumented hill climbing
    print("Running instrumented hill climbing...")
    perm, cost, initial_cost, history = instrumented_hill_climb(
        pi_init=pi_init,
        pi=index_obj.pi,
        w=index_obj.w,
        bucket_to_code=index_obj.bucket_to_code,
        n_bits=args.n_bits,
        max_iter=args.max_iter,
        sample_size=args.sample_size,
        random_state=args.random_state,
        index_obj=index_obj,
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth=ground_truth,
        k=args.k,
        hamming_radius=args.hamming_radius,
    )
    
    # Analyze history
    cost_improvement = initial_cost - cost
    cost_improvement_pct = (cost_improvement / initial_cost * 100) if initial_cost > 0 else 0.0
    
    recalls = [h["recall"] for h in history if h["recall"] is not None]
    recall_improvement = (max(recalls) - min(recalls)) if recalls else 0.0
    
    # Save results
    output_dict = {
        "configuration": {
            "n_samples": args.n_samples,
            "n_queries": args.n_queries,
            "dim": args.dim,
            "n_bits": args.n_bits,
            "k": args.k,
            "hamming_radius": args.hamming_radius,
            "max_iter": args.max_iter,
            "sample_size": args.sample_size,
        },
        "results": {
            "initial_cost": float(initial_cost),
            "final_cost": float(cost),
            "cost_improvement": float(cost_improvement),
            "cost_improvement_pct": float(cost_improvement_pct),
            "initial_recall": float(recalls[0]) if recalls else None,
            "final_recall": float(recalls[-1]) if recalls else None,
            "recall_improvement": float(recall_improvement),
            "n_iterations": len(history),
            "n_swaps_accepted": sum(1 for h in history if h["accepted"]),
        },
        "history": history,
    }
    
    with open(args.output, "w") as f:
        json.dump(output_dict, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print(f"Cost improvement: {cost_improvement:.2f} ({cost_improvement_pct:.2f}%)")
    if recalls:
        print(f"Recall improvement: {recall_improvement:.4f}")
        print(f"Final recall: {recalls[-1]:.4f}")


if __name__ == "__main__":
    from typing import Optional
    main()

