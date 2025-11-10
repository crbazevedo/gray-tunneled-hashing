#!/usr/bin/env python3
"""
Optimize directly for recall instead of J(φ).

This script tests direct recall optimization vs J(φ) optimization.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import argparse
import json
from typing import Dict, List, Optional, Tuple

from gray_tunneled_hashing.binary.lsh_families import HyperplaneLSH
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.distribution.j_phi_objective import hill_climb_j_phi
from gray_tunneled_hashing.distribution.recall_objective import (
    compute_recall_surrogate_cost,
    compute_recall_surrogate_cost_delta_swap,
)
from gray_tunneled_hashing.evaluation.metrics import recall_at_k
from gray_tunneled_hashing.api.query_pipeline import query_with_hamming_ball


def hill_climb_recall_surrogate(
    pi_init: np.ndarray,
    pi: np.ndarray,
    w: np.ndarray,
    bucket_to_code: np.ndarray,
    n_bits: int,
    hamming_radius: int = 1,
    max_iter: int = 100,
    sample_size: int = 256,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, float, float, list]:
    """
    Hill climb to minimize recall surrogate cost.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    N = len(pi_init)
    perm = pi_init.copy()
    initial_cost = compute_recall_surrogate_cost(
        perm, pi, w, bucket_to_code, n_bits, hamming_radius
    )
    cost = initial_cost
    cost_history = [initial_cost]
    
    K = len(pi)
    max_valid_embedding_idx = K - 1
    
    for iteration in range(max_iter):
        candidates = []
        for _ in range(sample_size):
            u, v = np.random.choice(N, size=2, replace=False)
            candidates.append((u, v))
        
        best_delta = 0.0
        best_swap = None
        
        for u, v in candidates:
            new_u_val = perm[v]
            new_v_val = perm[u]
            
            if new_u_val > max_valid_embedding_idx or new_v_val > max_valid_embedding_idx:
                continue
            
            delta = compute_recall_surrogate_cost_delta_swap(
                perm, pi, w, bucket_to_code, n_bits, u, v, hamming_radius
            )
            if delta < best_delta:
                best_delta = delta
                best_swap = (u, v)
        
        if best_swap is not None:
            u, v = best_swap
            temp_u, temp_v = perm[v], perm[u]
            if temp_u <= max_valid_embedding_idx and temp_v <= max_valid_embedding_idx:
                perm[u], perm[v] = temp_u, temp_v
                cost += best_delta
                cost_history.append(cost)
        else:
            break
    
    return perm, cost, initial_cost, cost_history


def main():
    parser = argparse.ArgumentParser(description="Compare recall optimization vs J(φ)")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of base samples")
    parser.add_argument("--n-queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--n-bits", type=int, default=8, help="Number of bits")
    parser.add_argument("--k", type=int, default=10, help="k for recall@k")
    parser.add_argument("--hamming-radius", type=int, default=1, help="Hamming ball radius")
    parser.add_argument("--max-iter", type=int, default=100, help="Max iterations")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="recall_optimization_comparison.json", help="Output file")
    
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
    
    # Optimize with J(φ)
    print("Optimizing with J(φ)...")
    perm_j_phi, cost_j_phi, initial_cost_j_phi, history_j_phi = hill_climb_j_phi(
        pi_init=pi_init,
        pi=index_obj.pi,
        w=index_obj.w,
        bucket_to_code=index_obj.bucket_to_code,
        n_bits=args.n_bits,
        max_iter=args.max_iter,
        sample_size=256,
        random_state=args.random_state,
    )
    
    # Optimize with recall surrogate
    print("Optimizing with recall surrogate...")
    perm_recall, cost_recall, initial_cost_recall, history_recall = hill_climb_recall_surrogate(
        pi_init=pi_init,
        pi=index_obj.pi,
        w=index_obj.w,
        bucket_to_code=index_obj.bucket_to_code,
        n_bits=args.n_bits,
        hamming_radius=args.hamming_radius,
        max_iter=args.max_iter,
        sample_size=256,
        random_state=args.random_state,
    )
    
    # Evaluate recall for both
    def compute_recall(permutation):
        query_codes = lsh.hash(queries)
        base_codes_lsh = lsh.hash(base_embeddings)
        
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
                n_bits=args.n_bits,
                hamming_radius=args.hamming_radius,
            )
            
            retrieved_indices = []
            for bucket_idx in result.candidate_indices:
                if bucket_idx in bucket_to_dataset_indices:
                    retrieved_indices.extend(bucket_to_dataset_indices[bucket_idx])
            
            retrieved_set = set(retrieved_indices[:args.k])
            gt_set = set(ground_truth[query_idx][:args.k])
            if len(gt_set) > 0:
                recalls.append(len(retrieved_set & gt_set) / len(gt_set))
        
        return np.mean(recalls) if recalls else 0.0
    
    recall_j_phi = compute_recall(perm_j_phi)
    recall_recall = compute_recall(perm_recall)
    
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
        },
        "j_phi_optimization": {
            "final_cost": float(cost_j_phi),
            "initial_cost": float(initial_cost_j_phi),
            "cost_improvement": float(initial_cost_j_phi - cost_j_phi),
            "recall": float(recall_j_phi),
        },
        "recall_optimization": {
            "final_cost": float(cost_recall),
            "initial_cost": float(initial_cost_recall),
            "cost_improvement": float(initial_cost_recall - cost_recall),
            "recall": float(recall_recall),
        },
        "comparison": {
            "recall_improvement": float(recall_recall - recall_j_phi),
            "best_method": "recall_surrogate" if recall_recall > recall_j_phi else "j_phi",
        },
    }
    
    with open(args.output, "w") as f:
        json.dump(output_dict, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print(f"J(φ) optimization recall: {recall_j_phi:.4f}")
    print(f"Recall optimization recall: {recall_recall:.4f}")
    print(f"Improvement: {recall_recall - recall_j_phi:.4f}")


if __name__ == "__main__":
    from typing import Optional, Tuple
    main()

