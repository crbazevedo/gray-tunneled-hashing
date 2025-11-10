#!/usr/bin/env python3
"""
Analyze optimization trajectory: J(φ) vs recall over iterations.

This script instruments optimization to track how J(φ) and recall evolve together.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import argparse
import json
from typing import Dict, List, Tuple
import time
from tqdm import tqdm

from gray_tunneled_hashing.binary.lsh_families import HyperplaneLSH
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.distribution.j_phi_objective import (
    compute_j_phi_cost,
    compute_j_phi_cost_delta_swap,
)
from gray_tunneled_hashing.evaluation.metrics import recall_at_k
from gray_tunneled_hashing.api.query_pipeline import query_with_hamming_ball


def instrumented_hill_climb_with_recall(
    pi_init: np.ndarray,
    pi: np.ndarray,
    w: np.ndarray,
    bucket_to_code: np.ndarray,
    n_bits: int,
    index_obj: any,
    base_embeddings: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
    hamming_radius: int = 1,
    max_iter: int = 100,
    sample_size: int = 256,
    recall_check_frequency: int = 5,  # Check recall every N iterations
    random_state: int = 42,
    verbose: bool = False,
    progress_bar: bool = True,
) -> Tuple[np.ndarray, float, float, List[Dict]]:
    """
    Instrumented hill climbing that tracks both J(φ) and recall.
    """
    np.random.seed(random_state)
    
    N = len(pi_init)
    perm = pi_init.copy()
    initial_cost = compute_j_phi_cost(perm, pi, w, bucket_to_code, n_bits)
    cost = initial_cost
    
    K = len(pi)
    max_valid_embedding_idx = K - 1
    
    history = []
    
    # Build bucket to dataset mapping (once, reused)
    from gray_tunneled_hashing.binary.lsh_families import create_lsh_family
    lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=base_embeddings.shape[1], random_state=random_state)
    base_codes_lsh = lsh.hash(base_embeddings)
    bucket_to_dataset_indices = {}
    for dataset_idx, code in enumerate(base_codes_lsh):
        code_tuple = tuple(code.astype(int).tolist())
        if code_tuple in index_obj.code_to_bucket:
            bucket_idx = index_obj.code_to_bucket[code_tuple]
            if bucket_idx not in bucket_to_dataset_indices:
                bucket_to_dataset_indices[bucket_idx] = []
            bucket_to_dataset_indices[bucket_idx].append(dataset_idx)
    
    def compute_recall(permutation):
        """Helper to compute recall for current permutation."""
        query_codes = lsh.hash(queries)
        recalls = []
        
        for query_idx, query_code in enumerate(query_codes[:20]):  # Sample for speed
            result = query_with_hamming_ball(
                query_code=query_code,
                permutation=permutation,
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
        
        return np.mean(recalls) if recalls else 0.0
    
    # Initial recall
    initial_recall = compute_recall(perm)
    
    iterator = tqdm(range(max_iter), desc="Optimizing", unit="iter", disable=not progress_bar) if progress_bar else range(max_iter)
    
    last_print_time = time.time()
    print_interval = 5.0  # Print every 5 seconds
    
    for iteration in iterator:
        iteration_start = time.time()
        
        # Sample random 2-swaps
        if verbose and iteration == 0:
            print(f"  Starting iteration {iteration}...")
        
        candidates = []
        for _ in range(sample_size):
            u, v = np.random.choice(N, size=2, replace=False)
            candidates.append((u, v))
        
        # Evaluate all candidates
        best_delta = 0.0
        best_swap = None
        
        for u, v in candidates:
            new_u_val = perm[v]
            new_v_val = perm[u]
            
            if new_u_val > max_valid_embedding_idx or new_v_val > max_valid_embedding_idx:
                continue
            
            delta = compute_j_phi_cost_delta_swap(
                perm, pi, w, bucket_to_code, n_bits, u, v, None, None, 0.0
            )
            if delta < best_delta:
                best_delta = delta
                best_swap = (u, v)
        
        # Apply best improving swap
        accepted = False
        if best_swap is not None:
            u, v = best_swap
            temp_u, temp_v = perm[v], perm[u]
            if temp_u <= max_valid_embedding_idx and temp_v <= max_valid_embedding_idx:
                perm[u], perm[v] = temp_u, temp_v
                cost += best_delta
                accepted = True
        
        # Compute recall periodically
        recall = None
        if iteration % recall_check_frequency == 0 or iteration == max_iter - 1:
            if verbose:
                print(f"  Computing recall at iteration {iteration}...")
            recall = compute_recall(perm)
        
        history.append({
            "iteration": iteration,
            "cost": float(cost),
            "delta": float(best_delta) if best_swap is not None else 0.0,
            "accepted": accepted,
            "recall": float(recall) if recall is not None else None,
            "time": time.time() - iteration_start,
        })
        
        # Print progress periodically (every N seconds or every iteration if verbose)
        current_time = time.time()
        should_print = (
            verbose and (iteration % 1 == 0) or  # Every iteration in verbose
            (current_time - last_print_time >= print_interval) or  # Every N seconds
            (iteration % 10 == 0) or  # Every 10 iterations
            recall is not None  # When recall is computed
        )
        
        if should_print:
            elapsed = current_time - last_print_time
            delta_str = f"{best_delta:.4f}" if best_swap is not None else "0.0000"
            recall_str = f"{recall:.4f}" if recall is not None else "N/A"
            print(f"[Iter {iteration:3d}] cost={cost:.4f}, delta={delta_str}, "
                  f"accepted={accepted}, recall={recall_str}, "
                  f"time={elapsed:.1f}s")
            last_print_time = current_time
        
        # Update progress bar
        if progress_bar:
            postfix = {"cost": f"{cost:.3f}"}
            if recall is not None:
                postfix["recall"] = f"{recall:.4f}"
            if best_swap is not None:
                postfix["delta"] = f"{best_delta:.4f}"
            iterator.set_postfix(postfix)
        
        if not accepted:
            if verbose:
                print(f"  No improvement found at iteration {iteration}, stopping.")
            break
    
    # Final recall
    final_recall = compute_recall(perm)
    history[-1]["recall"] = float(final_recall)
    
    return perm, cost, initial_cost, history


def main():
    parser = argparse.ArgumentParser(description="Analyze optimization trajectory")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of base samples")
    parser.add_argument("--n-queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--n-bits", type=int, default=8, help="Number of bits")
    parser.add_argument("--k", type=int, default=10, help="k for recall@k")
    parser.add_argument("--hamming-radius", type=int, default=1, help="Hamming ball radius")
    parser.add_argument("--max-iter", type=int, default=100, help="Max iterations")
    parser.add_argument("--recall-check-frequency", type=int, default=5, help="Check recall every N iterations")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="optimization_trajectory.json", help="Output file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    
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
    
    # Run instrumented optimization
    print("Running instrumented optimization...")
    perm, cost, initial_cost, history = instrumented_hill_climb_with_recall(
        pi_init=pi_init,
        pi=index_obj.pi,
        w=index_obj.w,
        bucket_to_code=index_obj.bucket_to_code,
        n_bits=args.n_bits,
        index_obj=index_obj,
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth=ground_truth,
        k=args.k,
        hamming_radius=args.hamming_radius,
        max_iter=args.max_iter,
        sample_size=256,
        recall_check_frequency=args.recall_check_frequency,
        random_state=args.random_state,
        verbose=args.verbose,
        progress_bar=not args.no_progress,
    )
    
    # Extract trajectory data
    iterations = [h["iteration"] for h in history]
    costs = [h["cost"] for h in history]
    recalls = [h["recall"] for h in history if h["recall"] is not None]
    recall_iterations = [h["iteration"] for h in history if h["recall"] is not None]
    
    # Analyze trajectory
    cost_improvement = initial_cost - cost
    recall_improvement = recalls[-1] - recalls[0] if len(recalls) > 1 else 0.0
    
    # Check if recall and cost are correlated
    if len(recalls) > 1:
        from scipy.stats import pearsonr
        recall_costs = [h["cost"] for h in history if h["recall"] is not None]
        correlation, p_value = pearsonr(recall_costs, recalls)
    else:
        correlation, p_value = 0.0, 1.0
    
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
            "recall_check_frequency": args.recall_check_frequency,
        },
        "trajectory": {
            "initial_cost": float(initial_cost),
            "final_cost": float(cost),
            "cost_improvement": float(cost_improvement),
            "initial_recall": float(recalls[0]) if recalls else None,
            "final_recall": float(recalls[-1]) if recalls else None,
            "recall_improvement": float(recall_improvement),
            "cost_recall_correlation": float(correlation),
            "correlation_p_value": float(p_value),
        },
        "history": history,
    }
    
    with open(args.output, "w") as f:
        json.dump(output_dict, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print(f"Cost improvement: {cost_improvement:.2f}")
    if recalls:
        print(f"Recall improvement: {recall_improvement:.4f}")
        print(f"Cost-Recall correlation: {correlation:.4f} (p={p_value:.4f})")


if __name__ == "__main__":
    main()

