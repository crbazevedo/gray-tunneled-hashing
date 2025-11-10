#!/usr/bin/env python3
"""
Theoretical analysis: Correlation between J(φ) and recall@k.

This script tests hypothesis H1: J(φ) is not a good surrogate for recall@k.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import argparse
import json
from typing import Dict, List, Tuple
from scipy.stats import pearsonr, spearmanr

from gray_tunneled_hashing.binary.lsh_families import HyperplaneLSH
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.distribution.j_phi_objective import compute_j_phi_cost
from gray_tunneled_hashing.evaluation.metrics import recall_at_k
from gray_tunneled_hashing.api.query_pipeline import query_with_hamming_ball


def generate_random_permutations(
    n_bits: int,
    K: int,
    n_permutations: int = 100,
    random_state: int = 42,
) -> List[np.ndarray]:
    """Generate random valid permutations."""
    np.random.seed(random_state)
    N = 2 ** n_bits
    permutations = []
    
    for _ in range(n_permutations):
        base_perm = np.random.permutation(K).astype(np.int32)
        perm = np.array([base_perm[i % K] for i in range(N)], dtype=np.int32)
        np.random.shuffle(perm)
        permutations.append(perm)
    
    return permutations


def evaluate_permutation(
    permutation: np.ndarray,
    index_obj: any,
    base_embeddings: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
    hamming_radius: int = 1,
) -> Dict[str, float]:
    """Evaluate a permutation: compute J(φ) and recall@k."""
    # Compute J(φ) cost
    j_phi_cost = compute_j_phi_cost(
        permutation=permutation,
        pi=index_obj.pi,
        w=index_obj.w,
        bucket_to_code=index_obj.bucket_to_code,
        n_bits=index_obj.n_bits,
        bucket_to_embedding_idx=None,
        semantic_distances=None,
        semantic_weight=0.0,
    )
    
    # Compute recall@k
    # Encode queries
    query_codes = index_obj.lsh.hash(queries)
    
    # Build bucket to dataset mapping
    base_codes_lsh = index_obj.lsh.hash(base_embeddings)
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
    for query_code in query_codes:
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
        gt_set = set(ground_truth[0][:k])  # Use first query's ground truth
        if len(gt_set) > 0:
            recall = len(retrieved_set & gt_set) / len(gt_set)
            recalls.append(recall)
    
    avg_recall = np.mean(recalls) if recalls else 0.0
    
    return {
        "j_phi_cost": j_phi_cost,
        "recall": avg_recall,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze J(φ) vs recall correlation")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of base samples")
    parser.add_argument("--n-queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--n-bits", type=int, default=8, help="Number of bits")
    parser.add_argument("--k", type=int, default=10, help="k for recall@k")
    parser.add_argument("--hamming-radius", type=int, default=1, help="Hamming ball radius")
    parser.add_argument("--n-permutations", type=int, default=50, help="Number of random permutations to test")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="j_phi_vs_recall_analysis.json", help="Output file")
    
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
    
    # Generate random permutations
    print(f"Generating {args.n_permutations} random permutations...")
    permutations = generate_random_permutations(
        n_bits=args.n_bits,
        K=index_obj.K,
        n_permutations=args.n_permutations,
        random_state=args.random_state,
    )
    
    # Evaluate each permutation
    print("Evaluating permutations...")
    results = []
    for i, perm in enumerate(permutations):
        if (i + 1) % 10 == 0:
            print(f"  Evaluated {i + 1}/{args.n_permutations} permutations...")
        
        metrics = evaluate_permutation(
            permutation=perm,
            index_obj=index_obj,
            base_embeddings=base_embeddings,
            queries=queries,
            ground_truth=ground_truth,
            k=args.k,
            hamming_radius=args.hamming_radius,
        )
        results.append(metrics)
    
    # Compute correlation
    j_phi_costs = [r["j_phi_cost"] for r in results]
    recalls = [r["recall"] for r in results]
    
    pearson_corr, pearson_p = pearsonr(j_phi_costs, recalls)
    spearman_corr, spearman_p = spearmanr(j_phi_costs, recalls)
    
    # Summary statistics
    summary = {
        "n_permutations": args.n_permutations,
        "j_phi_stats": {
            "mean": np.mean(j_phi_costs),
            "std": np.std(j_phi_costs),
            "min": np.min(j_phi_costs),
            "max": np.max(j_phi_costs),
        },
        "recall_stats": {
            "mean": np.mean(recalls),
            "std": np.std(recalls),
            "min": np.min(recalls),
            "max": np.max(recalls),
        },
        "correlations": {
            "pearson": {
                "correlation": float(pearson_corr),
                "p_value": float(pearson_p),
            },
            "spearman": {
                "correlation": float(spearman_corr),
                "p_value": float(spearman_p),
            },
        },
        "results": results,
    }
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print(f"Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.4f})")
    print(f"Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.4f})")
    
    if abs(pearson_corr) < 0.3:
        print("\n⚠️  WARNING: Low correlation suggests J(φ) is not a good surrogate for recall@k")
    else:
        print("\n✓ Correlation is moderate to high, J(φ) may be a reasonable surrogate")


if __name__ == "__main__":
    main()

