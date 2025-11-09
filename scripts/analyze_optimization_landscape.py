#!/usr/bin/env python3
"""
Analyze optimization landscape by sampling random permutations.

This script samples the space of permutations to understand the landscape structure.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import argparse
import json
from typing import Dict, List, Tuple
from scipy.stats import pearsonr

from gray_tunneled_hashing.binary.lsh_families import HyperplaneLSH
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.distribution.j_phi_objective import compute_j_phi_cost
from gray_tunneled_hashing.evaluation.metrics import recall_at_k
from gray_tunneled_hashing.api.query_pipeline import query_with_hamming_ball


def sample_permutations(
    N: int,
    K: int,
    n_samples: int = 100,
    random_state: int = 42,
) -> List[np.ndarray]:
    """Sample random valid permutations."""
    np.random.seed(random_state)
    permutations = []
    
    for _ in range(n_samples):
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
    """Evaluate a permutation: compute J(φ) and recall."""
    # Compute J(φ)
    j_phi = compute_j_phi_cost(
        permutation, index_obj.pi, index_obj.w, index_obj.bucket_to_code,
        index_obj.n_bits, None, None, 0.0
    )
    
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
    
    recall = np.mean(recalls) if recalls else 0.0
    
    return {
        "j_phi": float(j_phi),
        "recall": float(recall),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze optimization landscape")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of base samples")
    parser.add_argument("--n-queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--n-bits", type=int, default=8, help="Number of bits")
    parser.add_argument("--k", type=int, default=10, help="k for recall@k")
    parser.add_argument("--hamming-radius", type=int, default=1, help="Hamming ball radius")
    parser.add_argument("--n-permutations", type=int, default=100, help="Number of permutations to sample")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="optimization_landscape.json", help="Output file")
    
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
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth=ground_truth,
        lsh_family="hyperplane",
        n_bits=args.n_bits,
        k=args.k,
    )
    
    # Sample permutations
    print(f"Sampling {args.n_permutations} random permutations...")
    N = 2 ** args.n_bits
    K = index_obj.K
    permutations = sample_permutations(N, K, args.n_permutations, args.random_state)
    
    # Evaluate each permutation
    print("Evaluating permutations...")
    results = []
    for i, perm in enumerate(permutations):
        if (i + 1) % 10 == 0:
            print(f"  Evaluated {i + 1}/{args.n_permutations}...")
        
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
    
    # Analyze landscape
    j_phi_values = [r["j_phi"] for r in results]
    recall_values = [r["recall"] for r in results]
    
    correlation, p_value = pearsonr(j_phi_values, recall_values)
    
    # Statistics
    landscape_stats = {
        "j_phi": {
            "mean": float(np.mean(j_phi_values)),
            "std": float(np.std(j_phi_values)),
            "min": float(np.min(j_phi_values)),
            "max": float(np.max(j_phi_values)),
            "percentiles": {
                "25": float(np.percentile(j_phi_values, 25)),
                "50": float(np.percentile(j_phi_values, 50)),
                "75": float(np.percentile(j_phi_values, 75)),
            },
        },
        "recall": {
            "mean": float(np.mean(recall_values)),
            "std": float(np.std(recall_values)),
            "min": float(np.min(recall_values)),
            "max": float(np.max(recall_values)),
            "percentiles": {
                "25": float(np.percentile(recall_values, 25)),
                "50": float(np.percentile(recall_values, 50)),
                "75": float(np.percentile(recall_values, 75)),
            },
        },
        "correlation": {
            "pearson": float(correlation),
            "p_value": float(p_value),
        },
    }
    
    # Save results
    output_dict = {
        "configuration": {
            "n_samples": args.n_samples,
            "n_queries": args.n_queries,
            "dim": args.dim,
            "n_bits": args.n_bits,
            "k": args.k,
            "hamming_radius": args.hamming_radius,
            "n_permutations": args.n_permutations,
        },
        "landscape_stats": landscape_stats,
        "sample_results": results[:20],  # Save first 20 for reference
    }
    
    with open(args.output, "w") as f:
        json.dump(output_dict, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print(f"J(φ) range: [{landscape_stats['j_phi']['min']:.2f}, {landscape_stats['j_phi']['max']:.2f}]")
    print(f"Recall range: [{landscape_stats['recall']['min']:.4f}, {landscape_stats['recall']['max']:.4f}]")
    print(f"Correlation: {correlation:.4f} (p={p_value:.4f})")


if __name__ == "__main__":
    main()

