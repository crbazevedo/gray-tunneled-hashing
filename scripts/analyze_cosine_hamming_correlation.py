#!/usr/bin/env python3
"""
Analyze correlation between cosine distances and Hamming distances.

This helps determine the best objective function formulation for GTH.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import argparse
import json
from scipy.stats import pearsonr, spearmanr

from gray_tunneled_hashing.binary.lsh_families import HyperplaneLSH
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.distribution.cosine_objective import analyze_cosine_hamming_correlation
from gray_tunneled_hashing.evaluation.metrics import hamming_distance


def main():
    parser = argparse.ArgumentParser(description="Analyze cosine-Hamming correlation")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of base samples")
    parser.add_argument("--n-queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--n-bits", type=int, default=8, help="Number of bits")
    parser.add_argument("--k", type=int, default=10, help="k for recall@k")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="cosine_hamming_correlation.json", help="Output file")
    
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
    
    # Get bucket embeddings (use centroids or representative embeddings)
    # For now, use the first embedding in each bucket
    bucket_embeddings = []
    base_codes_lsh = lsh.hash(base_embeddings)
    for bucket_idx in range(index_obj.K):
        # Find first embedding in this bucket
        for dataset_idx, code in enumerate(base_codes_lsh):
            code_tuple = tuple(code.astype(int).tolist())
            if code_tuple in index_obj.code_to_bucket:
                if index_obj.code_to_bucket[code_tuple] == bucket_idx:
                    bucket_embeddings.append(base_embeddings[dataset_idx])
                    break
        else:
            # No embedding in this bucket, use zero vector
            bucket_embeddings.append(np.zeros(args.dim, dtype=np.float32))
    
    bucket_embeddings = np.array(bucket_embeddings)
    
    # Analyze correlation
    print("Analyzing cosine-Hamming correlation...")
    analysis = analyze_cosine_hamming_correlation(
        bucket_embeddings=bucket_embeddings,
        bucket_to_code=index_obj.bucket_to_code,
        n_bits=args.n_bits,
    )
    
    # Additional analysis: correlation after GTH optimization
    print("Computing correlation after GTH optimization...")
    # TODO: Run GTH optimization and analyze correlation of optimized permutation
    
    # Save results
    output_dict = {
        "configuration": {
            "n_samples": args.n_samples,
            "n_queries": args.n_queries,
            "dim": args.dim,
            "n_bits": args.n_bits,
            "k": args.k,
        },
        "correlation_analysis": {
            "pearson_correlation": float(analysis["correlation"]),
            "cosine_mean": float(analysis["cosine_mean"]),
            "cosine_std": float(analysis["cosine_std"]),
            "hamming_mean": float(analysis["hamming_mean"]),
            "hamming_std": float(analysis["hamming_std"]),
            "scale_factor": float(analysis["scale_factor"]),
        },
        "recommendations": {
            "use_cosine_objective": abs(analysis["correlation"]) > 0.3,
            "recommended_scale_factor": float(analysis["scale_factor"]),
            "recommended_weights": {
                "cosine_weight": 1.0 if abs(analysis["correlation"]) > 0.5 else 0.5,
                "hamming_weight": 1.0,
            },
        },
    }
    
    with open(args.output, "w") as f:
        json.dump(output_dict, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print(f"Pearson correlation: {analysis['correlation']:.4f}")
    print(f"Scale factor: {analysis['scale_factor']:.4f}")
    
    if abs(analysis["correlation"]) > 0.3:
        print("\n✓ Strong correlation suggests cosine objective may improve recall")
    else:
        print("\n⚠️  Weak correlation suggests cosine objective may not help much")


if __name__ == "__main__":
    main()

