#!/usr/bin/env python3
"""
Analyze contribution of each term in J(φ) to recall.

This script tests hypothesis H5: Optimization is focusing on wrong terms.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import argparse
import json
from typing import Dict, List, Tuple
from collections import defaultdict

from gray_tunneled_hashing.binary.lsh_families import HyperplaneLSH
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.distribution.j_phi_objective import compute_j_phi_cost
from gray_tunneled_hashing.evaluation.metrics import recall_at_k, hamming_distance
from gray_tunneled_hashing.api.query_pipeline import query_with_hamming_ball
from gray_tunneled_hashing.data.synthetic_generators import generate_hypercube_vertices


def analyze_term_contribution(
    permutation: np.ndarray,
    index_obj: any,
    base_embeddings: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
    hamming_radius: int = 1,
) -> Dict[str, any]:
    """
    Analyze contribution of each term in J(φ) to recall.
    
    For each pair (i, j) in J(φ) = Σ π_i · w_ij · d_H(φ(c_i), φ(c_j)):
    - Measure if this term affects recall
    - Categorize terms by d_H value
    - Measure recall-weighted cost
    """
    vertices = generate_hypercube_vertices(index_obj.n_bits)
    N = len(vertices)
    K = len(index_obj.pi)
    
    # Map bucket to vertex
    bucket_to_vertex = {}
    for vertex_idx in range(N):
        embedding_idx = permutation[vertex_idx]
        if embedding_idx < K:
            if embedding_idx not in bucket_to_vertex:
                bucket_to_vertex[embedding_idx] = vertex_idx
    
    # Build query-to-bucket mapping
    from gray_tunneled_hashing.binary.lsh_families import create_lsh_family
    lsh = create_lsh_family("hyperplane", n_bits=index_obj.n_bits, dim=base_embeddings.shape[1], random_state=42)
    query_codes = lsh.hash(queries)
    query_buckets = {}
    for query_idx, query_code in enumerate(query_codes):
        code_tuple = tuple(query_code.astype(int).tolist())
        if code_tuple in index_obj.code_to_bucket:
            bucket_idx = index_obj.code_to_bucket[code_tuple]
            query_buckets[query_idx] = bucket_idx
    
    # Build base embedding to bucket mapping
    from gray_tunneled_hashing.binary.lsh_families import create_lsh_family
    lsh = create_lsh_family("hyperplane", n_bits=index_obj.n_bits, dim=base_embeddings.shape[1], random_state=42)
    base_codes_lsh = lsh.hash(base_embeddings)
    base_buckets = {}
    for dataset_idx, code in enumerate(base_codes_lsh):
        code_tuple = tuple(code.astype(int).tolist())
        if code_tuple in index_obj.code_to_bucket:
            bucket_idx = index_obj.code_to_bucket[code_tuple]
            base_buckets[dataset_idx] = bucket_idx
    
    # Analyze terms by Hamming distance
    term_stats = defaultdict(lambda: {
        "count": 0,
        "total_weight": 0.0,
        "total_cost": 0.0,
        "recall_relevant": 0,  # How many terms involve query-neighbor pairs
    })
    
    # Recall-weighted cost: only terms where d_H <= radius
    recall_weighted_cost = 0.0
    total_j_phi = 0.0
    
    for i in range(K):
        if i not in bucket_to_vertex:
            continue
        
        code_i = vertices[bucket_to_vertex[i]]
        
        for j in range(K):
            if j not in bucket_to_vertex:
                continue
            
            code_j = vertices[bucket_to_vertex[j]]
            d_h = hamming_distance(code_i[np.newaxis, :], code_j[np.newaxis, :])[0, 0]
            
            # J(φ) term
            term = index_obj.pi[i] * index_obj.w[i, j] * d_h
            total_j_phi += term
            
            # Categorize by d_H
            term_stats[d_h]["count"] += 1
            term_stats[d_h]["total_weight"] += index_obj.pi[i] * index_obj.w[i, j]
            term_stats[d_h]["total_cost"] += term
            
            # Check if this term is recall-relevant (d_H <= radius)
            if d_h <= hamming_radius:
                recall_weighted_cost += term
                
                # Check if this pair (i, j) appears in query-neighbor pairs
                for query_idx, query_bucket in query_buckets.items():
                    if query_bucket == i:
                        # Check if any neighbor is in bucket j
                        for neighbor_idx in ground_truth[query_idx][:k]:
                            if neighbor_idx in base_buckets and base_buckets[neighbor_idx] == j:
                                term_stats[d_h]["recall_relevant"] += 1
                                break
    
    # Convert to lists for JSON serialization
    term_stats_list = [
        {
            "hamming_distance": d_h,
            "count": stats["count"],
            "total_weight": float(stats["total_weight"]),
            "total_cost": float(stats["total_cost"]),
            "avg_cost_per_term": float(stats["total_cost"] / stats["count"]) if stats["count"] > 0 else 0.0,
            "recall_relevant_count": stats["recall_relevant"],
            "recall_relevant_fraction": float(stats["recall_relevant"] / stats["count"]) if stats["count"] > 0 else 0.0,
        }
        for d_h, stats in sorted(term_stats.items())
    ]
    
    return {
        "total_j_phi": float(total_j_phi),
        "recall_weighted_cost": float(recall_weighted_cost),
        "recall_weighted_fraction": float(recall_weighted_cost / total_j_phi) if total_j_phi > 0 else 0.0,
        "term_stats": term_stats_list,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze J(φ) term contribution to recall")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of base samples")
    parser.add_argument("--n-queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--n-bits", type=int, default=8, help="Number of bits")
    parser.add_argument("--k", type=int, default=10, help="k for recall@k")
    parser.add_argument("--hamming-radius", type=int, default=1, help="Hamming ball radius")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="objective_contribution_analysis.json", help="Output file")
    
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
    
    # Get optimized permutation
    if hasattr(index_obj, 'hasher') and index_obj.hasher is not None:
        permutation = index_obj.hasher.get_assignment()
    else:
        # Use identity permutation as baseline
        N = 2 ** args.n_bits
        K = index_obj.K
        permutation = (np.arange(N, dtype=np.int32) % K).astype(np.int32)
    
    # Analyze term contribution
    print("Analyzing term contribution...")
    analysis = analyze_term_contribution(
        permutation=permutation,
        index_obj=index_obj,
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth=ground_truth,
        k=args.k,
        hamming_radius=args.hamming_radius,
    )
    
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
        "analysis": analysis,
        "recommendations": {
            "focus_on_low_hamming": analysis["recall_weighted_fraction"] < 0.5,
            "recommended_radius": args.hamming_radius,
            "optimization_suggestion": (
                "Consider recall-weighted objective" if analysis["recall_weighted_fraction"] < 0.5
                else "Current objective seems well-aligned with recall"
            ),
        },
    }
    
    with open(args.output, "w") as f:
        json.dump(output_dict, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print(f"Total J(φ): {analysis['total_j_phi']:.2f}")
    print(f"Recall-weighted cost: {analysis['recall_weighted_cost']:.2f}")
    print(f"Recall-weighted fraction: {analysis['recall_weighted_fraction']:.2%}")
    
    if analysis["recall_weighted_fraction"] < 0.5:
        print("\n⚠️  WARNING: Less than 50% of J(φ) comes from recall-relevant terms")
        print("   Consider using recall-weighted objective function")


if __name__ == "__main__":
    main()

