#!/usr/bin/env python3
"""
Analyze Hamming ball coverage of ground truth neighbors.

This script tests if Hamming ball expansion is working as expected.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import argparse
import json
from typing import Dict, List
from collections import defaultdict

from gray_tunneled_hashing.binary.lsh_families import HyperplaneLSH
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.evaluation.metrics import hamming_distance
from gray_tunneled_hashing.api.query_pipeline import query_with_hamming_ball
from gray_tunneled_hashing.data.synthetic_generators import generate_hypercube_vertices


def analyze_hamming_ball_coverage(
    permutation: np.ndarray,
    index_obj: any,
    base_embeddings: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
    hamming_radius: int = 1,
) -> Dict[str, any]:
    """Analyze how many ground truth neighbors are within Hamming ball."""
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
    
    # Need to get encoder from context
    # For now, recreate LSH
    from gray_tunneled_hashing.binary.lsh_families import create_lsh_family
    lsh = create_lsh_family("hyperplane", n_bits=index_obj.n_bits, dim=base_embeddings.shape[1], random_state=42)
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
    
    # Analyze coverage
    coverage_stats = {
        "total_queries": len(queries),
        "total_gt_neighbors": 0,
        "neighbors_in_ball": 0,
        "neighbors_not_in_ball": 0,
        "distance_distribution": defaultdict(int),
        "queries_with_full_coverage": 0,
        "queries_with_no_coverage": 0,
    }
    
    for query_idx, query_code in enumerate(query_codes):
        code_tuple = tuple(query_code.astype(int).tolist())
        if code_tuple not in index_obj.code_to_bucket:
            continue
        
        query_bucket = index_obj.code_to_bucket[code_tuple]
        
        # Query with Hamming ball
        result = query_with_hamming_ball(
            query_code=query_code,
            permutation=permutation,
            code_to_bucket=index_obj.code_to_bucket,
            bucket_to_code=index_obj.bucket_to_code,
            n_bits=index_obj.n_bits,
            hamming_radius=hamming_radius,
        )
        
        # Get retrieved dataset indices
        retrieved_indices = set()
        for bucket_idx in result.candidate_indices:
            if bucket_idx in bucket_to_dataset_indices:
                retrieved_indices.update(bucket_to_dataset_indices[bucket_idx])
        
        # Check ground truth neighbors
        gt_neighbors = set(ground_truth[query_idx][:k])
        coverage_stats["total_gt_neighbors"] += len(gt_neighbors)
        
        in_ball = gt_neighbors & retrieved_indices
        not_in_ball = gt_neighbors - retrieved_indices
        
        coverage_stats["neighbors_in_ball"] += len(in_ball)
        coverage_stats["neighbors_not_in_ball"] += len(not_in_ball)
        
        if len(in_ball) == len(gt_neighbors):
            coverage_stats["queries_with_full_coverage"] += 1
        if len(in_ball) == 0:
            coverage_stats["queries_with_no_coverage"] += 1
        
        # Compute distances for neighbors not in ball
        if query_bucket in bucket_to_vertex:
            query_vertex = bucket_to_vertex[query_bucket]
            query_code_vertex = vertices[query_vertex]
            
            for neighbor_idx in not_in_ball:
                neighbor_code = base_codes_lsh[neighbor_idx]
                neighbor_code_tuple = tuple(neighbor_code.astype(int).tolist())
                if neighbor_code_tuple in index_obj.code_to_bucket:
                    neighbor_bucket = index_obj.code_to_bucket[neighbor_code_tuple]
                    if neighbor_bucket in bucket_to_vertex:
                        neighbor_vertex = bucket_to_vertex[neighbor_bucket]
                        neighbor_code_vertex = vertices[neighbor_vertex]
                        
                        d_h = int(hamming_distance(
                            query_code_vertex[np.newaxis, :],
                            neighbor_code_vertex[np.newaxis, :]
                        )[0, 0])
                        coverage_stats["distance_distribution"][d_h] += 1
    
    # Compute coverage rate
    coverage_rate = (
        coverage_stats["neighbors_in_ball"] / coverage_stats["total_gt_neighbors"]
        if coverage_stats["total_gt_neighbors"] > 0 else 0.0
    )
    
    return {
        "coverage_rate": float(coverage_rate),
        "stats": {
            k: v if not isinstance(v, defaultdict) else dict(v)
            for k, v in coverage_stats.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze Hamming ball coverage")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of base samples")
    parser.add_argument("--n-queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--n-bits", type=int, default=8, help="Number of bits")
    parser.add_argument("--k", type=int, default=10, help="k for recall@k")
    parser.add_argument("--hamming-radius", type=int, default=1, help="Hamming ball radius")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="hamming_ball_coverage.json", help="Output file")
    
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
        N = 2 ** args.n_bits
        K = index_obj.K
        permutation = (np.arange(N, dtype=np.int32) % K).astype(np.int32)
    
    # Analyze coverage
    print("Analyzing Hamming ball coverage...")
    analysis = analyze_hamming_ball_coverage(
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
        "coverage_analysis": analysis,
        "recommendations": {
            "increase_radius": analysis["coverage_rate"] < 0.5,
            "recommended_radius": args.hamming_radius + 1 if analysis["coverage_rate"] < 0.5 else args.hamming_radius,
        },
    }
    
    with open(args.output, "w") as f:
        json.dump(output_dict, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print(f"Coverage rate: {analysis['coverage_rate']:.4f}")
    print(f"Neighbors in ball: {analysis['stats']['neighbors_in_ball']}/{analysis['stats']['total_gt_neighbors']}")


if __name__ == "__main__":
    main()

