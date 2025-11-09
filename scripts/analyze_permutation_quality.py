#!/usr/bin/env python3
"""
Analyze quality metrics of permutation.

This script computes various quality metrics that may correlate with recall.
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
from gray_tunneled_hashing.data.synthetic_generators import generate_hypercube_vertices
from sklearn.metrics.pairwise import cosine_distances


def compute_quality_metrics(
    permutation: np.ndarray,
    index_obj: any,
    base_embeddings: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
    hamming_radius: int = 1,
) -> Dict[str, any]:
    """Compute various quality metrics for the permutation."""
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
    
    # Get bucket embeddings
    base_codes_lsh = index_obj.lsh.hash(base_embeddings)
    bucket_embeddings_list = []
    for bucket_idx in range(K):
        for dataset_idx, code in enumerate(base_codes_lsh):
            code_tuple = tuple(code.astype(int).tolist())
            if code_tuple in index_obj.code_to_bucket:
                if index_obj.code_to_bucket[code_tuple] == bucket_idx:
                    bucket_embeddings_list.append(base_embeddings[dataset_idx])
                    break
        else:
            bucket_embeddings_list.append(np.zeros(base_embeddings.shape[1], dtype=np.float32))
    bucket_embeddings = np.array(bucket_embeddings_list)
    
    # Compute semantic distances
    semantic_distances = cosine_distances(bucket_embeddings)
    
    # Metric 1: Average Hamming distance for query-neighbor pairs
    query_codes = index_obj.lsh.hash(queries)
    query_neighbor_distances = []
    
    for query_idx, query_code in enumerate(query_codes):
        code_tuple = tuple(query_code.astype(int).tolist())
        if code_tuple in index_obj.code_to_bucket:
            query_bucket = index_obj.code_to_bucket[code_tuple]
            if query_bucket in bucket_to_vertex:
                query_vertex = bucket_to_vertex[query_bucket]
                query_code_vertex = vertices[query_vertex]
                
                for neighbor_idx in ground_truth[query_idx][:k]:
                    neighbor_code = base_codes_lsh[neighbor_idx]
                    neighbor_code_tuple = tuple(neighbor_code.astype(int).tolist())
                    if neighbor_code_tuple in index_obj.code_to_bucket:
                        neighbor_bucket = index_obj.code_to_bucket[neighbor_code_tuple]
                        if neighbor_bucket in bucket_to_vertex:
                            neighbor_vertex = bucket_to_vertex[neighbor_bucket]
                            neighbor_code_vertex = vertices[neighbor_vertex]
                            
                            d_h = hamming_distance(
                                query_code_vertex[np.newaxis, :],
                                neighbor_code_vertex[np.newaxis, :]
                            )[0, 0]
                            query_neighbor_distances.append(d_h)
    
    # Metric 2: Fraction of query-neighbor pairs within Hamming ball
    within_ball = sum(1 for d in query_neighbor_distances if d <= hamming_radius)
    coverage_fraction = within_ball / len(query_neighbor_distances) if query_neighbor_distances else 0.0
    
    # Metric 3: Gray-code score (Hamming-1 neighbors that are semantically close)
    hamming_1_pairs = 0
    semantically_close_pairs = 0
    semantic_threshold = np.percentile(semantic_distances[semantic_distances > 0], 25)
    
    for i in range(K):
        if i not in bucket_to_vertex:
            continue
        
        vertex_i = bucket_to_vertex[i]
        code_i = vertices[vertex_i]
        
        for vertex_j in range(N):
            if vertex_j == vertex_i:
                continue
            
            code_j = vertices[vertex_j]
            d_h = hamming_distance(code_i[np.newaxis, :], code_j[np.newaxis, :])[0, 0]
            
            if d_h == 1:
                embedding_j = permutation[vertex_j]
                if embedding_j < K:
                    hamming_1_pairs += 1
                    if semantic_distances[i, embedding_j] <= semantic_threshold:
                        semantically_close_pairs += 1
    
    gray_score = semantically_close_pairs / hamming_1_pairs if hamming_1_pairs > 0 else 0.0
    
    # Metric 4: Distance distribution statistics
    distance_distribution = defaultdict(int)
    for i in range(K):
        if i not in bucket_to_vertex:
            continue
        
        code_i = vertices[bucket_to_vertex[i]]
        for j in range(i + 1, K):
            if j not in bucket_to_vertex:
                continue
            
            code_j = vertices[bucket_to_vertex[j]]
            d_h = int(hamming_distance(code_i[np.newaxis, :], code_j[np.newaxis, :])[0, 0])
            distance_distribution[d_h] += index_obj.pi[i] * index_obj.w[i, j]
    
    return {
        "avg_query_neighbor_distance": float(np.mean(query_neighbor_distances)) if query_neighbor_distances else 0.0,
        "coverage_fraction": float(coverage_fraction),
        "gray_score": float(gray_score),
        "distance_distribution": {str(k): float(v) for k, v in distance_distribution.items()},
        "n_query_neighbor_pairs": len(query_neighbor_distances),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze permutation quality")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of base samples")
    parser.add_argument("--n-queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--n-bits", type=int, default=8, help="Number of bits")
    parser.add_argument("--k", type=int, default=10, help="k for recall@k")
    parser.add_argument("--hamming-radius", type=int, default=1, help="Hamming ball radius")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="permutation_quality_analysis.json", help="Output file")
    
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
    
    # Compute quality metrics
    print("Computing quality metrics...")
    metrics = compute_quality_metrics(
        permutation=permutation,
        index_obj=index_obj,
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
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
        "quality_metrics": metrics,
    }
    
    with open(args.output, "w") as f:
        json.dump(output_dict, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print(f"Coverage fraction: {metrics['coverage_fraction']:.4f}")
    print(f"Gray score: {metrics['gray_score']:.4f}")
    print(f"Avg query-neighbor distance: {metrics['avg_query_neighbor_distance']:.2f}")


if __name__ == "__main__":
    main()

