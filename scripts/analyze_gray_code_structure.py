#!/usr/bin/env python3
"""
Analyze Gray-code structure of permutation.

This script tests hypothesis H4: Permutation is not preserving Gray structure.
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
from gray_tunneled_hashing.evaluation.metrics import hamming_distance
from gray_tunneled_hashing.data.synthetic_generators import generate_hypercube_vertices
from sklearn.metrics.pairwise import cosine_distances


def compute_gray_score(
    permutation: np.ndarray,
    index_obj: any,
    bucket_embeddings: np.ndarray,
    n_bits: int,
) -> Dict[str, float]:
    """
    Compute Gray-code score: fraction of Hamming-1 neighbors that are semantically close.
    
    For each vertex v, check its Hamming-1 neighbors:
    - If neighbor u has semantically close embeddings, score += 1
    - Gray score = fraction of Hamming-1 pairs that are semantically close
    """
    vertices = generate_hypercube_vertices(n_bits)
    N = len(vertices)
    K = len(index_obj.pi)
    
    # Map bucket to vertex
    bucket_to_vertex = {}
    for vertex_idx in range(N):
        embedding_idx = permutation[vertex_idx]
        if embedding_idx < K:
            if embedding_idx not in bucket_to_vertex:
                bucket_to_vertex[embedding_idx] = vertex_idx
    
    # Compute semantic distances between buckets
    semantic_distances = cosine_distances(bucket_embeddings)
    semantic_threshold = np.percentile(semantic_distances[semantic_distances > 0], 25)  # Top 25% closest
    
    # Count Hamming-1 neighbors
    hamming_1_pairs = 0
    semantically_close_pairs = 0
    
    for i in range(K):
        if i not in bucket_to_vertex:
            continue
        
        vertex_i = bucket_to_vertex[i]
        code_i = vertices[vertex_i]
        
        # Find Hamming-1 neighbors
        for vertex_j in range(N):
            if vertex_j == vertex_i:
                continue
            
            code_j = vertices[vertex_j]
            d_h = hamming_distance(code_i[np.newaxis, :], code_j[np.newaxis, :])[0, 0]
            
            if d_h == 1:  # Hamming-1 neighbor
                # Find which bucket is at vertex_j
                embedding_j = permutation[vertex_j]
                if embedding_j < K:
                    hamming_1_pairs += 1
                    
                    # Check if semantically close
                    if semantic_distances[i, embedding_j] <= semantic_threshold:
                        semantically_close_pairs += 1
    
    gray_score = semantically_close_pairs / hamming_1_pairs if hamming_1_pairs > 0 else 0.0
    
    return {
        "gray_score": float(gray_score),
        "hamming_1_pairs": int(hamming_1_pairs),
        "semantically_close_pairs": int(semantically_close_pairs),
        "semantic_threshold": float(semantic_threshold),
    }


def compare_with_ideal_gray(
    permutation: np.ndarray,
    index_obj: any,
    bucket_embeddings: np.ndarray,
    n_bits: int,
) -> Dict[str, any]:
    """Compare current permutation with ideal Gray-code ordering."""
    # Generate ideal Gray-code sequence
    from gray_tunneled_hashing.algorithms.gray_tunneled_hasher import GrayTunneledHasher
    hasher = GrayTunneledHasher(n_bits=n_bits, init_strategy="identity")
    ideal_gray_perm = hasher._generate_gray_code_sequence(n_bits)
    
    # Compute Gray scores
    current_score = compute_gray_score(permutation, index_obj, bucket_embeddings, n_bits)
    
    # For ideal Gray, map buckets to Gray sequence
    K = len(index_obj.pi)
    ideal_perm = (ideal_gray_perm[:len(ideal_gray_perm)] % K).astype(np.int32)
    ideal_score = compute_gray_score(ideal_perm, index_obj, bucket_embeddings, n_bits)
    
    return {
        "current_gray_score": current_score["gray_score"],
        "ideal_gray_score": ideal_score["gray_score"],
        "score_ratio": float(current_score["gray_score"] / ideal_score["gray_score"]) if ideal_score["gray_score"] > 0 else 0.0,
        "current_details": current_score,
        "ideal_details": ideal_score,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze Gray-code structure")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of base samples")
    parser.add_argument("--n-queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--n-bits", type=int, default=8, help="Number of bits")
    parser.add_argument("--k", type=int, default=10, help="k for recall@k")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="gray_code_analysis.json", help="Output file")
    
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
    
    # Get bucket embeddings
    base_codes_lsh = index_obj.lsh.hash(base_embeddings)
    bucket_embeddings_list = []
    for bucket_idx in range(index_obj.K):
        for dataset_idx, code in enumerate(base_codes_lsh):
            code_tuple = tuple(code.astype(int).tolist())
            if code_tuple in index_obj.code_to_bucket:
                if index_obj.code_to_bucket[code_tuple] == bucket_idx:
                    bucket_embeddings_list.append(base_embeddings[dataset_idx])
                    break
        else:
            bucket_embeddings_list.append(np.zeros(args.dim, dtype=np.float32))
    bucket_embeddings = np.array(bucket_embeddings_list)
    
    # Get optimized permutation
    if hasattr(index_obj, 'hasher') and index_obj.hasher is not None:
        permutation = index_obj.hasher.get_assignment()
    else:
        N = 2 ** args.n_bits
        K = index_obj.K
        permutation = (np.arange(N, dtype=np.int32) % K).astype(np.int32)
    
    # Analyze Gray structure
    print("Analyzing Gray-code structure...")
    gray_analysis = compute_gray_score(permutation, index_obj, bucket_embeddings, args.n_bits)
    comparison = compare_with_ideal_gray(permutation, index_obj, bucket_embeddings, args.n_bits)
    
    # Save results
    output_dict = {
        "configuration": {
            "n_samples": args.n_samples,
            "n_queries": args.n_queries,
            "dim": args.dim,
            "n_bits": args.n_bits,
            "k": args.k,
        },
        "gray_analysis": gray_analysis,
        "comparison_with_ideal": comparison,
        "recommendations": {
            "needs_gray_regularization": comparison["current_gray_score"] < 0.5 * comparison["ideal_gray_score"],
            "recommendation": (
                "Add Gray-code regularization term to objective" 
                if comparison["current_gray_score"] < 0.5 * comparison["ideal_gray_score"]
                else "Gray structure is reasonably preserved"
            ),
        },
    }
    
    with open(args.output, "w") as f:
        json.dump(output_dict, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print(f"Current Gray score: {gray_analysis['gray_score']:.4f}")
    print(f"Ideal Gray score: {comparison['ideal_gray_score']:.4f}")
    print(f"Score ratio: {comparison['score_ratio']:.4f}")
    
    if comparison["current_gray_score"] < 0.5 * comparison["ideal_gray_score"]:
        print("\n⚠️  WARNING: Gray structure is not well preserved")
        print("   Consider adding Gray-code regularization term")


if __name__ == "__main__":
    main()

