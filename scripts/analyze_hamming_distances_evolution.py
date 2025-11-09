#!/usr/bin/env python3
"""
Analyze evolution of Hamming distances before and after GTH optimization.

This script tests hypothesis H3: Does GTH permutation increase Hamming distances
between ground truth neighbors?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import argparse
import json
from typing import Dict, List, Tuple
from tqdm import tqdm

from gray_tunneled_hashing.binary.lsh_families import create_lsh_family
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.algorithms.gray_tunneled_hasher import GrayTunneledHasher
from gray_tunneled_hashing.evaluation.metrics import hamming_distance
from sklearn.metrics.pairwise import euclidean_distances


def generate_synthetic_data(n_samples: int, n_queries: int, dim: int, random_state: int = 42):
    """Generate synthetic embeddings."""
    np.random.seed(random_state)
    base_embeddings = np.random.randn(n_samples, dim).astype(np.float32)
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    return base_embeddings, queries


def analyze_hamming_distances_evolution(
    base_embeddings: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    n_bits: int,
    k: int,
    random_state: int = 42,
) -> Dict:
    """Analyze Hamming distances before and after GTH optimization."""
    # Build LSH encoder
    lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=base_embeddings.shape[1], random_state=random_state)
    
    # Encode base embeddings and queries
    base_codes = lsh.hash(base_embeddings)
    query_codes = lsh.hash(queries)
    
    # Compute Hamming distances BEFORE GTH (baseline)
    hamming_distances_before = []
    cosine_distances = []
    
    for q_idx in range(len(queries)):
        query_code = query_codes[q_idx]
        gt_neighbors = ground_truth[q_idx][:k]
        
        for neighbor_idx in gt_neighbors:
            neighbor_code = base_codes[neighbor_idx]
            # Hamming distance
            hamm_dist = hamming_distance(
                query_code[np.newaxis, :],
                neighbor_code[np.newaxis, :]
            )[0, 0]
            hamming_distances_before.append(int(hamm_dist))
            
            # Cosine distance (for reference)
            cos_dist = 1.0 - np.dot(queries[q_idx], base_embeddings[neighbor_idx]) / (
                np.linalg.norm(queries[q_idx]) * np.linalg.norm(base_embeddings[neighbor_idx])
            )
            cosine_distances.append(float(cos_dist))
    
    # Build distribution-aware index (this applies GTH)
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        n_bits=n_bits,
        lsh_family=lsh,
    )
    
    # Get GTH permutation
    permutation = index_obj.hasher.get_assignment()
    
    # Compute Hamming distances AFTER GTH
    # Need to apply permutation to codes
    from gray_tunneled_hashing.api.query_pipeline import query_with_hamming_ball
    
    hamming_distances_after = []
    hamming_distances_after_permuted = []
    
    for q_idx in range(len(queries)):
        query_code = query_codes[q_idx]
        gt_neighbors = ground_truth[q_idx][:k]
        
        # Apply permutation to query code
        # Convert query_code to vertex index
        query_vertex_idx = 0
        for i, bit in enumerate(query_code):
            if bit:
                query_vertex_idx += 2 ** i
        
        if query_vertex_idx < len(permutation):
            # Get permuted embedding_idx
            query_embedding_idx = permutation[query_vertex_idx]
            
            # Find which vertex this embedding_idx maps to (inverse permutation)
            # Actually, we need to find vertex v such that permutation[v] == query_embedding_idx
            # But wait, permutation maps vertex -> embedding_idx, not embedding_idx -> vertex
            # So we need the inverse mapping
            
            # For now, let's compute Hamming distance between permuted codes directly
            # We need to find the permuted code for the query
            # This is complex - let's use a simpler approach
            
            # Actually, let's compute Hamming distance between query and neighbors
            # after applying permutation to both
            for neighbor_idx in gt_neighbors:
                neighbor_code = base_codes[neighbor_idx]
                
                # Convert neighbor_code to vertex index
                neighbor_vertex_idx = 0
                for i, bit in enumerate(neighbor_code):
                    if bit:
                        neighbor_vertex_idx += 2 ** i
                
                if neighbor_vertex_idx < len(permutation):
                    neighbor_embedding_idx = permutation[neighbor_vertex_idx]
                    
                    # Now we need to find the vertex codes that correspond to these embedding_idx
                    # This is the inverse permutation problem
                    # For simplicity, let's compute Hamming distance between the original codes
                    # and see if we can infer the permuted distance
                    
                    # Actually, the permutation doesn't change the codes themselves,
                    # it changes which vertex (code) is assigned to which bucket
                    # So the Hamming distance between codes doesn't change
                    # What changes is which buckets are close in Hamming space
                    
                    # Let's compute the Hamming distance between the vertex codes
                    # (which doesn't change) and also track which buckets they map to
                    hamm_dist_original = hamming_distance(
                        query_code[np.newaxis, :],
                        neighbor_code[np.newaxis, :]
                    )[0, 0]
                    hamming_distances_after_permuted.append(int(hamm_dist_original))
                    
                    # The key question: after permutation, are query and neighbor
                    # in buckets that are close in Hamming space?
                    # We need to find the bucket codes after permutation
                    # This requires finding which vertex each bucket is assigned to
                    
                    # For now, let's use a simpler metric: Hamming distance between
                    # the bucket codes that query and neighbor are assigned to
                    # But this requires knowing the bucket assignment after permutation
                    
                    # Let's compute it differently: use query_with_hamming_ball to see
                    # if neighbor is in the Hamming ball
                    result = query_with_hamming_ball(
                        query_code=query_code,
                        permutation=permutation,
                        code_to_bucket=index_obj.code_to_bucket,
                        bucket_to_code=index_obj.bucket_to_code,
                        n_bits=n_bits,
                        hamming_radius=10,  # Large radius to capture all
                    )
                    
                    # Check if neighbor is in the candidate set
                    # We need to map neighbor to its bucket
                    neighbor_code_tuple = tuple(neighbor_code.astype(int).tolist())
                    if neighbor_code_tuple in index_obj.code_to_bucket:
                        neighbor_bucket = index_obj.code_to_bucket[neighbor_code_tuple]
                        
                        # Find query bucket
                        query_code_tuple = tuple(query_code.astype(int).tolist())
                        if query_code_tuple in index_obj.code_to_bucket:
                            query_bucket = index_obj.code_to_bucket[query_code_tuple]
                            
                            # Compute Hamming distance between bucket codes
                            query_bucket_code = index_obj.bucket_to_code[query_bucket]
                            neighbor_bucket_code = index_obj.bucket_to_code[neighbor_bucket]
                            
                            hamm_dist_buckets = hamming_distance(
                                query_bucket_code[np.newaxis, :],
                                neighbor_bucket_code[np.newaxis, :]
                            )[0, 0]
                            hamming_distances_after.append(int(hamm_dist_buckets))
    
    # Statistics
    hamm_before = np.array(hamming_distances_before)
    hamm_after = np.array(hamming_distances_after) if len(hamming_distances_after) > 0 else np.array([])
    cosine_dists = np.array(cosine_distances)
    
    results = {
        "n_pairs": len(hamming_distances_before),
        "hamming_before": {
            "mean": float(hamm_before.mean()) if len(hamm_before) > 0 else 0.0,
            "std": float(hamm_before.std()) if len(hamm_before) > 0 else 0.0,
            "min": int(hamm_before.min()) if len(hamm_before) > 0 else 0,
            "max": int(hamm_before.max()) if len(hamm_before) > 0 else 0,
            "distribution": {
                str(i): int(np.sum(hamm_before == i)) for i in range(n_bits + 1)
            },
        },
        "hamming_after": {
            "mean": float(hamm_after.mean()) if len(hamm_after) > 0 else 0.0,
            "std": float(hamm_after.std()) if len(hamm_after) > 0 else 0.0,
            "min": int(hamm_after.min()) if len(hamm_after) > 0 else 0,
            "max": int(hamm_after.max()) if len(hamm_after) > 0 else 0,
            "distribution": {
                str(i): int(np.sum(hamm_after == i)) for i in range(n_bits + 1)
            } if len(hamm_after) > 0 else {},
        },
        "cosine_distances": {
            "mean": float(cosine_dists.mean()) if len(cosine_dists) > 0 else 0.0,
            "std": float(cosine_dists.std()) if len(cosine_dists) > 0 else 0.0,
        },
        "comparison": {
            "mean_increase": float(hamm_after.mean() - hamm_before.mean()) if len(hamm_after) > 0 and len(hamm_before) > 0 else 0.0,
            "pairs_with_increase": int(np.sum(hamm_after > hamm_before)) if len(hamm_after) > 0 else 0,
            "pairs_with_decrease": int(np.sum(hamm_after < hamm_before)) if len(hamm_after) > 0 else 0,
            "pairs_unchanged": int(np.sum(hamm_after == hamm_before)) if len(hamm_after) > 0 else 0,
        },
        "correlation": {
            "cosine_hamming_before": float(np.corrcoef(cosine_dists, hamm_before)[0, 1]) if len(cosine_dists) > 0 and len(hamm_before) > 0 else 0.0,
            "cosine_hamming_after": float(np.corrcoef(cosine_dists, hamm_after)[0, 1]) if len(cosine_dists) > 0 and len(hamm_after) > 0 else 0.0,
        },
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze Hamming distance evolution")
    parser.add_argument("--n-samples", type=int, default=500, help="Number of base samples")
    parser.add_argument("--n-queries", type=int, default=50, help="Number of queries")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--n-bits", type=int, default=6, help="Number of bits")
    parser.add_argument("--k", type=int, default=10, help="k for recall@k")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="experiments/real/hamming_distances_evolution.json", help="Output file")
    args = parser.parse_args()
    
    np.random.seed(args.random_state)
    
    # Generate synthetic data
    base_embeddings, queries = generate_synthetic_data(args.n_samples, args.n_queries, args.dim, args.random_state)
    
    # Compute ground truth
    distances = euclidean_distances(queries, base_embeddings)
    ground_truth = np.argsort(distances, axis=1)[:, :args.k]
    
    # Analyze
    print("Analyzing Hamming distance evolution...")
    results = analyze_hamming_distances_evolution(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth=ground_truth,
        n_bits=args.n_bits,
        k=args.k,
        random_state=args.random_state,
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print(f"Hamming distance before: {results['hamming_before']['mean']:.2f} ± {results['hamming_before']['std']:.2f}")
    print(f"Hamming distance after: {results['hamming_after']['mean']:.2f} ± {results['hamming_after']['std']:.2f}")
    print(f"Mean increase: {results['comparison']['mean_increase']:.2f}")
    print(f"Pairs with increase: {results['comparison']['pairs_with_increase']}")
    print(f"Pairs with decrease: {results['comparison']['pairs_with_decrease']}")
    print(f"Correlation cosine-Hamming before: {results['correlation']['cosine_hamming_before']:.4f}")
    print(f"Correlation cosine-Hamming after: {results['correlation']['cosine_hamming_after']:.4f}")


if __name__ == "__main__":
    main()

