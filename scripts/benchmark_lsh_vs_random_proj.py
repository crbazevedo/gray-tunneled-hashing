"""
Benchmark comparing LSH families vs. random projection with GTH.

This script compares:
- Hyperplane LSH + GTH
- p-stable LSH + GTH  
- Random projection + GTH
- Baselines (without GTH)

Measures recall@k and impact of Hamming ball radius.
"""

import numpy as np
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any

from gray_tunneled_hashing.binary.lsh_families import (
    HyperplaneLSH,
    PStableLSH,
    create_lsh_family,
)
from gray_tunneled_hashing.binary.baselines import (
    random_projection_binarize,
    apply_random_projection,
)
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.integrations.hamming_index import build_hamming_index
from gray_tunneled_hashing.evaluation.metrics import recall_at_k
from gray_tunneled_hashing.api.query_pipeline import query_with_hamming_ball


def run_experiment(
    base_embeddings: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    method: str,
    n_bits: int,
    n_codes: int,
    k: int,
    hamming_radius: int = 0,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run a single experiment.
    
    Args:
        base_embeddings: Base corpus embeddings
        queries: Query embeddings
        ground_truth: Ground truth neighbor indices
        method: Method name ("hyperplane", "p_stable", "random_proj", "baseline_hyperplane", etc.)
        n_bits: Number of bits
        n_codes: Number of codebook codes
        k: Number of neighbors to retrieve
        hamming_radius: Hamming ball radius (0 = exact match)
        random_state: Random seed
        
    Returns:
        Dictionary with results
    """
    dim = base_embeddings.shape[1]
    Q = queries.shape[0]
    
    start_time = time.time()
    
    if method.startswith("baseline_"):
        # Baseline without GTH
        method_type = method.replace("baseline_", "")
        
        if method_type == "hyperplane":
            lsh = HyperplaneLSH(n_bits=n_bits, dim=dim, random_state=random_state)
            base_codes = lsh.hash(base_embeddings)
            query_codes = lsh.hash(queries)
        elif method_type == "p_stable":
            lsh = PStableLSH(n_bits=n_bits, dim=dim, w=1.0, random_state=random_state)
            base_codes = lsh.hash(base_embeddings)
            query_codes = lsh.hash(queries)
        elif method_type == "random_proj":
            base_codes, proj_matrix = random_projection_binarize(
                base_embeddings, n_bits=n_bits, random_state=random_state
            )
            query_codes = apply_random_projection(queries, proj_matrix)
        else:
            raise ValueError(f"Unknown baseline method: {method_type}")
        
        # Build index and search
        index = build_hamming_index(base_codes, use_faiss=True)
        retrieved_indices, distances = index.search(query_codes, k)
        
        build_time = time.time() - start_time
        
        recall = recall_at_k(retrieved_indices, ground_truth, k=k)
        
        return {
            "method": method,
            "n_bits": n_bits,
            "n_codes": n_codes,
            "k": k,
            "hamming_radius": hamming_radius,
            "recall": float(recall),
            "build_time": build_time,
            "search_time": 0.0,  # Not measured separately for baseline
        }
    
    else:
        # With GTH
        if method == "hyperplane":
            lsh = HyperplaneLSH(n_bits=n_bits, dim=dim, random_state=random_state)
            encoder = None  # Will use lsh_family
        elif method == "p_stable":
            lsh = PStableLSH(n_bits=n_bits, dim=dim, w=1.0, random_state=random_state)
            encoder = None
        elif method == "random_proj":
            lsh = None
            # Create encoder function
            _, proj_matrix = random_projection_binarize(
                base_embeddings, n_bits=n_bits, random_state=random_state
            )
            encoder = lambda emb: apply_random_projection(emb, proj_matrix)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Build distribution-aware index
        index_obj = build_distribution_aware_index(
            base_embeddings=base_embeddings,
            queries=queries,
            ground_truth_neighbors=ground_truth,
            encoder=encoder,
            n_bits=n_bits,
            n_codes=n_codes,
            use_codebook=True,
            use_semantic_distances=False,
            lsh_family=lsh,
            block_size=4,
            max_two_swap_iters=20,
            num_tunneling_steps=0,  # Quick test
            mode="two_swap_only",
            random_state=random_state,
        )
        
        build_time = time.time() - start_time
        
        # Query with Hamming ball
        search_start = time.time()
        
        # Encode queries
        if lsh is not None:
            query_codes = lsh.hash(queries)
        else:
            query_codes = encoder(queries)
        
        # Map base embeddings to buckets for recall calculation
        # We need to know which embeddings belong to which bucket
        from gray_tunneled_hashing.binary.codebooks import build_codebook_kmeans, find_nearest_centroids
        
        # Get codebook assignments for base embeddings
        centroids, assignments = build_codebook_kmeans(
            embeddings=base_embeddings,
            n_codes=n_codes,
            random_state=random_state,
        )
        
        # Map bucket indices to dataset indices
        # bucket_to_dataset_indices[bucket_idx] = list of dataset indices in that bucket
        bucket_to_dataset_indices = {}
        for dataset_idx, bucket_idx in enumerate(assignments):
            if bucket_idx not in bucket_to_dataset_indices:
                bucket_to_dataset_indices[bucket_idx] = []
            bucket_to_dataset_indices[bucket_idx].append(dataset_idx)
        
        # Apply GTH permutation and expand Hamming ball
        all_retrieved = []
        for query_code in query_codes:
            result = query_with_hamming_ball(
                query_code=query_code,
                permutation=index_obj.permutation,
                code_to_bucket=index_obj.code_to_bucket,
                bucket_to_code=index_obj.bucket_to_code,
                n_bits=n_bits,
                hamming_radius=hamming_radius,
            )
            
            # Map bucket indices to dataset indices
            candidate_dataset_indices = []
            for bucket_idx in result.candidate_indices:
                if bucket_idx in bucket_to_dataset_indices:
                    candidate_dataset_indices.extend(bucket_to_dataset_indices[bucket_idx])
            
            # Remove duplicates and limit to k
            candidate_dataset_indices = list(dict.fromkeys(candidate_dataset_indices))[:k]
            
            # Pad or truncate to k
            if len(candidate_dataset_indices) < k:
                # Pad with -1 (invalid index)
                candidate_dataset_indices.extend([-1] * (k - len(candidate_dataset_indices)))
            else:
                candidate_dataset_indices = candidate_dataset_indices[:k]
            
            all_retrieved.append(candidate_dataset_indices)
        
        retrieved_indices = np.array(all_retrieved, dtype=np.int32)
        search_time = time.time() - search_start
        
        # Compute recall
        recall = recall_at_k(retrieved_indices, ground_truth, k=k)
        
        return {
            "method": method,
            "n_bits": n_bits,
            "n_codes": n_codes,
            "k": k,
            "hamming_radius": hamming_radius,
            "recall": float(recall),
            "build_time": build_time,
            "search_time": search_time,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LSH vs. random projection with GTH"
    )
    parser.add_argument(
        "--n-bits", type=int, default=8, help="Number of bits (default: 8)"
    )
    parser.add_argument(
        "--n-codes", type=int, default=32, help="Number of codebook codes (default: 32)"
    )
    parser.add_argument(
        "--k", type=int, default=5, help="Number of neighbors to retrieve (default: 5)"
    )
    parser.add_argument(
        "--hamming-radius", type=int, default=1, help="Hamming ball radius (default: 1)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=100, help="Number of base samples (default: 100)"
    )
    parser.add_argument(
        "--n-queries", type=int, default=20, help="Number of queries (default: 20)"
    )
    parser.add_argument(
        "--dim", type=int, default=16, help="Embedding dimension (default: 16)"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--n-runs", type=int, default=3, help="Number of runs for averaging (default: 3)"
    )
    parser.add_argument(
        "--output", type=str, default="benchmark_lsh_results.json", help="Output file"
    )
    
    args = parser.parse_args()
    
    # Run experiments with multiple runs
    methods = [
        "baseline_hyperplane",
        "baseline_p_stable",
        "baseline_random_proj",
        "hyperplane",
        "p_stable",
        "random_proj",
    ]
    
    all_results = []
    
    for method in methods:
        print(f"\nRunning {method} ({args.n_runs} runs)...")
        method_results = []
        
        for run_idx in range(args.n_runs):
            # Generate synthetic data for each run
            run_seed = args.random_state + run_idx
            np.random.seed(run_seed)
            base_embeddings = np.random.randn(args.n_samples, args.dim).astype(np.float32)
            queries = np.random.randn(args.n_queries, args.dim).astype(np.float32)
            
            # Generate ground truth using actual distances
            from sklearn.metrics.pairwise import euclidean_distances
            distances = euclidean_distances(queries, base_embeddings)
            ground_truth = np.argsort(distances, axis=1)[:, :args.k].astype(np.int32)
            
            try:
                result = run_experiment(
                    base_embeddings=base_embeddings,
                    queries=queries,
                    ground_truth=ground_truth,
                    method=method,
                    n_bits=args.n_bits,
                    n_codes=args.n_codes,
                    k=args.k,
                    hamming_radius=args.hamming_radius,
                    random_state=run_seed,
                )
                result["run"] = run_idx
                method_results.append(result)
                print(f"  Run {run_idx+1}/{args.n_runs}: Recall={result['recall']:.4f}, Build={result['build_time']:.2f}s")
            except Exception as e:
                print(f"  ✗ Run {run_idx+1} Error: {e}")
                import traceback
                traceback.print_exc()
        
        if method_results:
            # Compute statistics across runs
            recalls = [r["recall"] for r in method_results]
            build_times = [r["build_time"] for r in method_results]
            search_times = [r["search_time"] for r in method_results]
            
            summary = {
                "method": method,
                "n_bits": args.n_bits,
                "n_codes": args.n_codes,
                "k": args.k,
                "hamming_radius": args.hamming_radius,
                "n_runs": len(method_results),
                "recall_mean": float(np.mean(recalls)),
                "recall_std": float(np.std(recalls)),
                "recall_min": float(np.min(recalls)),
                "recall_max": float(np.max(recalls)),
                "build_time_mean": float(np.mean(build_times)),
                "build_time_std": float(np.std(build_times)),
                "search_time_mean": float(np.mean(search_times)) if search_times[0] > 0 else 0.0,
                "search_time_std": float(np.std(search_times)) if search_times[0] > 0 else 0.0,
                "runs": method_results,
            }
            all_results.append(summary)
            print(f"  ✓ Mean Recall: {summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}")
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    # Print summary
    print("\n=== Summary ===")
    for result in all_results:
        print(
            f"{result['method']:25s} | "
            f"Recall: {result['recall_mean']:.4f} ± {result['recall_std']:.4f} | "
            f"Build: {result['build_time_mean']:.2f}s"
        )


if __name__ == "__main__":
    main()

