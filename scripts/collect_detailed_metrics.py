"""
Collect detailed metrics for GTH recall analysis.

This script collects comprehensive metrics to understand the recall issue:
- Coverage rates
- Bucket distribution
- Permutation statistics
- Comparison with baseline
"""

import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter, defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gray_tunneled_hashing.binary.lsh_families import HyperplaneLSH
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.api.query_pipeline import query_with_hamming_ball
from gray_tunneled_hashing.integrations.hamming_index import build_hamming_index
from gray_tunneled_hashing.evaluation.metrics import recall_at_k
from sklearn.metrics.pairwise import euclidean_distances


def collect_metrics(
    base_embeddings: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    n_bits: int,
    n_codes: int,
    k: int = 5,
    hamming_radius: int = 1,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Collect comprehensive metrics for analysis."""
    
    metrics = {
        "configuration": {
            "n_samples": len(base_embeddings),
            "n_queries": len(queries),
            "dim": base_embeddings.shape[1],
            "n_bits": n_bits,
            "n_codes": n_codes,
            "k": k,
            "hamming_radius": hamming_radius,
        },
        "baseline": {},
        "gth": {},
        "comparison": {},
    }
    
    lsh = HyperplaneLSH(n_bits=n_bits, dim=base_embeddings.shape[1], random_state=random_state)
    
    # === BASELINE METRICS ===
    base_codes = lsh.hash(base_embeddings)
    query_codes = lsh.hash(queries)
    
    baseline_index = build_hamming_index(base_codes, use_faiss=True)
    baseline_retrieved, _ = baseline_index.search(query_codes, k)
    baseline_recall = recall_at_k(baseline_retrieved, ground_truth, k=k)
    
    # Baseline coverage: all codes are valid buckets
    baseline_unique_codes = len(set(tuple(code.astype(int).tolist()) for code in base_codes))
    baseline_coverage = 1.0  # 100% - all codes are buckets
    
    metrics["baseline"] = {
        "recall": float(baseline_recall),
        "unique_codes": baseline_unique_codes,
        "coverage_rate": baseline_coverage,
        "total_buckets": baseline_unique_codes,
    }
    
    # === GTH METRICS ===
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=None,
        n_bits=n_bits,
        n_codes=n_codes,
        use_codebook=True,
        use_semantic_distances=False,
        lsh_family=lsh,
        block_size=4,
        max_two_swap_iters=20,
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=random_state,
    )
    
    # Build bucket → dataset mapping
    base_codes_lsh = lsh.hash(base_embeddings)
    bucket_to_dataset_indices = {}
    
    for dataset_idx, code in enumerate(base_codes_lsh):
        code_tuple = tuple(code.astype(int).tolist())
        if code_tuple in index_obj.code_to_bucket:
            bucket_idx = index_obj.code_to_bucket[code_tuple]
            if bucket_idx not in bucket_to_dataset_indices:
                bucket_to_dataset_indices[bucket_idx] = []
            bucket_to_dataset_indices[bucket_idx].append(dataset_idx)
    
    # GTH coverage
    gth_mapped_count = sum(len(indices) for indices in bucket_to_dataset_indices.values())
    gth_coverage = gth_mapped_count / len(base_embeddings) if len(base_embeddings) > 0 else 0.0
    
    # Permutation analysis
    permutation = index_obj.permutation
    K = index_obj.K
    N = len(permutation)
    
    # Count invalid bucket indices
    invalid_bucket_count = sum(1 for b in permutation if b >= K)
    invalid_bucket_rate = invalid_bucket_count / N if N > 0 else 0.0
    
    # Bucket distribution
    bucket_to_vertices = defaultdict(list)
    for vertex_idx in range(N):
        bucket_idx = permutation[vertex_idx]
        if bucket_idx < K:
            bucket_to_vertices[bucket_idx].append(vertex_idx)
    
    buckets_with_vertices = len(bucket_to_vertices)
    empty_buckets = K - buckets_with_vertices
    
    # Consistency analysis
    permutation_buckets = set(b for b in permutation if b < K)
    code_to_bucket_buckets = set(index_obj.code_to_bucket.values())
    intersection = permutation_buckets & code_to_bucket_buckets
    consistency_rate = len(intersection) / len(permutation_buckets) if len(permutation_buckets) > 0 else 0.0
    
    # Query analysis
    all_retrieved = []
    total_candidates = 0
    total_invalid_buckets = 0
    total_buckets_not_in_code = 0
    
    for query_code in query_codes:
        result = query_with_hamming_ball(
            query_code=query_code,
            permutation=index_obj.permutation,
            code_to_bucket=index_obj.code_to_bucket,
            bucket_to_code=index_obj.bucket_to_code,
            n_bits=n_bits,
            hamming_radius=hamming_radius,
        )
        
        # Count invalid buckets
        valid_bucket_set = set(index_obj.code_to_bucket.values())
        for bucket_idx in result.candidate_indices:
            if bucket_idx >= K:
                total_invalid_buckets += 1
            elif bucket_idx not in valid_bucket_set:
                total_buckets_not_in_code += 1
        
        total_candidates += len(result.candidate_indices)
        
        candidate_dataset_indices = []
        for bucket_idx in result.candidate_indices:
            if bucket_idx in bucket_to_dataset_indices:
                candidate_dataset_indices.extend(bucket_to_dataset_indices[bucket_idx])
        
        candidate_dataset_indices = list(dict.fromkeys(candidate_dataset_indices))[:k]
        if len(candidate_dataset_indices) < k:
            candidate_dataset_indices.extend([-1] * (k - len(candidate_dataset_indices)))
        else:
            candidate_dataset_indices = candidate_dataset_indices[:k]
        
        all_retrieved.append(candidate_dataset_indices)
    
    retrieved_indices = np.array(all_retrieved, dtype=np.int32)
    gth_recall = recall_at_k(retrieved_indices, ground_truth, k=k)
    
    # Bucket size distribution
    bucket_sizes = [len(indices) for indices in bucket_to_dataset_indices.values()]
    
    metrics["gth"] = {
        "recall": float(gth_recall),
        "coverage_rate": float(gth_coverage),
        "mapped_embeddings": int(gth_mapped_count),
        "total_buckets": K,
        "populated_buckets": buckets_with_vertices,
        "empty_buckets": empty_buckets,
        "invalid_bucket_indices": int(invalid_bucket_count),
        "invalid_bucket_rate": float(invalid_bucket_rate),
        "consistency_rate": float(consistency_rate),
        "avg_candidates_per_query": float(total_candidates / len(queries)) if len(queries) > 0 else 0.0,
        "avg_invalid_buckets_per_query": float(total_invalid_buckets / len(queries)) if len(queries) > 0 else 0.0,
        "avg_buckets_not_in_code_per_query": float(total_buckets_not_in_code / len(queries)) if len(queries) > 0 else 0.0,
        "bucket_size_stats": {
            "min": int(np.min(bucket_sizes)) if bucket_sizes else 0,
            "max": int(np.max(bucket_sizes)) if bucket_sizes else 0,
            "mean": float(np.mean(bucket_sizes)) if bucket_sizes else 0.0,
            "std": float(np.std(bucket_sizes)) if bucket_sizes else 0.0,
        },
    }
    
    # === COMPARISON ===
    metrics["comparison"] = {
        "recall_difference": float(gth_recall - baseline_recall),
        "recall_ratio": float(gth_recall / baseline_recall) if baseline_recall > 0 else 0.0,
        "coverage_difference": float(gth_coverage - baseline_coverage),
        "bucket_count_ratio": float(K / baseline_unique_codes) if baseline_unique_codes > 0 else 0.0,
    }
    
    return metrics


def main():
    """Collect and save detailed metrics."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect detailed GTH metrics")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of base samples")
    parser.add_argument("--n-queries", type=int, default=20, help="Number of queries")
    parser.add_argument("--dim", type=int, default=16, help="Embedding dimension")
    parser.add_argument("--n-bits", type=int, default=6, help="Number of bits")
    parser.add_argument("--n-codes", type=int, default=16, help="Number of codebook codes")
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors")
    parser.add_argument("--hamming-radius", type=int, default=1, help="Hamming ball radius")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="experiments/real/detailed_metrics.json", help="Output file")
    
    args = parser.parse_args()
    
    # Generate synthetic data
    np.random.seed(args.random_state)
    base_embeddings = np.random.randn(args.n_samples, args.dim).astype(np.float32)
    queries = np.random.randn(args.n_queries, args.dim).astype(np.float32)
    
    # Generate ground truth
    distances = euclidean_distances(queries, base_embeddings)
    ground_truth = np.argsort(distances, axis=1)[:, :args.k].astype(np.int32)
    
    # Collect metrics
    print("Collecting detailed metrics...")
    metrics = collect_metrics(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth=ground_truth,
        n_bits=args.n_bits,
        n_codes=args.n_codes,
        k=args.k,
        hamming_radius=args.hamming_radius,
        random_state=args.random_state,
    )
    
    # Save metrics
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Metrics saved to {output_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Metrics Summary")
    print("=" * 70)
    print(f"Baseline recall: {metrics['baseline']['recall']:.4f}")
    print(f"GTH recall: {metrics['gth']['recall']:.4f}")
    print(f"Recall difference: {metrics['comparison']['recall_difference']:+.4f}")
    print(f"\nCoverage:")
    print(f"  Baseline: {metrics['baseline']['coverage_rate']:.1%}")
    print(f"  GTH: {metrics['gth']['coverage_rate']:.1%}")
    print(f"\nGTH Issues:")
    print(f"  Invalid bucket indices: {metrics['gth']['invalid_bucket_indices']} ({metrics['gth']['invalid_bucket_rate']:.1%})")
    print(f"  Empty buckets: {metrics['gth']['empty_buckets']}")
    print(f"  Consistency rate: {metrics['gth']['consistency_rate']:.1%}")
    print(f"  Avg invalid buckets per query: {metrics['gth']['avg_invalid_buckets_per_query']:.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

