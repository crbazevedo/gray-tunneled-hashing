#!/usr/bin/env python3
"""
Analyze Hamming ball coverage for different radii.

Tests radii 1, 2, 3, 4 and analyzes:
- Coverage: |candidates| / |dataset|
- Recall improvement vs baseline
- Search time overhead
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gray_tunneled_hashing.binary.lsh_families import create_lsh_family
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.api.query_pipeline import query_with_hamming_ball
from gray_tunneled_hashing.evaluation.metrics import recall_at_k
from gray_tunneled_hashing.data.synthetic_generators import generate_synthetic_dataset


def compute_baseline_recall(
    queries: np.ndarray,
    base_embeddings: np.ndarray,
    ground_truth: np.ndarray,
    lsh_encoder,
    k: int,
    hamming_radius: int,
) -> Dict[str, Any]:
    """Compute baseline recall."""
    query_codes = lsh_encoder(queries)
    base_codes = lsh_encoder(base_embeddings)
    
    start_time = time.time()
    recalls = []
    candidates_per_query = []
    
    for i in range(len(queries)):
        query_code = query_codes[i]
        hamming_dists = np.sum(query_code != base_codes, axis=1)
        candidates = np.where(hamming_dists <= hamming_radius)[0]
        candidates_per_query.append(len(candidates))
        
        if len(candidates) > 0:
            retrieved = set(candidates[:k])
            relevant = set(ground_truth[i][:k])
            recall = len(retrieved & relevant) / len(relevant) if len(relevant) > 0 else 0.0
            recalls.append(recall)
        else:
            recalls.append(0.0)
    
    search_time = time.time() - start_time
    
    return {
        "recall": float(np.mean(recalls)),
        "search_time_s": float(search_time),
        "avg_candidates": float(np.mean(candidates_per_query)),
        "coverage": float(np.mean(candidates_per_query) / len(base_embeddings)),
    }


def analyze_coverage_for_radius(
    queries: np.ndarray,
    base_embeddings: np.ndarray,
    ground_truth: np.ndarray,
    index_obj,
    lsh_encoder,
    k: int,
    hamming_radius: int,
) -> Dict[str, Any]:
    """Analyze coverage and recall for a specific Hamming radius."""
    # Build bucket_to_dataset_indices
    base_codes = lsh_encoder(base_embeddings)
    bucket_to_dataset_indices = {}
    for dataset_idx, code in enumerate(base_codes):
        code_tuple = tuple(code.astype(int).tolist())
        if code_tuple in index_obj.code_to_bucket:
            bucket_idx = index_obj.code_to_bucket[code_tuple]
            if bucket_idx not in bucket_to_dataset_indices:
                bucket_to_dataset_indices[bucket_idx] = []
            bucket_to_dataset_indices[bucket_idx].append(dataset_idx)
    
    permutation = index_obj.hasher.get_assignment()
    
    start_time = time.time()
    recalls = []
    candidates_per_query = []
    
    for i, query in enumerate(queries):
        result = query_with_hamming_ball(
            query_embedding=query,
            encoder=lsh_encoder,
            permutation=permutation,
            code_to_bucket=index_obj.code_to_bucket,
            bucket_to_dataset_indices=bucket_to_dataset_indices,
            hamming_radius=hamming_radius,
        )
        
        candidates_per_query.append(len(result.candidate_indices))
        
        recall = recall_at_k(
            retrieved_indices=result.candidate_indices[:k],
            ground_truth_indices=ground_truth[i],
            k=k,
        )
        recalls.append(recall)
    
    search_time = time.time() - start_time
    
    return {
        "recall": float(np.mean(recalls)),
        "recall_std": float(np.std(recalls)),
        "search_time_s": float(search_time),
        "avg_search_time_ms": float(search_time / len(queries) * 1000),
        "avg_candidates": float(np.mean(candidates_per_query)),
        "coverage": float(np.mean(candidates_per_query) / len(base_embeddings)),
    }


def analyze_coverage(
    base_embeddings: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    n_bits: int,
    n_codes: int,
    radii: List[int],
    k: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Analyze coverage for multiple radii."""
    lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=base_embeddings.shape[1], random_state=random_state)
    
    # Build index
    print("Building distribution-aware index...")
    build_start = time.time()
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        n_bits=n_bits,
        n_codes=n_codes,
        max_two_swap_iters=20,
        random_state=random_state,
    )
    build_time = time.time() - build_start
    
    # Compute baseline for radius=1
    baseline = compute_baseline_recall(
        queries=queries,
        base_embeddings=base_embeddings,
        ground_truth=ground_truth,
        lsh_encoder=lsh.hash,
        k=k,
        hamming_radius=1,
    )
    
    # Analyze each radius
    results = {}
    for radius in radii:
        print(f"Analyzing radius={radius}...")
        result = analyze_coverage_for_radius(
            queries=queries,
            base_embeddings=base_embeddings,
            ground_truth=ground_truth,
            index_obj=index_obj,
            lsh_encoder=lsh.hash,
            k=k,
            hamming_radius=radius,
        )
        
        result["recall_improvement"] = result["recall"] - baseline["recall"]
        result["relative_improvement_pct"] = (
            (result["recall"] - baseline["recall"]) / baseline["recall"] * 100
            if baseline["recall"] > 0 else 0.0
        )
        result["search_time_overhead"] = result["avg_search_time_ms"] - baseline.get("avg_search_time_ms", 0.0)
        
        results[f"radius_{radius}"] = result
    
    return {
        "build_time_s": float(build_time),
        "baseline": baseline,
        "results": results,
        "config": {
            "n_bits": n_bits,
            "n_codes": n_codes,
            "k": k,
            "n_base": len(base_embeddings),
            "n_queries": len(queries),
        },
    }


def generate_recommendations(analysis_results: Dict[str, Any]) -> List[str]:
    """Generate radius recommendations based on analysis."""
    results = analysis_results["results"]
    recommendations = []
    
    # Find radius with best recall/coverage trade-off
    best_radius = None
    best_score = -np.inf
    
    for radius_key, result in results.items():
        radius = int(radius_key.split("_")[1])
        recall = result["recall"]
        coverage = result["coverage"]
        search_time = result["avg_search_time_ms"]
        
        # Score: recall / (search_time + 1) to balance recall and speed
        score = recall / (search_time + 1.0)
        
        if score > best_score:
            best_score = score
            best_radius = radius
    
    if best_radius:
        recommendations.append(f"Optimal radius: {best_radius} (best recall/speed trade-off)")
    
    # Check if larger radius significantly improves recall
    radius_1 = results.get("radius_1", {})
    radius_2 = results.get("radius_2", {})
    radius_3 = results.get("radius_3", {})
    radius_4 = results.get("radius_4", {})
    
    if radius_2 and radius_1:
        improvement_2 = radius_2["recall"] - radius_1["recall"]
        if improvement_2 > 0.05:  # 5% improvement
            recommendations.append(f"Radius 2 provides {improvement_2:.1%} recall improvement over radius 1")
    
    if radius_3 and radius_2:
        improvement_3 = radius_3["recall"] - radius_2["recall"]
        if improvement_3 > 0.02:  # 2% improvement
            recommendations.append(f"Radius 3 provides {improvement_3:.1%} recall improvement over radius 2")
        elif improvement_3 < 0.01:
            recommendations.append("Radius 3 provides minimal improvement - radius 2 may be sufficient")
    
    # Coverage recommendations
    if radius_1 and radius_1["coverage"] < 0.1:
        recommendations.append("Low coverage (<10%) with radius=1 - consider larger radius or adaptive strategy")
    
    return recommendations


def main():
    parser = argparse.ArgumentParser(description="Analyze Hamming ball coverage for different radii")
    parser.add_argument("--n-base", type=int, default=1000, help="Number of base embeddings")
    parser.add_argument("--n-queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--n-bits", type=int, default=8, help="Number of bits")
    parser.add_argument("--n-codes", type=int, default=32, help="Number of codes")
    parser.add_argument("--k", type=int, default=10, help="k for recall@k")
    parser.add_argument("--radii", type=str, default="1,2,3,4", help="Comma-separated list of radii")
    parser.add_argument("--output-dir", type=str, default="experiments/real/reports", help="Output directory")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse radii
    radii = [int(x.strip()) for x in args.radii.split(",")]
    
    # Generate test data
    print("Generating test data...")
    base_embeddings, queries, ground_truth = generate_synthetic_dataset(
        n_base=args.n_base,
        n_queries=args.n_queries,
        dim=64,
        k=args.k,
        random_state=args.random_state,
    )
    
    # Analyze coverage
    print("Analyzing Hamming ball coverage...")
    analysis_results = analyze_coverage(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth=ground_truth,
        n_bits=args.n_bits,
        n_codes=args.n_codes,
        radii=radii,
        k=args.k,
        random_state=args.random_state,
    )
    
    # Generate recommendations
    recommendations = generate_recommendations(analysis_results)
    analysis_results["recommendations"] = recommendations
    
    # Save results
    json_path = output_dir / "hamming_ball_coverage_analysis.json"
    with open(json_path, "w") as f:
        json.dump(analysis_results, f, indent=2)
    
    # Generate markdown report
    md_path = output_dir / "HAMMING_BALL_COVERAGE_ANALYSIS.md"
    with open(md_path, "w") as f:
        f.write("# Hamming Ball Coverage Analysis\n\n")
        f.write(f"**Build time**: {analysis_results['build_time_s']:.2f} seconds\n\n")
        
        f.write("## Baseline (LSH without GTH, radius=1)\n\n")
        baseline = analysis_results["baseline"]
        f.write(f"- Recall: {baseline['recall']:.4f}\n")
        f.write(f"- Coverage: {baseline['coverage']:.2%}\n")
        f.write(f"- Avg candidates: {baseline['avg_candidates']:.1f}\n\n")
        
        f.write("## Results by Radius\n\n")
        f.write("| Radius | Recall | Coverage | Avg Candidates | Search Time (ms) | Improvement |\n")
        f.write("|--------|--------|----------|----------------|------------------|-------------|\n")
        
        for radius_key in sorted(analysis_results["results"].keys()):
            radius = int(radius_key.split("_")[1])
            result = analysis_results["results"][radius_key]
            f.write(f"| {radius} | {result['recall']:.4f} | {result['coverage']:.2%} | "
                   f"{result['avg_candidates']:.1f} | {result['avg_search_time_ms']:.2f} | "
                   f"{result['relative_improvement_pct']:+.1f}% |\n")
        f.write("\n")
        
        f.write("## Recommendations\n\n")
        for rec in recommendations:
            f.write(f"- {rec}\n")
        f.write("\n")
    
    print(f"\n✓ Analysis complete. Reports saved to {output_dir}")
    print(f"  - JSON: {json_path}")
    print(f"  - Markdown: {md_path}")


if __name__ == "__main__":
    main()
