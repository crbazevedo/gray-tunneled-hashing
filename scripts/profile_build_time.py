#!/usr/bin/env python3
"""
Profile build time for distribution-aware index construction.

Identifies bottlenecks and optimization opportunities.
"""

import argparse
import cProfile
import pstats
import io
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gray_tunneled_hashing.binary.lsh_families import create_lsh_family
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.data.synthetic_generators import generate_synthetic_dataset


def profile_build_time(
    base_embeddings: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    n_bits: int,
    n_codes: int,
    max_iters: int = 20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Profile build_distribution_aware_index execution."""
    lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=base_embeddings.shape[1], random_state=random_state)
    
    # Profile
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.time()
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        n_bits=n_bits,
        n_codes=n_codes,
        max_two_swap_iters=max_iters,
        random_state=random_state,
    )
    elapsed_time = time.time() - start_time
    
    profiler.disable()
    
    # Get profile stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions
    
    profile_output = s.getvalue()
    
    # Parse profile to extract key metrics
    lines = profile_output.split('\n')
    function_stats = []
    
    for line in lines[5:35]:  # Skip header, get top functions
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 5:
            try:
                ncalls = parts[0]
                tottime = float(parts[2])
                cumtime = float(parts[3])
                func_name = ' '.join(parts[5:])
                function_stats.append({
                    "ncalls": ncalls,
                    "tottime": tottime,
                    "cumtime": cumtime,
                    "function": func_name,
                })
            except (ValueError, IndexError):
                continue
    
    return {
        "total_time_s": float(elapsed_time),
        "top_functions": function_stats[:20],
        "full_profile": profile_output,
    }


def identify_bottlenecks(profile_results: Dict[str, Any]) -> Dict[str, Any]:
    """Identify bottlenecks from profile results."""
    top_functions = profile_results["top_functions"]
    
    # Categorize functions
    jphi_functions = []
    encoding_functions = []
    other_functions = []
    
    for func in top_functions:
        func_name = func["function"]
        if "j_phi" in func_name.lower() or "compute_j_phi" in func_name:
            jphi_functions.append(func)
        elif "encode" in func_name.lower() or "hash" in func_name.lower() or "lsh" in func_name.lower():
            encoding_functions.append(func)
        else:
            other_functions.append(func)
    
    # Compute time spent in each category
    total_cumtime = sum(f["cumtime"] for f in top_functions)
    jphi_time = sum(f["cumtime"] for f in jphi_functions)
    encoding_time = sum(f["cumtime"] for f in encoding_functions)
    other_time = sum(f["cumtime"] for f in other_functions)
    
    return {
        "jphi_functions": {
            "count": len(jphi_functions),
            "total_time": float(jphi_time),
            "percentage": float(jphi_time / total_cumtime * 100) if total_cumtime > 0 else 0.0,
            "top_functions": jphi_functions[:5],
        },
        "encoding_functions": {
            "count": len(encoding_functions),
            "total_time": float(encoding_time),
            "percentage": float(encoding_time / total_cumtime * 100) if total_cumtime > 0 else 0.0,
            "top_functions": encoding_functions[:5],
        },
        "other_functions": {
            "count": len(other_functions),
            "total_time": float(other_time),
            "percentage": float(other_time / total_cumtime * 100) if total_cumtime > 0 else 0.0,
            "top_functions": other_functions[:5],
        },
        "recommendations": generate_optimization_recommendations(jphi_functions, encoding_functions),
    }


def generate_optimization_recommendations(jphi_funcs: list, encoding_funcs: list) -> list:
    """Generate optimization recommendations based on bottlenecks."""
    recommendations = []
    
    # Check if J(φ) computation is a bottleneck
    jphi_total = sum(f["cumtime"] for f in jphi_funcs)
    if jphi_total > 10.0:  # More than 10 seconds
        recommendations.append({
            "category": "J(φ) computation",
            "issue": "J(φ) computation takes significant time",
            "suggestions": [
                "Cache LSH encodings (queries, base_embeddings) to avoid recomputation",
                "Reduce sample_size_pairs for faster iterations",
                "Batch delta computations",
                "Parallelize sampling in J(φ) computation",
            ],
        })
    
    # Check if encoding is a bottleneck
    encoding_total = sum(f["cumtime"] for f in encoding_funcs)
    if encoding_total > 5.0:  # More than 5 seconds
        recommendations.append({
            "category": "LSH encoding",
            "issue": "LSH encoding takes significant time",
            "suggestions": [
                "Cache encoded codes (queries, base_embeddings)",
                "Use vectorized encoding if available",
                "Pre-compute encodings before optimization",
            ],
        })
    
    # Check for repeated calls
    for func in jphi_funcs[:3]:
        if func["ncalls"] != "1" and int(func["ncalls"].split('/')[0]) > 100:
            recommendations.append({
                "category": "Repeated calls",
                "issue": f"Function {func['function']} called {func['ncalls']} times",
                "suggestions": [
                    "Cache results of repeated calls",
                    "Optimize delta computation to avoid full cost recalculation",
                ],
            })
    
    return recommendations


def main():
    parser = argparse.ArgumentParser(description="Profile build time for distribution-aware index")
    parser.add_argument("--n-base", type=int, default=1000, help="Number of base embeddings")
    parser.add_argument("--n-queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--n-bits", type=int, default=8, help="Number of bits")
    parser.add_argument("--n-codes", type=int, default=32, help="Number of codes")
    parser.add_argument("--max-iters", type=int, default=20, help="Max iterations")
    parser.add_argument("--output-dir", type=str, default="experiments/real/reports", help="Output directory")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate test data
    print("Generating test data...")
    base_embeddings, queries, ground_truth = generate_synthetic_dataset(
        n_base=args.n_base,
        n_queries=args.n_queries,
        dim=64,
        k=10,
        random_state=args.random_state,
    )
    
    # Profile
    print("Profiling build time...")
    profile_results = profile_build_time(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth=ground_truth,
        n_bits=args.n_bits,
        n_codes=args.n_codes,
        max_iters=args.max_iters,
        random_state=args.random_state,
    )
    
    # Identify bottlenecks
    print("Identifying bottlenecks...")
    bottlenecks = identify_bottlenecks(profile_results)
    
    # Save results
    results = {
        "profile": profile_results,
        "bottlenecks": bottlenecks,
        "config": {
            "n_base": args.n_base,
            "n_queries": args.n_queries,
            "n_bits": args.n_bits,
            "n_codes": args.n_codes,
            "max_iters": args.max_iters,
        },
    }
    
    json_path = output_dir / "build_time_profile.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate markdown report
    md_path = output_dir / "BUILD_TIME_OPTIMIZATION.md"
    with open(md_path, "w") as f:
        f.write("# Build Time Optimization Analysis\n\n")
        f.write(f"**Total build time**: {profile_results['total_time_s']:.2f} seconds\n\n")
        
        f.write("## Top Functions by Cumulative Time\n\n")
        f.write("| Function | Calls | Total Time | Cumulative Time |\n")
        f.write("|----------|-------|------------|-----------------|\n")
        for func in profile_results["top_functions"][:15]:
            f.write(f"| {func['function'][:60]} | {func['ncalls']} | {func['tottime']:.4f} | {func['cumtime']:.4f} |\n")
        f.write("\n")
        
        f.write("## Bottleneck Analysis\n\n")
        f.write("### J(φ) Functions\n\n")
        jphi = bottlenecks["jphi_functions"]
        f.write(f"- Count: {jphi['count']}\n")
        f.write(f"- Total time: {jphi['total_time']:.2f}s ({jphi['percentage']:.1f}%)\n")
        f.write("- Top functions:\n")
        for func in jphi["top_functions"]:
            f.write(f"  - {func['function']}: {func['cumtime']:.4f}s\n")
        f.write("\n")
        
        f.write("### Encoding Functions\n\n")
        encoding = bottlenecks["encoding_functions"]
        f.write(f"- Count: {encoding['count']}\n")
        f.write(f"- Total time: {encoding['total_time']:.2f}s ({encoding['percentage']:.1f}%)\n")
        f.write("- Top functions:\n")
        for func in encoding["top_functions"]:
            f.write(f"  - {func['function']}: {func['cumtime']:.4f}s\n")
        f.write("\n")
        
        f.write("## Optimization Recommendations\n\n")
        for rec in bottlenecks["recommendations"]:
            f.write(f"### {rec['category']}\n\n")
            f.write(f"**Issue**: {rec['issue']}\n\n")
            f.write("**Suggestions**:\n")
            for suggestion in rec["suggestions"]:
                f.write(f"- {suggestion}\n")
            f.write("\n")
    
    print(f"\n✓ Profiling complete. Reports saved to {output_dir}")
    print(f"  - JSON: {json_path}")
    print(f"  - Markdown: {md_path}")


if __name__ == "__main__":
    import json
    main()

