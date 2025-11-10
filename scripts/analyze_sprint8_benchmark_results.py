#!/usr/bin/env python3
"""
Analysis of Sprint 8 benchmark results.

Generates comparative reports and result tables.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any
import sys


def load_results(results_file: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(results_file, "r") as f:
        return json.load(f)


def print_summary_table(results: Dict[str, Any]):
    """Print summary comparison table."""
    print("=" * 100)
    print("Sprint 8 Benchmark - Summary Comparison")
    print("=" * 100)
    print()
    
    comparisons = results.get("comparisons", {})
    if not comparisons:
        print("No comparisons available.")
        return
    
    print(f"{'LSH Family':<15} {'n_bits':<8} {'Radius':<8} {'Baseline':<12} {'GTH':<12} {'Improvement':<15} {'Status':<10}")
    print("-" * 100)
    
    for comp_key, comp in sorted(comparisons.items()):
        lsh_family = comp_key.split("_")[0]
        n_bits = comp_key.split("nbits")[1].split("_")[0] if "nbits" in comp_key else "N/A"
        radius = comp_key.split("radius")[1] if "radius" in comp_key else "N/A"
        
        baseline = comp["baseline_recall"]
        gth = comp["gth_recall"]
        improvement = comp["relative_improvement_pct"]
        status = "✅ Better" if comp["is_better"] else "❌ Worse"
        
        print(f"{lsh_family:<15} {n_bits:<8} {radius:<8} {baseline:<12.4f} {gth:<12.4f} {improvement:>+14.2f}% {status:<10}")
    
    print()


def print_detailed_results(results: Dict[str, Any]):
    """Print detailed results for each configuration."""
    print("=" * 100)
    print("Detailed Results")
    print("=" * 100)
    print()
    
    # Baselines
    print("Baselines:")
    print("-" * 100)
    for key, result in sorted(results.get("baselines", {}).items()):
        print(f"  {key}:")
        print(f"    Recall: {result['recall']:.4f} ± {result.get('recall_std', 0):.4f}")
        print(f"    Search time: {result['avg_search_time_ms']:.2f} ms/query")
        print()
    
    # GTH Sprint 8
    print("GTH Sprint 8:")
    print("-" * 100)
    for key, result in sorted(results.get("gth_sprint8", {}).items()):
        print(f"  {key}:")
        print(f"    Recall: {result['recall']:.4f} ± {result.get('recall_std', 0):.4f}")
        print(f"    Build time: {result['build_time_s']:.2f} s")
        print(f"    Search time: {result['avg_search_time_ms']:.2f} ms/query")
        print(f"    J(φ) cost: {result['j_phi_cost']:.4f}")
        print(f"    J(φ) improvement: {result['j_phi_improvement']*100:.2f}%")
        print(f"    Hamming ball coverage: {result.get('hamming_ball_coverage', 0)*100:.2f}%")
        print()


def print_best_configurations(results: Dict[str, Any]):
    """Print best GTH configurations."""
    print("=" * 100)
    print("Best GTH Configurations")
    print("=" * 100)
    print()
    
    gth_results = results.get("gth_sprint8", {})
    if not gth_results:
        print("No GTH results available.")
        return
    
    # Sort by recall
    sorted_results = sorted(gth_results.items(), key=lambda x: x[1]["recall"], reverse=True)
    
    print(f"{'Configuration':<60} {'Recall':<12} {'Build Time':<12} {'J(φ) Impr.':<12}")
    print("-" * 100)
    
    for key, result in sorted_results[:10]:  # Top 10
        config_short = key[:60]
        recall = result["recall"]
        build_time = result["build_time_s"]
        j_phi_impr = result["j_phi_improvement"] * 100
        
        print(f"{config_short:<60} {recall:<12.4f} {build_time:<12.2f} {j_phi_impr:>+11.2f}%")
    
    print()


def generate_markdown_report(results: Dict[str, Any], output_file: str):
    """Generate markdown report."""
    report_lines = [
        "# Sprint 8 Benchmark Results",
        "",
        f"**Date**: {results['metadata'].get('date', 'N/A')}",
        f"**Dataset**: {results['metadata']['dataset']}",
        f"**N_base**: {results['metadata']['n_base']}",
        f"**N_queries**: {results['metadata']['n_queries']}",
        "",
        "## Summary Comparison",
        "",
        "| LSH Family | n_bits | Radius | Baseline | GTH | Improvement | Status |",
        "|------------|--------|--------|----------|-----|-------------|--------|",
    ]
    
    comparisons = results.get("comparisons", {})
    for comp_key, comp in sorted(comparisons.items()):
        lsh_family = comp_key.split("_")[0]
        n_bits = comp_key.split("nbits")[1].split("_")[0] if "nbits" in comp_key else "N/A"
        radius = comp_key.split("radius")[1] if "radius" in comp_key else "N/A"
        
        baseline = comp["baseline_recall"]
        gth = comp["gth_recall"]
        improvement = comp["relative_improvement_pct"]
        status = "✅" if comp["is_better"] else "❌"
        
        report_lines.append(
            f"| {lsh_family} | {n_bits} | {radius} | {baseline:.4f} | {gth:.4f} | {improvement:+.2f}% | {status} |"
        )
    
    report_lines.extend([
        "",
        "## Best GTH Configurations",
        "",
        "| Configuration | Recall | Build Time (s) | J(φ) Improvement |",
        "|---------------|--------|----------------|------------------|",
    ])
    
    gth_results = results.get("gth_sprint8", {})
    sorted_results = sorted(gth_results.items(), key=lambda x: x[1]["recall"], reverse=True)
    
    for key, result in sorted_results[:10]:
        recall = result["recall"]
        build_time = result["build_time_s"]
        j_phi_impr = result["j_phi_improvement"] * 100
        report_lines.append(
            f"| {key} | {recall:.4f} | {build_time:.2f} | {j_phi_impr:+.2f}% |"
        )
    
    # Save report
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))
    
    print(f"✓ Markdown report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Sprint 8 benchmark results")
    parser.add_argument("--input", type=str, required=True, help="Input JSON results file")
    parser.add_argument("--output", type=str, help="Output markdown report file (optional)")
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.input)
    
    # Print summary
    print_summary_table(results)
    print_best_configurations(results)
    print_detailed_results(results)
    
    # Generate markdown report if requested
    if args.output:
        generate_markdown_report(results, args.output)


if __name__ == "__main__":
    main()

