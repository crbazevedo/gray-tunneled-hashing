#!/usr/bin/env python3
"""
Investigate J(φ) correlation with recall.

Analyzes why J(φ) worsens but recall improves for n_bits=8.
Tests hypotheses:
- H1: J(φ) initial value differs significantly for n_bits=8
- H2: J(φ) optimization direction is wrong for n_bits=8
- H3: J(φ) doesn't capture query-time Hamming ball expansion
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gray_tunneled_hashing.binary.lsh_families import create_lsh_family
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.api.query_pipeline import query_with_hamming_ball
from gray_tunneled_hashing.evaluation.metrics import recall_at_k
from gray_tunneled_hashing.distribution.j_phi_objective import (
    compute_j_phi_cost_real_embeddings,
    compute_j_phi_cost_multi_radius,
    validate_radius_weights,
)
from gray_tunneled_hashing.data.synthetic_generators import generate_synthetic_dataset


def load_benchmark_results(results_file: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(results_file, "r") as f:
        return json.load(f)


def analyze_jphi_recall_correlation(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze correlation between J(φ) improvement and recall.
    
    Returns:
        Dictionary with correlation analysis results
    """
    gth_results = results.get("gth_sprint8", {})
    
    # Extract data
    jphi_improvements = []
    recalls = []
    n_bits_list = []
    configs = []
    
    for key, result in gth_results.items():
        jphi_impr = result.get("j_phi_improvement", 0.0)
        recall = result.get("recall", 0.0)
        n_bits = result.get("n_bits", 0)
        
        jphi_improvements.append(jphi_impr)
        recalls.append(recall)
        n_bits_list.append(n_bits)
        configs.append(key)
    
    jphi_improvements = np.array(jphi_improvements)
    recalls = np.array(recalls)
    n_bits_list = np.array(n_bits_list)
    
    # Overall correlation
    pearson_r, pearson_p = pearsonr(jphi_improvements, recalls)
    spearman_r, spearman_p = spearmanr(jphi_improvements, recalls)
    
    # Correlation by n_bits
    correlations_by_nbits = {}
    for n_bits in np.unique(n_bits_list):
        mask = n_bits_list == n_bits
        if np.sum(mask) < 2:
            continue
        
        jphi_subset = jphi_improvements[mask]
        recall_subset = recalls[mask]
        
        pearson_r_sub, pearson_p_sub = pearsonr(jphi_subset, recall_subset)
        spearman_r_sub, spearman_p_sub = spearmanr(jphi_subset, recall_subset)
        
        correlations_by_nbits[n_bits] = {
            "pearson_r": float(pearson_r_sub),
            "pearson_p": float(pearson_p_sub),
            "spearman_r": float(spearman_r_sub),
            "spearman_p": float(spearman_p_sub),
            "n_samples": int(np.sum(mask)),
        }
    
    return {
        "overall": {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "n_samples": len(jphi_improvements),
        },
        "by_n_bits": correlations_by_nbits,
        "data": {
            "jphi_improvements": jphi_improvements.tolist(),
            "recalls": recalls.tolist(),
            "n_bits": n_bits_list.tolist(),
            "configs": configs,
        },
    }


def test_hypothesis_h1(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test H1: J(φ) initial value differs significantly for n_bits=8.
    
    Compares initial J(φ) values for n_bits=6 vs n_bits=8.
    """
    gth_results = results.get("gth_sprint8", {})
    
    initial_costs_6 = []
    initial_costs_8 = []
    
    for key, result in gth_results.items():
        n_bits = result.get("n_bits", 0)
        jphi_initial = result.get("j_phi_initial", 0.0)
        
        if n_bits == 6:
            initial_costs_6.append(jphi_initial)
        elif n_bits == 8:
            initial_costs_8.append(jphi_initial)
    
    if len(initial_costs_6) == 0 or len(initial_costs_8) == 0:
        return {"error": "Insufficient data for n_bits=6 or n_bits=8"}
    
    initial_costs_6 = np.array(initial_costs_6)
    initial_costs_8 = np.array(initial_costs_8)
    
    # Statistical test
    from scipy.stats import ttest_ind, mannwhitneyu
    
    t_stat, t_p = ttest_ind(initial_costs_6, initial_costs_8)
    u_stat, u_p = mannwhitneyu(initial_costs_6, initial_costs_8, alternative='two-sided')
    
    return {
        "n_bits_6": {
            "mean": float(np.mean(initial_costs_6)),
            "std": float(np.std(initial_costs_6)),
            "n": len(initial_costs_6),
        },
        "n_bits_8": {
            "mean": float(np.mean(initial_costs_8)),
            "std": float(np.std(initial_costs_8)),
            "n": len(initial_costs_8),
        },
        "t_test": {
            "statistic": float(t_stat),
            "p_value": float(t_p),
            "significant": t_p < 0.05,
        },
        "mannwhitney_u": {
            "statistic": float(u_stat),
            "p_value": float(u_p),
            "significant": u_p < 0.05,
        },
        "conclusion": "H1 supported" if (t_p < 0.05 or u_p < 0.05) else "H1 not supported",
    }


def test_hypothesis_h2(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test H2: J(φ) optimization direction is wrong for n_bits=8.
    
    Analyzes whether J(φ) improvement correlates with recall improvement
    differently for n_bits=6 vs n_bits=8.
    """
    gth_results = results.get("gth_sprint8", {})
    
    data_6 = {"jphi_impr": [], "recall": []}
    data_8 = {"jphi_impr": [], "recall": []}
    
    for key, result in gth_results.items():
        n_bits = result.get("n_bits", 0)
        jphi_impr = result.get("j_phi_improvement", 0.0)
        recall = result.get("recall", 0.0)
        
        if n_bits == 6:
            data_6["jphi_impr"].append(jphi_impr)
            data_6["recall"].append(recall)
        elif n_bits == 8:
            data_8["jphi_impr"].append(jphi_impr)
            data_8["recall"].append(recall)
    
    if len(data_6["jphi_impr"]) < 2 or len(data_8["jphi_impr"]) < 2:
        return {"error": "Insufficient data"}
    
    # Compute correlations
    corr_6 = pearsonr(data_6["jphi_impr"], data_6["recall"])
    corr_8 = pearsonr(data_8["jphi_impr"], data_8["recall"])
    
    return {
        "n_bits_6": {
            "pearson_r": float(corr_6[0]),
            "pearson_p": float(corr_6[1]),
            "n_samples": len(data_6["jphi_impr"]),
        },
        "n_bits_8": {
            "pearson_r": float(corr_8[0]),
            "pearson_p": float(corr_8[1]),
            "n_samples": len(data_8["jphi_impr"]),
        },
        "difference": {
            "r_diff": float(corr_6[0] - corr_8[0]),
            "conclusion": "H2 supported" if (corr_6[0] > 0 and corr_8[0] < 0) else "H2 partially supported" if corr_6[0] > corr_8[0] else "H2 not supported",
        },
    }


def test_hypothesis_h3(
    base_embeddings: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    n_bits: int,
    n_codes: int,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Test H3: J(φ) doesn't capture query-time Hamming ball expansion.
    
    Compares J(φ) computed with single radius vs multi-radius,
    and checks if multi-radius better correlates with recall.
    """
    lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=base_embeddings.shape[1], random_state=random_state)
    
    # Build index with single-radius objective
    index_single = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        n_bits=n_bits,
        n_codes=n_codes,
        max_two_swap_iters=20,
        random_state=random_state,
        hamming_radii=None,  # Single radius
    )
    
    # Build index with multi-radius objective [1, 2, 3]
    index_multi = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        n_bits=n_bits,
        n_codes=n_codes,
        max_two_swap_iters=20,
        random_state=random_state,
        hamming_radii=[1, 2, 3],
    )
    
    # Compute recall for both
    from gray_tunneled_hashing.api.query_pipeline import query_with_hamming_ball
    
    recalls_single = []
    recalls_multi = []
    
    for i, query in enumerate(queries):
        # Single-radius index
        result_single = query_with_hamming_ball(
            query_embedding=query,
            encoder=lsh.hash,
            permutation=index_single.hasher.get_assignment(),
            code_to_bucket=index_single.code_to_bucket,
            bucket_to_dataset_indices={k: list(range(len(base_embeddings))) for k in range(len(index_single.pi))},
            hamming_radius=1,
        )
        recall_single = recall_at_k(
            retrieved_indices=result_single.candidate_indices[:10],
            ground_truth_indices=ground_truth[i],
            k=10,
        )
        recalls_single.append(recall_single)
        
        # Multi-radius index
        result_multi = query_with_hamming_ball(
            query_embedding=query,
            encoder=lsh.hash,
            permutation=index_multi.hasher.get_assignment(),
            code_to_bucket=index_multi.code_to_bucket,
            bucket_to_dataset_indices={k: list(range(len(base_embeddings))) for k in range(len(index_multi.pi))},
            hamming_radius=1,
        )
        recall_multi = recall_at_k(
            retrieved_indices=result_multi.candidate_indices[:10],
            ground_truth_indices=ground_truth[i],
            k=10,
        )
        recalls_multi.append(recall_multi)
    
    return {
        "single_radius": {
            "mean_recall": float(np.mean(recalls_single)),
            "std_recall": float(np.std(recalls_single)),
        },
        "multi_radius": {
            "mean_recall": float(np.mean(recalls_multi)),
            "std_recall": float(np.std(recalls_multi)),
        },
        "improvement": float(np.mean(recalls_multi) - np.mean(recalls_single)),
        "conclusion": "H3 supported" if np.mean(recalls_multi) > np.mean(recalls_single) else "H3 not supported",
    }


def generate_plots(analysis_results: Dict[str, Any], output_dir: Path):
    """Generate plots for J(φ) correlation analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data = analysis_results.get("data", {})
    if not data:
        return
    
    jphi_impr = np.array(data["jphi_improvements"])
    recalls = np.array(data["recalls"])
    n_bits = np.array(data["n_bits"])
    
    # Plot 1: Overall scatter
    plt.figure(figsize=(10, 6))
    plt.scatter(jphi_impr, recalls, alpha=0.6)
    plt.xlabel("J(φ) Improvement")
    plt.ylabel("Recall")
    plt.title("J(φ) Improvement vs Recall (All Configurations)")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "jphi_recall_scatter_overall.png", dpi=150)
    plt.close()
    
    # Plot 2: By n_bits
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, n_bits_val in enumerate([6, 8]):
        mask = n_bits == n_bits_val
        if np.sum(mask) == 0:
            continue
        
        axes[idx].scatter(jphi_impr[mask], recalls[mask], alpha=0.6)
        axes[idx].set_xlabel("J(φ) Improvement")
        axes[idx].set_ylabel("Recall")
        axes[idx].set_title(f"n_bits={n_bits_val}")
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "jphi_recall_scatter_by_nbits.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Investigate J(φ) correlation with recall")
    parser.add_argument("--benchmark-results", type=str, required=True, help="Path to benchmark results JSON")
    parser.add_argument("--output-dir", type=str, default="experiments/real/reports", help="Output directory for reports")
    parser.add_argument("--generate-data", action="store_true", help="Generate new test data for H3")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load benchmark results
    print("Loading benchmark results...")
    results = load_benchmark_results(args.benchmark_results)
    
    # Analyze correlation
    print("Analyzing J(φ) correlation with recall...")
    correlation_analysis = analyze_jphi_recall_correlation(results)
    
    # Test hypotheses
    print("Testing H1: J(φ) initial value differs for n_bits=8...")
    h1_results = test_hypothesis_h1(results)
    
    print("Testing H2: J(φ) optimization direction wrong for n_bits=8...")
    h2_results = test_hypothesis_h2(results)
    
    h3_results = None
    if args.generate_data:
        print("Testing H3: J(φ) doesn't capture Hamming ball expansion...")
        # Generate test data
        base_embeddings, queries, ground_truth = generate_synthetic_dataset(
            n_base=1000, n_queries=100, dim=64, k=10, random_state=42
        )
        h3_results = test_hypothesis_h3(
            base_embeddings=base_embeddings,
            queries=queries,
            ground_truth=ground_truth,
            n_bits=8,
            n_codes=32,
        )
    
    # Generate plots
    print("Generating plots...")
    generate_plots(correlation_analysis, output_dir)
    
    # Generate report
    report = {
        "correlation_analysis": correlation_analysis,
        "hypothesis_h1": h1_results,
        "hypothesis_h2": h2_results,
        "hypothesis_h3": h3_results,
    }
    
    # Save JSON report
    json_path = output_dir / "jphi_correlation_analysis.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    
    # Generate markdown report
    md_path = output_dir / "JPHI_CORRELATION_ANALYSIS.md"
    with open(md_path, "w") as f:
        f.write("# J(φ) Correlation Analysis\n\n")
        f.write("## Overall Correlation\n\n")
        overall = correlation_analysis["overall"]
        f.write(f"- Pearson r: {overall['pearson_r']:.4f} (p={overall['pearson_p']:.4f})\n")
        f.write(f"- Spearman r: {overall['spearman_r']:.4f} (p={overall['spearman_p']:.4f})\n")
        f.write(f"- N samples: {overall['n_samples']}\n\n")
        
        f.write("## Correlation by n_bits\n\n")
        for n_bits, corr_data in correlation_analysis["by_n_bits"].items():
            f.write(f"### n_bits={n_bits}\n\n")
            f.write(f"- Pearson r: {corr_data['pearson_r']:.4f} (p={corr_data['pearson_p']:.4f})\n")
            f.write(f"- Spearman r: {corr_data['spearman_r']:.4f} (p={corr_data['spearman_p']:.4f})\n")
            f.write(f"- N samples: {corr_data['n_samples']}\n\n")
        
        f.write("## Hypothesis Tests\n\n")
        f.write("### H1: J(φ) initial value differs for n_bits=8\n\n")
        if "error" not in h1_results:
            f.write(f"- n_bits=6: mean={h1_results['n_bits_6']['mean']:.4f}, std={h1_results['n_bits_6']['std']:.4f}\n")
            f.write(f"- n_bits=8: mean={h1_results['n_bits_8']['mean']:.4f}, std={h1_results['n_bits_8']['std']:.4f}\n")
            f.write(f"- t-test p-value: {h1_results['t_test']['p_value']:.4f}\n")
            f.write(f"- Conclusion: {h1_results['conclusion']}\n\n")
        else:
            f.write(f"Error: {h1_results['error']}\n\n")
        
        f.write("### H2: J(φ) optimization direction wrong for n_bits=8\n\n")
        if "error" not in h2_results:
            f.write(f"- n_bits=6 correlation: r={h2_results['n_bits_6']['pearson_r']:.4f}\n")
            f.write(f"- n_bits=8 correlation: r={h2_results['n_bits_8']['pearson_r']:.4f}\n")
            f.write(f"- Difference: {h2_results['difference']['r_diff']:.4f}\n")
            f.write(f"- Conclusion: {h2_results['difference']['conclusion']}\n\n")
        else:
            f.write(f"Error: {h2_results['error']}\n\n")
        
        if h3_results:
            f.write("### H3: J(φ) doesn't capture Hamming ball expansion\n\n")
            f.write(f"- Single-radius recall: {h3_results['single_radius']['mean_recall']:.4f}\n")
            f.write(f"- Multi-radius recall: {h3_results['multi_radius']['mean_recall']:.4f}\n")
            f.write(f"- Improvement: {h3_results['improvement']:.4f}\n")
            f.write(f"- Conclusion: {h3_results['conclusion']}\n\n")
    
    print(f"\n✓ Analysis complete. Reports saved to {output_dir}")
    print(f"  - JSON: {json_path}")
    print(f"  - Markdown: {md_path}")


if __name__ == "__main__":
    main()

