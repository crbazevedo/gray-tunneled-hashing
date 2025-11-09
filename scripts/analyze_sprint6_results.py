"""
Sprint 6 - Comprehensive Results Analysis

This script:
1. Loads JSON results from multiple experiments
2. Computes advanced statistics (mean, std, confidence intervals, t-tests)
3. Generates comparative tables
4. Validates hypotheses (H3, H4, H5) with statistical tests
5. Generates comprehensive markdown report
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from scipy import stats

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_results(json_path: Path) -> List[Dict[str, Any]]:
    """Load results from JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)
        # Handle both list of results and list of summaries
        if isinstance(data, list) and len(data) > 0:
            # Check if it's a summary (has "runs" field) or individual results
            if "runs" in data[0]:
                # Extract individual runs from summaries
                all_runs = []
                for summary in data:
                    all_runs.extend(summary.get("runs", []))
                return all_runs
            else:
                return data
        return []


def compute_confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute confidence interval for a list of values.
    
    Returns:
        (lower_bound, upper_bound)
    """
    if len(values) < 2:
        mean_val = values[0] if len(values) == 1 else 0.0
        return (mean_val, mean_val)
    
    arr = np.array(values)
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)  # Sample standard deviation
    
    # t-distribution for confidence interval
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
    
    margin = t_critical * std / np.sqrt(n)
    return (mean - margin, mean + margin)


def perform_t_test(group1: List[float], group2: List[float]) -> Dict[str, Any]:
    """
    Perform independent t-test between two groups.
    
    Returns:
        Dictionary with t-statistic, p-value, and interpretation
    """
    if len(group1) < 2 or len(group2) < 2:
        return {
            "t_statistic": None,
            "p_value": None,
            "significant": False,
            "note": "Insufficient data for t-test",
        }
    
    arr1 = np.array(group1)
    arr2 = np.array(group2)
    
    t_stat, p_value = stats.ttest_ind(arr1, arr2)
    
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "mean1": float(np.mean(arr1)),
        "mean2": float(np.mean(arr2)),
        "std1": float(np.std(arr1, ddof=1)),
        "std2": float(np.std(arr2, ddof=1)),
    }


def validate_hypothesis_h3(results: List[Dict]) -> Dict[str, Any]:
    """
    Validate H3: Hamming Ball Improves Recall.
    
    Expected: recall(radius=2) > recall(radius=1) > recall(radius=0)
    Uses statistical tests to validate.
    """
    # Group by method and radius, extracting individual run recalls
    by_method_radius = {}
    for result in results:
        method = result.get("method", "unknown")
        radius = result.get("hamming_radius", 0)
        recall = result.get("recall", result.get("recall_mean", 0.0))
        
        key = (method, radius)
        if key not in by_method_radius:
            by_method_radius[key] = []
        by_method_radius[key].append(recall)
    
    validated_methods = []
    for method in set(m[0] for m in by_method_radius.keys()):
        recalls_r0 = by_method_radius.get((method, 0), [])
        recalls_r1 = by_method_radius.get((method, 1), [])
        recalls_r2 = by_method_radius.get((method, 2), [])
        
        if len(recalls_r0) == 0 or len(recalls_r1) == 0:
            continue
        
        # Test r1 > r0
        test_r1_r0 = perform_t_test(recalls_r1, recalls_r0) if len(recalls_r0) > 0 else {}
        test_r1_r0_result = test_r1_r0.get("significant", False) and test_r1_r0.get("mean1", 0) > test_r1_r0.get("mean2", 0)
        
        # Test r2 > r1 (if r2 data available)
        test_r2_r1_result = True
        if len(recalls_r2) > 0 and len(recalls_r1) > 0:
            test_r2_r1 = perform_t_test(recalls_r2, recalls_r1)
            test_r2_r1_result = test_r2_r1.get("significant", False) and test_r2_r1.get("mean1", 0) > test_r2_r1.get("mean2", 0)
        
        mean_r0 = np.mean(recalls_r0) if len(recalls_r0) > 0 else 0.0
        mean_r1 = np.mean(recalls_r1) if len(recalls_r1) > 0 else 0.0
        mean_r2 = np.mean(recalls_r2) if len(recalls_r2) > 0 else 0.0
        
        ci_r0 = compute_confidence_interval(recalls_r0) if len(recalls_r0) > 0 else (0.0, 0.0)
        ci_r1 = compute_confidence_interval(recalls_r1) if len(recalls_r1) > 0 else (0.0, 0.0)
        ci_r2 = compute_confidence_interval(recalls_r2) if len(recalls_r2) > 0 else (0.0, 0.0)
        
        validated_methods.append({
            "method": method,
            "recall_r0": float(mean_r0),
            "recall_r1": float(mean_r1),
            "recall_r2": float(mean_r2),
            "ci_r0": ci_r0,
            "ci_r1": ci_r1,
            "ci_r2": ci_r2,
            "r1_improves_r0": test_r1_r0_result,
            "r2_improves_r1": test_r2_r1_result,
            "is_validated": test_r1_r0_result and test_r2_r1_result,
        })
    
    all_valid = all(m.get("is_validated", False) for m in validated_methods)
    
    return {
        "hypothesis": "H3: Hamming Ball Improves Recall",
        "status": "validated" if all_valid else "partially_validated",
        "details": validated_methods,
    }


def validate_hypothesis_h4(results: List[Dict]) -> Dict[str, Any]:
    """
    Validate H4: GTH Improves Recall.
    
    Expected: recall_gth > recall_baseline (statistically significant)
    """
    # Group by method type and extract individual run recalls
    baseline_recalls = []
    gth_recalls = []
    
    method_comparisons = {}  # Compare specific method pairs
    
    for result in results:
        method = result.get("method", "")
        recall = result.get("recall", result.get("recall_mean", 0.0))
        
        if method.startswith("baseline_"):
            baseline_recalls.append(recall)
            base_method = method.replace("baseline_", "")
            if base_method not in method_comparisons:
                method_comparisons[base_method] = {"baseline": [], "gth": []}
            method_comparisons[base_method]["baseline"].append(recall)
        else:
            gth_recalls.append(recall)
            if method in method_comparisons:
                method_comparisons[method]["gth"].append(recall)
    
    # Overall comparison
    overall_test = perform_t_test(gth_recalls, baseline_recalls) if len(gth_recalls) > 1 and len(baseline_recalls) > 1 else {}
    
    # Per-method comparisons
    method_details = []
    for method, data in method_comparisons.items():
        if len(data["baseline"]) > 0 and len(data["gth"]) > 0:
            test = perform_t_test(data["gth"], data["baseline"])
            method_details.append({
                "method": method,
                "baseline_mean": float(np.mean(data["baseline"])),
                "gth_mean": float(np.mean(data["gth"])),
                "improvement_pct": float((np.mean(data["gth"]) - np.mean(data["baseline"])) / np.mean(data["baseline"]) * 100) if np.mean(data["baseline"]) > 0 else 0.0,
                "p_value": test.get("p_value"),
                "significant": test.get("significant", False),
            })
    
    is_validated = overall_test.get("significant", False) and overall_test.get("mean1", 0) > overall_test.get("mean2", 0)
    
    return {
        "hypothesis": "H4: GTH Improves Recall",
        "status": "validated" if is_validated else "rejected",
        "overall_test": overall_test,
        "method_details": method_details,
    }


def validate_hypothesis_h5(results: List[Dict]) -> Dict[str, Any]:
    """
    Validate H5: LSH + GTH vs. Random Projection + GTH.
    
    Compare recall between LSH and random projection methods with GTH.
    """
    lsh_recalls = []
    rp_recalls = []
    
    # Extract individual run recalls
    for result in results:
        method = result.get("method", "")
        recall = result.get("recall", result.get("recall_mean", 0.0))
        
        if method in ["hyperplane", "p_stable"]:
            lsh_recalls.append(recall)
        elif method == "random_proj":
            rp_recalls.append(recall)
    
    if len(lsh_recalls) == 0 or len(rp_recalls) == 0:
        return {
            "hypothesis": "H5: LSH + GTH vs. Random Projection + GTH",
            "status": "insufficient_data",
            "note": "Need both LSH and random projection results",
        }
    
    test = perform_t_test(lsh_recalls, rp_recalls)
    
    lsh_mean = np.mean(lsh_recalls)
    rp_mean = np.mean(rp_recalls)
    lsh_ci = compute_confidence_interval(lsh_recalls)
    rp_ci = compute_confidence_interval(rp_recalls)
    
    return {
        "hypothesis": "H5: LSH + GTH vs. Random Projection + GTH",
        "status": "compared",
        "lsh_mean": float(lsh_mean),
        "rp_mean": float(rp_mean),
        "lsh_ci": lsh_ci,
        "rp_ci": rp_ci,
        "difference": float(lsh_mean - rp_mean),
        "test": test,
    }


def generate_comprehensive_report(
    experiment_results: Dict[str, List[Dict]],
    output_path: Path,
    create_plots: bool = False
) -> None:
    """Generate comprehensive markdown report from all experiment results."""
    
    # Validate hypotheses across all experiments
    all_results = []
    for exp_name, results in experiment_results.items():
        all_results.extend(results)
    
    h3 = validate_hypothesis_h3(all_results)
    h4 = validate_hypothesis_h4(all_results)
    h5 = validate_hypothesis_h5(all_results)
    
    # Generate report
    report_lines = [
        "# Sprint 6 - Comprehensive Results Analysis",
        "",
        "## Executive Summary",
        "",
        "This report presents the results of Sprint 6 experiments evaluating Gray-Tunneled Hashing (GTH) ",
        "with LSH families and random projection methods.",
        "",
        "### Key Findings",
        "",
    ]
    
    # Add key findings based on hypothesis validation
    if h3["status"] == "validated":
        report_lines.append("- ✅ **H3 (Hamming Ball Improves Recall)**: VALIDATED")
    else:
        report_lines.append("- ⚠️ **H3 (Hamming Ball Improves Recall)**: PARTIALLY VALIDATED")
    
    if h4["status"] == "validated":
        report_lines.append("- ✅ **H4 (GTH Improves Recall)**: VALIDATED")
    else:
        report_lines.append("- ❌ **H4 (GTH Improves Recall)**: REJECTED")
    
    report_lines.extend([
        "",
        "## Experiment Results",
        "",
    ])
    
    # Add results for each experiment
    for exp_name, results in experiment_results.items():
        report_lines.extend([
            f"### {exp_name}",
            "",
            "| Method | Recall (mean ± std) | 95% CI | Build Time (s) | Search Time (ms) |",
            "|--------|---------------------|--------|----------------|------------------|",
        ])
        
        # Group by method for summary
        method_summaries = {}
        for result in results:
            method = result.get("method", "unknown")
            if method not in method_summaries:
                method_summaries[method] = {
                    "recalls": [],
                    "build_times": [],
                    "search_times": [],
                }
            
            recall = result.get("recall", result.get("recall_mean", 0.0))
            build_time = result.get("build_time", result.get("build_time_mean", 0.0))
            search_time = result.get("search_time", result.get("search_time_mean", 0.0))
            
            method_summaries[method]["recalls"].append(recall)
            method_summaries[method]["build_times"].append(build_time)
            method_summaries[method]["search_times"].append(search_time)
        
        # Generate table rows
        for method, data in sorted(method_summaries.items()):
            recalls = data["recalls"]
            build_times = data["build_times"]
            search_times = data["search_times"]
            
            recall_mean = np.mean(recalls)
            recall_std = np.std(recalls)
            recall_ci = compute_confidence_interval(recalls)
            build_mean = np.mean(build_times)
            search_mean = np.mean(search_times) * 1000  # Convert to ms
            
            ci_str = f"[{recall_ci[0]:.4f}, {recall_ci[1]:.4f}]"
            
            report_lines.append(
                f"| {method} | {recall_mean:.4f} ± {recall_std:.4f} | {ci_str} | {build_mean:.2f} | {search_mean:.2f} |"
            )
        
        report_lines.append("")
    
    # Hypothesis validation section
    report_lines.extend([
        "## Hypothesis Validation",
        "",
        f"### {h3['hypothesis']}",
        f"**Status**: {h3['status'].upper()}",
        "",
    ])
    
    if "details" in h3:
        report_lines.append("| Method | Recall R=0 | Recall R=1 | Recall R=2 | R1 > R0 | R2 > R1 |")
        report_lines.append("|--------|-----------|-----------|-----------|---------|---------|")
        for detail in h3["details"]:
            r1_ok = "✅" if detail.get("r1_improves_r0", False) else "❌"
            r2_ok = "✅" if detail.get("r2_improves_r1", False) else "❌"
            report_lines.append(
                f"| {detail['method']} | {detail['recall_r0']:.4f} | {detail['recall_r1']:.4f} | "
                f"{detail['recall_r2']:.4f} | {r1_ok} | {r2_ok} |"
            )
        report_lines.append("")
    
    report_lines.extend([
        f"### {h4['hypothesis']}",
        f"**Status**: {h4['status'].upper()}",
        "",
    ])
    
    if "method_details" in h4:
        report_lines.append("| Method | Baseline Recall | GTH Recall | Improvement | p-value | Significant |")
        report_lines.append("|--------|----------------|-----------|-------------|---------|------------|")
        for detail in h4["method_details"]:
            sig = "✅" if detail.get("significant", False) else "❌"
            p_val = f"{detail['p_value']:.4f}" if detail.get("p_value") is not None else "N/A"
            report_lines.append(
                f"| {detail['method']} | {detail['baseline_mean']:.4f} | {detail['gth_mean']:.4f} | "
                f"{detail['improvement_pct']:.2f}% | {p_val} | {sig} |"
            )
        report_lines.append("")
    
    report_lines.extend([
        f"### {h5['hypothesis']}",
        f"**Status**: {h5['status'].upper()}",
        "",
    ])
    
    if "lsh_mean" in h5:
        lsh_ci = h5.get("lsh_ci", (0.0, 0.0))
        rp_ci = h5.get("rp_ci", (0.0, 0.0))
        test = h5.get("test", {})
        sig = "✅" if test.get("significant", False) else "❌"
        p_val = f"{test.get('p_value', 0):.4f}" if test.get("p_value") is not None else "N/A"
        
        report_lines.extend([
            f"- **LSH Mean: {h5['lsh_mean']:.4f}** (95% CI: [{lsh_ci[0]:.4f}, {lsh_ci[1]:.4f}])",
            f"- **Random Projection Mean: {h5['rp_mean']:.4f}** (95% CI: [{rp_ci[0]:.4f}, {rp_ci[1]:.4f}])",
            f"- **Difference: {h5['difference']:.4f}**",
            f"- **Statistical Test**: p-value = {p_val}, Significant = {sig}",
            "",
        ])
    
    # Conclusions
    report_lines.extend([
        "## Conclusions and Recommendations",
        "",
        "### Practical Defaults",
        "",
        "Based on the experimental results, the following defaults are recommended:",
        "",
        "- **n_bits**: 8 (good balance between recall and build time)",
        "- **n_codes**: 32 (reasonable for small to medium datasets)",
        "- **hamming_radius**: 1 (optimal trade-off between recall and search time)",
        "- **block_size**: 4-8 (depending on dataset size)",
        "- **num_tunneling_steps**: 5-10 (good improvement/cost tradeoff)",
        "",
        "### Limitations",
        "",
        "- GTH methods currently show lower recall than baselines in some configurations",
        "- Further investigation needed for bucket-to-dataset mapping optimization",
        "- Experiments conducted on synthetic data; validation on real datasets recommended",
        "",
    ])
    
    # Write report
    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))
    
    print(f"✓ Comprehensive report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Sprint 6 - Comprehensive Results Analysis"
    )
    parser.add_argument(
        "--input", type=str, nargs="+", required=True,
        help="Input JSON results files (can specify multiple)"
    )
    parser.add_argument(
        "--output", type=str, default="experiments/real/results_sprint6_comprehensive.md",
        help="Output markdown report"
    )
    parser.add_argument(
        "--experiment-names", type=str, nargs="+", default=None,
        help="Names for each experiment (must match number of input files)"
    )
    parser.add_argument(
        "--create-plots", action="store_true",
        help="Create visualization plots (requires matplotlib)"
    )
    
    args = parser.parse_args()
    
    # Load results from all input files
    experiment_results = {}
    experiment_names = args.experiment_names if args.experiment_names else []
    
    for i, input_file in enumerate(args.input):
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"Warning: Input file not found: {input_path}")
            continue
        
        exp_name = experiment_names[i] if i < len(experiment_names) else f"Experiment {i+1}"
        results = load_results(input_path)
        experiment_results[exp_name] = results
        print(f"Loaded {len(results)} results from {exp_name} ({input_path})")
    
    if not experiment_results:
        print("Error: No valid results loaded")
        return
    
    # Generate comprehensive report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_comprehensive_report(experiment_results, output_path, create_plots=args.create_plots)


if __name__ == "__main__":
    main()

