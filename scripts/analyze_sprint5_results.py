"""
Analyze results from Sprint 5 experiments.

This script:
1. Loads JSON results from benchmark experiments
2. Computes statistics (mean, std, min, max, percentiles)
3. Generates comparative tables
4. Validates hypotheses (H2, H3, H4, H5)
5. Generates markdown report
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from scipy import stats


def load_results(json_path: Path) -> List[Dict[str, Any]]:
    """Load results from JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute statistics for a list of values."""
    if len(values) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
        }
    
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
    }


def validate_hypothesis_h2(results: List[Dict]) -> Dict[str, Any]:
    """
    Validate H2: GTH Preserves Collisions LSH.
    
    Expected: 100% preservation rate
    """
    # Look for collision preservation results
    # For now, return placeholder
    return {
        "hypothesis": "H2: GTH Preserves Collisions",
        "status": "pending",
        "note": "Requires collision preservation validation results",
    }


def validate_hypothesis_h3(results: List[Dict]) -> Dict[str, Any]:
    """
    Validate H3: Hamming Ball Improves Recall.
    
    Expected: recall(radius=2) > recall(radius=1) > recall(radius=0)
    """
    # Group by method and radius
    by_method_radius = {}
    for result in results:
        method = result.get("method", "unknown")
        radius = result.get("hamming_radius", 0)
        key = (method, radius)
        if key not in by_method_radius:
            by_method_radius[key] = []
        by_method_radius[key].append(result.get("recall_mean", 0.0))
    
    # Check monotonicity for each method
    validated_methods = []
    for method in set(m[0] for m in by_method_radius.keys()):
        recalls_r0 = by_method_radius.get((method, 0), [0.0])
        recalls_r1 = by_method_radius.get((method, 1), [0.0])
        recalls_r2 = by_method_radius.get((method, 2), [0.0])
        
        mean_r0 = np.mean(recalls_r0)
        mean_r1 = np.mean(recalls_r1)
        mean_r2 = np.mean(recalls_r2)
        
        is_monotonic = mean_r2 > mean_r1 > mean_r0
        
        validated_methods.append({
            "method": method,
            "recall_r0": mean_r0,
            "recall_r1": mean_r1,
            "recall_r2": mean_r2,
            "is_monotonic": is_monotonic,
        })
    
    all_valid = all(m["is_monotonic"] for m in validated_methods)
    
    return {
        "hypothesis": "H3: Hamming Ball Improves Recall",
        "status": "validated" if all_valid else "rejected",
        "details": validated_methods,
    }


def validate_hypothesis_h4(results: List[Dict]) -> Dict[str, Any]:
    """
    Validate H4: GTH Improves Recall.
    
    Expected: recall_gth > recall_baseline (statistically significant)
    """
    # Group by method type (baseline vs. with GTH)
    baseline_recalls = []
    gth_recalls = []
    
    for result in results:
        method = result.get("method", "")
        recall = result.get("recall_mean", 0.0)
        
        if method.startswith("baseline_"):
            baseline_recalls.append(recall)
        else:
            gth_recalls.append(recall)
    
    if len(baseline_recalls) == 0 or len(gth_recalls) == 0:
        return {
            "hypothesis": "H4: GTH Improves Recall",
            "status": "insufficient_data",
            "note": "Need both baseline and GTH results",
        }
    
    # Statistical test
    baseline_mean = np.mean(baseline_recalls)
    gth_mean = np.mean(gth_recalls)
    
    # Simple t-test if we have enough data
    if len(baseline_recalls) > 1 and len(gth_recalls) > 1:
        t_stat, p_value = stats.ttest_ind(gth_recalls, baseline_recalls)
        is_significant = p_value < 0.05 and gth_mean > baseline_mean
    else:
        is_significant = gth_mean > baseline_mean
        p_value = None
    
    return {
        "hypothesis": "H4: GTH Improves Recall",
        "status": "validated" if is_significant else "rejected",
        "baseline_mean": float(baseline_mean),
        "gth_mean": float(gth_mean),
        "improvement": float((gth_mean - baseline_mean) / baseline_mean * 100) if baseline_mean > 0 else 0.0,
        "p_value": float(p_value) if p_value is not None else None,
    }


def validate_hypothesis_h5(results: List[Dict]) -> Dict[str, Any]:
    """
    Validate H5: LSH + GTH vs. Random Projection + GTH.
    
    Compare recall between LSH and random projection methods with GTH.
    """
    lsh_recalls = []
    rp_recalls = []
    
    for result in results:
        method = result.get("method", "")
        recall = result.get("recall_mean", 0.0)
        
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
    
    lsh_mean = np.mean(lsh_recalls)
    rp_mean = np.mean(rp_recalls)
    
    return {
        "hypothesis": "H5: LSH + GTH vs. Random Projection + GTH",
        "status": "compared",
        "lsh_mean": float(lsh_mean),
        "rp_mean": float(rp_mean),
        "difference": float(lsh_mean - rp_mean),
    }


def generate_report(results: List[Dict], output_path: Path) -> None:
    """Generate markdown report from results."""
    # Validate hypotheses
    h2 = validate_hypothesis_h2(results)
    h3 = validate_hypothesis_h3(results)
    h4 = validate_hypothesis_h4(results)
    h5 = validate_hypothesis_h5(results)
    
    # Generate report
    report_lines = [
        "# Sprint 5 Results Analysis",
        "",
        "## Summary Statistics",
        "",
        "### By Method",
        "",
        "| Method | Recall (mean ± std) | Build Time (s) | Search Time (ms) |",
        "|--------|---------------------|----------------|------------------|",
    ]
    
    for result in results:
        method = result.get("method", "unknown")
        recall_mean = result.get("recall_mean", 0.0)
        recall_std = result.get("recall_std", 0.0)
        build_time = result.get("build_time_mean", 0.0)
        search_time = result.get("search_time_mean", 0.0) * 1000  # Convert to ms
        
        report_lines.append(
            f"| {method} | {recall_mean:.4f} ± {recall_std:.4f} | {build_time:.2f} | {search_time:.2f} |"
        )
    
    report_lines.extend([
        "",
        "## Hypothesis Validation",
        "",
        f"### {h2['hypothesis']}",
        f"**Status**: {h2['status']}",
        "",
        f"### {h3['hypothesis']}",
        f"**Status**: {h3['status']}",
        "",
        f"### {h4['hypothesis']}",
        f"**Status**: {h4['status']}",
        f"**Improvement**: {h4.get('improvement', 0.0):.2f}%",
        "",
        f"### {h5['hypothesis']}",
        f"**Status**: {h5['status']}",
        f"**LSH Mean**: {h5.get('lsh_mean', 0.0):.4f}",
        f"**RP Mean**: {h5.get('rp_mean', 0.0):.4f}",
        "",
    ])
    
    # Write report
    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))
    
    print(f"✓ Report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Sprint 5 experiment results"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input JSON results file"
    )
    parser.add_argument(
        "--output", type=str, default="results_sprint5.md", help="Output markdown report"
    )
    
    args = parser.parse_args()
    
    # Load results
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    results = load_results(input_path)
    print(f"Loaded {len(results)} result entries")
    
    # Generate report
    output_path = Path(args.output)
    generate_report(results, output_path)


if __name__ == "__main__":
    main()

