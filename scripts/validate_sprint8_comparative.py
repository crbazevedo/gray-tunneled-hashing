#!/usr/bin/env python3
"""
Script de validação comparativa para Sprint 8.

Compara resultados antes e depois das mudanças da Sprint 8.
"""

import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime


def run_comparative_tests() -> dict:
    """Run comparative tests and collect results."""
    print("=" * 80)
    print("Sprint 8 Comparative Validation")
    print("=" * 80)
    print()
    
    # Focus on recall comparative tests
    test_files = [
        "tests/test_sprint8_recall_comparative.py",
        "tests/test_sprint8_performance.py",
    ]
    
    results = {}
    total_start = time.time()
    
    for test_file in test_files:
        test_path = Path(test_file)
        if not test_path.exists():
            print(f"⚠️  Skipping {test_file} (not found)")
            continue
        
        print(f"Running {test_file}...")
        start_time = time.time()
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes timeout
            )
            elapsed = time.time() - start_time
            success = result.returncode == 0
            
            # Parse output for key metrics
            output = result.stdout + result.stderr
            metrics = parse_test_output(output)
            
            results[test_file] = {
                "success": success,
                "elapsed": elapsed,
                "metrics": metrics,
                "output": output[-500:] if len(output) > 500 else output,  # Last 500 chars
            }
            
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"  {status} ({elapsed:.2f}s)")
            
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            results[test_file] = {
                "success": False,
                "elapsed": elapsed,
                "metrics": {},
                "output": "Timeout after 600 seconds",
            }
            print(f"  ❌ TIMEOUT ({elapsed:.2f}s)")
        except Exception as e:
            elapsed = time.time() - start_time
            results[test_file] = {
                "success": False,
                "elapsed": elapsed,
                "metrics": {},
                "output": str(e),
            }
            print(f"  ❌ ERROR ({elapsed:.2f}s): {e}")
        
        print()
    
    total_time = time.time() - total_start
    
    # Generate summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_time": total_time,
        "results": results,
    }
    
    return summary


def parse_test_output(output: str) -> dict:
    """Parse test output for key metrics."""
    metrics = {}
    
    # Look for recall values
    import re
    recall_pattern = r"recall[:\s]+([\d.]+)"
    recalls = re.findall(recall_pattern, output, re.IGNORECASE)
    if recalls:
        metrics["recalls"] = [float(r) for r in recalls]
        metrics["mean_recall"] = sum(metrics["recalls"]) / len(metrics["recalls"])
    
    # Look for time values
    time_pattern = r"time[:\s]+([\d.]+)s"
    times = re.findall(time_pattern, output, re.IGNORECASE)
    if times:
        metrics["times"] = [float(t) for t in times]
    
    # Look for baseline vs GTH comparisons
    baseline_pattern = r"baseline[:\s]+([\d.]+)"
    gth_pattern = r"GTH[:\s]+([\d.]+)"
    baselines = re.findall(baseline_pattern, output, re.IGNORECASE)
    gths = re.findall(gth_pattern, output, re.IGNORECASE)
    if baselines and gths:
        metrics["baseline_recalls"] = [float(b) for b in baselines]
        metrics["gth_recalls"] = [float(g) for g in gths]
        if len(metrics["baseline_recalls"]) == len(metrics["gth_recalls"]):
            improvements = [
                g - b for g, b in zip(metrics["gth_recalls"], metrics["baseline_recalls"])
            ]
            metrics["improvements"] = improvements
            metrics["mean_improvement"] = sum(improvements) / len(improvements) if improvements else 0.0
    
    return metrics


def print_summary(summary: dict):
    """Print summary of comparative results."""
    print("=" * 80)
    print("Comparative Summary")
    print("=" * 80)
    print()
    
    total_passed = sum(1 for r in summary["results"].values() if r["success"])
    total_failed = len(summary["results"]) - total_passed
    
    print(f"Total test files: {len(summary['results'])}")
    print(f"  ✅ Passed: {total_passed}")
    print(f"  ❌ Failed: {total_failed}")
    print(f"Total time: {summary['total_time']:.2f}s")
    print()
    
    # Print metrics for each test file
    for test_file, result in summary["results"].items():
        print(f"{test_file}:")
        print(f"  Status: {'✅ PASSED' if result['success'] else '❌ FAILED'}")
        print(f"  Time: {result['elapsed']:.2f}s")
        
        if result["metrics"]:
            print("  Metrics:")
            for key, value in result["metrics"].items():
                if isinstance(value, list):
                    print(f"    {key}: {value}")
                elif isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")
        print()
    
    # Overall assessment
    print("=" * 80)
    if total_failed == 0:
        print("✅ All comparative tests PASSED")
        print()
        print("Key Findings:")
        
        # Aggregate recall improvements
        all_improvements = []
        for result in summary["results"].values():
            if "metrics" in result and "improvements" in result["metrics"]:
                all_improvements.extend(result["metrics"]["improvements"])
        
        if all_improvements:
            mean_improvement = sum(all_improvements) / len(all_improvements)
            if mean_improvement > 0:
                print(f"  ✅ GTH shows improvement over baseline: {mean_improvement:+.4f} average")
            elif mean_improvement >= -0.05:
                print(f"  ⚠️  GTH is similar to baseline: {mean_improvement:+.4f} average")
            else:
                print(f"  ❌ GTH is worse than baseline: {mean_improvement:+.4f} average")
    else:
        print("❌ Some comparative tests FAILED")
        print("Review the output above for details.")
    
    print("=" * 80)


def save_results(summary: dict, output_file: str = "sprint8_comparative_results.json"):
    """Save results to JSON file."""
    output_path = Path(output_file)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main():
    """Main entry point."""
    summary = run_comparative_tests()
    print_summary(summary)
    save_results(summary)
    
    # Exit with appropriate code
    total_failed = sum(1 for r in summary["results"].values() if not r["success"])
    sys.exit(1 if total_failed > 0 else 0)


if __name__ == "__main__":
    main()

