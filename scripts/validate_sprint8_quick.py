#!/usr/bin/env python3
"""Quick validation script for Sprint 8 changes.

Executes all quick tests and generates a summary report.
Returns exit code 0 if all tests pass, 1 otherwise.
"""

import sys
import subprocess
import time
from pathlib import Path


def run_tests():
    """Run all quick tests for Sprint 8."""
    test_files = [
        "tests/test_sprint8_data_structure.py",
        "tests/test_sprint8_jphi_real.py",
        "tests/test_sprint8_query_pipeline.py",
        "tests/test_sprint8_e2e_basic.py",
    ]
    
    print("=" * 80)
    print("Sprint 8 Quick Validation")
    print("=" * 80)
    print()
    
    results = {}
    total_start = time.time()
    
    for test_file in test_files:
        if not Path(test_file).exists():
            print(f"⚠️  Test file not found: {test_file}")
            results[test_file] = {"status": "skipped", "reason": "file not found"}
            continue
        
        print(f"Running {test_file}...")
        start = time.time()
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_file, "-v"],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout per test file
            )
            
            elapsed = time.time() - start
            
            if result.returncode == 0:
                print(f"  ✅ PASSED ({elapsed:.2f}s)")
                results[test_file] = {"status": "passed", "time": elapsed}
            else:
                print(f"  ❌ FAILED ({elapsed:.2f}s)")
                print(f"  Error output:")
                print(result.stdout[-500:])  # Last 500 chars
                print(result.stderr[-500:])
                results[test_file] = {"status": "failed", "time": elapsed, "output": result.stdout + result.stderr}
        
        except subprocess.TimeoutExpired:
            print(f"  ⏱️  TIMEOUT (>5 minutes)")
            results[test_file] = {"status": "timeout", "time": 300}
        
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results[test_file] = {"status": "error", "error": str(e)}
        
        print()
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    
    passed = sum(1 for r in results.values() if r["status"] == "passed")
    failed = sum(1 for r in results.values() if r["status"] in ["failed", "error", "timeout"])
    skipped = sum(1 for r in results.values() if r["status"] == "skipped")
    
    print(f"Total tests: {len(test_files)}")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  ⏭️  Skipped: {skipped}")
    print(f"Total time: {total_elapsed:.2f}s")
    print()
    
    # Detailed results
    for test_file, result in results.items():
        status_icon = {
            "passed": "✅",
            "failed": "❌",
            "error": "❌",
            "timeout": "⏱️",
            "skipped": "⏭️",
        }.get(result["status"], "❓")
        
        time_str = f" ({result.get('time', 0):.2f}s)" if "time" in result else ""
        print(f"  {status_icon} {test_file}{time_str}")
    
    print()
    
    # Return exit code
    if failed > 0:
        print("❌ Validation FAILED")
        return 1
    elif skipped > 0 and passed == 0:
        print("⚠️  All tests were skipped")
        return 1
    else:
        print("✅ Validation PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(run_tests())

