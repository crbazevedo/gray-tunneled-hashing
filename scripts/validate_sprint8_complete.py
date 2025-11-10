#!/usr/bin/env python3
"""
Script de validação completa para Sprint 8.

Executa todos os testes completos (não apenas os rápidos).
"""

import subprocess
import sys
import time
from pathlib import Path


def run_test_file(test_file: str) -> tuple[bool, float, str]:
    """Run a test file and return (success, time, output)."""
    start_time = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout per file
        )
        elapsed = time.time() - start_time
        success = result.returncode == 0
        return success, elapsed, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return False, elapsed, "Timeout after 300 seconds"
    except Exception as e:
        elapsed = time.time() - start_time
        return False, elapsed, str(e)


def main():
    """Run all complete validation tests."""
    print("=" * 80)
    print("Sprint 8 Complete Validation")
    print("=" * 80)
    print()
    
    # All test files (including complete versions)
    test_files = [
        "tests/test_sprint8_data_structure.py",
        "tests/test_sprint8_data_structure_complete.py",
        "tests/test_sprint8_jphi_real.py",
        "tests/test_sprint8_jphi_real_complete.py",
        "tests/test_sprint8_query_pipeline.py",
        "tests/test_sprint8_query_pipeline_complete.py",
        "tests/test_sprint8_e2e_basic.py",
        "tests/test_sprint8_integration_complete.py",
        "tests/test_sprint8_recall_comparative.py",
        "tests/test_sprint8_performance.py",
        "tests/test_sprint8_regression.py",
    ]
    
    results = []
    total_start = time.time()
    
    for test_file in test_files:
        test_path = Path(test_file)
        if not test_path.exists():
            print(f"⚠️  Skipping {test_file} (not found)")
            continue
        
        print(f"Running {test_file}...")
        success, elapsed, output = run_test_file(test_file)
        
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {status} ({elapsed:.2f}s)")
        
        if not success:
            # Print last 20 lines of output for debugging
            lines = output.split("\n")
            print("  Error output:")
            for line in lines[-20:]:
                if line.strip():
                    print(f"    {line}")
        
        results.append((test_file, success, elapsed))
        print()
    
    total_time = time.time() - total_start
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    
    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed
    
    print(f"Total tests: {len(results)}")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  ⏭️  Skipped: 0")
    print(f"Total time: {total_time:.2f}s")
    print()
    
    # Detailed results
    for test_file, success, elapsed in results:
        status = "✅" if success else "❌"
        print(f"  {status} {test_file} ({elapsed:.2f}s)")
    
    print()
    
    if failed > 0:
        print("❌ Validation FAILED")
        sys.exit(1)
    else:
        print("✅ Validation PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()

