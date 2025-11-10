"""
Sprint 6 - Experiment 3: Grid Search over Parameters

This script performs a systematic grid search over:
- n_bits: {6, 8, 10}
- n_codes: {16, 32, 64}
- hamming_radius: {0, 1, 2}
- block_size: {4, 8}
- num_tunneling_steps: {0, 5, 10}

Uses the improved benchmark script with parallelization and checkpointing.
"""

import argparse
import subprocess
import json
import itertools
from pathlib import Path
from typing import List, Dict, Any
import time

# Grid search parameters
N_BITS_VALUES = [6, 8, 10]
N_CODES_VALUES = [16, 32, 64]
HAMMING_RADIUS_VALUES = [0, 1, 2]
BLOCK_SIZE_VALUES = [4, 8]
NUM_TUNNELING_STEPS_VALUES = [0, 5, 10]

# Fixed parameters
K = 5
N_SAMPLES = 100
N_QUERIES = 20
DIM = 16
N_RUNS = 3
RANDOM_STATE = 42
N_WORKERS = 4  # Parallel workers for each benchmark run


def generate_configs() -> List[Dict[str, Any]]:
    """Generate all configurations for grid search."""
    configs = []
    
    for n_bits in N_BITS_VALUES:
        for n_codes in N_CODES_VALUES:
            for hamming_radius in HAMMING_RADIUS_VALUES:
                for block_size in BLOCK_SIZE_VALUES:
                    for num_tunneling_steps in NUM_TUNNELING_STEPS_VALUES:
                        configs.append({
                            "n_bits": n_bits,
                            "n_codes": n_codes,
                            "hamming_radius": hamming_radius,
                            "block_size": block_size,
                            "num_tunneling_steps": num_tunneling_steps,
                        })
    
    return configs


def run_single_config(config: Dict[str, Any], output_dir: Path, resume: bool = False) -> bool:
    """
    Run benchmark for a single configuration.
    
    Returns:
        True if successful, False otherwise
    """
    # Create output filename
    config_str = (
        f"nbits{config['n_bits']}_ncodes{config['n_codes']}_"
        f"radius{config['hamming_radius']}_block{config['block_size']}_"
        f"tunnel{config['num_tunneling_steps']}"
    )
    output_file = output_dir / f"experiment3_{config_str}.json"
    checkpoint_file = output_dir / f"experiment3_{config_str}.checkpoint.json"
    
    # Check if already completed
    if output_file.exists() and resume:
        try:
            with open(output_file, "r") as f:
                data = json.load(f)
                if data and len(data) > 0:
                    print(f"  ✓ Skipping {config_str} (already completed)")
                    return True
        except:
            pass
    
    # Determine mode based on num_tunneling_steps
    if config["num_tunneling_steps"] == 0:
        mode = "two_swap_only"
    else:
        mode = "full"
    
    # Build command
    cmd = [
        "python", "scripts/benchmark_lsh_vs_random_proj.py",
        "--n-bits", str(config["n_bits"]),
        "--n-codes", str(config["n_codes"]),
        "--k", str(K),
        "--hamming-radius", str(config["hamming_radius"]),
        "--n-samples", str(N_SAMPLES),
        "--n-queries", str(N_QUERIES),
        "--dim", str(DIM),
        "--random-state", str(RANDOM_STATE),
        "--n-runs", str(N_RUNS),
        "--n-workers", str(N_WORKERS),
        "--block-size", str(config["block_size"]),
        "--num-tunneling-steps", str(config["num_tunneling_steps"]),
        "--mode", mode,
        "--output", str(output_file),
        "--checkpoint", str(checkpoint_file),
    ]
    
    if resume:
        cmd.append("--resume")
    
    print(f"\n{'='*70}")
    print(f"Running config: {config_str}")
    print(f"{'='*70}")
    print(f"  n_bits: {config['n_bits']}")
    print(f"  n_codes: {config['n_codes']}")
    print(f"  hamming_radius: {config['hamming_radius']}")
    print(f"  block_size: {config['block_size']}")
    print(f"  num_tunneling_steps: {config['num_tunneling_steps']}")
    print(f"  mode: {mode}")
    print(f"  Output: {output_file}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout per config
        )
        
        if result.returncode == 0:
            print(f"  ✓ Success: {config_str}")
            return True
        else:
            print(f"  ✗ Failed: {config_str}")
            print(f"  Error: {result.stderr[:500]}")
            return False
    
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout: {config_str}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {config_str} - {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Sprint 6 - Experiment 3: Grid Search over Parameters"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/real",
        help="Output directory for results (default: experiments/real)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results"
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Maximum number of configurations to run (for testing)"
    )
    parser.add_argument(
        "--config-index",
        type=int,
        default=None,
        help="Run only a specific config index (for parallel execution)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all configurations
    configs = generate_configs()
    total_configs = len(configs)
    
    print(f"\n{'='*70}")
    print(f"Sprint 6 - Experiment 3: Grid Search")
    print(f"{'='*70}")
    print(f"Total configurations: {total_configs}")
    print(f"Grid size:")
    print(f"  n_bits: {len(N_BITS_VALUES)} values")
    print(f"  n_codes: {len(N_CODES_VALUES)} values")
    print(f"  hamming_radius: {len(HAMMING_RADIUS_VALUES)} values")
    print(f"  block_size: {len(BLOCK_SIZE_VALUES)} values")
    print(f"  num_tunneling_steps: {len(NUM_TUNNELING_STEPS_VALUES)} values")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")
    
    # Limit configs if specified
    if args.max_configs:
        configs = configs[:args.max_configs]
        print(f"Limited to {len(configs)} configurations (--max-configs={args.max_configs})\n")
    
    # Run specific config if specified
    if args.config_index is not None:
        if 0 <= args.config_index < len(configs):
            configs = [configs[args.config_index]]
            print(f"Running only config index {args.config_index}\n")
        else:
            print(f"Error: config_index {args.config_index} out of range [0, {len(configs)-1}]")
            return
    
    # Run configurations
    start_time = time.time()
    successful = 0
    failed = 0
    
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Processing configuration...")
        if run_single_config(config, output_dir, resume=args.resume):
            successful += 1
        else:
            failed += 1
    
    total_time = time.time() - start_time
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Grid Search Summary")
    print(f"{'='*70}")
    print(f"Total configurations: {len(configs)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Average time per config: {total_time/len(configs):.1f}s")
    print(f"{'='*70}\n")
    
    # Create summary file
    summary_file = output_dir / "experiment3_grid_search_summary.json"
    summary = {
        "total_configs": len(configs),
        "successful": successful,
        "failed": failed,
        "total_time_seconds": total_time,
        "configs": configs,
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary saved to {summary_file}")


if __name__ == "__main__":
    main()

