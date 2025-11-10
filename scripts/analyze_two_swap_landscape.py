#!/usr/bin/env python3
"""Analyze 2-swap local minima distribution on real embeddings."""

import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from gray_tunneled_hashing.data.real_datasets import load_embeddings
from gray_tunneled_hashing.binary.codebooks import build_codebook_kmeans
from gray_tunneled_hashing.algorithms.gray_tunneled_hasher import GrayTunneledHasher


def analyze_two_swap_landscape(
    dataset_name: str,
    n_bits: int,
    n_codes: int,
    n_samples: int = 20,
    max_two_swap_iters: int = 50,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Analyze distribution of 2-swap local minima.
    
    Args:
        dataset_name: Dataset name
        n_bits: Number of bits
        n_codes: Number of codebook vectors
        n_samples: Number of random initializations to try
        max_two_swap_iters: Max iterations for 2-swap
        random_state: Random seed
        
    Returns:
        Dictionary with statistics
    """
    print("=" * 70)
    print("2-Swap Landscape Analysis")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Dataset: {dataset_name}")
    print(f"  n_bits: {n_bits}")
    print(f"  n_codes: {n_codes}")
    print(f"  n_samples: {n_samples}")
    print(f"  max_two_swap_iters: {max_two_swap_iters}\n")
    
    # Load embeddings
    print("Loading embeddings...")
    base_embeddings = load_embeddings(dataset_name, split="base")
    print(f"  ✓ Base embeddings: {base_embeddings.shape}\n")
    
    # Build codebook
    print("Building codebook...")
    centroids, assignments = build_codebook_kmeans(
        base_embeddings,
        n_codes=n_codes,
        random_state=random_state,
    )
    print(f"  ✓ Centroids: {centroids.shape}\n")
    
    # Prepare centroids for hasher
    max_codes = 2 ** n_bits
    if n_codes < max_codes:
        n_pad = max_codes - n_codes
        padding = np.tile(centroids[-1:], (n_pad, 1))
        centroids_for_hasher = np.vstack([centroids, padding])
    else:
        centroids_for_hasher = centroids[:max_codes]
    
    # Sample multiple initializations
    print(f"Running {n_samples} 2-swap optimizations...")
    initial_costs = []
    final_costs = []
    improvements = []
    
    for i in range(n_samples):
        seed = random_state + i
        np.random.seed(seed)
        
        print(f"  Sample {i+1}/{n_samples}...", end=" ", flush=True)
        
        # Create hasher with two_swap_only mode
        hasher = GrayTunneledHasher(
            n_bits=n_bits,
            block_size=8,  # Not used in two_swap_only
            max_two_swap_iters=max_two_swap_iters,
            num_tunneling_steps=0,
            two_swap_sample_size=256,
            init_strategy="random",
            random_state=seed,
            mode="two_swap_only",
            track_history=True,
        )
        
        hasher.fit(centroids_for_hasher)
        
        # Extract initial and final costs from history
        if hasher.track_history and len(hasher.cost_history_) > 0:
            if isinstance(hasher.cost_history_[0], dict):
                initial_cost = hasher.cost_history_[0]["cost"]
                final_cost = hasher.cost_history_[-1]["cost"]
            else:
                initial_cost = hasher.cost_history_[0]
                final_cost = hasher.cost_history_[-1]
        else:
            # Fallback: compute from current state
            initial_cost = None  # Would need to track separately
            final_cost = hasher.cost_
        
        if initial_cost is not None:
            initial_costs.append(initial_cost)
            final_costs.append(final_cost)
            improvements.append(initial_cost - final_cost)
            print(f"Initial: {initial_cost:.6f}, Final: {final_cost:.6f}, Improvement: {initial_cost - final_cost:.6f}")
        else:
            final_costs.append(final_cost)
            print(f"Final: {final_cost:.6f}")
    
    print("\n" + "=" * 70)
    print("Statistics")
    print("=" * 70)
    
    # Compute statistics
    stats = {
        "dataset": dataset_name,
        "n_bits": n_bits,
        "n_codes": n_codes,
        "n_samples": n_samples,
        "max_two_swap_iters": max_two_swap_iters,
    }
    
    if initial_costs:
        stats["initial_cost"] = {
            "mean": float(np.mean(initial_costs)),
            "std": float(np.std(initial_costs)),
            "min": float(np.min(initial_costs)),
            "max": float(np.max(initial_costs)),
            "median": float(np.median(initial_costs)),
        }
        print(f"\nInitial Cost:")
        print(f"  Mean: {stats['initial_cost']['mean']:.6f}")
        print(f"  Std:  {stats['initial_cost']['std']:.6f}")
        print(f"  Min:  {stats['initial_cost']['min']:.6f}")
        print(f"  Max:  {stats['initial_cost']['max']:.6f}")
        print(f"  Median: {stats['initial_cost']['median']:.6f}")
    
    if final_costs:
        stats["final_cost"] = {
            "mean": float(np.mean(final_costs)),
            "std": float(np.std(final_costs)),
            "min": float(np.min(final_costs)),
            "max": float(np.max(final_costs)),
            "median": float(np.median(final_costs)),
        }
        print(f"\nFinal Cost (2-swap local minima):")
        print(f"  Mean: {stats['final_cost']['mean']:.6f}")
        print(f"  Std:  {stats['final_cost']['std']:.6f}")
        print(f"  Min:  {stats['final_cost']['min']:.6f}")
        print(f"  Max:  {stats['final_cost']['max']:.6f}")
        print(f"  Median: {stats['final_cost']['median']:.6f}")
    
    if improvements:
        stats["improvement"] = {
            "mean": float(np.mean(improvements)),
            "std": float(np.std(improvements)),
            "min": float(np.min(improvements)),
            "max": float(np.max(improvements)),
            "median": float(np.median(improvements)),
        }
        print(f"\nImprovement (Initial - Final):")
        print(f"  Mean: {stats['improvement']['mean']:.6f}")
        print(f"  Std:  {stats['improvement']['std']:.6f}")
        print(f"  Min:  {stats['improvement']['min']:.6f}")
        print(f"  Max:  {stats['improvement']['max']:.6f}")
        print(f"  Median: {stats['improvement']['median']:.6f}")
    
    # Store raw data for further analysis
    stats["raw_initial_costs"] = [float(c) for c in initial_costs] if initial_costs else []
    stats["raw_final_costs"] = [float(c) for c in final_costs]
    stats["raw_improvements"] = [float(i) for i in improvements] if improvements else []
    
    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze 2-swap local minima distribution"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="quora",
        help="Dataset name",
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        default=64,
        help="Number of bits",
    )
    parser.add_argument(
        "--n-codes",
        type=int,
        default=512,
        help="Number of codebook vectors",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="Number of random initializations to try",
    )
    parser.add_argument(
        "--max-two-swap-iters",
        type=int,
        default=50,
        help="Max iterations for 2-swap",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "experiments" / "real" / "landscape_two_swap_stats_sprint3.json",
        help="Output JSON file",
    )
    
    args = parser.parse_args()
    
    stats = analyze_two_swap_landscape(
        dataset_name=args.dataset,
        n_bits=args.n_bits,
        n_codes=args.n_codes,
        n_samples=args.n_samples,
        max_two_swap_iters=args.max_two_swap_iters,
        random_state=args.random_state,
    )
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ Results saved to: {args.output}")


if __name__ == "__main__":
    main()

