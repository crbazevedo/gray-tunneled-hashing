#!/usr/bin/env python3
"""Analyze tunneling improvement effect on 2-swap local minima."""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from gray_tunneled_hashing.data.real_datasets import load_embeddings
from gray_tunneled_hashing.binary.codebooks import build_codebook_kmeans
from gray_tunneled_hashing.algorithms.gray_tunneled_hasher import GrayTunneledHasher
from gray_tunneled_hashing.algorithms.qap_objective import qap_cost, generate_hypercube_edges


def analyze_tunneling_effect(
    dataset_name: str,
    n_bits: int,
    n_codes: int,
    block_size: int = 8,
    num_tunneling_steps: int = 10,
    landscape_stats_path: Path = None,
    n_local_minima: int = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Analyze how much tunneling improves 2-swap local minima.
    
    Args:
        dataset_name: Dataset name
        n_bits: Number of bits
        n_codes: Number of codebook vectors
        block_size: Block size for tunneling
        num_tunneling_steps: Number of tunneling steps to apply
        landscape_stats_path: Path to JSON file with 2-swap landscape stats (optional)
        n_local_minima: Number of local minima to test (if landscape_stats_path not provided)
        random_state: Random seed
        
    Returns:
        Dictionary with statistics
    """
    print("=" * 70)
    print("Tunneling Effect Analysis")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Dataset: {dataset_name}")
    print(f"  n_bits: {n_bits}")
    print(f"  n_codes: {n_codes}")
    print(f"  block_size: {block_size}")
    print(f"  num_tunneling_steps: {num_tunneling_steps}\n")
    
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
    
    # Compute distance matrix and edges (needed for cost computation)
    D = np.zeros((max_codes, max_codes), dtype=np.float64)
    for i in range(max_codes):
        for j in range(max_codes):
            D[i, j] = np.linalg.norm(centroids_for_hasher[i] - centroids_for_hasher[j]) ** 2
    
    edges = generate_hypercube_edges(n_bits)
    
    # Get local minima from previous analysis or generate new ones
    local_minima_pis = []
    
    if landscape_stats_path and landscape_stats_path.exists():
        print(f"Loading local minima from: {landscape_stats_path}")
        with open(landscape_stats_path, "r") as f:
            landscape_stats = json.load(f)
        
        # We need to regenerate the permutations, so we'll create new ones
        # with the same random seeds
        n_local_minima = len(landscape_stats.get("raw_final_costs", []))
        print(f"  Found {n_local_minima} local minima to test\n")
        
        # Regenerate by running 2-swap with same seeds
        for i in range(n_local_minima):
            seed = random_state + i
            hasher = GrayTunneledHasher(
                n_bits=n_bits,
                block_size=8,
                max_two_swap_iters=50,
                num_tunneling_steps=0,
                two_swap_sample_size=256,
                init_strategy="random",
                random_state=seed,
                mode="two_swap_only",
                track_history=False,
            )
            hasher.fit(centroids_for_hasher)
            local_minima_pis.append(hasher.get_assignment())
    else:
        # Generate new local minima
        if n_local_minima is None:
            n_local_minima = 10
        
        print(f"Generating {n_local_minima} local minima...")
        for i in range(n_local_minima):
            seed = random_state + i
            hasher = GrayTunneledHasher(
                n_bits=n_bits,
                block_size=8,
                max_two_swap_iters=50,
                num_tunneling_steps=0,
                two_swap_sample_size=256,
                init_strategy="random",
                random_state=seed,
                mode="two_swap_only",
                track_history=False,
            )
            hasher.fit(centroids_for_hasher)
            local_minima_pis.append(hasher.get_assignment())
        print(f"  ✓ Generated {n_local_minima} local minima\n")
    
    # Apply tunneling to each local minimum
    print(f"Applying tunneling steps to each local minimum...")
    tunneling_improvements = []
    improvement_flags = []
    
    for i, pi_local_min in enumerate(local_minima_pis):
        print(f"  Local minimum {i+1}/{len(local_minima_pis)}...", end=" ", flush=True)
        
        # Compute cost at local minimum
        cost_before = qap_cost(pi_local_min, D, edges)
        
        # Apply tunneling
        hasher = GrayTunneledHasher(
            n_bits=n_bits,
            block_size=block_size,
            max_two_swap_iters=0,  # No 2-swap, start from local minimum
            num_tunneling_steps=num_tunneling_steps,
            two_swap_sample_size=256,
            init_strategy="identity",  # Will override with pi_local_min
            random_state=random_state + 1000 + i,
            mode="full",
            track_history=False,
            block_selection_strategy="random",
        )
        
        # Manually set pi_ to start from local minimum
        # We need to override the fit method's initialization
        # For simplicity, we'll use a workaround: set pi_init directly
        # Actually, we need to modify the hasher to accept an initial pi
        # For now, we'll use a simpler approach: create hasher and manually set pi
        
        # Set up hasher internals manually
        hasher.D_ = D
        hasher.edges_ = edges
        hasher.pi_ = pi_local_min.copy()
        hasher.cost_ = cost_before
        hasher.is_fitted = True
        
        # Run tunneling steps manually
        from gray_tunneled_hashing.algorithms.block_moves import tunneling_step
        from gray_tunneled_hashing.algorithms.block_selection import get_block_selection_fn
        
        pi = pi_local_min.copy()
        block_selection_fn = get_block_selection_fn("random")
        
        for step in range(num_tunneling_steps):
            pi_new, delta = tunneling_step(
                pi=pi,
                D=D,
                edges=edges,
                block_size=block_size,
                num_blocks=10,
                random_state=None,
                block_selection_fn=block_selection_fn,
            )
            if delta < -1e-10:
                pi = pi_new
        
        cost_after = qap_cost(pi, D, edges)
        improvement = cost_before - cost_after
        tunneling_improvements.append(improvement)
        improvement_flags.append(improvement > 1e-10)
        
        print(f"Before: {cost_before:.6f}, After: {cost_after:.6f}, Improvement: {improvement:.6f}")
    
    print("\n" + "=" * 70)
    print("Statistics")
    print("=" * 70)
    
    # Compute statistics
    stats = {
        "dataset": dataset_name,
        "n_bits": n_bits,
        "n_codes": n_codes,
        "block_size": block_size,
        "num_tunneling_steps": num_tunneling_steps,
        "n_local_minima": len(local_minima_pis),
    }
    
    if tunneling_improvements:
        stats["tunneling_improvement"] = {
            "mean": float(np.mean(tunneling_improvements)),
            "std": float(np.std(tunneling_improvements)),
            "min": float(np.min(tunneling_improvements)),
            "max": float(np.max(tunneling_improvements)),
            "median": float(np.median(tunneling_improvements)),
        }
        stats["improvement_rate"] = float(np.mean(improvement_flags))
        
        print(f"\nTunneling Improvement:")
        print(f"  Mean: {stats['tunneling_improvement']['mean']:.6f}")
        print(f"  Std:  {stats['tunneling_improvement']['std']:.6f}")
        print(f"  Min:  {stats['tunneling_improvement']['min']:.6f}")
        print(f"  Max:  {stats['tunneling_improvement']['max']:.6f}")
        print(f"  Median: {stats['tunneling_improvement']['median']:.6f}")
        print(f"\nImprovement Rate: {stats['improvement_rate']*100:.1f}% of local minima improved")
    
    stats["raw_improvements"] = [float(i) for i in tunneling_improvements]
    
    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze tunneling improvement effect"
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
        "--block-size",
        type=int,
        default=8,
        help="Block size for tunneling",
    )
    parser.add_argument(
        "--num-tunneling-steps",
        type=int,
        default=10,
        help="Number of tunneling steps",
    )
    parser.add_argument(
        "--landscape-stats",
        type=Path,
        default=Path(__file__).parent.parent / "experiments" / "real" / "landscape_two_swap_stats_sprint3.json",
        help="Path to 2-swap landscape stats JSON (optional)",
    )
    parser.add_argument(
        "--n-local-minima",
        type=int,
        default=None,
        help="Number of local minima to test (if landscape_stats not provided)",
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
        default=Path(__file__).parent.parent / "experiments" / "real" / "landscape_tunneling_stats_sprint3.json",
        help="Output JSON file",
    )
    
    args = parser.parse_args()
    
    stats = analyze_tunneling_effect(
        dataset_name=args.dataset,
        n_bits=args.n_bits,
        n_codes=args.n_codes,
        block_size=args.block_size,
        num_tunneling_steps=args.num_tunneling_steps,
        landscape_stats_path=args.landscape_stats if args.landscape_stats.exists() else None,
        n_local_minima=args.n_local_minima,
        random_state=args.random_state,
    )
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ Results saved to: {args.output}")


if __name__ == "__main__":
    main()

