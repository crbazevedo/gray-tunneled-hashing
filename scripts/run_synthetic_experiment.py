#!/usr/bin/env python3
"""Run a synthetic experiment comparing baseline vs Gray-Tunneled Hashing."""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from gray_tunneled_hashing.data.synthetic_generators import PlantedModelConfig
from gray_tunneled_hashing.algorithms.gray_tunneled_hasher import GrayTunneledHasher
from gray_tunneled_hashing.algorithms.qap_objective import (
    generate_hypercube_edges,
    qap_cost,
)


def compute_permutation_distance(pi1: np.ndarray, pi2: np.ndarray) -> float:
    """
    Compute distance between two permutations.
    
    Returns fraction of vertices where assignments differ.
    """
    if len(pi1) != len(pi2):
        raise ValueError("Permutations must have same length")
    return np.mean(pi1 != pi2)


def run_experiment(
    n_bits: int,
    dim: int,
    sigma: float,
    block_size: int,
    max_two_swap_iters: int,
    num_tunneling_steps: int,
    random_state: int,
):
    """
    Run synthetic experiment comparing baseline vs Gray-Tunneled Hashing.
    
    Args:
        n_bits: Hypercube dimension
        dim: Embedding dimension
        sigma: Noise standard deviation
        block_size: Block size for tunneling
        max_two_swap_iters: Max iterations for 2-swap
        num_tunneling_steps: Number of tunneling steps
        random_state: Random seed
    """
    print("=" * 70)
    print("Gray-Tunneled Hashing - Synthetic Experiment (Sprint 1)")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  - n_bits: {n_bits} (N = {2**n_bits} vertices)")
    print(f"  - dim: {dim}")
    print(f"  - sigma: {sigma}")
    print(f"  - block_size: {block_size}")
    print(f"  - max_two_swap_iters: {max_two_swap_iters}")
    print(f"  - num_tunneling_steps: {num_tunneling_steps}")
    print(f"  - random_state: {random_state}\n")
    
    # Step 1: Generate planted model instance
    print("Step 1: Generating planted model instance...")
    config = PlantedModelConfig(
        n_bits=n_bits,
        dim=dim,
        sigma=sigma,
        random_state=random_state,
    )
    vertices, phi, w = config.generate()
    N = 2 ** n_bits
    print(f"  ✓ Generated {N} embeddings of dimension {dim}")
    
    # For planted π*, treat identity as ground truth (phi index alignment)
    # In the planted model, phi[i] corresponds to vertex i ideally
    pi_star = np.arange(N, dtype=np.int32)
    
    # Step 2: Compute distance matrix
    print("\nStep 2: Computing distance matrix...")
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            D[i, j] = np.linalg.norm(w[i] - w[j]) ** 2
    print(f"  ✓ Distance matrix shape: {D.shape}")
    
    edges = generate_hypercube_edges(n_bits)
    print(f"  ✓ Generated {len(edges)} hypercube edges")
    
    # Step 3: Baseline - random permutation
    print("\nStep 3: Computing baseline (random assignment)...")
    np.random.seed(random_state + 100)
    pi_rand = np.random.permutation(N).astype(np.int32)
    baseline_cost = qap_cost(pi_rand, D, edges)
    baseline_dist_to_star = compute_permutation_distance(pi_rand, pi_star)
    print(f"  ✓ Random baseline cost: {baseline_cost:.6f}")
    print(f"  ✓ Distance to π*: {baseline_dist_to_star:.4f}")
    
    # Step 4: Baseline - identity permutation
    print("\nStep 4: Computing baseline (identity assignment)...")
    pi_identity = np.arange(N, dtype=np.int32)
    identity_cost = qap_cost(pi_identity, D, edges)
    identity_dist_to_star = compute_permutation_distance(pi_identity, pi_star)
    print(f"  ✓ Identity baseline cost: {identity_cost:.6f}")
    print(f"  ✓ Distance to π*: {identity_dist_to_star:.4f}")
    
    # Step 5: Gray-Tunneled Hashing
    print("\nStep 5: Running Gray-Tunneled Hashing optimization...")
    hasher = GrayTunneledHasher(
        n_bits=n_bits,
        block_size=block_size,
        max_two_swap_iters=max_two_swap_iters,
        num_tunneling_steps=num_tunneling_steps,
        two_swap_sample_size=min(256, N * (N - 1) // 2),
        init_strategy="random",
        random_state=random_state + 200,
    )
    
    hasher.fit(w)
    gt_cost = hasher.cost_
    pi_gt = hasher.get_assignment()
    gt_dist_to_star = compute_permutation_distance(pi_gt, pi_star)
    
    print(f"  ✓ Gray-Tunneled cost: {gt_cost:.6f}")
    print(f"  ✓ Distance to π*: {gt_dist_to_star:.4f}")
    print(f"  ✓ Cost history length: {len(hasher.cost_history_)}")
    
    # Step 6: Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nCosts:")
    print(f"  Random baseline:    {baseline_cost:12.6f}")
    print(f"  Identity baseline:  {identity_cost:12.6f}")
    print(f"  Gray-Tunneled:      {gt_cost:12.6f}")
    
    print(f"\nRelative improvements:")
    improvement_vs_random = (baseline_cost - gt_cost) / baseline_cost * 100
    improvement_vs_identity = (identity_cost - gt_cost) / identity_cost * 100
    print(f"  vs Random:   {improvement_vs_random:+7.2f}%")
    print(f"  vs Identity: {improvement_vs_identity:+7.2f}%")
    
    print(f"\nDistance to planted π* (fraction of mismatched vertices):")
    print(f"  Random baseline:    {baseline_dist_to_star:.4f}")
    print(f"  Identity baseline:  {identity_dist_to_star:.4f}")
    print(f"  Gray-Tunneled:      {gt_dist_to_star:.4f}")
    
    print(f"\nCost history (first 5, last 5):")
    history = hasher.cost_history_
    if len(history) <= 5:
        print(f"  {history}")
    else:
        print(f"  First 5:  {history[:5]}")
        print(f"  Last 5:   {history[-5:]}")
    
    print("\n" + "=" * 70)
    print("Experiment completed successfully!")
    print("=" * 70)


def main():
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run synthetic experiment comparing baseline vs Gray-Tunneled Hashing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        default=5,
        help="Number of bits (hypercube dimension, default: 5)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=8,
        help="Embedding dimension (default: 8)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.1,
        help="Noise standard deviation (default: 0.1)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=8,
        help="Block size for tunneling (default: 8)",
    )
    parser.add_argument(
        "--max-two-swap-iters",
        type=int,
        default=50,
        help="Maximum iterations for 2-swap hill climbing (default: 50)",
    )
    parser.add_argument(
        "--num-tunneling-steps",
        type=int,
        default=10,
        help="Number of tunneling steps (default: 10)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    
    args = parser.parse_args()
    
    run_experiment(
        n_bits=args.n_bits,
        dim=args.dim,
        sigma=args.sigma,
        block_size=args.block_size,
        max_two_swap_iters=args.max_two_swap_iters,
        num_tunneling_steps=args.num_tunneling_steps,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
