#!/usr/bin/env python3
"""Demo CLI application for Gray-Tunneled Hashing."""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gray_tunneled_hashing import GrayTunneledHasher, generate_synthetic_embeddings
from gray_tunneled_hashing.evaluation.metrics import hamming_distance, hamming_preservation_score
import numpy as np


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Demo CLI for Gray-Tunneled Hashing experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=50,
        help="Number of synthetic embeddings to generate (default: 50)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=64,
        help="Dimensionality of embeddings (default: 64)",
    )
    parser.add_argument(
        "--code-length",
        type=int,
        default=32,
        help="Length of binary codes (default: 32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )
    
    args = parser.parse_args()
    
    print("Gray-Tunneled Hashing Demo CLI")
    print("-" * 40)
    print(f"Parameters:")
    print(f"  Points: {args.n_points}")
    print(f"  Dimension: {args.dim}")
    print(f"  Code length: {args.code_length}")
    if args.seed is not None:
        print(f"  Seed: {args.seed}")
    print()
    
    # Generate synthetic embeddings
    print("Generating embeddings...")
    embeddings = generate_synthetic_embeddings(
        n_points=args.n_points,
        dim=args.dim,
        seed=args.seed,
    )
    print(f"  ✓ Generated {embeddings.shape}")
    
    # Initialize and fit hasher
    print("Fitting hasher...")
    hasher = GrayTunneledHasher(code_length=args.code_length)
    hasher.fit(embeddings)
    print("  ✓ Hasher fitted")
    
    # Encode
    print("Encoding to binary codes...")
    codes = hasher.encode(embeddings)
    print(f"  ✓ Encoded to {codes.shape}")
    
    # Print summary metrics
    print("\nSummary:")
    metrics = hasher.evaluate(embeddings, codes)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Compute some sample Hamming distances
    print("\nSample Hamming distances:")
    sample_size = min(5, args.n_points)
    hamm_dist = hamming_distance(codes[:sample_size], codes[:sample_size])
    print(f"  First {sample_size} points (diagonal excluded):")
    for i in range(sample_size):
        for j in range(i + 1, sample_size):
            print(f"    Point {i} <-> Point {j}: {hamm_dist[i, j]}")
    
    print("\n✓ Demo completed successfully!")


if __name__ == "__main__":
    main()

