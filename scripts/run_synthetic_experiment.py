#!/usr/bin/env python3
"""Run a synthetic experiment with Gray-Tunneled Hashing."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from gray_tunneled_hashing import GrayTunneledHasher, generate_synthetic_embeddings
from gray_tunneled_hashing.evaluation.metrics import hamming_preservation_score, hamming_distance


def run_experiment(n_points: int = 100, dim: int = 128, code_length: int = 64):
    """
    Run a synthetic experiment with the Gray-Tunneled Hasher.
    
    Args:
        n_points: Number of synthetic embeddings to generate
        dim: Dimensionality of embeddings
        code_length: Length of binary codes
    """
    print("=" * 60)
    print("Gray-Tunneled Hashing - Synthetic Experiment")
    print("=" * 60)
    print(f"\nParameters:")
    print(f"  - Number of points: {n_points}")
    print(f"  - Embedding dimension: {dim}")
    print(f"  - Code length: {code_length}\n")
    
    # Generate synthetic embeddings
    print("Generating synthetic embeddings...")
    embeddings = generate_synthetic_embeddings(n_points=n_points, dim=dim, seed=42)
    print(f"  ✓ Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
    
    # Initialize and fit hasher
    print("\nInitializing and fitting hasher...")
    hasher = GrayTunneledHasher(code_length=code_length)
    hasher.fit(embeddings)
    print(f"  ✓ Hasher fitted successfully")
    
    # Encode embeddings
    print("\nEncoding embeddings to binary codes...")
    codes = hasher.encode(embeddings)
    print(f"  ✓ Encoded to binary codes of shape {codes.shape}")
    
    # Evaluate
    print("\nEvaluating encoding quality...")
    metrics = hasher.evaluate(embeddings, codes)
    print(f"  Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"    - {key}: {value:.4f}")
        else:
            print(f"    - {key}: {value}")
    
    # Compute Hamming distances
    print("\nComputing Hamming distances...")
    hamm_dist_matrix = hamming_distance(codes, codes)
    print(f"  ✓ Computed pairwise Hamming distances (shape: {hamm_dist_matrix.shape})")
    
    # Compute original distances (Euclidean)
    print("\nComputing original embedding distances...")
    # Compute pairwise Euclidean distances
    orig_dist_matrix = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(n_points):
            orig_dist_matrix[i, j] = np.linalg.norm(embeddings[i] - embeddings[j])
    
    # Flatten upper triangle (excluding diagonal) for comparison
    mask = np.triu(np.ones((n_points, n_points)), k=1).astype(bool)
    orig_dist_flat = orig_dist_matrix[mask]
    hamm_dist_flat = hamm_dist_matrix[mask]
    
    # Compute preservation score
    preservation_score = hamming_preservation_score(orig_dist_flat, hamm_dist_flat)
    print(f"  ✓ Distance preservation score: {preservation_score:.4f}")
    
    print("\n" + "=" * 60)
    print("Experiment completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_experiment()

