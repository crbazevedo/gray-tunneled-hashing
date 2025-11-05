"""Synthetic data generation for testing and experimentation."""

import numpy as np
from typing import Optional


def generate_synthetic_embeddings(
    n_points: int,
    dim: int,
    seed: Optional[int] = None,
    distribution: str = "normal",
) -> np.ndarray:
    """
    Generate synthetic embeddings for testing and experimentation.
    
    Args:
        n_points: Number of embedding vectors to generate
        dim: Dimensionality of each embedding
        seed: Random seed for reproducibility (default: None)
        distribution: Distribution type ('normal' or 'uniform', default: 'normal')
        
    Returns:
        Array of shape (n_points, dim) with synthetic embeddings
        
    Examples:
        >>> embeddings = generate_synthetic_embeddings(100, 64)
        >>> embeddings.shape
        (100, 64)
    """
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    if dim <= 0:
        raise ValueError("dim must be positive")
    
    if seed is not None:
        np.random.seed(seed)
    
    if distribution == "normal":
        embeddings = np.random.randn(n_points, dim).astype(np.float32)
    elif distribution == "uniform":
        embeddings = np.random.uniform(-1, 1, size=(n_points, dim)).astype(np.float32)
    else:
        raise ValueError(f"Unknown distribution: {distribution}. Use 'normal' or 'uniform'")
    
    return embeddings

