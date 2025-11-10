"""Baseline binary encoding methods."""

import numpy as np
from typing import Optional, Tuple


def sign_binarize(embeddings: np.ndarray) -> np.ndarray:
    """
    Binarize embeddings using sign thresholding.
    
    Each bit i is set to 1 if embedding[i] > 0, else 0.
    This uses all dimensions of the embedding, so the output has
    shape (N, dim) where dim is the original embedding dimension.
    
    Args:
        embeddings: Input embeddings of shape (N, dim)
        
    Returns:
        Binary codes of shape (N, dim) with dtype bool
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
    
    return (embeddings > 0).astype(bool)


def random_projection_binarize(
    embeddings: np.ndarray,
    n_bits: int,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Binarize embeddings using random projection.
    
    Projects embeddings onto n_bits random hyperplanes and binarizes
    based on which side of each hyperplane each embedding falls.
    
    Args:
        embeddings: Input embeddings of shape (N, dim)
        n_bits: Number of bits in the binary code
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (binary_codes, projection_matrix) where:
        - binary_codes: Array of shape (N, n_bits) with dtype bool
        - projection_matrix: Array of shape (n_bits, dim) used for projection
          (save this to encode queries with the same projection)
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
    
    if n_bits <= 0:
        raise ValueError(f"n_bits must be positive, got {n_bits}")
    
    N, dim = embeddings.shape
    
    # Generate random projection matrix
    if random_state is not None:
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random
    
    # Random projection: each row is a random hyperplane normal
    projection_matrix = rng.randn(n_bits, dim).astype(np.float32)
    
    # Project embeddings: (N, dim) @ (dim, n_bits) -> (N, n_bits)
    projections = embeddings @ projection_matrix.T
    
    # Binarize: positive values -> 1, negative/zero -> 0
    binary_codes = (projections > 0).astype(bool)
    
    return binary_codes, projection_matrix


def apply_random_projection(
    embeddings: np.ndarray,
    projection_matrix: np.ndarray,
) -> np.ndarray:
    """
    Apply a pre-computed random projection to embeddings.
    
    This is used to encode queries with the same projection matrix
    used for the base corpus.
    
    Args:
        embeddings: Input embeddings of shape (Q, dim)
        projection_matrix: Projection matrix of shape (n_bits, dim)
        
    Returns:
        Binary codes of shape (Q, n_bits) with dtype bool
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
    
    if projection_matrix.ndim != 2:
        raise ValueError(
            f"Expected 2D projection matrix, got shape {projection_matrix.shape}"
        )
    
    # Project and binarize
    projections = embeddings @ projection_matrix.T
    binary_codes = (projections > 0).astype(bool)
    
    return binary_codes

