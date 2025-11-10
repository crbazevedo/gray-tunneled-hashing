"""
Locality-Sensitive Hashing (LSH) families for binary encoding.

This module implements LSH families as described in the theoretical paper:
- Hyperplane LSH for cosine similarity
- p-stable LSH for ℓ₂ distance

These are proper LSH families with theoretical collision probability guarantees,
unlike simple random projections which don't have explicit LSH properties.
"""

import numpy as np
from typing import Optional, Tuple, Callable
from dataclasses import dataclass


@dataclass
class LSHFamily:
    """Base class for LSH families."""
    n_bits: int
    dim: int
    random_state: Optional[int] = None
    
    def hash(self, vectors: np.ndarray) -> np.ndarray:
        """
        Hash vectors to binary codes.
        
        Args:
            vectors: Input vectors of shape (N, dim)
            
        Returns:
            Binary codes of shape (N, n_bits) with dtype bool
        """
        raise NotImplementedError
    
    def collision_probability(self, distance: float) -> float:
        """
        Theoretical collision probability for a given distance.
        
        Args:
            distance: Distance between two vectors
            
        Returns:
            Probability that two vectors at this distance collide (hash to same code)
        """
        raise NotImplementedError


class HyperplaneLSH(LSHFamily):
    """
    Hyperplane LSH for cosine similarity.
    
    For vectors x, y with cosine similarity cos(θ), the collision probability
    is approximately 1 - θ/π for a single hyperplane.
    
    For n_bits hyperplanes, we use AND composition: vectors collide if they
    are on the same side of all hyperplanes.
    """
    
    def __init__(
        self,
        n_bits: int,
        dim: int,
        random_state: Optional[int] = None,
    ):
        """
        Initialize hyperplane LSH.
        
        Args:
            n_bits: Number of bits (number of hyperplanes)
            dim: Dimension of input vectors
            random_state: Random seed
        """
        self.n_bits = n_bits
        self.dim = dim
        self.random_state = random_state
        
        # Generate random hyperplanes (normal vectors)
        if random_state is not None:
            np.random.seed(random_state)
        
        # Sample from unit sphere (normalize Gaussian vectors)
        hyperplanes = np.random.randn(n_bits, dim).astype(np.float32)
        norms = np.linalg.norm(hyperplanes, axis=1, keepdims=True)
        self.hyperplanes = hyperplanes / (norms + 1e-10)  # Normalize to unit vectors
    
    def hash(self, vectors: np.ndarray) -> np.ndarray:
        """
        Hash vectors using hyperplane LSH.
        
        For each hyperplane, compute sign(dot(v, h)) to get one bit.
        
        Args:
            vectors: Input vectors of shape (N, dim)
            
        Returns:
            Binary codes of shape (N, n_bits) with dtype bool
        """
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Vectors must have dimension {self.dim}, got {vectors.shape[1]}"
            )
        
        # Project onto hyperplanes: (N, dim) @ (n_bits, dim).T = (N, n_bits)
        projections = vectors @ self.hyperplanes.T
        
        # Sign thresholding: positive -> 1, negative -> 0
        codes = (projections >= 0).astype(bool)
        
        return codes
    
    def collision_probability(self, cosine_similarity: float) -> float:
        """
        Theoretical collision probability for cosine similarity.
        
        For a single hyperplane, P[collision] = 1 - arccos(sim) / π
        For n_bits hyperplanes with AND composition, P[all collide] = (1 - arccos(sim)/π)^n_bits
        
        Args:
            cosine_similarity: Cosine similarity between two vectors (in [-1, 1])
            
        Returns:
            Probability that both vectors hash to the same code
        """
        # Clamp to valid range
        cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
        
        # Angle between vectors
        theta = np.arccos(cosine_similarity)
        
        # Single hyperplane collision probability
        p_single = 1.0 - (theta / np.pi)
        
        # AND composition: all hyperplanes must agree
        p_all = p_single ** self.n_bits
        
        return float(p_all)


class PStableLSH(LSHFamily):
    """
    p-stable LSH for ℓ₂ (Euclidean) distance.
    
    Uses 2-stable distribution (Gaussian) for ℓ₂ distance.
    For vectors x, y with ℓ₂ distance d, the collision probability
    decreases as d increases.
    
    Uses multiple hash functions with AND composition.
    """
    
    def __init__(
        self,
        n_bits: int,
        dim: int,
        w: float = 1.0,
        random_state: Optional[int] = None,
    ):
        """
        Initialize p-stable LSH.
        
        Args:
            n_bits: Number of bits (number of hash functions)
            dim: Dimension of input vectors
            w: Width parameter for quantization (larger w = more collisions)
            random_state: Random seed
        """
        self.n_bits = n_bits
        self.dim = dim
        self.w = w
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Sample random vectors from Gaussian (2-stable distribution)
        self.a = np.random.randn(n_bits, dim).astype(np.float32)
        
        # Random offsets for quantization
        self.b = np.random.uniform(0, w, size=n_bits).astype(np.float32)
    
    def hash(self, vectors: np.ndarray) -> np.ndarray:
        """
        Hash vectors using p-stable LSH.
        
        For each hash function: h(x) = floor((a·x + b) / w)
        Then take modulo 2 to get binary bit.
        
        Args:
            vectors: Input vectors of shape (N, dim)
            
        Returns:
            Binary codes of shape (N, n_bits) with dtype bool
        """
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Vectors must have dimension {self.dim}, got {vectors.shape[1]}"
            )
        
        # Project onto random vectors: (N, dim) @ (n_bits, dim).T = (N, n_bits)
        projections = vectors @ self.a.T
        
        # Add offsets and quantize
        quantized = np.floor((projections + self.b) / self.w).astype(np.int32)
        
        # Modulo 2 to get binary bits
        codes = (quantized % 2 == 1).astype(bool)
        
        return codes
    
    def collision_probability(self, distance: float) -> float:
        """
        Theoretical collision probability for ℓ₂ distance.
        
        For p-stable LSH with width w, the collision probability
        for distance d is approximately:
        P[collision] ≈ 1 / (1 + d/w)
        
        For n_bits hash functions with AND composition:
        P[all collide] ≈ (1 / (1 + d/w))^n_bits
        
        Args:
            distance: ℓ₂ distance between two vectors
            
        Returns:
            Probability that both vectors hash to the same code
        """
        if distance < 0:
            raise ValueError(f"Distance must be non-negative, got {distance}")
        
        # Single hash function collision probability
        p_single = 1.0 / (1.0 + distance / self.w)
        
        # AND composition: all hash functions must agree
        p_all = p_single ** self.n_bits
        
        return float(p_all)


def create_lsh_family(
    family: str,
    n_bits: int,
    dim: int,
    random_state: Optional[int] = None,
    **kwargs,
) -> LSHFamily:
    """
    Factory function to create LSH family instances.
    
    Args:
        family: LSH family name ("hyperplane" or "p_stable")
        n_bits: Number of bits
        dim: Dimension of input vectors
        random_state: Random seed
        **kwargs: Additional family-specific parameters
            - For "p_stable": w (width parameter, default: 1.0)
        
    Returns:
        LSHFamily instance
        
    Examples:
        >>> lsh = create_lsh_family("hyperplane", n_bits=64, dim=128, random_state=42)
        >>> codes = lsh.hash(vectors)
        >>> prob = lsh.collision_probability(0.9)  # cosine similarity
    """
    if family == "hyperplane":
        return HyperplaneLSH(n_bits=n_bits, dim=dim, random_state=random_state)
    elif family == "p_stable":
        w = kwargs.get("w", 1.0)
        return PStableLSH(n_bits=n_bits, dim=dim, w=w, random_state=random_state)
    else:
        raise ValueError(
            f"Unknown LSH family: {family}. "
            f"Supported families: 'hyperplane', 'p_stable'"
        )


def validate_lsh_properties(
    lsh: LSHFamily,
    vectors: np.ndarray,
    similarity_fn: Callable[[np.ndarray, np.ndarray], float],
    n_samples: int = 1000,
) -> dict:
    """
    Validate LSH properties empirically.
    
    Tests that collision probability decreases as distance increases
    (or similarity decreases).
    
    Args:
        lsh: LSH family instance
        vectors: Input vectors of shape (N, dim)
        similarity_fn: Function to compute similarity/distance between two vectors
        n_samples: Number of random pairs to sample for validation
        
    Returns:
        Dictionary with validation results:
        - collision_rates: Array of collision rates for different similarity bins
        - theoretical_probs: Array of theoretical probabilities
        - correlation: Correlation between empirical and theoretical
    """
    if vectors.shape[0] < 2:
        raise ValueError("Need at least 2 vectors for validation")
    
    # Sample random pairs
    pairs = []
    similarities = []
    for _ in range(n_samples):
        i, j = np.random.choice(vectors.shape[0], size=2, replace=False)
        pairs.append((i, j))
        sim = similarity_fn(vectors[i], vectors[j])
        similarities.append(sim)
    
    similarities = np.array(similarities)
    pairs = np.array(pairs)
    
    # Hash all vectors
    codes = lsh.hash(vectors)
    
    # Compute collisions
    collisions = []
    for i, j in pairs:
        collide = np.array_equal(codes[i], codes[j])
        collisions.append(collide)
    collisions = np.array(collisions)
    
    # Bin by similarity
    n_bins = 10
    bins = np.linspace(similarities.min(), similarities.max(), n_bins + 1)
    bin_indices = np.digitize(similarities, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    collision_rates = []
    theoretical_probs = []
    bin_centers = []
    
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if mask.sum() == 0:
            continue
        
        bin_similarities = similarities[mask]
        bin_collisions = collisions[mask]
        
        # Empirical collision rate
        rate = bin_collisions.mean()
        collision_rates.append(rate)
        
        # Theoretical probability (average over bin)
        avg_sim = bin_similarities.mean()
        if isinstance(lsh, HyperplaneLSH):
            prob = lsh.collision_probability(avg_sim)
        else:  # PStableLSH
            # For p-stable, similarity_fn should return distance
            prob = lsh.collision_probability(avg_sim)
        theoretical_probs.append(prob)
        
        bin_centers.append(bin_similarities.mean())
    
    # Compute correlation
    if len(collision_rates) > 1:
        correlation = np.corrcoef(collision_rates, theoretical_probs)[0, 1]
    else:
        correlation = np.nan
    
    return {
        "collision_rates": np.array(collision_rates),
        "theoretical_probs": np.array(theoretical_probs),
        "bin_centers": np.array(bin_centers),
        "correlation": correlation,
    }

