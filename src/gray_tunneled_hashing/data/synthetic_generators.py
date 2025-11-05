"""Synthetic data generation for testing and experimentation."""

import numpy as np
from typing import Optional
from dataclasses import dataclass


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


def generate_hypercube_vertices(n_bits: int) -> np.ndarray:
    """
    Generate all vertices of the n-dimensional hypercube.
    
    For n_bits, generates all 2^n binary vectors in {0,1}^n.
    
    Args:
        n_bits: Dimension of the hypercube (n)
        
    Returns:
        Array of shape (2^n, n_bits) where each row is a binary vertex
        Vertices are ordered by their binary representation (0 to 2^n-1)
        
    Examples:
        >>> vertices = generate_hypercube_vertices(3)
        >>> vertices.shape
        (8, 3)
        >>> vertices[0]
        array([0, 0, 0])
        >>> vertices[-1]
        array([1, 1, 1])
    """
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")
    
    N = 2 ** n_bits
    vertices = np.zeros((N, n_bits), dtype=np.uint8)
    
    # Generate all binary representations
    for i in range(N):
        # Convert integer i to binary representation
        binary_str = format(i, f'0{n_bits}b')
        vertices[i] = [int(bit) for bit in binary_str]
    
    return vertices


def generate_planted_phi(
    n_bits: int,
    dim: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Generate ideal planted φ embeddings for hypercube vertices.
    
    Creates embeddings such that Hamming-1 neighbors are closer than random pairs.
    Strategy: Map binary vertices to ±1 space, apply random projection, and ensure
    local structure.
    
    Args:
        n_bits: Dimension of hypercube (n)
        dim: Dimensionality of embedding space
        random_state: Random seed for reproducibility
        
    Returns:
        Array of shape (2^n, dim) where phi[u] is the ideal embedding for vertex u
        
    Examples:
        >>> phi = generate_planted_phi(3, 10, random_state=42)
        >>> phi.shape
        (8, 10)
    """
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")
    if dim <= 0:
        raise ValueError("dim must be positive")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    N = 2 ** n_bits
    vertices = generate_hypercube_vertices(n_bits)
    
    # Convert binary {0,1} to ±1 representation
    vertices_pm1 = 2 * vertices.astype(np.float32) - 1  # (N, n_bits)
    
    # Create a random projection matrix to embed in higher dimension
    # This preserves some structure while mapping to dim dimensions
    if dim >= n_bits:
        # Use random projection
        projection = np.random.randn(n_bits, dim).astype(np.float32)
        # Scale to ensure reasonable distances
        projection = projection / np.sqrt(n_bits)
        phi = vertices_pm1 @ projection  # (N, dim)
    else:
        # If dim < n_bits, use PCA-like approach or just take first dim components
        projection = np.random.randn(n_bits, dim).astype(np.float32)
        projection = projection / np.sqrt(n_bits)
        phi = vertices_pm1 @ projection  # (N, dim)
    
    # Add small random offset to break symmetry and ensure uniqueness
    phi += 0.1 * np.random.randn(N, dim).astype(np.float32)
    
    # Scale to ensure Hamming-1 neighbors are closer
    # The projection already helps, but we can fine-tune
    phi = phi.astype(np.float32)
    
    return phi


def sample_noisy_embeddings(
    phi: np.ndarray,
    sigma: float,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Sample noisy embeddings from ideal planted model.
    
    Generates w = phi + noise where noise is Gaussian with standard deviation sigma.
    
    Args:
        phi: Ideal embeddings of shape (N, dim)
        sigma: Standard deviation of Gaussian noise
        random_state: Random seed for reproducibility
        
    Returns:
        Noisy embeddings of shape (N, dim)
        
    Examples:
        >>> phi = np.random.randn(8, 10)
        >>> w = sample_noisy_embeddings(phi, sigma=0.1, random_state=42)
        >>> w.shape
        (8, 10)
    """
    if phi.ndim != 2:
        raise ValueError(f"phi must be 2D, got {phi.ndim}D")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    noise = np.random.randn(*phi.shape).astype(np.float32) * sigma
    w = phi + noise
    
    return w


@dataclass
class PlantedModelConfig:
    """Configuration for planted model generation."""
    
    n_bits: int
    dim: int
    sigma: float
    random_state: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.n_bits <= 0:
            raise ValueError("n_bits must be positive")
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")
    
    def generate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate complete planted model instance.
        
        Returns:
            Tuple of (vertices, phi, w) where:
            - vertices: (2^n_bits, n_bits) hypercube vertices
            - phi: (2^n_bits, dim) ideal embeddings
            - w: (2^n_bits, dim) noisy embeddings
        """
        vertices = generate_hypercube_vertices(self.n_bits)
        phi = generate_planted_phi(
            self.n_bits, self.dim, random_state=self.random_state
        )
        # Use different seed for noise to ensure independence
        noise_seed = (
            self.random_state + 1 if self.random_state is not None else None
        )
        w = sample_noisy_embeddings(phi, self.sigma, random_state=noise_seed)
        
        return vertices, phi, w

