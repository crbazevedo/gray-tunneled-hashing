"""QAP objective and 2-swap hill-climbing for hypercube assignment."""

import numpy as np
from typing import Optional, Tuple


def generate_hypercube_edges(n_bits: int) -> np.ndarray:
    """
    Generate all edges of the n-dimensional hypercube.
    
    Returns all pairs of vertices that are Hamming distance 1 apart.
    
    Args:
        n_bits: Dimension of the hypercube (n)
        
    Returns:
        Array of shape (E, 2) where E = n * 2^(n-1) and each row is
        a pair of vertex indices [u, v] representing an edge.
        Vertices are indexed from 0 to 2^n - 1.
        
    Examples:
        >>> edges = generate_hypercube_edges(3)
        >>> edges.shape
        (12, 2)
        >>> # Each edge connects vertices differing by 1 bit
    """
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")
    
    N = 2 ** n_bits
    edges = []
    
    # Generate all vertices
    vertices = []
    for i in range(N):
        binary_str = format(i, f'0{n_bits}b')
        vertices.append([int(bit) for bit in binary_str])
    vertices = np.array(vertices, dtype=np.uint8)
    
    # Find all Hamming-1 neighbors
    for u in range(N):
        for v in range(u + 1, N):
            # Check if Hamming distance is 1
            hamming_dist = np.sum(vertices[u] != vertices[v])
            if hamming_dist == 1:
                edges.append([u, v])
    
    return np.array(edges, dtype=np.int32)


def qap_cost(pi: np.ndarray, D: np.ndarray, edges: np.ndarray) -> float:
    """
    Compute QAP cost for a given permutation.
    
    Computes f(π) = sum_{(u,v) in edges} D[π(u), π(v)]
    
    Args:
        pi: Permutation array of shape (N,) where pi[u] is the index
            of the embedding assigned to vertex u
        D: Distance matrix of shape (N, N) where D[i, j] is the squared
           distance between embeddings i and j
        edges: Array of shape (E, 2) where each row is an edge [u, v]
        
    Returns:
        Total QAP cost (float)
        
    Examples:
        >>> pi = np.array([0, 1, 2, 3])
        >>> D = np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]])
        >>> edges = np.array([[0, 1], [1, 2], [2, 3]])
        >>> cost = qap_cost(pi, D, edges)
    """
    if pi.ndim != 1:
        raise ValueError(f"pi must be 1D, got {pi.ndim}D")
    if D.ndim != 2:
        raise ValueError(f"D must be 2D, got {D.ndim}D")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"edges must be 2D with shape (E, 2), got {edges.shape}")
    
    N = len(pi)
    if D.shape[0] != N or D.shape[1] != N:
        raise ValueError(f"D shape {D.shape} incompatible with pi length {N}")
    
    # Compute cost for each edge
    cost = 0.0
    for edge in edges:
        u, v = edge[0], edge[1]
        if u >= N or v >= N:
            raise ValueError(f"Edge vertex indices out of range: {u}, {v}")
        i, j = pi[u], pi[v]
        cost += D[i, j]
    
    return float(cost)


def sample_two_swaps(
    N: int,
    sample_size: int,
    random_state: Optional[int] = None,
) -> list[Tuple[int, int]]:
    """
    Sample random pairs of vertices for 2-swap moves.
    
    Args:
        N: Number of vertices
        sample_size: Number of pairs to sample
        random_state: Random seed for reproducibility
        
    Returns:
        List of (u, v) tuples where u < v, representing candidate swaps
    """
    if N < 2:
        raise ValueError(f"N must be at least 2, got {N}")
    if sample_size <= 0:
        raise ValueError(f"sample_size must be positive, got {sample_size}")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate all possible pairs
    all_pairs = []
    for u in range(N):
        for v in range(u + 1, N):
            all_pairs.append((u, v))
    
    # Sample from all pairs
    if sample_size >= len(all_pairs):
        return all_pairs
    else:
        indices = np.random.choice(len(all_pairs), size=sample_size, replace=False)
        return [all_pairs[i] for i in indices]


def hill_climb_two_swap(
    pi_init: np.ndarray,
    D: np.ndarray,
    edges: np.ndarray,
    max_iter: int = 100,
    sample_size: int = 256,
    random_state: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, list[float]]:
    """
    Hill-climbing optimization using 2-swap moves.
    
    Monotonically decreases cost by applying improving 2-swaps.
    
    Args:
        pi_init: Initial permutation of shape (N,)
        D: Distance matrix of shape (N, N)
        edges: Hypercube edges of shape (E, 2)
        max_iter: Maximum number of iterations
        sample_size: Number of 2-swaps to sample per iteration
        random_state: Random seed for reproducibility
        verbose: Whether to print progress
        
    Returns:
        Tuple of (pi_best, cost_history) where:
        - pi_best: Best permutation found
        - cost_history: List of costs at each iteration
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    N = len(pi_init)
    pi = pi_init.copy()
    cost_history = []
    
    current_cost = qap_cost(pi, D, edges)
    cost_history.append(current_cost)
    
    if verbose:
        print(f"Initial cost: {current_cost:.6f}")
    
    for iteration in range(max_iter):
        # Sample candidate 2-swaps
        swaps = sample_two_swaps(N, sample_size, random_state=None)
        
        best_delta = 0.0
        best_swap = None
        
        # Evaluate all candidate swaps
        for u, v in swaps:
            # Compute cost delta for swapping pi[u] and pi[v]
            # We compute: new_cost - old_cost for all edges affected by the swap
            
            # Temporarily apply swap to compute new cost
            pi_swapped = pi.copy()
            pi_swapped[u], pi_swapped[v] = pi_swapped[v], pi_swapped[u]
            
            # Compute cost difference
            old_cost = qap_cost(pi, D, edges)
            new_cost = qap_cost(pi_swapped, D, edges)
            delta = new_cost - old_cost
            
            if delta < best_delta:
                best_delta = delta
                best_swap = (u, v)
        
        # Apply best improving swap
        if best_delta < -1e-10:  # Small tolerance for numerical errors
            u, v = best_swap
            pi[u], pi[v] = pi[v], pi[u]
            current_cost += best_delta
            cost_history.append(current_cost)
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}: cost = {current_cost:.6f}, delta = {best_delta:.6f}")
        else:
            # No improving move found
            if verbose:
                print(f"No improving move found at iteration {iteration + 1}")
            break
    
    return pi, cost_history

