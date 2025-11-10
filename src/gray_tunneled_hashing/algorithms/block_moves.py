"""Block moves and tunneling for QAP optimization."""

import numpy as np
from typing import Optional, Tuple, Callable
from itertools import permutations

from gray_tunneled_hashing.algorithms.qap_objective import qap_cost
from gray_tunneled_hashing.algorithms.block_selection import select_block_random


def select_block(
    N: int,
    block_size: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Select a random block of vertices.
    
    Args:
        N: Total number of vertices
        block_size: Size of the block to select
        random_state: Random seed for reproducibility
        
    Returns:
        Sorted array of block_size distinct vertex indices
        
    Examples:
        >>> block = select_block(16, 4, random_state=42)
        >>> len(block)
        4
        >>> np.all(block < 16)
        True
    """
    if block_size > N:
        raise ValueError(f"block_size {block_size} cannot exceed N {N}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    block = np.sort(np.random.choice(N, size=block_size, replace=False))
    return block


def block_reoptimize(
    pi: np.ndarray,
    D: np.ndarray,
    edges: np.ndarray,
    block_vertices: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Reoptimize assignment within a block using brute force.
    
    For small blocks, enumerates all permutations of embeddings assigned to
    block vertices and selects the best one.
    
    Args:
        pi: Current permutation of shape (N,)
        D: Distance matrix of shape (N, N)
        edges: Hypercube edges of shape (E, 2)
        block_vertices: Array of vertex indices in the block
        
    Returns:
        Tuple of (pi_new, delta_cost) where:
        - pi_new: Updated permutation (only changed within block)
        - delta_cost: Change in cost (negative means improvement)
        
    Examples:
        >>> pi = np.array([0, 1, 2, 3])
        >>> D = np.random.rand(4, 4)
        >>> edges = np.array([[0, 1], [0, 2], [1, 3], [2, 3]])
        >>> block = np.array([0, 1])
        >>> pi_new, delta = block_reoptimize(pi, D, edges, block)
    """
    if len(block_vertices) == 0:
        return pi.copy(), 0.0
    
    # For large blocks, brute force is too expensive
    # Limit to reasonable size (e.g., 8)
    if len(block_vertices) > 8:
        # For now, just return unchanged
        # In future, could use approximate optimization
        return pi.copy(), 0.0
    
    N = len(pi)
    block_size = len(block_vertices)
    
    # Get embeddings currently assigned to block vertices
    I_B = pi[block_vertices].copy()
    
    # Compute current cost
    current_cost = qap_cost(pi, D, edges)
    
    # Find edges within block and edges crossing block boundary
    block_set = set(block_vertices)
    
    # Try all permutations of I_B assigned to block_vertices
    best_pi = pi.copy()
    best_cost = current_cost
    
    for perm in permutations(I_B):
        # Create new permutation
        pi_candidate = pi.copy()
        for i, vertex in enumerate(block_vertices):
            pi_candidate[vertex] = perm[i]
        
        # Compute cost
        cost = qap_cost(pi_candidate, D, edges)
        
        if cost < best_cost:
            best_cost = cost
            best_pi = pi_candidate
    
    delta_cost = best_cost - current_cost
    
    return best_pi, delta_cost


def tunneling_step(
    pi: np.ndarray,
    D: np.ndarray,
    edges: np.ndarray,
    block_size: int,
    num_blocks: int = 10,
    random_state: Optional[int] = None,
    block_selection_fn: Optional[Callable] = None,
) -> Tuple[np.ndarray, float]:
    """
    Perform one tunneling step: try multiple blocks and apply best improving move.
    
    Args:
        pi: Current permutation of shape (N,)
        D: Distance matrix of shape (N, N)
        edges: Hypercube edges of shape (E, 2)
        block_size: Size of blocks to try
        num_blocks: Number of candidate blocks to sample
        random_state: Random seed for reproducibility
        block_selection_fn: Function to select blocks. If None, uses random selection.
                           Signature: (N, block_size, random_state) -> np.ndarray
        
    Returns:
        Tuple of (pi_new, best_delta) where:
        - pi_new: Updated permutation (may be unchanged if no improvement)
        - best_delta: Best cost delta found (negative means improvement)
        
    Examples:
        >>> pi = np.array([0, 1, 2, 3])
        >>> D = np.random.rand(4, 4)
        >>> edges = np.array([[0, 1], [0, 2], [1, 3], [2, 3]])
        >>> pi_new, delta = tunneling_step(pi, D, edges, block_size=2, num_blocks=5)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    N = len(pi)
    best_pi = pi.copy()
    best_delta = 0.0
    
    # Use provided block selection function or default to random
    if block_selection_fn is None:
        block_selection_fn = lambda N, block_size, random_state=None: select_block(
            N, block_size, random_state
        )
    
    # Try multiple blocks
    for i in range(num_blocks):
        # Select block using provided function
        block_vertices = block_selection_fn(N, block_size, random_state=None)
        
        # Reoptimize within block
        pi_candidate, delta = block_reoptimize(pi, D, edges, block_vertices)
        
        # Keep track of best improvement
        if delta < best_delta:
            best_delta = delta
            best_pi = pi_candidate.copy()
    
    # Only return improved permutation if we found an improvement
    if best_delta < -1e-10:  # Small tolerance for numerical errors
        return best_pi, best_delta
    else:
        return pi.copy(), 0.0

