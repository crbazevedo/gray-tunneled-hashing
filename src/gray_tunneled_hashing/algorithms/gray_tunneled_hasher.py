"""Gray-Tunneled Hashing algorithm implementation."""

import numpy as np
from typing import Optional

from gray_tunneled_hashing.algorithms.qap_objective import (
    generate_hypercube_edges,
    qap_cost,
    hill_climb_two_swap,
)
from gray_tunneled_hashing.algorithms.block_moves import tunneling_step


class GrayTunneledHasher:
    """
    Gray-Tunneled Hashing algorithm for binary vector encoding.
    
    For Sprint 1, this implements the core QAP optimization with 2-swap
    hill-climbing and block tunneling on synthetic planted instances.
    
    Attributes:
        n_bits: Number of bits (dimension of hypercube)
        block_size: Size of blocks for tunneling moves
        max_two_swap_iters: Maximum iterations for 2-swap hill climbing
        num_tunneling_steps: Number of tunneling steps to perform
        two_swap_sample_size: Number of 2-swap candidates to sample per iteration
        random_state: Random seed for reproducibility
        pi_: Final permutation (assignment of embeddings to vertices)
        cost_: Final QAP cost
        cost_history_: History of costs during optimization
        is_fitted: Whether the hasher has been fitted
    """

    def __init__(
        self,
        n_bits: int,
        block_size: int = 8,
        max_two_swap_iters: int = 100,
        num_tunneling_steps: int = 10,
        two_swap_sample_size: int = 256,
        init_strategy: str = "random",
        random_state: Optional[int] = None,
    ):
        """
        Initialize the Gray-Tunneled Hasher.
        
        Args:
            n_bits: Number of bits (hypercube dimension), determines N = 2^n_bits
            block_size: Size of blocks for tunneling moves (default: 8)
            max_two_swap_iters: Maximum iterations for 2-swap hill climbing (default: 100)
            num_tunneling_steps: Number of tunneling steps to perform (default: 10)
            two_swap_sample_size: Number of 2-swap candidates to sample per iteration (default: 256)
            init_strategy: Initialization strategy ('random' or 'identity', default: 'random')
            random_state: Random seed for reproducibility (default: None)
        """
        if n_bits <= 0:
            raise ValueError(f"n_bits must be positive, got {n_bits}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if init_strategy not in ["random", "identity"]:
            raise ValueError(f"init_strategy must be 'random' or 'identity', got {init_strategy}")
        
        self.n_bits = n_bits
        self.N = 2 ** n_bits
        self.block_size = block_size
        self.max_two_swap_iters = max_two_swap_iters
        self.num_tunneling_steps = num_tunneling_steps
        self.two_swap_sample_size = two_swap_sample_size
        self.init_strategy = init_strategy
        self.random_state = random_state
        
        self.is_fitted = False
        self.pi_ = None
        self.cost_ = None
        self.cost_history_ = None
        self.edges_ = None
        self.D_ = None

    def fit(self, embeddings: np.ndarray) -> "GrayTunneledHasher":
        """
        Fit the hasher to embeddings by optimizing QAP assignment.
        
        For Sprint 1, embeddings should be synthetic w of shape (N, dim) with N = 2**n_bits.
        
        Steps:
        1. Compute distance matrix D
        2. Generate hypercube edges
        3. Initialize permutation (identity or random)
        4. Run hill_climb_two_swap
        5. Run tunneling steps
        6. Save results
        
        Args:
            embeddings: Training embeddings of shape (N, dim) where N = 2**n_bits
            
        Returns:
            self (for method chaining)
        """
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got {embeddings.ndim}D")
        
        if embeddings.shape[0] != self.N:
            raise ValueError(
                f"embeddings must have shape ({self.N}, dim) where N = 2**n_bits = 2**{self.n_bits}, "
                f"got shape {embeddings.shape}"
            )
        
        # Set random seed
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Step 1: Compute distance matrix D
        dim = embeddings.shape[1]
        D = np.zeros((self.N, self.N), dtype=np.float64)
        for i in range(self.N):
            for j in range(self.N):
                D[i, j] = np.linalg.norm(embeddings[i] - embeddings[j]) ** 2
        
        self.D_ = D
        
        # Step 2: Generate hypercube edges
        edges = generate_hypercube_edges(self.n_bits)
        self.edges_ = edges
        
        # Step 3: Initialize permutation
        if self.init_strategy == "identity":
            pi_init = np.arange(self.N, dtype=np.int32)
        else:  # random
            pi_init = np.random.permutation(self.N).astype(np.int32)
        
        # Step 4: Run 2-swap hill climbing
        pi, cost_history = hill_climb_two_swap(
            pi_init=pi_init,
            D=D,
            edges=edges,
            max_iter=self.max_two_swap_iters,
            sample_size=self.two_swap_sample_size,
            random_state=self.random_state,
        )
        
        # Step 5: Run tunneling steps
        for step in range(self.num_tunneling_steps):
            pi_new, delta = tunneling_step(
                pi=pi,
                D=D,
                edges=edges,
                block_size=self.block_size,
                num_blocks=10,
                random_state=None,  # Use different random state for each step
            )
            
            if delta < -1e-10:  # Improvement found
                pi = pi_new
                current_cost = qap_cost(pi, D, edges)
                cost_history.append(current_cost)
            else:
                # No improvement, continue with current pi
                pass
        
        # Step 6: Save results
        self.pi_ = pi
        self.cost_ = qap_cost(pi, D, edges)
        self.cost_history_ = cost_history
        self.is_fitted = True
        
        return self

    def get_assignment(self) -> np.ndarray:
        """
        Get the final permutation (assignment of embeddings to hypercube vertices).
        
        Returns:
            Permutation array of shape (N,) where pi[u] is the index of the
            embedding assigned to vertex u
            
        Raises:
            ValueError: If hasher has not been fitted yet
        """
        if not self.is_fitted:
            raise ValueError("Hasher must be fitted before getting assignment. Call fit() first.")
        
        return self.pi_.copy()

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Encode embeddings into binary codes using the optimized assignment.
        
        For Sprint 1, this is a minimal implementation. The main goal is
        assignment optimization, not encoding.
        
        Args:
            embeddings: Embeddings to encode, shape (n_samples, n_features)
            
        Returns:
            Binary codes of shape (n_samples, n_bits)
        """
        if not self.is_fitted:
            raise ValueError("Hasher must be fitted before encoding. Call fit() first.")
        
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got {embeddings.ndim}D")
        
        # For Sprint 1, minimal encoding: use hypercube vertices
        # Map each embedding to its assigned vertex's binary code
        from gray_tunneled_hashing.data.synthetic_generators import generate_hypercube_vertices
        
        vertices = generate_hypercube_vertices(self.n_bits)
        n_samples = embeddings.shape[0]
        
        # For now, return a placeholder encoding
        # In future sprints, this will use the assignment π to map embeddings to codes
        codes = np.zeros((n_samples, self.n_bits), dtype=np.uint8)
        
        # Simple heuristic: find closest vertex for each embedding
        for i in range(n_samples):
            # Find vertex with index matching assignment (for Sprint 1, simplified)
            # This is a placeholder - proper encoding will use π inverse mapping
            closest_idx = i % self.N  # Placeholder
            codes[i] = vertices[closest_idx]
        
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        Decode binary codes back to embeddings (approximate).
        
        This is a stub implementation. Full decoding will be implemented later.
        
        Args:
            codes: Binary codes of shape (n_samples, code_length)
            
        Returns:
            Approximate embeddings (for now, returns zeros)
        """
        if codes.ndim != 2:
            raise ValueError(f"codes must be 2D, got {codes.ndim}D")
        
        # Stub: return zeros with appropriate shape
        n_samples = codes.shape[0]
        n_features = self.D_.shape[0] if self.D_ is not None else 128
        return np.zeros((n_samples, n_features))

    def evaluate(self, embeddings: np.ndarray, codes: np.ndarray) -> dict:
        """
        Evaluate the quality of the encoding.
        
        Args:
            embeddings: Original embeddings
            codes: Binary codes produced by encode()
            
        Returns:
            Dictionary with evaluation metrics
        """
        if embeddings.ndim != 2 or codes.ndim != 2:
            raise ValueError("embeddings and codes must be 2D")
        
        if embeddings.shape[0] != codes.shape[0]:
            raise ValueError("embeddings and codes must have same number of samples")
        
        metrics = {
            "n_samples": embeddings.shape[0],
            "embedding_dim": embeddings.shape[1],
            "code_length": codes.shape[1],
            "final_qap_cost": float(self.cost_) if self.is_fitted else None,
            "mean_code_value": float(np.mean(codes)),
            "code_sparsity": float(np.mean(codes == 0)),
        }
        
        return metrics
