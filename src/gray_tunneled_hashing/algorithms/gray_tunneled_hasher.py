"""Gray-Tunneled Hashing algorithm implementation."""

import numpy as np
import time
from typing import Optional, Literal, Dict, Any, Callable, Tuple

from gray_tunneled_hashing.algorithms.qap_objective import (
    generate_hypercube_edges,
    qap_cost,
    hill_climb_two_swap,
)
from gray_tunneled_hashing.algorithms.block_moves import tunneling_step
from gray_tunneled_hashing.algorithms.block_selection import (
    get_block_selection_fn,
    select_block_random,
)


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
        mode: Literal["trivial", "two_swap_only", "full"] = "full",
        track_history: bool = False,
        block_selection_strategy: Literal["random", "cluster"] = "random",
        cluster_assignments: Optional[np.ndarray] = None,
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
            mode: Optimization mode (default: 'full')
                - 'trivial': Simple mapping (identity or Gray-code), no optimization
                - 'two_swap_only': 2-swap hill climb only, no tunneling
                - 'full': 2-swap + tunneling (current behavior)
            track_history: If True, store detailed cost history with timestamps (default: False)
            block_selection_strategy: Strategy for selecting blocks ('random' or 'cluster', default: 'random')
            cluster_assignments: Cluster assignments for cluster-based block selection (required if strategy='cluster')
        """
        if n_bits <= 0:
            raise ValueError(f"n_bits must be positive, got {n_bits}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if init_strategy not in ["random", "identity"]:
            raise ValueError(f"init_strategy must be 'random' or 'identity', got {init_strategy}")
        if mode not in ["trivial", "two_swap_only", "full"]:
            raise ValueError(f"mode must be 'trivial', 'two_swap_only', or 'full', got {mode}")
        if block_selection_strategy not in ["random", "cluster"]:
            raise ValueError(
                f"block_selection_strategy must be 'random' or 'cluster', "
                f"got {block_selection_strategy}"
            )
        if block_selection_strategy == "cluster" and cluster_assignments is None:
            raise ValueError(
                "cluster_assignments is required when block_selection_strategy='cluster'"
            )
        
        self.n_bits = n_bits
        self.N = 2 ** n_bits
        self.block_size = block_size
        self.max_two_swap_iters = max_two_swap_iters
        self.num_tunneling_steps = num_tunneling_steps
        self.two_swap_sample_size = two_swap_sample_size
        self.init_strategy = init_strategy
        self.random_state = random_state
        self.mode = mode
        self.track_history = track_history
        self.block_selection_strategy = block_selection_strategy
        self.cluster_assignments = cluster_assignments
        
        self.is_fitted = False
        self.pi_ = None
        self.cost_ = None
        self.cost_history_ = None
        self.edges_ = None
        self.D_ = None

    def fit(
        self,
        embeddings: np.ndarray,
        D: Optional[np.ndarray] = None,
    ) -> "GrayTunneledHasher":
        """
        Fit the hasher to embeddings by optimizing QAP assignment.
        
        For Sprint 1, embeddings should be synthetic w of shape (N, dim) with N = 2**n_bits.
        For Sprint 2+, embeddings can be centroids from codebook (n_codes <= 2**n_bits).
        
        Steps (depending on mode):
        1. Compute distance matrix D (or use provided D)
        2. Generate hypercube edges
        3. Initialize permutation (identity, random, or Gray-code)
        4. Run optimization based on mode:
           - trivial: Use simple mapping
           - two_swap_only: Run hill_climb_two_swap only
           - full: Run hill_climb_two_swap + tunneling
        5. Save results
        
        Args:
            embeddings: Training embeddings of shape (N, dim) where N = 2**n_bits (or n_codes)
            D: Optional custom distance matrix of shape (N, N). If None, computed from embeddings.
            
        Returns:
            self (for method chaining)
        """
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got {embeddings.ndim}D")
        
        # Allow N != 2**n_bits for codebook scenarios (n_codes < 2**n_bits)
        # But for now, we'll enforce it for backward compatibility
        # In future, we can relax this
        if embeddings.shape[0] != self.N:
            raise ValueError(
                f"embeddings must have shape ({self.N}, dim) where N = 2**n_bits = 2**{self.n_bits}, "
                f"got shape {embeddings.shape}"
            )
        
        # Set random seed
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Step 1: Compute or use provided distance matrix D
        if D is not None:
            if D.shape != (self.N, self.N):
                raise ValueError(
                    f"Provided D must have shape ({self.N}, {self.N}), got {D.shape}"
                )
            self.D_ = D.astype(np.float64)
        else:
            dim = embeddings.shape[1]
            D = np.zeros((self.N, self.N), dtype=np.float64)
            for i in range(self.N):
                for j in range(self.N):
                    D[i, j] = np.linalg.norm(embeddings[i] - embeddings[j]) ** 2
            self.D_ = D
        
        # Step 2: Generate hypercube edges
        edges = generate_hypercube_edges(self.n_bits)
        self.edges_ = edges
        
        # Initialize cost history
        if self.track_history:
            self.cost_history_ = []  # List of dicts with metadata
        else:
            self.cost_history_ = []  # List of floats for backward compatibility
        
        # Step 3: Initialize permutation based on mode
        if self.mode == "trivial":
            # Trivial mode: use identity or Gray-code mapping
            pi = self._create_trivial_mapping(embeddings)
            cost = qap_cost(pi, D, edges)
            
            if self.track_history:
                self.cost_history_.append({
                    "cost": float(cost),
                    "step": "trivial_mapping",
                    "iteration": 0,
                    "timestamp": time.time(),
                })
            else:
                self.cost_history_.append(cost)
        
        elif self.mode == "two_swap_only":
            # Two-swap only mode
            if self.init_strategy == "identity":
                pi_init = np.arange(self.N, dtype=np.int32)
            else:  # random
                pi_init = np.random.permutation(self.N).astype(np.int32)
            
            initial_cost = qap_cost(pi_init, D, edges)
            if self.track_history:
                self.cost_history_.append({
                    "cost": float(initial_cost),
                    "step": "initialization",
                    "iteration": 0,
                    "timestamp": time.time(),
                })
            else:
                self.cost_history_.append(initial_cost)
            
            # Run 2-swap hill climbing
            pi, cost_history_two_swap = hill_climb_two_swap(
                pi_init=pi_init,
                D=D,
                edges=edges,
                max_iter=self.max_two_swap_iters,
                sample_size=self.two_swap_sample_size,
                random_state=self.random_state,
            )
            
            # Merge cost history
            if self.track_history:
                for i, cost_val in enumerate(cost_history_two_swap[1:], start=1):
                    self.cost_history_.append({
                        "cost": float(cost_val),
                        "step": f"two_swap_iter_{i}",
                        "iteration": i,
                        "timestamp": time.time(),
                    })
            else:
                self.cost_history_.extend(cost_history_two_swap[1:])
        
        else:  # mode == "full"
            # Full mode: 2-swap + tunneling
            if self.init_strategy == "identity":
                pi_init = np.arange(self.N, dtype=np.int32)
            else:  # random
                pi_init = np.random.permutation(self.N).astype(np.int32)
            
            initial_cost = qap_cost(pi_init, D, edges)
            if self.track_history:
                self.cost_history_.append({
                    "cost": float(initial_cost),
                    "step": "initialization",
                    "iteration": 0,
                    "timestamp": time.time(),
                })
            else:
                self.cost_history_.append(initial_cost)
            
            # Step 4: Run 2-swap hill climbing
            pi, cost_history_two_swap = hill_climb_two_swap(
                pi_init=pi_init,
                D=D,
                edges=edges,
                max_iter=self.max_two_swap_iters,
                sample_size=self.two_swap_sample_size,
                random_state=self.random_state,
            )
            
            # Merge 2-swap cost history
            if self.track_history:
                for i, cost_val in enumerate(cost_history_two_swap[1:], start=1):
                    self.cost_history_.append({
                        "cost": float(cost_val),
                        "step": f"two_swap_iter_{i}",
                        "iteration": i,
                        "timestamp": time.time(),
                    })
            else:
                self.cost_history_.extend(cost_history_two_swap[1:])
            
            # Step 5: Run tunneling steps
            # Get block selection function
            block_selection_fn = None
            if self.block_selection_strategy == "cluster" and self.cluster_assignments is not None:
                # Create cluster-based block selection function
                block_selection_fn = get_block_selection_fn(
                    "cluster",
                    pi=pi,
                    cluster_assignments=self.cluster_assignments,
                )
            else:
                block_selection_fn = get_block_selection_fn("random")
            
            for step in range(self.num_tunneling_steps):
                pi_new, delta = tunneling_step(
                    pi=pi,
                    D=D,
                    edges=edges,
                    block_size=self.block_size,
                    num_blocks=10,
                    random_state=None,  # Use different random state for each step
                    block_selection_fn=block_selection_fn,
                )
                
                if delta < -1e-10:  # Improvement found
                    pi = pi_new
                    current_cost = qap_cost(pi, D, edges)
                    
                    if self.track_history:
                        self.cost_history_.append({
                            "cost": float(current_cost),
                            "step": f"tunneling_step_{step}",
                            "iteration": len(self.cost_history_),
                            "timestamp": time.time(),
                        })
                    else:
                        self.cost_history_.append(current_cost)
                else:
                    # No improvement, but still record
                    if self.track_history:
                        current_cost = qap_cost(pi, D, edges)
                        self.cost_history_.append({
                            "cost": float(current_cost),
                            "step": f"tunneling_step_{step}_no_improvement",
                            "iteration": len(self.cost_history_),
                            "timestamp": time.time(),
                        })
        
        # Step 6: Save results
        self.pi_ = pi
        self.cost_ = qap_cost(pi, D, edges)
        self.is_fitted = True
        
        return self
    
    def _create_trivial_mapping(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Create trivial mapping (identity or Gray-code order).
        
        Args:
            embeddings: Embeddings of shape (N, dim)
            
        Returns:
            Permutation array pi
        """
        N = embeddings.shape[0]
        
        if self.init_strategy == "identity":
            # Simple identity mapping
            return np.arange(N, dtype=np.int32)
        else:
            # Gray-code ordering: sort by first dimension, assign in Gray-code order
            from gray_tunneled_hashing.data.synthetic_generators import generate_hypercube_vertices
            
            # Sort embeddings by first dimension
            embedding_scores = embeddings[:, 0]
            sorted_indices = np.argsort(embedding_scores)
            
            # Generate Gray-code sequence
            gray_sequence = self._generate_gray_code_sequence(self.n_bits)
            
            # Create permutation: pi[vertex_idx] = embedding_index
            pi = np.zeros(N, dtype=np.int32)
            for i, vertex_idx in enumerate(gray_sequence):
                pi[vertex_idx] = sorted_indices[i]
            
            return pi
    
    def _generate_gray_code_sequence(self, n_bits: int) -> np.ndarray:
        """Generate Gray code sequence recursively."""
        if n_bits == 0:
            return np.array([0], dtype=np.int32)
        if n_bits == 1:
            return np.array([0, 1], dtype=np.int32)
        
        # Recursive Gray code generation
        gray_n_minus_1 = self._generate_gray_code_sequence(n_bits - 1)
        gray_n = np.concatenate([
            gray_n_minus_1,
            gray_n_minus_1[::-1] + (2 ** (n_bits - 1))
        ])
        return gray_n.astype(np.int32)
    
    def fit_with_traffic(
        self,
        bucket_embeddings: np.ndarray,
        pi: np.ndarray,
        w: np.ndarray,
        queries: Optional[np.ndarray] = None,  # NEW: Required for real embeddings objective
        base_embeddings: Optional[np.ndarray] = None,  # NEW: Required for real embeddings objective
        ground_truth_neighbors: Optional[np.ndarray] = None,  # NEW: Required for real embeddings objective
        encoder: Optional[Callable] = None,  # NEW: LSH encoder function
        code_to_bucket: Optional[Dict] = None,  # NEW: Mapping from codes to buckets
        use_semantic_distances: bool = True,
        optimize_j_phi_directly: bool = True,
        optimization_method: Literal["hill_climb", "simulated_annealing", "memetic"] = "hill_climb",
        use_cosine_objective: bool = False,
        cosine_weight: float = 1.0,
        hamming_weight: float = 1.0,
        distance_metric: str = "cosine",
        use_real_embeddings_objective: bool = True,  # NEW: Use real embeddings objective (Sprint 8)
        sample_size_pairs: Optional[int] = None,  # NEW: Sample size for pairs in cost computation
    ) -> "GrayTunneledHasher":
        """
        Fit with distribution-aware traffic weights.
        
        This method integrates traffic statistics (query prior π_i and neighbor weights w_ij)
        into the optimization. It can use either:
        1. QAP optimization (legacy, may not guarantee J(φ*) ≤ J(φ₀))
        2. Direct J(φ) optimization (recommended, guarantees J(φ*) ≤ J(φ₀))
        
        Args:
            bucket_embeddings: Bucket representative embeddings of shape (K, dim) where K <= 2**n_bits
            pi: Query prior of shape (K,) - fraction of queries in each bucket
            w: Neighbor weights of shape (K, K) - probability neighbor in j given query in i
            use_semantic_distances: If True, multiply by semantic distance (default: True)
            optimize_j_phi_directly: If True, optimize J(φ) directly instead of QAP (default: True)
            
        Returns:
            self (for method chaining)
        """
        from gray_tunneled_hashing.distribution.traffic_stats import build_weighted_distance_matrix
        from gray_tunneled_hashing.distribution.j_phi_objective import (
            compute_j_phi_cost,
            hill_climb_j_phi,
        )
        from gray_tunneled_hashing.data.synthetic_generators import generate_hypercube_vertices
        
        if bucket_embeddings.ndim != 2:
            raise ValueError(f"Expected 2D bucket_embeddings, got shape {bucket_embeddings.shape}")
        if pi.ndim != 1:
            raise ValueError(f"Expected 1D pi, got shape {pi.shape}")
        if w.ndim != 2 or w.shape[0] != w.shape[1]:
            raise ValueError(f"Expected square w, got shape {w.shape}")
        if bucket_embeddings.shape[0] != pi.shape[0] or pi.shape[0] != w.shape[0]:
            raise ValueError(
                f"Shape mismatch: bucket_embeddings={bucket_embeddings.shape}, "
                f"pi={pi.shape}, w={w.shape}"
            )
        
        K_original = len(pi)  # Store original K before any padding/subsampling
        
        # Get bucket_to_code - check if it was stored (from pipeline)
        if hasattr(self, 'bucket_to_code_'):
            bucket_to_code_original = self.bucket_to_code_
        else:
            # Fallback: create dummy mapping (not recommended)
            vertices = generate_hypercube_vertices(self.n_bits)
            N_hypercube = len(vertices)
            bucket_to_code_original = vertices[:min(K_original, N_hypercube)]
            if K_original > N_hypercube:
                padding = np.tile(vertices[-1:], (K_original - N_hypercube, 1))
                bucket_to_code_original = np.vstack([bucket_to_code_original, padding])
        
        if optimize_j_phi_directly:
            # Direct J(φ) optimization
            # Use original K for pi, w, bucket_to_code (no padding for J(φ) computation)
            # But we still need to pad for permutation space (N = 2**n_bits)
            
            # Calculate semantic distances if requested
            semantic_distances = None
            semantic_weight = 0.0
            if use_semantic_distances:
                # Compute semantic distance matrix between bucket embeddings
                K = len(bucket_embeddings)
                semantic_distances = np.zeros((K, K), dtype=np.float64)
                for i in range(K):
                    for j in range(K):
                        # Squared L2 distance (normalized by dimension for stability)
                        d_semantic = np.linalg.norm(bucket_embeddings[i] - bucket_embeddings[j]) ** 2
                        semantic_distances[i, j] = d_semantic
                
                # Normalize semantic distances to be on similar scale as Hamming distances
                # Hamming distances are in [0, n_bits], so normalize semantic to similar range
                if semantic_distances.max() > 0:
                    semantic_distances = semantic_distances / semantic_distances.max() * self.n_bits
                
                # Set semantic weight (can be tuned, default: 0.5 to balance Hamming and semantic)
                semantic_weight = 0.5
            
            # Initialize permutation as array of binary codes (K, n_bits)
            # NEW: permutation[bucket_idx] = novo_código_binário
            K_actual = len(pi)  # Number of actual buckets
            
            if self.init_strategy == "random":
                # Random initialization: generate K random binary codes
                # Each code is a random bit vector of length n_bits
                pi_init = np.random.randint(0, 2, size=(K_actual, self.n_bits), dtype=np.uint8)
            else:
                # Identity: use original bucket codes as initial permutation
                # This means no permutation initially (each bucket keeps its original code)
                pi_init = bucket_to_code_original.copy().astype(np.uint8)
                # Ensure shape is (K, n_bits)
                if pi_init.shape != (K_actual, self.n_bits):
                    # If bucket_to_code_original has wrong shape, generate identity codes
                    from gray_tunneled_hashing.data.synthetic_generators import generate_hypercube_vertices
                    vertices = generate_hypercube_vertices(self.n_bits)
                    # Use first K vertices as identity codes
                    pi_init = vertices[:K_actual].copy().astype(np.uint8)
            
            # Store initial permutation before optimization
            self.pi_init_ = pi_init.copy()
            
            # NEW (Sprint 8): Use real embeddings objective if requested and parameters provided
            if use_real_embeddings_objective and queries is not None and base_embeddings is not None and \
               ground_truth_neighbors is not None and encoder is not None and code_to_bucket is not None:
                # Use new objective over real embeddings
                from gray_tunneled_hashing.distribution.j_phi_objective import (
                    hill_climb_j_phi_real_embeddings,
                )
                
                if optimization_method == "hill_climb":
                    pi_optimized, cost, initial_cost, cost_history = hill_climb_j_phi_real_embeddings(
                        pi_init=pi_init,
                        pi=pi,
                        w=w,
                        queries=queries,
                        base_embeddings=base_embeddings,
                        ground_truth_neighbors=ground_truth_neighbors,
                        encoder=encoder,
                        code_to_bucket=code_to_bucket,
                        n_bits=self.n_bits,
                        max_iter=self.max_two_swap_iters,
                        sample_size=self.two_swap_sample_size,
                        random_state=self.random_state,
                        sample_size_pairs=sample_size_pairs,
                        verbose=self.track_history,
                    )
                else:
                    # For now, only hill_climb is supported with real embeddings objective
                    # TODO: Implement SA and memetic with real embeddings
                    raise ValueError(
                        f"optimization_method '{optimization_method}' not yet supported "
                        f"with use_real_embeddings_objective=True. Use 'hill_climb'."
                    )
                
                # Store results
                self.pi_ = pi_optimized
                self.cost_ = cost
                self.initial_cost_ = initial_cost
                if self.track_history:
                    self.cost_history_ = [
                        {"cost": c, "step": "j_phi_real_optimization", "iteration": i, "timestamp": time.time()}
                        for i, c in enumerate(cost_history)
                    ]
                self.is_fitted = True
                
                return self
            
            # Choose optimization method (legacy: old objective)
            if use_cosine_objective:
                # Use cosine-based objective
                from gray_tunneled_hashing.distribution.cosine_objective import (
                    compute_j_phi_cosine_cost,
                )
                from gray_tunneled_hashing.algorithms.simulated_annealing import (
                    simulated_annealing_j_phi,
                    memetic_algorithm_j_phi,
                )
                
                if optimization_method == "simulated_annealing":
                    pi_optimized, cost, initial_cost, cost_history = simulated_annealing_j_phi(
                        pi_init=pi_init,
                        pi=pi,
                        w=w,
                        bucket_to_code=bucket_to_code_original,
                        n_bits=self.n_bits,
                        initial_temperature=1000.0,
                        cooling_rate=0.95,
                        min_temperature=0.01,
                        max_iter=self.max_two_swap_iters,
                        sample_size=self.two_swap_sample_size,
                        use_block_tunneling=(self.mode == "full"),
                        block_size=self.block_size,
                        num_tunneling_steps=self.num_tunneling_steps,
                        random_state=self.random_state,
                        bucket_to_embedding_idx=None,
                        semantic_distances=None,  # Not used with cosine objective
                        semantic_weight=0.0,
                        bucket_embeddings=bucket_embeddings,
                        use_cosine_objective=True,
                        cosine_weight=cosine_weight,
                        hamming_weight=hamming_weight,
                        distance_metric=distance_metric,
                        enable_logging=self.track_history,
                    )
                elif optimization_method == "memetic":
                    pi_optimized, cost, initial_cost, cost_history = memetic_algorithm_j_phi(
                        pi_init=pi_init,
                        pi=pi,
                        w=w,
                        bucket_to_code=bucket_to_code_original,
                        n_bits=self.n_bits,
                        population_size=10,
                        num_generations=max(1, self.max_two_swap_iters // 20),
                        sa_iterations=20,
                        local_search_iterations=10,
                        random_state=self.random_state,
                        bucket_to_embedding_idx=None,
                        semantic_distances=None,
                        semantic_weight=0.0,
                        bucket_embeddings=bucket_embeddings,
                        enable_logging=self.track_history,
                    )
                else:
                    # Fallback to hill climbing with cosine objective
                    # TODO: Implement hill_climb with cosine objective
                    pi_optimized, cost, initial_cost, cost_history = hill_climb_j_phi(
                        pi_init=pi_init,
                        pi=pi,
                        w=w,
                        bucket_to_code=bucket_to_code_original,
                        n_bits=self.n_bits,
                        max_iter=self.max_two_swap_iters,
                        sample_size=self.two_swap_sample_size,
                        random_state=self.random_state,
                        bucket_to_embedding_idx=None,
                        semantic_distances=semantic_distances,
                        semantic_weight=semantic_weight,
                    )
            else:
                # Use standard J(φ) objective
                if optimization_method == "simulated_annealing":
                    from gray_tunneled_hashing.algorithms.simulated_annealing import (
                        simulated_annealing_j_phi,
                    )
                    pi_optimized, cost, initial_cost, cost_history = simulated_annealing_j_phi(
                        pi_init=pi_init,
                        pi=pi,
                        w=w,
                        bucket_to_code=bucket_to_code_original,
                        n_bits=self.n_bits,
                        initial_temperature=1000.0,
                        cooling_rate=0.95,
                        min_temperature=0.01,
                        max_iter=self.max_two_swap_iters,
                        sample_size=self.two_swap_sample_size,
                        use_block_tunneling=(self.mode == "full"),
                        block_size=self.block_size,
                        num_tunneling_steps=self.num_tunneling_steps,
                        random_state=self.random_state,
                        bucket_to_embedding_idx=None,
                        semantic_distances=semantic_distances,
                        semantic_weight=semantic_weight,
                        enable_logging=self.track_history,
                    )
                elif optimization_method == "memetic":
                    from gray_tunneled_hashing.algorithms.simulated_annealing import (
                        memetic_algorithm_j_phi,
                    )
                    pi_optimized, cost, initial_cost, cost_history = memetic_algorithm_j_phi(
                        pi_init=pi_init,
                        pi=pi,
                        w=w,
                        bucket_to_code=bucket_to_code_original,
                        n_bits=self.n_bits,
                        population_size=10,
                        num_generations=max(1, self.max_two_swap_iters // 20),
                        sa_iterations=20,
                        local_search_iterations=10,
                        random_state=self.random_state,
                        bucket_to_embedding_idx=None,
                        semantic_distances=semantic_distances,
                        semantic_weight=semantic_weight,
                        bucket_embeddings=None,  # Not used with standard J(φ)
                        enable_logging=self.track_history,
                    )
                else:
                    # Default: hill climbing
                    pi_optimized, cost, initial_cost, cost_history = hill_climb_j_phi(
                        pi_init=pi_init,
                        pi=pi,  # Original K
                        w=w,    # Original K x K
                        bucket_to_code=bucket_to_code_original,  # Original K
                        n_bits=self.n_bits,
                        max_iter=self.max_two_swap_iters,
                        sample_size=self.two_swap_sample_size,
                        random_state=self.random_state,
                        bucket_to_embedding_idx=None,  # bucket i maps to embedding i
                        semantic_distances=semantic_distances,
                        semantic_weight=semantic_weight,
                    )
            
            # Store results
            self.pi_ = pi_optimized
            self.cost_ = cost
            self.initial_cost_ = initial_cost  # J(φ₀)
            if self.track_history:
                self.cost_history_ = [
                    {"cost": c, "step": "j_phi_optimization", "iteration": i, "timestamp": time.time()}
                    for i, c in enumerate(cost_history)
                ]
            self.is_fitted = True
            
            return self
        else:
            # Legacy QAP optimization (may not guarantee J(φ*) ≤ J(φ₀))
            from gray_tunneled_hashing.distribution.traffic_stats import build_weighted_distance_matrix
            
            # Build weighted distance matrix
            D_weighted = build_weighted_distance_matrix(
                pi=pi,
                w=w,
                bucket_embeddings=bucket_embeddings,
                use_semantic_distances=use_semantic_distances,
            )
            
            # Pad to 2**n_bits if needed
            if K < self.N:
                D_padded = np.zeros((self.N, self.N), dtype=np.float64)
                D_padded[:K, :K] = D_weighted
                dummy_weight = np.mean(D_weighted[D_weighted > 0]) * 0.01 if np.any(D_weighted > 0) else 1e-6
                D_padded[K:, :] = dummy_weight
                D_padded[:, K:] = dummy_weight
                D_padded[K:, K:] = 0
                
                embeddings_padded = np.zeros((self.N, bucket_embeddings.shape[1]), dtype=bucket_embeddings.dtype)
                embeddings_padded[:K] = bucket_embeddings
                embeddings_padded[K:] = bucket_embeddings[-1:]
                D_weighted = D_padded
                bucket_embeddings = embeddings_padded
            elif K > self.N:
                top_indices = np.argsort(pi)[-self.N:]
                D_weighted = D_weighted[np.ix_(top_indices, top_indices)]
                bucket_embeddings = bucket_embeddings[top_indices]
            
            return self.fit(embeddings=bucket_embeddings, D=D_weighted)

    def get_initial_permutation(self) -> Optional[np.ndarray]:
        """
        Get the initial permutation used for optimization.
        
        Returns:
            Initial permutation array of shape (N,) or None if not fitted
        """
        if hasattr(self, 'pi_init_'):
            return self.pi_init_.copy()
        return None
    
    def get_assignment(self) -> np.ndarray:
        """
        Get the final permutation (assignment of buckets to binary codes).
        
        Returns:
            Permutation array of shape (K, n_bits) where K is the number of buckets.
            pi_[bucket_idx] is the new binary code assigned to bucket bucket_idx.
            
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
