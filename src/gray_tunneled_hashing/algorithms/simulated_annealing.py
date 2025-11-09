"""
Simulated Annealing for QAP optimization with 2-swap and block operators.

This implements a memetic algorithm combining:
- Simulated annealing (global search)
- Hill climbing 2-swap (local search)
- Block tunneling operators (local search)
"""

import numpy as np
from typing import Optional, Tuple, Callable, List, Dict, Any
import math
import time

from gray_tunneled_hashing.distribution.j_phi_objective import (
    compute_j_phi_cost,
    compute_j_phi_cost_delta_swap,
)
from gray_tunneled_hashing.algorithms.block_moves import tunneling_step


def simulated_annealing_j_phi(
    pi_init: np.ndarray,
    pi: np.ndarray,
    w: np.ndarray,
    bucket_to_code: np.ndarray,
    n_bits: int,
    initial_temperature: float = 1000.0,
    cooling_rate: float = 0.95,
    min_temperature: float = 0.01,
    max_iter: int = 1000,
    sample_size: int = 256,
    use_block_tunneling: bool = True,
    block_size: int = 4,
    num_tunneling_steps: int = 5,
    tunneling_frequency: int = 10,  # Apply tunneling every N iterations
    random_state: Optional[int] = None,
    bucket_to_embedding_idx: Optional[np.ndarray] = None,
    semantic_distances: Optional[np.ndarray] = None,
    semantic_weight: float = 0.0,
    bucket_embeddings: Optional[np.ndarray] = None,
    use_cosine_objective: bool = False,
    cosine_weight: float = 1.0,
    hamming_weight: float = 1.0,
    distance_metric: str = "cosine",
    enable_logging: bool = False,
) -> Tuple[np.ndarray, float, float, List[Dict[str, Any]]]:
    """
    Simulated annealing to minimize J(φ) with 2-swap moves and block tunneling.
    
    This is a memetic algorithm combining:
    - Simulated annealing (global meta-heuristic)
    - Hill climbing 2-swap (local search)
    - Block tunneling (local search for escaping local minima)
    
    Args:
        pi_init: Initial permutation of shape (N,) where N = 2**n_bits
        pi: Query prior of shape (K,) where K <= N
        w: Neighbor weights of shape (K, K)
        bucket_to_code: Original bucket codes of shape (K, n_bits)
        n_bits: Number of bits
        initial_temperature: Starting temperature for SA
        cooling_rate: Temperature decay rate (0 < cooling_rate < 1)
        min_temperature: Minimum temperature to stop
        max_iter: Maximum iterations
        sample_size: Number of swaps to sample per iteration
        use_block_tunneling: If True, apply block tunneling periodically
        block_size: Size of blocks for tunneling
        num_tunneling_steps: Number of tunneling steps per application
        tunneling_frequency: Apply tunneling every N iterations
        random_state: Random seed
        bucket_to_embedding_idx: Optional bucket to embedding mapping
        semantic_distances: Optional semantic distance matrix of shape (K, K)
        semantic_weight: Weight for semantic term in J(φ)
        enable_logging: If True, return detailed logging information
        
    Returns:
        (best_permutation, best_cost, initial_cost, history)
        where history is a list of dicts with iteration details
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    N = len(pi_init)
    perm = pi_init.copy()
    best_perm = perm.copy()
    
    # Compute initial cost
    if use_cosine_objective and bucket_embeddings is not None:
        from gray_tunneled_hashing.distribution.cosine_objective import (
            compute_j_phi_cosine_cost,
        )
        initial_cost = compute_j_phi_cosine_cost(
            perm, pi, w, bucket_to_code, bucket_embeddings, n_bits,
            bucket_to_embedding_idx, cosine_weight, hamming_weight, distance_metric
        )
    else:
        initial_cost = compute_j_phi_cost(
            perm, pi, w, bucket_to_code, n_bits, bucket_to_embedding_idx,
            semantic_distances, semantic_weight
        )
    cost = initial_cost
    best_cost = initial_cost
    
    # Cost computation function
    def compute_cost(permutation):
        if use_cosine_objective and bucket_embeddings is not None:
            from gray_tunneled_hashing.distribution.cosine_objective import (
                compute_j_phi_cosine_cost,
            )
            return compute_j_phi_cosine_cost(
                permutation, pi, w, bucket_to_code, bucket_embeddings, n_bits,
                bucket_to_embedding_idx, cosine_weight, hamming_weight, distance_metric
            )
        else:
            return compute_j_phi_cost(
                permutation, pi, w, bucket_to_code, n_bits, bucket_to_embedding_idx,
                semantic_distances, semantic_weight
            )
    
    # Get K to validate swaps
    K = len(pi)
    if bucket_to_embedding_idx is None:
        max_valid_embedding_idx = K - 1
    else:
        max_valid_embedding_idx = bucket_to_embedding_idx.max() if len(bucket_to_embedding_idx) > 0 else K - 1
    
    # Initialize temperature
    temperature = initial_temperature
    
    # History tracking
    history = []
    n_acceptances = 0
    n_rejections = 0
    n_improvements = 0
    
    last_print_time = time.time()
    print_interval = 10.0  # Print every 10 seconds
    
    for iteration in range(max_iter):
        iter_start_time = time.time()
        iteration_start_time = time.time()
        
        # Sample random 2-swaps
        candidates = []
        for _ in range(sample_size):
            u, v = np.random.choice(N, size=2, replace=False)
            candidates.append((u, v))
        
        # Evaluate all candidates
        best_delta = float('inf')
        best_swap = None
        
        for u, v in candidates:
            # Check if swap maintains validity constraint
            new_u_val = perm[v]
            new_v_val = perm[u]
            
            if new_u_val > max_valid_embedding_idx or new_v_val > max_valid_embedding_idx:
                continue
            
            # Compute delta efficiently
            if use_cosine_objective and bucket_embeddings is not None:
                from gray_tunneled_hashing.distribution.cosine_objective import (
                    compute_j_phi_cosine_cost_delta_swap,
                )
                delta = compute_j_phi_cosine_cost_delta_swap(
                    perm, pi, w, bucket_to_code, bucket_embeddings, n_bits, u, v,
                    bucket_to_embedding_idx, cosine_weight, hamming_weight, distance_metric
                )
            else:
                delta = compute_j_phi_cost_delta_swap(
                    perm, pi, w, bucket_to_code, n_bits, u, v, bucket_to_embedding_idx,
                    semantic_distances, semantic_weight
                )
            if delta < best_delta:
                best_delta = delta
                best_swap = (u, v)
        
        # Apply move (accept or reject based on SA criterion)
        accepted = False
        if best_swap is not None:
            if best_delta < 0:
                # Always accept improving moves
                u, v = best_swap
                temp_u, temp_v = perm[v], perm[u]
                if temp_u <= max_valid_embedding_idx and temp_v <= max_valid_embedding_idx:
                    perm[u], perm[v] = temp_u, temp_v
                    # Recompute cost to ensure accuracy
                    cost = compute_cost(perm)
                    accepted = True
                    n_improvements += 1
                    n_acceptances += 1
                    
                    # Update best if improved
                    if cost < best_cost:
                        best_cost = cost
                        best_perm = perm.copy()
            else:
                # Accept worsening moves with probability exp(-delta / T)
                acceptance_prob = math.exp(-best_delta / temperature)
                if np.random.random() < acceptance_prob:
                    u, v = best_swap
                    temp_u, temp_v = perm[v], perm[u]
                    if temp_u <= max_valid_embedding_idx and temp_v <= max_valid_embedding_idx:
                        perm[u], perm[v] = temp_u, temp_v
                        # Recompute cost to ensure accuracy
                        cost = compute_cost(perm)
                        accepted = True
                        n_acceptances += 1
                    else:
                        n_rejections += 1
                else:
                    n_rejections += 1
        
        # Apply block tunneling periodically (local search)
        tunneling_improvement = 0.0
        if use_block_tunneling and (iteration + 1) % tunneling_frequency == 0:
            # Note: Block tunneling currently uses QAP cost, not J(φ)
            # For now, we'll skip it or implement J(φ)-aware tunneling
            # TODO: Implement J(φ)-aware block tunneling
            pass
        
        # Cool down temperature
        temperature = max(min_temperature, temperature * cooling_rate)
        
        # Record history
        if enable_logging:
            history.append({
                "iteration": iteration,
                "cost": cost,
                "best_cost": best_cost,
                "temperature": temperature,
                "accepted": accepted,
                "delta": best_delta if best_swap is not None else 0.0,
                "acceptance_rate": n_acceptances / (n_acceptances + n_rejections + 1e-10),
                "tunneling_improvement": tunneling_improvement,
                "time": time.time() - iteration_start_time,
            })
        
        # Print progress periodically
        current_time = time.time()
        if enable_logging or (current_time - last_print_time >= print_interval):
            elapsed = current_time - last_print_time
            improvement = ((initial_cost - best_cost) / initial_cost * 100) if initial_cost > 0 else 0
            acceptance_rate = n_acceptances / (n_acceptances + n_rejections + 1e-10)
            print(f"  [SA Iter {iteration:3d}] cost={best_cost:.4f} ({improvement:.1f}% improvement), "
                  f"T={temperature:.2f}, accept_rate={acceptance_rate:.2f}, "
                  f"delta={best_delta:.4f if best_swap else 0:.4f}, time={elapsed:.1f}s")
            last_print_time = current_time
        
        # Early stopping if temperature too low and no recent improvements
        if temperature <= min_temperature and iteration > 100:
            if n_improvements == 0:
                if enable_logging:
                    print(f"  Temperature too low ({temperature:.4f}) and no improvements, stopping.")
                break
    
    return best_perm, best_cost, initial_cost, history


def memetic_algorithm_j_phi(
    pi_init: np.ndarray,
    pi: np.ndarray,
    w: np.ndarray,
    bucket_to_code: np.ndarray,
    n_bits: int,
    population_size: int = 10,
    num_generations: int = 50,
    sa_iterations: int = 100,
    local_search_iterations: int = 50,
    crossover_rate: float = 0.7,
    mutation_rate: float = 0.1,
    elite_size: int = 2,
    random_state: Optional[int] = None,
    bucket_to_embedding_idx: Optional[np.ndarray] = None,
    semantic_distances: Optional[np.ndarray] = None,
    semantic_weight: float = 0.0,
    bucket_embeddings: Optional[np.ndarray] = None,
    enable_logging: bool = False,
) -> Tuple[np.ndarray, float, float, List[Dict[str, Any]]]:
    """
    Memetic algorithm combining simulated annealing with genetic operators.
    
    This algorithm:
    1. Maintains a population of permutations
    2. Applies simulated annealing to each individual (local search)
    3. Uses crossover and mutation (genetic operators)
    4. Applies hill climbing 2-swap to offspring (local search)
    5. Selects best individuals for next generation
    
    Args:
        pi_init: Initial permutation of shape (N,)
        pi: Query prior of shape (K,)
        w: Neighbor weights of shape (K, K)
        bucket_to_code: Original bucket codes of shape (K, n_bits)
        n_bits: Number of bits
        population_size: Number of individuals in population
        num_generations: Number of generations
        sa_iterations: Simulated annealing iterations per individual
        local_search_iterations: Hill climbing iterations for offspring
        crossover_rate: Probability of crossover
        mutation_rate: Probability of mutation
        elite_size: Number of elite individuals to preserve
        random_state: Random seed
        bucket_to_embedding_idx: Optional bucket to embedding mapping
        semantic_distances: Optional semantic distance matrix
        semantic_weight: Weight for semantic term
        enable_logging: If True, return detailed logging
        
    Returns:
        (best_permutation, best_cost, initial_cost, history)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    N = len(pi_init)
    K = len(pi)
    
    # Initialize population
    population = []
    population_costs = []
    
    for i in range(population_size):
        if i == 0:
            # First individual: use provided initial permutation
            individual = pi_init.copy()
        else:
            # Other individuals: random valid permutation
            base_perm = np.random.permutation(K).astype(np.int32)
            individual = np.array([base_perm[j % K] for j in range(N)], dtype=np.int32)
            np.random.shuffle(individual)
        
        cost = compute_j_phi_cost(
            individual, pi, w, bucket_to_code, n_bits, bucket_to_embedding_idx,
            semantic_distances, semantic_weight
        )
        population.append(individual)
        population_costs.append(cost)
    
    initial_cost = min(population_costs)
    best_cost = initial_cost
    best_perm = population[np.argmin(population_costs)].copy()
    
    history = []
    
    for generation in range(num_generations):
        generation_start_time = time.time()
        
        # Apply local search (simulated annealing) to each individual
        improved_population = []
        improved_costs = []
        
        for individual, cost in zip(population, population_costs):
            # Short SA run for local improvement
            # Note: simulated_annealing_j_phi doesn't support bucket_embeddings yet
            # For now, use standard J(φ) optimization
            improved_perm, improved_cost, _, _ = simulated_annealing_j_phi(
                individual, pi, w, bucket_to_code, n_bits,
                initial_temperature=100.0,
                cooling_rate=0.9,
                min_temperature=0.1,
                max_iter=sa_iterations,
                sample_size=128,
                use_block_tunneling=False,  # Skip tunneling in memetic for speed
                random_state=None,
                bucket_to_embedding_idx=bucket_to_embedding_idx,
                semantic_distances=semantic_distances,
                semantic_weight=semantic_weight,
                enable_logging=False,
            )
            improved_population.append(improved_perm)
            improved_costs.append(improved_cost)
        
        # Update best
        gen_best_idx = np.argmin(improved_costs)
        if improved_costs[gen_best_idx] < best_cost:
            best_cost = improved_costs[gen_best_idx]
            best_perm = improved_population[gen_best_idx].copy()
        
        # Selection: keep elite + select rest based on fitness
        elite_indices = np.argsort(improved_costs)[:elite_size]
        elite_population = [improved_population[i] for i in elite_indices]
        elite_costs = [improved_costs[i] for i in elite_indices]
        
        # Create new population through crossover and mutation
        new_population = elite_population.copy()
        new_costs = elite_costs.copy()
        
        while len(new_population) < population_size:
            # Select parents (tournament selection)
            parent1_idx = np.random.choice(len(improved_population))
            parent2_idx = np.random.choice(len(improved_population))
            parent1 = improved_population[parent1_idx]
            parent2 = improved_population[parent2_idx]
            
            # Crossover
            if np.random.random() < crossover_rate:
                # Order crossover (OX) for permutations
                child = _order_crossover(parent1, parent2, K)
            else:
                child = parent1.copy()
            
            # Mutation
            if np.random.random() < mutation_rate:
                # Random 2-swap mutation
                u, v = np.random.choice(N, size=2, replace=False)
                child[u], child[v] = child[v], child[u]
            
            # Local search on offspring (hill climbing)
            child = _hill_climb_local_search(
                child, pi, w, bucket_to_code, n_bits,
                max_iter=local_search_iterations,
                bucket_to_embedding_idx=bucket_to_embedding_idx,
                semantic_distances=semantic_distances,
                semantic_weight=semantic_weight,
            )
            
            child_cost = compute_j_phi_cost(
                child, pi, w, bucket_to_code, n_bits, bucket_to_embedding_idx,
                semantic_distances, semantic_weight
            )
            
            new_population.append(child)
            new_costs.append(child_cost)
        
        population = new_population
        population_costs = new_costs
        
        # Record history
        if enable_logging:
            history.append({
                "generation": generation,
                "best_cost": best_cost,
                "avg_cost": np.mean(population_costs),
                "std_cost": np.std(population_costs),
                "time": time.time() - generation_start_time,
            })
    
    return best_perm, best_cost, initial_cost, history


def _order_crossover(parent1: np.ndarray, parent2: np.ndarray, K: int) -> np.ndarray:
    """Order crossover (OX) for permutation."""
    N = len(parent1)
    # Select random segment from parent1
    start = np.random.randint(0, N)
    end = np.random.randint(start, N + 1)
    
    child = np.zeros(N, dtype=np.int32)
    child[start:end] = parent1[start:end]
    
    # Fill remaining positions from parent2
    parent2_values = []
    for val in parent2:
        if val not in child[start:end]:
            parent2_values.append(val)
    
    idx = 0
    for i in range(N):
        if i < start or i >= end:
            child[i] = parent2_values[idx]
            idx += 1
    
    return child


def _hill_climb_local_search(
    perm: np.ndarray,
    pi: np.ndarray,
    w: np.ndarray,
    bucket_to_code: np.ndarray,
    n_bits: int,
    max_iter: int = 50,
    sample_size: int = 128,
    bucket_to_embedding_idx: Optional[np.ndarray] = None,
    semantic_distances: Optional[np.ndarray] = None,
    semantic_weight: float = 0.0,
) -> np.ndarray:
    """Quick hill climbing local search."""
    N = len(perm)
    K = len(pi)
    
    if bucket_to_embedding_idx is None:
        max_valid_embedding_idx = K - 1
    else:
        max_valid_embedding_idx = bucket_to_embedding_idx.max() if len(bucket_to_embedding_idx) > 0 else K - 1
    
    for _ in range(max_iter):
        # Sample random swaps
        candidates = []
        for _ in range(sample_size):
            u, v = np.random.choice(N, size=2, replace=False)
            candidates.append((u, v))
        
        best_delta = 0.0
        best_swap = None
        
        for u, v in candidates:
            new_u_val = perm[v]
            new_v_val = perm[u]
            
            if new_u_val > max_valid_embedding_idx or new_v_val > max_valid_embedding_idx:
                continue
            
            delta = compute_j_phi_cost_delta_swap(
                perm, pi, w, bucket_to_code, n_bits, u, v, bucket_to_embedding_idx,
                semantic_distances, semantic_weight
            )
            if delta < best_delta:
                best_delta = delta
                best_swap = (u, v)
        
        if best_swap is not None:
            u, v = best_swap
            perm[u], perm[v] = perm[v], perm[u]
        else:
            break
    
    return perm

