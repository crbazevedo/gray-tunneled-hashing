"""
Testes comparativos de recall para Sprint 8.

Compara GTH com baselines (LSH sem GTH, Random Projection) e testa
diferentes configurações (raios Hamming, n_bits, n_codes).
"""

import numpy as np
import pytest
from sklearn.metrics.pairwise import euclidean_distances

from gray_tunneled_hashing.binary.lsh_families import create_lsh_family
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.api.query_pipeline import query_with_hamming_ball
from gray_tunneled_hashing.evaluation.metrics import recall_at_k


def compute_baseline_recall_lsh(
    queries: np.ndarray,
    base_embeddings: np.ndarray,
    ground_truth: np.ndarray,
    lsh_encoder,
    k: int,
    hamming_radius: int = 1,
) -> float:
    """Compute baseline recall using LSH without GTH."""
    query_codes = lsh_encoder(queries)
    base_codes = lsh_encoder(base_embeddings)
    
    recalls = []
    for i in range(len(queries)):
        query_code = query_codes[i]
        # Find all base codes within Hamming radius
        hamming_dists = np.sum(query_code != base_codes, axis=1)
        candidates = np.where(hamming_dists <= hamming_radius)[0]
        
        # Compute recall
        if len(candidates) > 0:
            retrieved = set(candidates[:k])
            relevant = set(ground_truth[i][:k])
            recall = len(retrieved & relevant) / len(relevant) if len(relevant) > 0 else 0.0
            recalls.append(recall)
        else:
            recalls.append(0.0)
    
    return np.mean(recalls) if recalls else 0.0


def compute_gth_recall(
    queries: np.ndarray,
    base_embeddings: np.ndarray,
    ground_truth: np.ndarray,
    index_obj,
    lsh_encoder,
    k: int,
    hamming_radius: int = 1,
) -> float:
    """Compute GTH recall using optimized permutation."""
    query_codes = lsh_encoder(queries)
    base_codes = lsh_encoder(base_embeddings)
    
    # Build bucket_to_dataset_indices mapping
    bucket_to_dataset_indices = {}
    for dataset_idx, code in enumerate(base_codes):
        code_tuple = tuple(code.astype(int).tolist())
        if code_tuple in index_obj.code_to_bucket:
            bucket_idx = index_obj.code_to_bucket[code_tuple]
            if bucket_idx not in bucket_to_dataset_indices:
                bucket_to_dataset_indices[bucket_idx] = []
            bucket_to_dataset_indices[bucket_idx].append(dataset_idx)
    
    # Get permutation
    permutation = index_obj.hasher.get_assignment()
    
    # Compute GTH recall
    recalls = []
    for i in range(len(queries)):
        query_code = query_codes[i].astype(bool)
        
        result = query_with_hamming_ball(
            query_code=query_code,
            permutation=permutation,
            code_to_bucket=index_obj.code_to_bucket,
            bucket_to_code=index_obj.bucket_to_code,
            n_bits=index_obj.n_bits,
            hamming_radius=hamming_radius,
        )
        
        # Get dataset indices from bucket indices
        retrieved_indices = []
        for bucket_idx in result.candidate_indices:
            if bucket_idx in bucket_to_dataset_indices:
                retrieved_indices.extend(bucket_to_dataset_indices[bucket_idx])
        
        # Compute recall
        retrieved = set(retrieved_indices[:k])
        relevant = set(ground_truth[i][:k])
        recall = len(retrieved & relevant) / len(relevant) if len(relevant) > 0 else 0.0
        recalls.append(recall)
    
    return np.mean(recalls) if recalls else 0.0


def test_recall_vs_baseline_hyperplane_lsh():
    """Compare GTH recall with baseline Hyperplane LSH."""
    n_bits = 6
    dim = 16
    N = 100
    Q = 20
    k = 5
    hamming_radius = 1
    
    # Generate synthetic data
    np.random.seed(42)
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    
    # Compute ground truth
    distances = euclidean_distances(queries, base_embeddings)
    ground_truth = np.argsort(distances, axis=1)[:, :k]
    
    # Create LSH encoder
    lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=dim, random_state=42)
    
    # Baseline recall
    baseline_recall = compute_baseline_recall_lsh(
        queries=queries,
        base_embeddings=base_embeddings,
        ground_truth=ground_truth,
        lsh_encoder=lsh.hash,
        k=k,
        hamming_radius=hamming_radius,
    )
    
    # GTH recall
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        n_bits=n_bits,
        n_codes=32,
        use_codebook=True,
        max_two_swap_iters=10,
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=42,
    )
    
    gth_recall = compute_gth_recall(
        queries=queries,
        base_embeddings=base_embeddings,
        ground_truth=ground_truth,
        index_obj=index_obj,
        lsh_encoder=lsh.hash,
        k=k,
        hamming_radius=hamming_radius,
    )
    
    # GTH should not be significantly worse than baseline
    # (allowing small tolerance for randomness)
    assert gth_recall >= baseline_recall - 0.1, \
        f"GTH recall ({gth_recall:.4f}) should not be much worse than baseline ({baseline_recall:.4f})"
    
    print(f"Baseline recall: {baseline_recall:.4f}, GTH recall: {gth_recall:.4f}")


def test_recall_vs_baseline_random_projection():
    """Compare GTH recall with baseline p-stable LSH (alternative to hyperplane)."""
    n_bits = 6
    dim = 16
    N = 100
    Q = 20
    k = 5
    hamming_radius = 1
    
    # Generate synthetic data
    np.random.seed(42)
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    
    # Compute ground truth
    distances = euclidean_distances(queries, base_embeddings)
    ground_truth = np.argsort(distances, axis=1)[:, :k]
    
    # Create p-stable LSH encoder (alternative to hyperplane)
    lsh = create_lsh_family("p_stable", n_bits=n_bits, dim=dim, random_state=42)
    
    # Baseline recall
    baseline_recall = compute_baseline_recall_lsh(
        queries=queries,
        base_embeddings=base_embeddings,
        ground_truth=ground_truth,
        lsh_encoder=lsh.hash,
        k=k,
        hamming_radius=hamming_radius,
    )
    
    # GTH recall
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        n_bits=n_bits,
        n_codes=32,
        use_codebook=True,
        max_two_swap_iters=10,
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=42,
    )
    
    gth_recall = compute_gth_recall(
        queries=queries,
        base_embeddings=base_embeddings,
        ground_truth=ground_truth,
        index_obj=index_obj,
        lsh_encoder=lsh.hash,
        k=k,
        hamming_radius=hamming_radius,
    )
    
    # GTH should not be significantly worse than baseline
    assert gth_recall >= baseline_recall - 0.1, \
        f"GTH recall ({gth_recall:.4f}) should not be much worse than baseline ({baseline_recall:.4f})"
    
    print(f"Baseline (p-stable) recall: {baseline_recall:.4f}, GTH recall: {gth_recall:.4f}")


@pytest.mark.parametrize("hamming_radius", [1, 2, 3])
def test_recall_different_hamming_radii(hamming_radius):
    """Test recall with different Hamming radii."""
    n_bits = 6
    dim = 16
    N = 100
    Q = 20
    k = 5
    
    # Generate synthetic data
    np.random.seed(42)
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    
    # Compute ground truth
    distances = euclidean_distances(queries, base_embeddings)
    ground_truth = np.argsort(distances, axis=1)[:, :k]
    
    # Create LSH encoder
    lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=dim, random_state=42)
    
    # Baseline recall
    baseline_recall = compute_baseline_recall_lsh(
        queries=queries,
        base_embeddings=base_embeddings,
        ground_truth=ground_truth,
        lsh_encoder=lsh.hash,
        k=k,
        hamming_radius=hamming_radius,
    )
    
    # GTH recall
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        n_bits=n_bits,
        n_codes=32,
        use_codebook=True,
        max_two_swap_iters=10,
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=42,
    )
    
    gth_recall = compute_gth_recall(
        queries=queries,
        base_embeddings=base_embeddings,
        ground_truth=ground_truth,
        index_obj=index_obj,
        lsh_encoder=lsh.hash,
        k=k,
        hamming_radius=hamming_radius,
    )
    
    # Both should be valid recall values
    assert 0.0 <= baseline_recall <= 1.0
    assert 0.0 <= gth_recall <= 1.0
    
    # Larger radius should generally give better recall (or at least not worse)
    # (This is a sanity check, not a strict requirement)
    print(f"Radius {hamming_radius}: Baseline={baseline_recall:.4f}, GTH={gth_recall:.4f}")


@pytest.mark.parametrize("n_bits", [4, 6, 8])
def test_recall_different_n_bits(n_bits):
    """Test recall with different n_bits."""
    dim = 16
    N = 100
    Q = 20
    k = 5
    hamming_radius = 1
    
    # Generate synthetic data
    np.random.seed(42)
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    
    # Compute ground truth
    distances = euclidean_distances(queries, base_embeddings)
    ground_truth = np.argsort(distances, axis=1)[:, :k]
    
    # Create LSH encoder
    lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=dim, random_state=42)
    
    # Baseline recall
    baseline_recall = compute_baseline_recall_lsh(
        queries=queries,
        base_embeddings=base_embeddings,
        ground_truth=ground_truth,
        lsh_encoder=lsh.hash,
        k=k,
        hamming_radius=hamming_radius,
    )
    
    # GTH recall
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        n_bits=n_bits,
        n_codes=min(32, 2**n_bits),  # Don't exceed number of possible codes
        use_codebook=True,
        max_two_swap_iters=10,
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=42,
    )
    
    gth_recall = compute_gth_recall(
        queries=queries,
        base_embeddings=base_embeddings,
        ground_truth=ground_truth,
        index_obj=index_obj,
        lsh_encoder=lsh.hash,
        k=k,
        hamming_radius=hamming_radius,
    )
    
    # Both should be valid recall values
    assert 0.0 <= baseline_recall <= 1.0
    assert 0.0 <= gth_recall <= 1.0
    
    # GTH should not be significantly worse
    assert gth_recall >= baseline_recall - 0.15, \
        f"GTH recall ({gth_recall:.4f}) should not be much worse than baseline ({baseline_recall:.4f}) for n_bits={n_bits}"
    
    print(f"n_bits={n_bits}: Baseline={baseline_recall:.4f}, GTH={gth_recall:.4f}")


@pytest.mark.parametrize("n_codes", [16, 32, 64])
def test_recall_different_n_codes(n_codes):
    """Test recall with different n_codes."""
    n_bits = 6
    dim = 16
    N = 100
    Q = 20
    k = 5
    hamming_radius = 1
    
    # Generate synthetic data
    np.random.seed(42)
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    
    # Compute ground truth
    distances = euclidean_distances(queries, base_embeddings)
    ground_truth = np.argsort(distances, axis=1)[:, :k]
    
    # Create LSH encoder
    lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=dim, random_state=42)
    
    # Baseline recall (independent of n_codes)
    baseline_recall = compute_baseline_recall_lsh(
        queries=queries,
        base_embeddings=base_embeddings,
        ground_truth=ground_truth,
        lsh_encoder=lsh.hash,
        k=k,
        hamming_radius=hamming_radius,
    )
    
    # GTH recall with different n_codes
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        n_bits=n_bits,
        n_codes=n_codes,
        use_codebook=True,
        max_two_swap_iters=10,
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=42,
    )
    
    gth_recall = compute_gth_recall(
        queries=queries,
        base_embeddings=base_embeddings,
        ground_truth=ground_truth,
        index_obj=index_obj,
        lsh_encoder=lsh.hash,
        k=k,
        hamming_radius=hamming_radius,
    )
    
    # Both should be valid recall values
    assert 0.0 <= baseline_recall <= 1.0
    assert 0.0 <= gth_recall <= 1.0
    
    # GTH should not be significantly worse
    assert gth_recall >= baseline_recall - 0.1, \
        f"GTH recall ({gth_recall:.4f}) should not be much worse than baseline ({baseline_recall:.4f}) for n_codes={n_codes}"
    
    print(f"n_codes={n_codes}: Baseline={baseline_recall:.4f}, GTH={gth_recall:.4f}")


def test_recall_improvement_after_optimization():
    """Test that recall improves (or at least doesn't worsen) after optimization."""
    n_bits = 6
    dim = 16
    N = 100
    Q = 20
    k = 5
    hamming_radius = 1
    
    # Generate synthetic data
    np.random.seed(42)
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    
    # Compute ground truth
    distances = euclidean_distances(queries, base_embeddings)
    ground_truth = np.argsort(distances, axis=1)[:, :k]
    
    # Create LSH encoder
    lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=dim, random_state=42)
    
    # Build index with minimal optimization (identity permutation)
    index_obj_minimal = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        n_bits=n_bits,
        n_codes=32,
        use_codebook=True,
        max_two_swap_iters=0,  # No optimization
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=42,
    )
    
    recall_minimal = compute_gth_recall(
        queries=queries,
        base_embeddings=base_embeddings,
        ground_truth=ground_truth,
        index_obj=index_obj_minimal,
        lsh_encoder=lsh.hash,
        k=k,
        hamming_radius=hamming_radius,
    )
    
    # Build index with optimization
    index_obj_optimized = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        n_bits=n_bits,
        n_codes=32,
        use_codebook=True,
        max_two_swap_iters=20,  # More optimization
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=42,
    )
    
    recall_optimized = compute_gth_recall(
        queries=queries,
        base_embeddings=base_embeddings,
        ground_truth=ground_truth,
        index_obj=index_obj_optimized,
        lsh_encoder=lsh.hash,
        k=k,
        hamming_radius=hamming_radius,
    )
    
    # Optimized should be at least as good as minimal (allowing small tolerance)
    assert recall_optimized >= recall_minimal - 0.05, \
        f"Optimized recall ({recall_optimized:.4f}) should not be worse than minimal ({recall_minimal:.4f})"
    
    print(f"Minimal optimization recall: {recall_minimal:.4f}, Optimized recall: {recall_optimized:.4f}")


def test_recall_statistical_significance():
    """Test recall with multiple random seeds to check consistency."""
    n_bits = 6
    dim = 16
    N = 100
    Q = 20
    k = 5
    hamming_radius = 1
    
    recalls_baseline = []
    recalls_gth = []
    
    for seed in [42, 43, 44]:
        # Generate synthetic data
        np.random.seed(seed)
        base_embeddings = np.random.randn(N, dim).astype(np.float32)
        queries = np.random.randn(Q, dim).astype(np.float32)
        
        # Compute ground truth
        distances = euclidean_distances(queries, base_embeddings)
        ground_truth = np.argsort(distances, axis=1)[:, :k]
        
        # Create LSH encoder
        lsh = create_lsh_family("hyperplane", n_bits=n_bits, dim=dim, random_state=seed)
        
        # Baseline recall
        baseline_recall = compute_baseline_recall_lsh(
            queries=queries,
            base_embeddings=base_embeddings,
            ground_truth=ground_truth,
            lsh_encoder=lsh.hash,
            k=k,
            hamming_radius=hamming_radius,
        )
        recalls_baseline.append(baseline_recall)
        
        # GTH recall
        index_obj = build_distribution_aware_index(
            base_embeddings=base_embeddings,
            queries=queries,
            ground_truth_neighbors=ground_truth,
            encoder=lsh.hash,
            n_bits=n_bits,
            n_codes=32,
            use_codebook=True,
            max_two_swap_iters=10,
            num_tunneling_steps=0,
            mode="two_swap_only",
            random_state=seed,
        )
        
        gth_recall = compute_gth_recall(
            queries=queries,
            base_embeddings=base_embeddings,
            ground_truth=ground_truth,
            index_obj=index_obj,
            lsh_encoder=lsh.hash,
            k=k,
            hamming_radius=hamming_radius,
        )
        recalls_gth.append(gth_recall)
    
    # Compute statistics
    mean_baseline = np.mean(recalls_baseline)
    mean_gth = np.mean(recalls_gth)
    std_baseline = np.std(recalls_baseline)
    std_gth = np.std(recalls_gth)
    
    # Both should be valid
    assert 0.0 <= mean_baseline <= 1.0
    assert 0.0 <= mean_gth <= 1.0
    
    # GTH should not be significantly worse on average
    assert mean_gth >= mean_baseline - 0.15, \
        f"GTH mean recall ({mean_gth:.4f} ± {std_gth:.4f}) should not be much worse than baseline ({mean_baseline:.4f} ± {std_baseline:.4f})"
    
    print(f"Baseline: {mean_baseline:.4f} ± {std_baseline:.4f}")
    print(f"GTH: {mean_gth:.4f} ± {std_gth:.4f}")

