"""Smoke test for Gray-Tunneled end-to-end pipeline on real-like data."""

import numpy as np
import pytest

from gray_tunneled_hashing.data.synthetic_generators import generate_hypercube_vertices
from gray_tunneled_hashing.binary.codebooks import (
    build_codebook_kmeans,
    encode_with_codebook,
    find_nearest_centroids,
)
from gray_tunneled_hashing.algorithms.gray_tunneled_hasher import GrayTunneledHasher
from gray_tunneled_hashing.integrations.hamming_index import build_hamming_index
from gray_tunneled_hashing.evaluation.metrics import recall_at_k


def test_gray_tunneled_real_smoke():
    """
    Smoke test: Run full Gray-Tunneled pipeline on synthetic data.
    
    This test verifies that the complete pipeline works without errors,
    even if the recall is not necessarily high (it's just a smoke test).
    """
    # Create small synthetic dataset
    np.random.seed(42)
    n_base = 100
    n_queries = 10
    dim = 16
    n_bits = 5  # 2^5 = 32 codes
    n_codes = 32  # Must equal 2**n_bits for GrayTunneledHasher
    k = 5
    
    base_embeddings = np.random.randn(n_base, dim).astype(np.float32)
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    
    # Compute ground truth (simple: nearest neighbors by L2)
    ground_truth = np.zeros((n_queries, k), dtype=np.int32)
    for i, query in enumerate(queries):
        distances = np.sum((base_embeddings - query) ** 2, axis=1)
        ground_truth[i] = np.argsort(distances)[:k]
    
    # Step 1: Build codebook
    centroids, assignments = build_codebook_kmeans(
        base_embeddings, n_codes=n_codes, random_state=42
    )
    
    assert centroids.shape == (n_codes, dim)
    assert assignments.shape == (n_base,)
    
    # Step 2: Run Gray-Tunneled Hasher
    hasher = GrayTunneledHasher(
        n_bits=n_bits,
        block_size=4,
        max_two_swap_iters=10,
        num_tunneling_steps=2,
        random_state=42,
    )
    
    hasher.fit(centroids)
    
    assert hasher.is_fitted
    assert hasher.cost_ is not None
    assert hasher.cost_ >= 0
    
    # Step 3: Create centroid-to-code mapping
    pi = hasher.get_assignment()
    vertices = generate_hypercube_vertices(n_bits)
    
    centroid_to_code = {}
    for centroid_idx in range(n_codes):
        vertex_indices = np.where(pi == centroid_idx)[0]
        if len(vertex_indices) > 0:
            vertex_idx = vertex_indices[0]
            binary_code = vertices[vertex_idx].astype(bool)
            centroid_to_code[centroid_idx] = binary_code
    
    assert len(centroid_to_code) == n_codes
    
    # Step 4: Encode embeddings
    base_codes = encode_with_codebook(
        base_embeddings, centroids, centroid_to_code, assignments=assignments
    )
    
    query_assignments = find_nearest_centroids(queries, centroids)
    query_codes = encode_with_codebook(
        queries, centroids, centroid_to_code, assignments=query_assignments
    )
    
    assert base_codes.shape == (n_base, n_bits)
    assert query_codes.shape == (n_queries, n_bits)
    
    # Step 5: Build index and search
    index = build_hamming_index(base_codes, use_faiss=False)
    retrieved_indices, distances = index.search(query_codes, k=k)
    
    assert retrieved_indices.shape == (n_queries, k)
    assert distances.shape == (n_queries, k)
    
    # Step 6: Compute recall (should be > 0, but not necessarily high)
    recall = recall_at_k(retrieved_indices, ground_truth, k=k)
    
    assert recall >= 0.0
    assert recall <= 1.0
    # Smoke test: just verify recall is defined (not necessarily high)
    assert not np.isnan(recall)


def test_gray_tunneled_pipeline_components():
    """Test that all components can be imported and instantiated."""
    # Just verify imports work and basic instantiation
    from gray_tunneled_hashing.binary.codebooks import build_codebook_kmeans
    from gray_tunneled_hashing.algorithms.gray_tunneled_hasher import GrayTunneledHasher
    from gray_tunneled_hashing.integrations.hamming_index import build_hamming_index
    
    # Small test data
    embeddings = np.random.randn(20, 8).astype(np.float32)
    
    # Test codebook
    centroids, assignments = build_codebook_kmeans(embeddings, n_codes=4, random_state=42)
    assert centroids.shape == (4, 8)
    
    # Test hasher (need 2**n_bits centroids)
    n_bits_test = 4
    n_codes_test = 2 ** n_bits_test  # 16
    centroids_test, _ = build_codebook_kmeans(embeddings, n_codes=n_codes_test, random_state=42)
    
    hasher = GrayTunneledHasher(n_bits=n_bits_test, random_state=42)
    hasher.fit(centroids_test)
    assert hasher.is_fitted
    
    # Test index
    codes = np.random.rand(10, 8) > 0.5
    index = build_hamming_index(codes, use_faiss=False)
    assert index.n_samples == 10

