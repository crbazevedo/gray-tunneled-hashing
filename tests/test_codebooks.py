"""Tests for codebook construction and encoding."""

import numpy as np
import pytest

from gray_tunneled_hashing.binary.codebooks import (
    build_codebook_kmeans,
    encode_with_codebook,
    find_nearest_centroids,
)


def test_build_codebook_kmeans():
    """Test k-means codebook construction."""
    # Create simple test data with clear clusters
    np.random.seed(42)
    cluster1 = np.random.randn(20, 10) + np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
    cluster2 = np.random.randn(20, 10) + np.array([-5, -5, -5, -5, -5, -5, -5, -5, -5, -5])
    embeddings = np.vstack([cluster1, cluster2]).astype(np.float32)
    
    centroids, assignments = build_codebook_kmeans(
        embeddings, n_codes=2, random_state=42
    )
    
    assert centroids.shape == (2, 10)
    assert assignments.shape == (40,)
    assert np.all(assignments >= 0)
    assert np.all(assignments < 2)
    assert centroids.dtype == np.float32
    assert assignments.dtype == np.int32


def test_build_codebook_kmeans_deterministic():
    """Test that same seed produces same results."""
    embeddings = np.random.randn(50, 20).astype(np.float32)
    
    centroids1, assignments1 = build_codebook_kmeans(
        embeddings, n_codes=5, random_state=42
    )
    centroids2, assignments2 = build_codebook_kmeans(
        embeddings, n_codes=5, random_state=42
    )
    
    assert np.allclose(centroids1, centroids2)
    assert np.array_equal(assignments1, assignments2)


def test_build_codebook_kmeans_invalid_n_codes():
    """Test with invalid n_codes."""
    embeddings = np.random.randn(10, 20).astype(np.float32)
    
    with pytest.raises(ValueError, match="must be positive"):
        build_codebook_kmeans(embeddings, n_codes=0)
    
    with pytest.raises(ValueError, match="cannot exceed"):
        build_codebook_kmeans(embeddings, n_codes=20)


def test_find_nearest_centroids():
    """Test finding nearest centroids."""
    embeddings = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [9.0, 9.0],
        [10.0, 10.0],
    ]).astype(np.float32)
    
    centroids = np.array([
        [1.5, 1.5],  # Close to first two
        [9.5, 9.5],  # Close to last two
    ]).astype(np.float32)
    
    indices = find_nearest_centroids(embeddings, centroids)
    
    assert indices.shape == (4,)
    assert indices.dtype == np.int32
    assert indices[0] == 0  # First embedding closer to centroid 0
    assert indices[1] == 0
    assert indices[2] == 1
    assert indices[3] == 1


def test_encode_with_codebook():
    """Test encoding with codebook."""
    embeddings = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
    ]).astype(np.float32)
    
    centroids = np.array([
        [1.5, 1.5],
        [9.5, 9.5],
    ]).astype(np.float32)
    
    # Map centroid 0 -> [True, False], centroid 1 -> [False, True]
    centroid_to_code = {
        0: np.array([True, False]),
        1: np.array([False, True]),
    }
    
    # Pre-computed assignments (both embeddings map to centroid 0)
    assignments = np.array([0, 0], dtype=np.int32)
    
    codes = encode_with_codebook(embeddings, centroids, centroid_to_code, assignments)
    
    assert codes.shape == (2, 2)
    assert codes.dtype == bool
    assert np.array_equal(codes[0], [True, False])
    assert np.array_equal(codes[1], [True, False])


def test_encode_with_codebook_no_assignments():
    """Test encoding without pre-computed assignments."""
    embeddings = np.array([
        [1.0, 1.0],
        [9.0, 9.0],
    ]).astype(np.float32)
    
    centroids = np.array([
        [1.5, 1.5],
        [9.5, 9.5],
    ]).astype(np.float32)
    
    centroid_to_code = {
        0: np.array([True, False]),
        1: np.array([False, True]),
    }
    
    codes = encode_with_codebook(embeddings, centroids, centroid_to_code, assignments=None)
    
    assert codes.shape == (2, 2)
    # First embedding should map to centroid 0, second to centroid 1
    assert np.array_equal(codes[0], [True, False])
    assert np.array_equal(codes[1], [False, True])

