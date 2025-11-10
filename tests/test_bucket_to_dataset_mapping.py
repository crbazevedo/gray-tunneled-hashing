"""
Tests for bucket to dataset index mapping validation.
"""

import numpy as np
import pytest

from gray_tunneled_hashing.binary.codebooks import build_codebook_kmeans, find_nearest_centroids


def test_bucket_mapping_completeness():
    """Test that all embeddings are assigned to buckets."""
    np.random.seed(42)
    n_samples = 100
    n_codes = 16
    dim = 16
    
    embeddings = np.random.randn(n_samples, dim).astype(np.float32)
    centroids, assignments = build_codebook_kmeans(
        embeddings=embeddings,
        n_codes=n_codes,
        random_state=42,
    )
    
    assert len(assignments) == n_samples, "All embeddings must be assigned"
    assert np.all((assignments >= 0) & (assignments < n_codes)), "All assignments must be valid"


def test_bucket_mapping_consistency():
    """Test that assignments are consistent with find_nearest_centroids."""
    np.random.seed(42)
    n_samples = 100
    n_codes = 16
    dim = 16
    
    embeddings = np.random.randn(n_samples, dim).astype(np.float32)
    centroids, assignments = build_codebook_kmeans(
        embeddings=embeddings,
        n_codes=n_codes,
        random_state=42,
    )
    
    # Verify assignments match find_nearest_centroids
    test_assignments = find_nearest_centroids(embeddings, centroids)
    assert np.array_equal(assignments, test_assignments), "Assignments must be consistent"


def test_bucket_to_dataset_indices_mapping():
    """Test bucket to dataset indices mapping construction."""
    np.random.seed(42)
    n_samples = 100
    n_codes = 16
    dim = 16
    
    embeddings = np.random.randn(n_samples, dim).astype(np.float32)
    centroids, assignments = build_codebook_kmeans(
        embeddings=embeddings,
        n_codes=n_codes,
        random_state=42,
    )
    
    # Build mapping
    bucket_to_dataset_indices = {}
    for dataset_idx, bucket_idx in enumerate(assignments):
        if bucket_idx not in bucket_to_dataset_indices:
            bucket_to_dataset_indices[bucket_idx] = []
        bucket_to_dataset_indices[bucket_idx].append(dataset_idx)
    
    # Verify all embeddings are in mapping
    total_mapped = sum(len(indices) for indices in bucket_to_dataset_indices.values())
    assert total_mapped == n_samples, "All embeddings must be in mapping"
    
    # Verify no duplicates
    for bucket_idx, indices in bucket_to_dataset_indices.items():
        assert len(indices) == len(set(indices)), f"Bucket {bucket_idx} has duplicate indices"


def test_bucket_coverage():
    """Test that buckets have reasonable coverage."""
    np.random.seed(42)
    n_samples = 100
    n_codes = 16
    dim = 16
    
    embeddings = np.random.randn(n_samples, dim).astype(np.float32)
    centroids, assignments = build_codebook_kmeans(
        embeddings=embeddings,
        n_codes=n_codes,
        random_state=42,
    )
    
    unique_buckets = len(np.unique(assignments))
    coverage = unique_buckets / n_codes
    
    # At least 50% of buckets should be used
    assert coverage >= 0.5, f"Only {coverage:.1%} of buckets used, expected >= 50%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

