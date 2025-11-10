"""Tests for experimental metrics."""

import pytest
import numpy as np
from gray_tunneled_hashing.experiments.metrics import (
    compute_recall_at_k,
    compute_collision_preservation_rate,
    compute_hamming_ball_coverage,
    compute_improvement_over_baseline,
)


def test_recall_at_k_calculation():
    """Test recall@k calculation with known cases."""
    # Perfect recall
    retrieved = np.array([[0, 1, 2], [3, 4, 5]])
    ground_truth = np.array([[0, 1, 2], [3, 4, 5]])
    recall = compute_recall_at_k(retrieved, ground_truth, k=3)
    assert recall == 1.0
    
    # Partial recall
    retrieved = np.array([[0, 1, 2], [3, 4, 6]])
    ground_truth = np.array([[0, 1, 2], [3, 4, 5]])
    recall = compute_recall_at_k(retrieved, ground_truth, k=3)
    assert 0.0 < recall < 1.0
    
    # Zero recall
    retrieved = np.array([[10, 11, 12], [13, 14, 15]])
    ground_truth = np.array([[0, 1, 2], [3, 4, 5]])
    recall = compute_recall_at_k(retrieved, ground_truth, k=3)
    assert recall == 0.0


def test_collision_preservation_rate():
    """Test collision preservation rate calculation."""
    # Perfect preservation
    collisions_before = {(0, 1), (2, 3), (4, 5)}
    collisions_after = {(0, 1), (2, 3), (4, 5)}
    rate = compute_collision_preservation_rate(collisions_before, collisions_after)
    assert rate == 100.0
    
    # Partial preservation
    collisions_before = {(0, 1), (2, 3), (4, 5)}
    collisions_after = {(0, 1), (2, 3)}
    rate = compute_collision_preservation_rate(collisions_before, collisions_after)
    assert rate == pytest.approx(66.666, abs=0.1)
    
    # No collisions
    collisions_before = set()
    collisions_after = set()
    rate = compute_collision_preservation_rate(collisions_before, collisions_after)
    assert rate == 100.0


def test_hamming_ball_coverage():
    """Test Hamming ball coverage calculation."""
    n_bits = 4
    center_code = np.array([0, 0, 0, 0], dtype=bool)
    
    # Radius 0: only center
    coverage = compute_hamming_ball_coverage(center_code, radius=0, n_bits=n_bits)
    assert coverage == 1
    
    # Radius 1: center + n_bits neighbors
    coverage = compute_hamming_ball_coverage(center_code, radius=1, n_bits=n_bits)
    assert coverage == n_bits + 1  # 1 center + 4 neighbors
    
    # Radius 2: more neighbors
    coverage = compute_hamming_ball_coverage(center_code, radius=2, n_bits=n_bits)
    # Should be C(4,0) + C(4,1) + C(4,2) = 1 + 4 + 6 = 11
    assert coverage == 11


def test_improvement_metric():
    """Test improvement over baseline calculation."""
    # Positive improvement
    improvement = compute_improvement_over_baseline(recall_gth=0.8, recall_baseline=0.6)
    assert improvement == pytest.approx(33.333, abs=0.1)
    
    # Negative improvement (worse)
    improvement = compute_improvement_over_baseline(recall_gth=0.4, recall_baseline=0.6)
    assert improvement == pytest.approx(-33.333, abs=0.1)
    
    # No improvement
    improvement = compute_improvement_over_baseline(recall_gth=0.6, recall_baseline=0.6)
    assert improvement == 0.0
    
    # Baseline is zero
    improvement = compute_improvement_over_baseline(recall_gth=0.5, recall_baseline=0.0)
    assert improvement == float('inf')
    
    # Both are zero
    improvement = compute_improvement_over_baseline(recall_gth=0.0, recall_baseline=0.0)
    assert improvement == 0.0

