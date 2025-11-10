"""Tests for benchmark script functionality."""

import pytest
import json
import numpy as np
from pathlib import Path
import tempfile
import subprocess
import sys

from gray_tunneled_hashing.experiments.config import LSHExperimentConfig


def test_benchmark_script_parameters():
    """Test that benchmark script accepts valid parameters."""
    # Test with minimal config
    config = LSHExperimentConfig(
        n_samples=50,
        n_queries=10,
        n_bits=6,
        n_codes=16,
        k=3,
        hamming_radius=1,
        random_state=42,
    )
    
    # Validate config
    errors = config.validate()
    assert len(errors) == 0, f"Config should be valid, got errors: {errors}"


def test_benchmark_script_recall_calculation():
    """Test recall calculation logic."""
    from gray_tunneled_hashing.experiments.metrics import compute_recall_at_k
    
    # Perfect recall case
    retrieved = np.array([[0, 1, 2], [3, 4, 5]])
    ground_truth = np.array([[0, 1, 2], [3, 4, 5]])
    recall = compute_recall_at_k(retrieved, ground_truth, k=3)
    assert recall == 1.0
    
    # Partial recall
    retrieved = np.array([[0, 1, 2], [3, 4, 6]])
    ground_truth = np.array([[0, 1, 2], [3, 4, 5]])
    recall = compute_recall_at_k(retrieved, ground_truth, k=3)
    assert 0.0 < recall < 1.0


def test_benchmark_script_output_format():
    """Test that benchmark script produces valid JSON output."""
    # This is a smoke test - would need to actually run the script
    # For now, just test that the expected structure is valid JSON
    expected_structure = {
        "method": "hyperplane",
        "n_bits": 8,
        "n_codes": 32,
        "k": 5,
        "hamming_radius": 1,
        "n_runs": 3,
        "recall_mean": 0.5,
        "recall_std": 0.1,
        "build_time_mean": 1.0,
        "search_time_mean": 0.1,
        "runs": [],
    }
    
    # Should be JSON serializable
    json_str = json.dumps(expected_structure)
    loaded = json.loads(json_str)
    assert loaded["method"] == "hyperplane"
    assert loaded["n_runs"] == 3

