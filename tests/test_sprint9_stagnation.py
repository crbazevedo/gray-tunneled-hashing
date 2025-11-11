"""Tests for Sprint 9 stagnation detection."""

import numpy as np
import pytest
from gray_tunneled_hashing.distribution.j_phi_objective import detect_stagnation


def test_detect_stagnation_no_stagnation():
    """Test that stagnation is not detected when improving."""
    cost_history = [10.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0]
    
    is_stagnant = detect_stagnation(cost_history, window=10, threshold=0.001)
    assert not is_stagnant


def test_detect_stagnation_stagnant():
    """Test that stagnation is detected when not improving."""
    cost_history = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    
    is_stagnant = detect_stagnation(cost_history, window=10, threshold=0.001)
    assert is_stagnant


def test_detect_stagnation_small_improvement():
    """Test stagnation with very small improvement."""
    cost_history = [10.0, 9.999, 9.998, 9.997, 9.996, 9.995, 9.994, 9.993, 9.992, 9.991, 9.990]
    
    # Relative improvement: (10.0 - 9.990) / 10.0 = 0.001 = 0.1%
    # With threshold=0.001, this should be detected as stagnant
    is_stagnant = detect_stagnation(cost_history, window=10, threshold=0.001)
    assert is_stagnant


def test_detect_stagnation_insufficient_history():
    """Test that insufficient history returns False."""
    cost_history = [10.0, 9.5, 9.0]  # Only 3 values, need 11 for window=10
    
    is_stagnant = detect_stagnation(cost_history, window=10, threshold=0.001)
    assert not is_stagnant


def test_detect_stagnation_custom_threshold():
    """Test with custom threshold."""
    cost_history = [10.0, 9.8, 9.6, 9.4, 9.2, 9.0, 8.8, 8.6, 8.4, 8.2, 8.0]
    
    # Relative improvement: (10.0 - 8.0) / 10.0 = 0.2 = 20%
    # With threshold=0.25, this should NOT be stagnant
    is_stagnant = detect_stagnation(cost_history, window=10, threshold=0.25)
    assert not is_stagnant
    
    # With threshold=0.15, this SHOULD be stagnant
    is_stagnant = detect_stagnation(cost_history, window=10, threshold=0.15)
    assert is_stagnant


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

