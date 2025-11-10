"""Smoke test for Sprint 3 sweep script."""

import pytest
import tempfile
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scripts.run_sweep_sprint3 import run_sweep


@pytest.fixture
def minimal_config(tmp_path):
    """Create a minimal config file for testing."""
    config = {
        "datasets": [
            {
                "name": "quora",
                "n_bits": [5],  # Small for testing
                "n_codes": [8],  # 2^3, small for testing
                "block_sizes": [4],
                "num_tunneling_steps": [0],  # Skip tunneling for speed
                "gt_modes": ["trivial"],
                "block_selection_strategies": ["random"],
                "k_values": [5],
                "random_state": 42,
                "max_two_swap_iters": 5,
                "two_swap_sample_size": 10,
            }
        ]
    }
    
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    return config_path


def test_sweep_config_loading(minimal_config):
    """Test that config file loads correctly."""
    import yaml
    
    with open(minimal_config, "r") as f:
        config = yaml.safe_load(f)
    
    assert "datasets" in config
    assert len(config["datasets"]) > 0


@pytest.mark.skip(reason="Requires actual dataset files - run manually with real data")
def test_sweep_script_runs(minimal_config, tmp_path):
    """Smoke test that sweep script can run (requires dataset files)."""
    # This test requires actual dataset files in experiments/real/data/
    # Skip by default, run manually when dataset is available
    
    output_dir = tmp_path / "results"
    
    try:
        run_sweep(
            config_path=minimal_config,
            output_dir=output_dir,
            use_faiss=False,  # Use Python backend for portability
            verbose=False,
        )
        
        # Check that results files were created
        json_path = output_dir / "results_sprint3_sweep.json"
        csv_path = output_dir / "results_sprint3_sweep.csv"
        
        # At least one should exist if the script ran successfully
        assert json_path.exists() or csv_path.exists()
        
    except FileNotFoundError:
        # Expected if dataset files don't exist
        pytest.skip("Dataset files not found - this is expected in CI")


def test_sweep_imports():
    """Test that all imports in sweep script work."""
    import scripts.run_sweep_sprint3
    
    # If we get here, imports work
    assert True

