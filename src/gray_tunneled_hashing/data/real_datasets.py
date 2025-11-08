"""Loaders for real-world embedding datasets."""

import numpy as np
from pathlib import Path
from typing import Optional


def _get_data_dir() -> Path:
    """Get the experiments/real/data directory."""
    # Assume we're in the repo root when scripts are run
    repo_root = Path(__file__).parent.parent.parent.parent
    return repo_root / "experiments" / "real" / "data"


def load_embeddings(name: str, split: str = "base") -> np.ndarray:
    """
    Load base embeddings from a real dataset.
    
    Args:
        name: Dataset name (e.g., "quora")
        split: Split to load ("base" for corpus, "queries" for queries)
        
    Returns:
        Embeddings array of shape (N, dim) for base or (Q, dim) for queries
        
    Raises:
        FileNotFoundError: If the embeddings file doesn't exist
        ValueError: If split is invalid
    """
    if split not in ["base", "queries"]:
        raise ValueError(f"split must be 'base' or 'queries', got '{split}'")
    
    data_dir = _get_data_dir()
    
    if split == "base":
        filename = f"{name}_base_embeddings.npy"
    else:  # queries
        filename = f"{name}_queries_embeddings.npy"
    
    filepath = data_dir / filename
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {filepath}\n"
            f"Please ensure embeddings are generated and saved in {data_dir}"
        )
    
    embeddings = np.load(filepath)
    
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D array, got shape {embeddings.shape}. "
            f"Embeddings should be (N, dim) for base or (Q, dim) for queries."
        )
    
    return embeddings.astype(np.float32)


def load_queries_and_ground_truth(
    name: str,
    k: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load queries embeddings and ground truth kNN indices.
    
    Args:
        name: Dataset name (e.g., "quora")
        k: Number of neighbors in ground truth (if None, uses all available)
        
    Returns:
        Tuple of (queries, gt_indices) where:
        - queries: Array of shape (Q, dim) with query embeddings
        - gt_indices: Array of shape (Q, k) with indices of k nearest neighbors
                      in the base corpus for each query
                      
    Raises:
        FileNotFoundError: If files don't exist
    """
    data_dir = _get_data_dir()
    
    # Load queries
    queries = load_embeddings(name, split="queries")
    
    # Load ground truth
    gt_filepath = data_dir / f"{name}_ground_truth_indices.npy"
    
    if not gt_filepath.exists():
        raise FileNotFoundError(
            f"Ground truth file not found: {gt_filepath}\n"
            f"Please run scripts/compute_float_ground_truth.py first to generate ground truth."
        )
    
    gt_indices = np.load(gt_filepath)
    
    if gt_indices.ndim != 2:
        raise ValueError(
            f"Expected 2D ground truth array, got shape {gt_indices.shape}. "
            f"Ground truth should be (Q, k) where Q is number of queries."
        )
    
    if queries.shape[0] != gt_indices.shape[0]:
        raise ValueError(
            f"Mismatch: queries has {queries.shape[0]} queries, "
            f"but ground truth has {gt_indices.shape[0]} rows."
        )
    
    # Optionally slice to k if specified
    if k is not None:
        if k > gt_indices.shape[1]:
            raise ValueError(
                f"Requested k={k} but ground truth only has {gt_indices.shape[1]} neighbors."
            )
        gt_indices = gt_indices[:, :k]
    
    return queries, gt_indices.astype(np.int32)


def list_available_datasets() -> list[str]:
    """
    List available datasets in the data directory.
    
    Returns:
        List of dataset names (without suffixes)
    """
    data_dir = _get_data_dir()
    
    if not data_dir.exists():
        return []
    
    # Find all base embeddings files
    datasets = set()
    for file in data_dir.glob("*_base_embeddings.npy"):
        name = file.stem.replace("_base_embeddings", "")
        datasets.add(name)
    
    return sorted(list(datasets))

