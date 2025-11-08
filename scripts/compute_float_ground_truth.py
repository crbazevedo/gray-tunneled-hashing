#!/usr/bin/env python3
"""Compute exact kNN ground truth using float embeddings."""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from gray_tunneled_hashing.data.real_datasets import load_embeddings, _get_data_dir


def compute_ground_truth_faiss(
    base_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Compute ground truth using FAISS Flat L2 index.
    
    Args:
        base_embeddings: Base corpus embeddings (N, dim)
        query_embeddings: Query embeddings (Q, dim)
        k: Number of neighbors to retrieve
        
    Returns:
        Array of shape (Q, k) with indices of k nearest neighbors
    """
    try:
        import faiss
        
        # Normalize embeddings for cosine similarity (or use L2)
        # For L2 distance, we don't normalize
        dim = base_embeddings.shape[1]
        
        # Create FAISS index
        index = faiss.IndexFlatL2(dim)
        index.add(base_embeddings.astype(np.float32))
        
        # Search
        distances, indices = index.search(query_embeddings.astype(np.float32), k)
        
        return indices
        
    except ImportError:
        raise ImportError(
            "FAISS not available. Install with: pip install faiss-cpu\n"
            "Or use --method brute-force"
        )


def compute_ground_truth_brute_force(
    base_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Compute ground truth using brute-force numpy.
    
    Args:
        base_embeddings: Base corpus embeddings (N, dim)
        query_embeddings: Query embeddings (Q, dim)
        k: Number of neighbors to retrieve
        
    Returns:
        Array of shape (Q, k) with indices of k nearest neighbors
    """
    # Compute pairwise squared L2 distances
    # distances[i, j] = ||query[i] - base[j]||^2
    distances = np.zeros((query_embeddings.shape[0], base_embeddings.shape[0]))
    
    for i, query in enumerate(query_embeddings):
        # Squared L2 distance
        dists = np.sum((base_embeddings - query) ** 2, axis=1)
        distances[i] = dists
    
    # Get k nearest neighbors
    indices = np.argsort(distances, axis=1)[:, :k]
    
    return indices


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compute exact kNN ground truth for a dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., 'quora')",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=100,
        help="Number of neighbors to compute (default: 100)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["faiss", "brute-force", "auto"],
        default="auto",
        help="Method to use: 'faiss' (FAISS L2), 'brute-force' (numpy), 'auto' (try FAISS, fallback to brute-force)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: experiments/real/data/{dataset}_ground_truth_indices.npy)",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Compute Float Ground Truth")
    print("=" * 70)
    print(f"\nDataset: {args.dataset}")
    print(f"k: {args.k}")
    print(f"Method: {args.method}\n")
    
    # Load embeddings
    print("Loading embeddings...")
    try:
        base_embeddings = load_embeddings(args.dataset, split="base")
        query_embeddings = load_embeddings(args.dataset, split="queries")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    print(f"  ✓ Base embeddings: {base_embeddings.shape}")
    print(f"  ✓ Query embeddings: {query_embeddings.shape}")
    
    # Compute ground truth
    print(f"\nComputing ground truth (k={args.k})...")
    
    if args.method == "auto":
        # Try FAISS first
        try:
            gt_indices = compute_ground_truth_faiss(base_embeddings, query_embeddings, args.k)
            print("  ✓ Using FAISS")
        except ImportError:
            print("  ⚠ FAISS not available, using brute-force")
            gt_indices = compute_ground_truth_brute_force(base_embeddings, query_embeddings, args.k)
    elif args.method == "faiss":
        gt_indices = compute_ground_truth_faiss(base_embeddings, query_embeddings, args.k)
    else:  # brute-force
        gt_indices = compute_ground_truth_brute_force(base_embeddings, query_embeddings, args.k)
    
    print(f"  ✓ Ground truth shape: {gt_indices.shape}")
    
    # Save
    if args.output is None:
        data_dir = _get_data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)
        output_path = data_dir / f"{args.dataset}_ground_truth_indices.npy"
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path, gt_indices)
    print(f"\n✓ Saved ground truth to: {output_path}")
    print("\n" + "=" * 70)
    print("Ground truth computation completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

