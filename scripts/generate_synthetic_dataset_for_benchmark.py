"""Generate synthetic dataset for distribution-aware benchmark."""

import argparse
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gray_tunneled_hashing.evaluation.metrics import recall_at_k


def generate_synthetic_dataset(
    n_base: int = 1000,
    n_queries: int = 100,
    dim: int = 64,
    k: int = 10,
    output_dir: Path = None,
    dataset_name: str = "synthetic",
    random_state: int = 42,
):
    """
    Generate synthetic dataset with skewed query distribution for distribution-aware testing.
    
    Creates:
    - Base embeddings with some clustering structure
    - Queries with skewed distribution (some clusters queried more)
    - Ground truth neighbors
    
    Args:
        n_base: Number of base embeddings
        n_queries: Number of queries
        dim: Embedding dimension
        k: Number of neighbors for ground truth
        output_dir: Directory to save files
        dataset_name: Name for the dataset
        random_state: Random seed
    """
    np.random.seed(random_state)
    
    if output_dir is None:
        output_dir = Path("experiments/real/data")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating synthetic dataset: {dataset_name}")
    print(f"  Base embeddings: {n_base}")
    print(f"  Queries: {n_queries}")
    print(f"  Dimension: {dim}")
    print(f"  k: {k}")
    
    # Generate base embeddings with clusters
    n_clusters = 5
    cluster_centers = np.random.randn(n_clusters, dim).astype(np.float32)
    cluster_sizes = np.random.multinomial(n_base, [1/n_clusters] * n_clusters)
    
    base_embeddings = []
    cluster_labels = []
    for i, (center, size) in enumerate(zip(cluster_centers, cluster_sizes)):
        cluster_emb = center + np.random.randn(size, dim).astype(np.float32) * 0.3
        base_embeddings.append(cluster_emb)
        cluster_labels.extend([i] * size)
    
    base_embeddings = np.vstack(base_embeddings)
    cluster_labels = np.array(cluster_labels)
    
    # Shuffle
    perm = np.random.permutation(n_base)
    base_embeddings = base_embeddings[perm]
    cluster_labels = cluster_labels[perm]
    
    print(f"  Generated {len(base_embeddings)} base embeddings")
    
    # Generate queries with skewed distribution
    # Cluster 0 and 2 get more queries (skewed traffic)
    query_weights = np.array([0.4, 0.1, 0.3, 0.1, 0.1])  # Skewed
    query_cluster_probs = query_weights / query_weights.sum()
    
    query_clusters = np.random.choice(n_clusters, size=n_queries, p=query_cluster_probs)
    queries = []
    for cluster_idx in query_clusters:
        center = cluster_centers[cluster_idx]
        query = center + np.random.randn(1, dim).astype(np.float32) * 0.2
        queries.append(query)
    
    queries = np.vstack(queries)
    
    print(f"  Generated {len(queries)} queries (skewed distribution)")
    print(f"    Cluster distribution: {np.bincount(query_clusters, minlength=n_clusters)}")
    
    # Compute ground truth neighbors (brute force)
    print("  Computing ground truth neighbors...")
    ground_truth = []
    for query in queries:
        distances = np.linalg.norm(base_embeddings - query, axis=1)
        nearest_indices = np.argsort(distances)[:k]
        ground_truth.append(nearest_indices)
    
    ground_truth = np.array(ground_truth, dtype=np.int32)
    
    print(f"  Computed ground truth for {len(ground_truth)} queries")
    
    # Save files
    base_file = output_dir / f"{dataset_name}_base_embeddings.npy"
    queries_file = output_dir / f"{dataset_name}_queries_embeddings.npy"
    gt_file = output_dir / f"{dataset_name}_ground_truth_indices.npy"
    
    np.save(base_file, base_embeddings)
    np.save(queries_file, queries)
    np.save(gt_file, ground_truth)
    
    print(f"\nSaved files:")
    print(f"  Base: {base_file}")
    print(f"  Queries: {queries_file}")
    print(f"  Ground truth: {gt_file}")
    
    return base_embeddings, queries, ground_truth


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic dataset for distribution-aware benchmark"
    )
    parser.add_argument(
        "--n-base",
        type=int,
        default=1000,
        help="Number of base embeddings (default: 1000)",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=100,
        help="Number of queries (default: 100)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=64,
        help="Embedding dimension (default: 64)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of neighbors for ground truth (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/real/data",
        help="Output directory (default: experiments/real/data)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="synthetic",
        help="Dataset name (default: synthetic)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    
    args = parser.parse_args()
    
    generate_synthetic_dataset(
        n_base=args.n_base,
        n_queries=args.n_queries,
        dim=args.dim,
        k=args.k,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()

