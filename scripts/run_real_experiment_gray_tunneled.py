#!/usr/bin/env python3
"""Run Gray-Tunneled experiment on real embeddings."""

import sys
import argparse
import json
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from gray_tunneled_hashing.data.real_datasets import (
    load_embeddings,
    load_queries_and_ground_truth,
)
from gray_tunneled_hashing.data.synthetic_generators import generate_hypercube_vertices
from gray_tunneled_hashing.binary.codebooks import (
    build_codebook_kmeans,
    encode_with_codebook,
    find_nearest_centroids,
)
from gray_tunneled_hashing.algorithms.gray_tunneled_hasher import GrayTunneledHasher
from gray_tunneled_hashing.integrations.hamming_index import build_hamming_index
from gray_tunneled_hashing.evaluation.metrics import recall_at_k


def run_gray_tunneled_experiment(
    dataset_name: str,
    n_bits: int,
    n_codes: int,
    k: int,
    block_size: int,
    max_two_swap_iters: int,
    num_tunneling_steps: int,
    random_state: int,
    use_faiss: bool = True,
):
    """
    Run Gray-Tunneled experiment on real embeddings.
    
    Args:
        dataset_name: Name of the dataset
        n_bits: Number of bits for binary codes
        n_codes: Number of codebook vectors (should be <= 2**n_bits)
        k: Recall@k value
        block_size: Block size for tunneling
        max_two_swap_iters: Max iterations for 2-swap
        num_tunneling_steps: Number of tunneling steps
        random_state: Random seed
        use_faiss: Whether to try FAISS (fallback to Python if not available)
        
    Returns:
        Dictionary with results
    """
    print("=" * 70)
    print("Gray-Tunneled Experiment")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Dataset: {dataset_name}")
    print(f"  n_bits: {n_bits}")
    print(f"  n_codes: {n_codes}")
    print(f"  k (recall@k): {k}")
    print(f"  block_size: {block_size}")
    print(f"  max_two_swap_iters: {max_two_swap_iters}")
    print(f"  num_tunneling_steps: {num_tunneling_steps}")
    print(f"  Random state: {random_state}")
    print(f"  Use FAISS: {use_faiss}\n")
    
    # Check n_codes <= 2**n_bits
    max_codes = 2 ** n_bits
    if n_codes > max_codes:
        raise ValueError(
            f"n_codes={n_codes} cannot exceed 2**n_bits={max_codes}"
        )
    
    # Step 1: Load embeddings
    print("Step 1: Loading embeddings...")
    try:
        base_embeddings = load_embeddings(dataset_name, split="base")
        queries, ground_truth = load_queries_and_ground_truth(dataset_name, k=k)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    print(f"  ✓ Base embeddings: {base_embeddings.shape}")
    print(f"  ✓ Queries: {queries.shape}")
    print(f"  ✓ Ground truth: {ground_truth.shape}")
    
    # Step 2: Build codebook
    print(f"\nStep 2: Building codebook (k-means, n_codes={n_codes})...")
    start_time = time.time()
    
    centroids, assignments = build_codebook_kmeans(
        base_embeddings,
        n_codes=n_codes,
        random_state=random_state,
    )
    
    codebook_time = time.time() - start_time
    print(f"  ✓ Centroids: {centroids.shape}")
    print(f"  ✓ Assignments: {assignments.shape}")
    print(f"  ✓ Codebook time: {codebook_time:.2f}s")
    
    # Step 3: Run Gray-Tunneled Hasher on centroids
    print(f"\nStep 3: Running Gray-Tunneled optimization on centroids...")
    start_time = time.time()
    
    hasher = GrayTunneledHasher(
        n_bits=n_bits,
        block_size=block_size,
        max_two_swap_iters=max_two_swap_iters,
        num_tunneling_steps=num_tunneling_steps,
        two_swap_sample_size=min(256, n_codes * (n_codes - 1) // 2),
        init_strategy="random",
        random_state=random_state + 100,
    )
    
    hasher.fit(centroids)
    
    optimization_time = time.time() - start_time
    print(f"  ✓ Optimization cost: {hasher.cost_:.6f}")
    print(f"  ✓ Optimization time: {optimization_time:.2f}s")
    print(f"  ✓ Cost history length: {len(hasher.cost_history_)}")
    
    # Step 4: Create centroid to binary code mapping
    print(f"\nStep 4: Creating centroid-to-code mapping...")
    
    # Get permutation from hasher
    pi = hasher.get_assignment()  # Shape: (2**n_bits,)
    
    # Generate hypercube vertices
    vertices = generate_hypercube_vertices(n_bits)  # Shape: (2**n_bits, n_bits)
    
    # Map centroids to binary codes
    # pi[vertex_index] = centroid_index
    # So for centroid i, we need to find vertex j such that pi[j] == i
    centroid_to_code = {}
    
    for centroid_idx in range(n_codes):
        # Find vertex index where pi[vertex_idx] == centroid_idx
        vertex_indices = np.where(pi == centroid_idx)[0]
        
        if len(vertex_indices) == 0:
            raise ValueError(
                f"Centroid {centroid_idx} not found in permutation pi"
            )
        
        # Use first vertex (or could use best if multiple)
        vertex_idx = vertex_indices[0]
        binary_code = vertices[vertex_idx].astype(bool)
        
        centroid_to_code[centroid_idx] = binary_code
    
    print(f"  ✓ Mapped {len(centroid_to_code)} centroids to binary codes")
    
    # Step 5: Encode embeddings via codebook
    print(f"\nStep 5: Encoding embeddings via codebook...")
    start_time = time.time()
    
    base_codes = encode_with_codebook(
        base_embeddings,
        centroids,
        centroid_to_code,
        assignments=assignments,
    )
    
    # For queries, find nearest centroids
    query_assignments = find_nearest_centroids(queries, centroids)
    query_codes = encode_with_codebook(
        queries,
        centroids,
        centroid_to_code,
        assignments=query_assignments,
    )
    
    encode_time = time.time() - start_time
    print(f"  ✓ Base codes: {base_codes.shape}")
    print(f"  ✓ Query codes: {query_codes.shape}")
    print(f"  ✓ Encoding time: {encode_time:.2f}s")
    
    # Step 6: Build index
    print(f"\nStep 6: Building Hamming index...")
    start_time = time.time()
    
    try:
        index = build_hamming_index(base_codes, use_faiss=use_faiss)
        print(f"  ✓ Index built (backend: {index.backend})")
    except Exception as e:
        print(f"  ✗ Error building index: {e}")
        sys.exit(1)
    
    build_time = time.time() - start_time
    print(f"  ✓ Build time: {build_time:.2f}s")
    
    # Step 7: Search
    print(f"\nStep 7: Searching for k={k} neighbors...")
    start_time = time.time()
    
    retrieved_indices, distances = index.search(query_codes, k)
    
    search_time = time.time() - start_time
    avg_search_time = search_time / len(queries)
    print(f"  ✓ Retrieved indices: {retrieved_indices.shape}")
    print(f"  ✓ Total search time: {search_time:.4f}s")
    print(f"  ✓ Avg search time per query: {avg_search_time*1000:.2f}ms")
    
    # Step 8: Compute recall
    print(f"\nStep 8: Computing recall@k...")
    recall = recall_at_k(retrieved_indices, ground_truth, k=k)
    print(f"  ✓ Recall@{k}: {recall:.4f} ({recall*100:.2f}%)")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nMethod: Gray-Tunneled")
    print(f"n_bits: {n_bits}")
    print(f"n_codes: {n_codes}")
    print(f"Recall@{k}: {recall:.4f}")
    print(f"Codebook time: {codebook_time:.2f}s")
    print(f"Optimization time: {optimization_time:.2f}s")
    print(f"Encoding time: {encode_time:.2f}s")
    print(f"Index build time: {build_time:.2f}s")
    print(f"Search time: {search_time:.4f}s ({avg_search_time*1000:.2f}ms per query)")
    print(f"Backend: {index.backend}")
    print(f"Final QAP cost: {hasher.cost_:.6f}")
    
    # Save results
    results = {
        "dataset": dataset_name,
        "method": "gray_tunneled",
        "n_bits": n_bits,
        "n_codes": n_codes,
        "k": k,
        "recall_at_k": float(recall),
        "codebook_time": codebook_time,
        "optimization_time": optimization_time,
        "encode_time": encode_time,
        "build_time": build_time,
        "search_time": search_time,
        "avg_search_time_ms": avg_search_time * 1000,
        "backend": index.backend,
        "final_qap_cost": float(hasher.cost_),
        "n_base": base_embeddings.shape[0],
        "n_queries": queries.shape[0],
        "dim": base_embeddings.shape[1],
        "block_size": block_size,
        "max_two_swap_iters": max_two_swap_iters,
        "num_tunneling_steps": num_tunneling_steps,
    }
    
    results_dir = Path(__file__).parent.parent / "experiments" / "real"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / f"results_gray_tunneled_{dataset_name}_bits{n_bits}_codes{n_codes}_k{k}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    print("\n" + "=" * 70)
    print("Experiment completed!")
    print("=" * 70)
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Gray-Tunneled experiment on real embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., 'quora')",
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        default=64,
        help="Number of bits for binary codes (default: 64)",
    )
    parser.add_argument(
        "--n-codes",
        type=int,
        required=True,
        help="Number of codebook vectors (must be <= 2**n_bits)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Recall@k value (default: 10)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=8,
        help="Block size for tunneling (default: 8)",
    )
    parser.add_argument(
        "--max-two-swap-iters",
        type=int,
        default=50,
        help="Maximum iterations for 2-swap (default: 50)",
    )
    parser.add_argument(
        "--num-tunneling-steps",
        type=int,
        default=10,
        help="Number of tunneling steps (default: 10)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--no-faiss",
        action="store_true",
        help="Force use of Python backend (disable FAISS)",
    )
    
    args = parser.parse_args()
    
    run_gray_tunneled_experiment(
        dataset_name=args.dataset,
        n_bits=args.n_bits,
        n_codes=args.n_codes,
        k=args.k,
        block_size=args.block_size,
        max_two_swap_iters=args.max_two_swap_iters,
        num_tunneling_steps=args.num_tunneling_steps,
        random_state=args.random_state,
        use_faiss=not args.no_faiss,
    )


if __name__ == "__main__":
    main()

