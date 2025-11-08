#!/usr/bin/env python3
"""Run baseline binary experiment on real embeddings."""

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
from gray_tunneled_hashing.binary.baselines import (
    sign_binarize,
    random_projection_binarize,
    apply_random_projection,
)
from gray_tunneled_hashing.integrations.hamming_index import build_hamming_index
from gray_tunneled_hashing.evaluation.metrics import recall_at_k


def run_baseline_experiment(
    dataset_name: str,
    method: str,
    n_bits: int,
    k: int,
    random_state: int,
    use_faiss: bool = True,
):
    """
    Run baseline binary experiment.
    
    Args:
        dataset_name: Name of the dataset
        method: Binarization method ('sign' or 'random_proj')
        n_bits: Number of bits for binary codes (only for random_proj)
        k: Recall@k value
        random_state: Random seed
        use_faiss: Whether to try FAISS (fallback to Python if not available)
        
    Returns:
        Dictionary with results
    """
    print("=" * 70)
    print("Baseline Binary Experiment")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Method: {method}")
    if method == "random_proj":
        print(f"  n_bits: {n_bits}")
    print(f"  k (recall@k): {k}")
    print(f"  Random state: {random_state}")
    print(f"  Use FAISS: {use_faiss}\n")
    
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
    
    # Step 2: Binarize
    print(f"\nStep 2: Binarizing embeddings ({method})...")
    start_time = time.time()
    
    if method == "sign":
        base_codes = sign_binarize(base_embeddings)
        query_codes = sign_binarize(queries)
        projection_matrix = None
    elif method == "random_proj":
        base_codes, projection_matrix = random_projection_binarize(
            base_embeddings, n_bits=n_bits, random_state=random_state
        )
        query_codes = apply_random_projection(queries, projection_matrix)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    binarize_time = time.time() - start_time
    print(f"  ✓ Base codes: {base_codes.shape}, dtype={base_codes.dtype}")
    print(f"  ✓ Query codes: {query_codes.shape}, dtype={query_codes.dtype}")
    print(f"  ✓ Binarization time: {binarize_time:.2f}s")
    
    # Step 3: Build index
    print(f"\nStep 3: Building Hamming index...")
    start_time = time.time()
    
    try:
        index = build_hamming_index(base_codes, use_faiss=use_faiss)
        print(f"  ✓ Index built (backend: {index.backend})")
    except Exception as e:
        print(f"  ✗ Error building index: {e}")
        sys.exit(1)
    
    build_time = time.time() - start_time
    print(f"  ✓ Build time: {build_time:.2f}s")
    
    # Step 4: Search
    print(f"\nStep 4: Searching for k={k} neighbors...")
    start_time = time.time()
    
    retrieved_indices, distances = index.search(query_codes, k)
    
    search_time = time.time() - start_time
    avg_search_time = search_time / len(queries)
    print(f"  ✓ Retrieved indices: {retrieved_indices.shape}")
    print(f"  ✓ Total search time: {search_time:.4f}s")
    print(f"  ✓ Avg search time per query: {avg_search_time*1000:.2f}ms")
    
    # Step 5: Compute recall
    print(f"\nStep 5: Computing recall@k...")
    recall = recall_at_k(retrieved_indices, ground_truth, k=k)
    print(f"  ✓ Recall@{k}: {recall:.4f} ({recall*100:.2f}%)")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nMethod: {method}")
    if method == "random_proj":
        print(f"n_bits: {n_bits}")
    print(f"Recall@{k}: {recall:.4f}")
    print(f"Binarization time: {binarize_time:.2f}s")
    print(f"Index build time: {build_time:.2f}s")
    print(f"Search time: {search_time:.4f}s ({avg_search_time*1000:.2f}ms per query)")
    print(f"Backend: {index.backend}")
    
    # Save results
    results = {
        "dataset": dataset_name,
        "method": method,
        "n_bits": n_bits if method == "random_proj" else base_codes.shape[1],
        "k": k,
        "recall_at_k": float(recall),
        "binarize_time": binarize_time,
        "build_time": build_time,
        "search_time": search_time,
        "avg_search_time_ms": avg_search_time * 1000,
        "backend": index.backend,
        "n_base": base_embeddings.shape[0],
        "n_queries": queries.shape[0],
        "dim": base_embeddings.shape[1],
    }
    
    results_dir = Path(__file__).parent.parent / "experiments" / "real"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / f"results_baseline_{method}_{dataset_name}_k{k}.json"
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
        description="Run baseline binary experiment on real embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., 'quora')",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["sign", "random_proj"],
        required=True,
        help="Binarization method: 'sign' or 'random_proj'",
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        default=64,
        help="Number of bits for binary codes (only for random_proj, default: 64)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Recall@k value (default: 10)",
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
    
    if args.method == "sign" and args.n_bits != 64:
        print("Warning: --n-bits is ignored for 'sign' method (uses all dimensions)")
    
    run_baseline_experiment(
        dataset_name=args.dataset,
        method=args.method,
        n_bits=args.n_bits,
        k=args.k,
        random_state=args.random_state,
        use_faiss=not args.no_faiss,
    )


if __name__ == "__main__":
    main()

