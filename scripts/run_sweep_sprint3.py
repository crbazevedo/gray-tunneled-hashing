#!/usr/bin/env python3
"""Unified sweep script for Sprint 3 hyperparameter tuning and ablation."""

import sys
import argparse
import json
import csv
import time
import itertools
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
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
from gray_tunneled_hashing.binary.codebooks import (
    build_codebook_kmeans,
    encode_with_codebook,
    find_nearest_centroids,
)
from gray_tunneled_hashing.algorithms.gray_tunneled_hasher import GrayTunneledHasher
from gray_tunneled_hashing.integrations.hamming_index import build_hamming_index
from gray_tunneled_hashing.evaluation.metrics import recall_at_k


def run_baseline_experiment(
    dataset_name: str,
    method: str,
    n_bits: int,
    k: int,
    random_state: int,
    use_faiss: bool = True,
) -> Dict[str, Any]:
    """
    Run baseline binary experiment.
    
    Returns:
        Dictionary with results
    """
    # Load data
    base_embeddings = load_embeddings(dataset_name, split="base")
    queries, ground_truth = load_queries_and_ground_truth(dataset_name, k=k)
    
    # Binarize
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
    
    # Build index
    start_time = time.time()
    index = build_hamming_index(base_codes, use_faiss=use_faiss)
    build_time = time.time() - start_time
    
    # Search
    start_time = time.time()
    retrieved_indices, distances = index.search(query_codes, k)
    search_time = time.time() - start_time
    avg_search_time = search_time / len(queries)
    
    # Compute recall
    recall = recall_at_k(retrieved_indices, ground_truth, k=k)
    
    return {
        "dataset": dataset_name,
        "method": f"baseline_{method}",
        "n_bits": n_bits if method == "random_proj" else base_codes.shape[1],
        "n_codes": None,
        "block_size": None,
        "num_tunneling_steps": None,
        "gt_mode": None,
        "block_selection_strategy": None,
        "k": k,
        "recall_at_k": float(recall),
        "build_time": binarize_time + build_time,
        "search_time": search_time,
        "avg_search_time_ms": avg_search_time * 1000,
        "backend": index.backend,
        "final_qap_cost": None,
        "n_base": base_embeddings.shape[0],
        "n_queries": queries.shape[0],
        "dim": base_embeddings.shape[1],
    }


def run_gray_tunneled_experiment(
    dataset_name: str,
    n_bits: int,
    n_codes: int,
    k: int,
    block_size: int,
    max_two_swap_iters: int,
    num_tunneling_steps: int,
    gt_mode: str,
    block_selection_strategy: str,
    random_state: int,
    use_faiss: bool = True,
) -> Dict[str, Any]:
    """
    Run Gray-Tunneled experiment.
    
    Returns:
        Dictionary with results
    """
    # Check n_codes <= 2**n_bits
    max_codes = 2 ** n_bits
    if n_codes > max_codes:
        raise ValueError(
            f"n_codes={n_codes} cannot exceed 2**n_bits={max_codes}"
        )
    
    # Load data
    base_embeddings = load_embeddings(dataset_name, split="base")
    queries, ground_truth = load_queries_and_ground_truth(dataset_name, k=k)
    
    # Build codebook
    start_time = time.time()
    centroids, assignments = build_codebook_kmeans(
        base_embeddings,
        n_codes=n_codes,
        random_state=random_state,
    )
    codebook_time = time.time() - start_time
    
    # For trivial mode, we still need to fit the hasher but with minimal optimization
    # For two_swap_only and full, we fit normally
    
    # Prepare centroids for hasher (need exactly 2**n_bits)
    # We'll pad or subsample if needed
    if n_codes < max_codes:
        # Pad with duplicates or random vectors
        n_pad = max_codes - n_codes
        # Duplicate last centroid
        padding = np.tile(centroids[-1:], (n_pad, 1))
        centroids_for_hasher = np.vstack([centroids, padding])
    elif n_codes == max_codes:
        centroids_for_hasher = centroids
    else:
        # This shouldn't happen due to check above
        centroids_for_hasher = centroids[:max_codes]
    
    # Fit hasher
    start_time = time.time()
    
    # For cluster-based block selection, we need cluster assignments
    cluster_assignments = None
    if block_selection_strategy == "cluster":
        # Use codebook assignments as cluster assignments
        # But we need to map them to the hasher's embedding indices
        # For now, we'll use assignments directly (this might need refinement)
        cluster_assignments = assignments
    
    hasher = GrayTunneledHasher(
        n_bits=n_bits,
        block_size=block_size,
        max_two_swap_iters=max_two_swap_iters,
        num_tunneling_steps=num_tunneling_steps if gt_mode == "full" else 0,
        two_swap_sample_size=256,
        init_strategy="random",
        random_state=random_state,
        mode=gt_mode,
        track_history=False,  # Can be enabled for detailed analysis
        block_selection_strategy=block_selection_strategy if gt_mode == "full" else "random",
        cluster_assignments=cluster_assignments,
    )
    
    hasher.fit(centroids_for_hasher)
    optimization_time = time.time() - start_time
    
    # Get assignment and create mapping
    pi = hasher.get_assignment()
    
    # Create centroid-to-code mapping
    # pi[u] = embedding index, so we need to map centroids to codes
    from gray_tunneled_hashing.data.synthetic_generators import generate_hypercube_vertices
    vertices = generate_hypercube_vertices(n_bits)
    
    # Map centroids to vertex indices
    centroid_to_code_map = np.zeros(n_codes, dtype=np.int32)
    for i in range(n_codes):
        vertices_with_centroid = np.where(pi == i)[0]
        if len(vertices_with_centroid) > 0:
            centroid_to_code_map[i] = vertices_with_centroid[0]
        else:
            # If not found (padded centroid), use fallback
            centroid_to_code_map[i] = i % max_codes
    
    # Convert to dictionary format expected by encode_with_codebook
    centroid_to_code = {}
    for i in range(n_codes):
        vertex_idx = centroid_to_code_map[i]
        binary_code = vertices[vertex_idx]
        centroid_to_code[i] = binary_code
    
    # Encode base embeddings
    start_time = time.time()
    base_codes = encode_with_codebook(
        base_embeddings,
        centroids,
        centroid_to_code,
        assignments=assignments,
    )
    encode_time = time.time() - start_time
    
    # Encode queries
    query_centroid_indices = find_nearest_centroids(queries, centroids)
    query_codes_binary = np.array([centroid_to_code[idx] for idx in query_centroid_indices], dtype=bool)
    
    # base_codes is already binary (from encode_with_codebook)
    base_codes_binary = base_codes
    
    # Build index
    start_time = time.time()
    index = build_hamming_index(base_codes_binary, use_faiss=use_faiss)
    build_time = time.time() - start_time
    
    # Search
    start_time = time.time()
    retrieved_indices, distances = index.search(query_codes_binary, k)
    search_time = time.time() - start_time
    avg_search_time = search_time / len(queries)
    
    # Compute recall
    recall = recall_at_k(retrieved_indices, ground_truth, k=k)
    
    return {
        "dataset": dataset_name,
        "method": f"gray_tunneled_{gt_mode}",
        "n_bits": n_bits,
        "n_codes": n_codes,
        "block_size": block_size if gt_mode == "full" else None,
        "num_tunneling_steps": num_tunneling_steps if gt_mode == "full" else None,
        "gt_mode": gt_mode,
        "block_selection_strategy": block_selection_strategy if gt_mode == "full" else None,
        "k": k,
        "recall_at_k": float(recall),
        "build_time": codebook_time + optimization_time + encode_time + build_time,
        "search_time": search_time,
        "avg_search_time_ms": avg_search_time * 1000,
        "backend": index.backend,
        "final_qap_cost": float(hasher.cost_) if hasher.cost_ is not None else None,
        "n_base": base_embeddings.shape[0],
        "n_queries": queries.shape[0],
        "dim": base_embeddings.shape[1],
        "codebook_time": codebook_time,
        "optimization_time": optimization_time,
        "encode_time": encode_time,
    }


def run_sweep(config_path: Path, output_dir: Path, use_faiss: bool = True, verbose: bool = True):
    """
    Run hyperparameter sweep from config file.
    
    Args:
        config_path: Path to YAML config file
        output_dir: Directory to save results
        use_faiss: Whether to use FAISS if available
        verbose: Whether to print progress
    """
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    datasets = config.get("datasets", [])
    if not datasets:
        raise ValueError("No datasets found in config")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    total_experiments = 0
    
    # Count total experiments for progress tracking
    for dataset_config in datasets:
        dataset_name = dataset_config["name"]
        n_bits_list = dataset_config["n_bits"]
        n_codes_list = dataset_config["n_codes"]
        block_sizes = dataset_config.get("block_sizes", [8])
        num_tunneling_steps_list = dataset_config.get("num_tunneling_steps", [10])
        gt_modes = dataset_config.get("gt_modes", ["full"])
        block_selection_strategies = dataset_config.get("block_selection_strategies", ["random"])
        k_values = dataset_config.get("k_values", [10])
        random_state = dataset_config.get("random_state", 42)
        
        # Count baselines
        total_experiments += len(n_bits_list) * len(k_values) * 2  # sign + random_proj
        
        # Count GT experiments
        for n_bits, n_codes, block_size, num_tunneling_steps, gt_mode, block_strategy, k in itertools.product(
            n_bits_list,
            n_codes_list,
            block_sizes,
            num_tunneling_steps_list,
            gt_modes,
            block_selection_strategies,
            k_values,
        ):
            # Skip invalid combinations
            if gt_mode == "trivial" and (block_size is not None or num_tunneling_steps != 0):
                continue
            if gt_mode == "two_swap_only" and num_tunneling_steps != 0:
                continue
            if gt_mode != "full" and block_strategy != "random":
                continue
            
            total_experiments += 1
    
    print(f"Total experiments to run: {total_experiments}")
    print(f"Results will be saved to: {output_dir}\n")
    
    experiment_num = 0
    
    # Run experiments
    for dataset_config in datasets:
        dataset_name = dataset_config["name"]
        n_bits_list = dataset_config["n_bits"]
        n_codes_list = dataset_config["n_codes"]
        block_sizes = dataset_config.get("block_sizes", [8])
        num_tunneling_steps_list = dataset_config.get("num_tunneling_steps", [10])
        gt_modes = dataset_config.get("gt_modes", ["full"])
        block_selection_strategies = dataset_config.get("block_selection_strategies", ["random"])
        k_values = dataset_config.get("k_values", [10])
        random_state = dataset_config.get("random_state", 42)
        max_two_swap_iters = dataset_config.get("max_two_swap_iters", 50)
        two_swap_sample_size = dataset_config.get("two_swap_sample_size", 256)
        
        print(f"\n{'=' * 70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 70}\n")
        
        # Run baselines
        for n_bits, k in itertools.product(n_bits_list, k_values):
            # Sign baseline
            experiment_num += 1
            if verbose:
                print(f"[{experiment_num}/{total_experiments}] Baseline: sign, n_bits={n_bits}, k={k}")
            try:
                result = run_baseline_experiment(
                    dataset_name=dataset_name,
                    method="sign",
                    n_bits=n_bits,  # Not used for sign, but included for consistency
                    k=k,
                    random_state=random_state,
                    use_faiss=use_faiss,
                )
                all_results.append(result)
                if verbose:
                    print(f"  ✓ Recall@{k}: {result['recall_at_k']:.4f}")
            except Exception as e:
                print(f"  ✗ Error: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
            
            # Random projection baseline
            experiment_num += 1
            if verbose:
                print(f"[{experiment_num}/{total_experiments}] Baseline: random_proj, n_bits={n_bits}, k={k}")
            try:
                result = run_baseline_experiment(
                    dataset_name=dataset_name,
                    method="random_proj",
                    n_bits=n_bits,
                    k=k,
                    random_state=random_state,
                    use_faiss=use_faiss,
                )
                all_results.append(result)
                if verbose:
                    print(f"  ✓ Recall@{k}: {result['recall_at_k']:.4f}")
            except Exception as e:
                print(f"  ✗ Error: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
        
        # Run Gray-Tunneled experiments
        for n_bits, n_codes, block_size, num_tunneling_steps, gt_mode, block_strategy, k in itertools.product(
            n_bits_list,
            n_codes_list,
            block_sizes,
            num_tunneling_steps_list,
            gt_modes,
            block_selection_strategies,
            k_values,
        ):
            # Skip invalid combinations
            if gt_mode == "trivial":
                # For trivial, ignore block_size and num_tunneling_steps
                block_size = None
                num_tunneling_steps = 0
                block_strategy = "random"
            elif gt_mode == "two_swap_only":
                num_tunneling_steps = 0
                block_strategy = "random"
            elif gt_mode != "full":
                continue
            
            experiment_num += 1
            if verbose:
                print(
                    f"[{experiment_num}/{total_experiments}] GT: {gt_mode}, "
                    f"n_bits={n_bits}, n_codes={n_codes}, block_size={block_size}, "
                    f"tunneling_steps={num_tunneling_steps}, block_strategy={block_strategy}, k={k}"
                )
            
            try:
                result = run_gray_tunneled_experiment(
                    dataset_name=dataset_name,
                    n_bits=n_bits,
                    n_codes=n_codes,
                    k=k,
                    block_size=block_size or 8,  # Default if None
                    max_two_swap_iters=max_two_swap_iters,
                    num_tunneling_steps=num_tunneling_steps,
                    gt_mode=gt_mode,
                    block_selection_strategy=block_strategy,
                    random_state=random_state,
                    use_faiss=use_faiss,
                )
                all_results.append(result)
                if verbose:
                    print(f"  ✓ Recall@{k}: {result['recall_at_k']:.4f}, Cost: {result.get('final_qap_cost', 'N/A')}")
            except Exception as e:
                print(f"  ✗ Error: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
    
    # Save results
    print(f"\n{'=' * 70}")
    print("Saving results...")
    print(f"{'=' * 70}\n")
    
    # Save JSON
    json_path = output_dir / "results_sprint3_sweep.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Saved JSON: {json_path}")
    
    # Save CSV
    csv_path = output_dir / "results_sprint3_sweep.csv"
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"✓ Saved CSV: {csv_path}")
    
    print(f"\nTotal results: {len(all_results)}")
    print(f"Results saved to: {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Sprint 3 hyperparameter sweep from YAML config"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "experiments" / "real" / "configs_sprint3.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "experiments" / "real",
        help="Directory to save results",
    )
    parser.add_argument(
        "--no-faiss",
        action="store_true",
        help="Disable FAISS (use pure Python backend)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    
    args = parser.parse_args()
    
    if not args.config.exists():
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)
    
    run_sweep(
        config_path=args.config,
        output_dir=args.output_dir,
        use_faiss=not args.no_faiss,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

