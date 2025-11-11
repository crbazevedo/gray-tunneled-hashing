#!/usr/bin/env python3
"""
Benchmark completo para Sprint 8 - Comparação GTH vs Baselines com dados reais.

Este script executa benchmark comparando:
- Baselines: Hyperplane LSH, p-stable LSH, Random Projection (sem GTH)
- GTH Sprint 8: Com nova estrutura (K, n_bits) e objetivo sobre embeddings reais

Métricas coletadas:
- Recall@k
- Build time
- Search time
- J(φ) cost e improvement
- Hamming ball coverage
- Candidates per query
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from itertools import product
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gray_tunneled_hashing.binary.lsh_families import create_lsh_family
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.api.query_pipeline import query_with_hamming_ball
from gray_tunneled_hashing.evaluation.metrics import recall_at_k, hamming_distance
from gray_tunneled_hashing.distribution.j_phi_objective import compute_j_phi_cost_real_embeddings


def parse_list_arg(arg_str: str, dtype=int) -> List:
    """Parse comma-separated list argument."""
    if not arg_str:
        return []
    return [dtype(x.strip()) for x in arg_str.split(",")]


def load_or_generate_data(dataset_name: str, n_base: int = 1000, n_queries: int = 100, 
                         dim: int = 64, k: int = 10, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load real data or generate synthetic."""
    if dataset_name == "synthetic":
        print(f"Generating synthetic dataset: N={n_base}, Q={n_queries}, dim={dim}, k={k}")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "generate_synthetic_dataset_for_benchmark",
            project_root / "scripts" / "generate_synthetic_dataset_for_benchmark.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        base_embeddings, queries, ground_truth = module.generate_synthetic_dataset(
            n_base=n_base, n_queries=n_queries, dim=dim, k=k, random_state=random_state
        )
        return base_embeddings, queries, ground_truth
    else:
        # Try to load real dataset
        try:
            from gray_tunneled_hashing.data.real_datasets import (
                load_embeddings,
                load_queries_and_ground_truth,
            )
            base_embeddings = load_embeddings(dataset_name, split="base")
            queries, ground_truth = load_queries_and_ground_truth(dataset_name, k=k)
            return base_embeddings, queries, ground_truth
        except FileNotFoundError:
            print(f"Dataset '{dataset_name}' not found, falling back to synthetic")
            return load_or_generate_data("synthetic", n_base, n_queries, dim, k, random_state)


def compute_baseline_recall(
    queries: np.ndarray,
    base_embeddings: np.ndarray,
    ground_truth: np.ndarray,
    lsh_encoder,
    k: int,
    hamming_radius: int = 1,
) -> Dict[str, Any]:
    """Compute baseline recall using LSH without GTH."""
    query_codes = lsh_encoder(queries)
    base_codes = lsh_encoder(base_embeddings)
    
    start_time = time.time()
    recalls = []
    candidates_per_query = []
    
    for i in range(len(queries)):
        query_code = query_codes[i]
        # Find all base codes within Hamming radius
        hamming_dists = np.sum(query_code != base_codes, axis=1)
        candidates = np.where(hamming_dists <= hamming_radius)[0]
        candidates_per_query.append(len(candidates))
        
        # Compute recall
        if len(candidates) > 0:
            retrieved = set(candidates[:k])
            relevant = set(ground_truth[i][:k])
            recall = len(retrieved & relevant) / len(relevant) if len(relevant) > 0 else 0.0
            recalls.append(recall)
        else:
            recalls.append(0.0)
    
    search_time = time.time() - start_time
    avg_recall = np.mean(recalls) if recalls else 0.0
    std_recall = np.std(recalls) if len(recalls) > 1 else 0.0
    
    return {
        "recall": float(avg_recall),
        "recall_std": float(std_recall),
        "search_time_s": float(search_time),
        "avg_search_time_ms": float(search_time / len(queries) * 1000),
        "avg_candidates_per_query": float(np.mean(candidates_per_query)),
        "hamming_radius": hamming_radius,
    }


def compute_gth_recall_sprint8(
    queries: np.ndarray,
    base_embeddings: np.ndarray,
    ground_truth: np.ndarray,
    index_obj,
    lsh_encoder,
    k: int,
    hamming_radius: int = 1,
) -> Dict[str, Any]:
    """Compute GTH recall using Sprint 8 structure."""
    query_codes = lsh_encoder(queries)
    base_codes = lsh_encoder(base_embeddings)
    
    # Build bucket_to_dataset_indices mapping
    bucket_to_dataset_indices = {}
    for dataset_idx, code in enumerate(base_codes):
        code_tuple = tuple(code.astype(int).tolist())
        if code_tuple in index_obj.code_to_bucket:
            bucket_idx = index_obj.code_to_bucket[code_tuple]
            if bucket_idx not in bucket_to_dataset_indices:
                bucket_to_dataset_indices[bucket_idx] = []
            bucket_to_dataset_indices[bucket_idx].append(dataset_idx)
    
    # Get permutation (Sprint 8: shape (K, n_bits))
    permutation = index_obj.hasher.get_assignment()
    
    start_time = time.time()
    recalls = []
    candidates_per_query = []
    neighbors_in_ball = 0
    total_neighbors = 0
    
    for i in range(len(queries)):
        query_code = query_codes[i].astype(bool)
        
        result = query_with_hamming_ball(
            query_code=query_code,
            permutation=permutation,
            code_to_bucket=index_obj.code_to_bucket,
            bucket_to_code=index_obj.bucket_to_code,
            n_bits=index_obj.n_bits,
            hamming_radius=hamming_radius,
        )
        
        candidates_per_query.append(result.n_candidates)
        
        # Get dataset indices from bucket indices
        retrieved_indices = []
        for bucket_idx in result.candidate_indices:
            if bucket_idx in bucket_to_dataset_indices:
                retrieved_indices.extend(bucket_to_dataset_indices[bucket_idx])
        
        # Compute recall
        retrieved = set(retrieved_indices[:k])
        relevant = set(ground_truth[i][:k])
        total_neighbors += len(relevant)
        neighbors_in_ball += len(retrieved & relevant)
        
        recall = len(retrieved & relevant) / len(relevant) if len(relevant) > 0 else 0.0
        recalls.append(recall)
    
    search_time = time.time() - start_time
    avg_recall = np.mean(recalls) if recalls else 0.0
    std_recall = np.std(recalls) if len(recalls) > 1 else 0.0
    coverage = neighbors_in_ball / total_neighbors if total_neighbors > 0 else 0.0
    
    return {
        "recall": float(avg_recall),
        "recall_std": float(std_recall),
        "search_time_s": float(search_time),
        "avg_search_time_ms": float(search_time / len(queries) * 1000),
        "avg_candidates_per_query": float(np.mean(candidates_per_query)),
        "hamming_ball_coverage": float(coverage),
        "hamming_radius": hamming_radius,
    }


def run_baseline_experiment(
    queries: np.ndarray,
    base_embeddings: np.ndarray,
    ground_truth: np.ndarray,
    lsh_family_name: str,
    n_bits: int,
    k: int,
    hamming_radius: int,
    random_state: int,
) -> Dict[str, Any]:
    """Run baseline experiment (LSH without GTH)."""
    lsh = create_lsh_family(lsh_family_name, n_bits=n_bits, dim=base_embeddings.shape[1], random_state=random_state)
    
    result = compute_baseline_recall(
        queries=queries,
        base_embeddings=base_embeddings,
        ground_truth=ground_truth,
        lsh_encoder=lsh.hash,
        k=k,
        hamming_radius=hamming_radius,
    )
    
    result.update({
        "method": f"baseline_{lsh_family_name}",
        "n_bits": n_bits,
        "k": k,
        "build_time_s": 0.0,  # Baselines are instant
    })
    
    return result


def run_gth_experiment_sprint8(
    queries: np.ndarray,
    base_embeddings: np.ndarray,
    ground_truth: np.ndarray,
    lsh_family_name: str,
    n_bits: int,
    n_codes: int,
    k: int,
    hamming_radius: int,
    max_two_swap_iters: int,
    num_tunneling_steps: int,
    mode: str,
    random_state: int,
    # NEW (Sprint 9): Multi-radius and tunneling parameters
    hamming_radii: Optional[List[int]] = None,
    radius_weights: Optional[np.ndarray] = None,
    tunneling_on_stagnation: bool = False,
    tunneling_probability: float = 0.0,
    stagnation_window: int = 10,
    stagnation_threshold: float = 0.001,
) -> Dict[str, Any]:
    """Run GTH experiment with Sprint 8 structure."""
    lsh = create_lsh_family(lsh_family_name, n_bits=n_bits, dim=base_embeddings.shape[1], random_state=random_state)
    
    # Build distribution-aware index (Sprint 8 uses real embeddings objective by default)
    # NEW (Sprint 9): Pass multi-radius and tunneling parameters
    build_start = time.time()
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        n_bits=n_bits,
        n_codes=n_codes,
        use_codebook=True,
        max_two_swap_iters=max_two_swap_iters,
        num_tunneling_steps=num_tunneling_steps,
        mode=mode,
        random_state=random_state,
        # NEW (Sprint 9): Multi-radius objective
        hamming_radii=hamming_radii,
        radius_weights=radius_weights,
        # NEW (Sprint 9): Tunneling support
        tunneling_on_stagnation=tunneling_on_stagnation,
        tunneling_probability=tunneling_probability,
        stagnation_window=stagnation_window,
        stagnation_threshold=stagnation_threshold,
    )
    build_time = time.time() - build_start
    
    # Compute J(φ) cost (final)
    permutation = index_obj.hasher.get_assignment()
    j_phi_cost = compute_j_phi_cost_real_embeddings(
        permutation=permutation,
        pi=index_obj.pi,
        w=index_obj.w,
        queries=queries,
        base_embeddings=base_embeddings,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        code_to_bucket=index_obj.code_to_bucket,
        n_bits=n_bits,
    )
    
    # Compute initial J(φ) cost (identity permutation - bucket_to_code)
    initial_permutation = index_obj.bucket_to_code.astype(np.uint8)
    j_phi_initial = compute_j_phi_cost_real_embeddings(
        permutation=initial_permutation,
        pi=index_obj.pi,
        w=index_obj.w,
        queries=queries,
        base_embeddings=base_embeddings,
        ground_truth_neighbors=ground_truth,
        encoder=lsh.hash,
        code_to_bucket=index_obj.code_to_bucket,
        n_bits=n_bits,
    )
    
    j_phi_improvement = (j_phi_initial - j_phi_cost) / j_phi_initial if j_phi_initial > 0 else 0.0
    
    # Compute recall
    recall_result = compute_gth_recall_sprint8(
        queries=queries,
        base_embeddings=base_embeddings,
        ground_truth=ground_truth,
        index_obj=index_obj,
        lsh_encoder=lsh.hash,
        k=k,
        hamming_radius=hamming_radius,
    )
    
    result = {
        "method": f"gth_sprint8_{lsh_family_name}",
        "n_bits": n_bits,
        "n_codes": n_codes,
        "k": k,
        "max_two_swap_iters": max_two_swap_iters,
        "num_tunneling_steps": num_tunneling_steps,
        "mode": mode,
        "build_time_s": float(build_time),
        "j_phi_cost": float(j_phi_cost),
        "j_phi_initial": float(j_phi_initial),
        "j_phi_improvement": float(j_phi_improvement),
    }
    result.update(recall_result)
    
    return result


def run_benchmark(
    dataset_name: str,
    n_bits_list: List[int],
    n_codes_list: List[int],
    k_list: List[int],
    hamming_radius_list: List[int],
    max_iters_list: List[int],
    tunneling_steps_list: List[int],
    mode_list: List[str],
    n_runs: int,
    random_state: int,
    quick: bool = False,
    # NEW (Sprint 9): Multi-radius and tunneling parameters
    hamming_radii: Optional[List[int]] = None,
    radius_weights: Optional[np.ndarray] = None,
    tunneling_on_stagnation: bool = False,
    tunneling_probability: float = 0.0,
    stagnation_window: int = 10,
    stagnation_threshold: float = 0.001,
) -> Dict[str, Any]:
    """Run complete benchmark."""
    print("=" * 80)
    print("Sprint 8 Benchmark - GTH vs Baselines")
    print("=" * 80)
    print()
    
    # Load or generate data
    base_embeddings, queries, ground_truth = load_or_generate_data(
        dataset_name=dataset_name,
        n_base=1000 if quick else 5000,
        n_queries=100 if quick else 500,
        dim=64,
        k=max(k_list) if k_list else 10,
        random_state=random_state,
    )
    
    print(f"Dataset: {dataset_name}")
    print(f"Base embeddings: {base_embeddings.shape}")
    print(f"Queries: {queries.shape}")
    print(f"Ground truth: {ground_truth.shape}")
    print()
    
    results = {
        "metadata": {
            "sprint": "8",
            "dataset": dataset_name,
            "n_base": int(base_embeddings.shape[0]),
            "n_queries": int(queries.shape[0]),
            "dim": int(base_embeddings.shape[1]),
            "n_runs": n_runs,
            "random_state": random_state,
        },
        "baselines": {},
        "gth_sprint8": {},
        "comparisons": {},
    }
    
    # Run baselines
    print("Running baselines...")
    lsh_families = ["hyperplane", "p_stable"]
    
    for lsh_family in tqdm(lsh_families, desc="Baselines"):
        for n_bits in n_bits_list:
            for radius in hamming_radius_list:
                key = f"{lsh_family}_nbits{n_bits}_radius{radius}"
                baseline_results = []
                
                for run in range(n_runs):
                    result = run_baseline_experiment(
                        queries=queries,
                        base_embeddings=base_embeddings,
                        ground_truth=ground_truth,
                        lsh_family_name=lsh_family,
                        n_bits=n_bits,
                        k=max(k_list) if k_list else 10,
                        hamming_radius=radius,
                        random_state=random_state + run,
                    )
                    baseline_results.append(result)
                
                # Average results
                avg_result = {
                    "recall": float(np.mean([r["recall"] for r in baseline_results])),
                    "recall_std": float(np.std([r["recall"] for r in baseline_results])),
                    "search_time_s": float(np.mean([r["search_time_s"] for r in baseline_results])),
                    "avg_search_time_ms": float(np.mean([r["avg_search_time_ms"] for r in baseline_results])),
                    "avg_candidates_per_query": float(np.mean([r["avg_candidates_per_query"] for r in baseline_results])),
                }
                avg_result.update(baseline_results[0])  # Add metadata
                results["baselines"][key] = avg_result
    
    # Run GTH Sprint 8
    print("\nRunning GTH Sprint 8...")
    configs = list(product(
        lsh_families,
        n_bits_list,
        n_codes_list,
        k_list if k_list else [10],
        hamming_radius_list,
        max_iters_list,
        tunneling_steps_list,
        mode_list,
    ))
    
    for config in tqdm(configs, desc="GTH Sprint 8"):
        lsh_family, n_bits, n_codes, k, radius, max_iters, tunneling_steps, mode = config
        
        # Skip if n_codes > 2**n_bits
        if n_codes > 2**n_bits:
            continue
        
        key = f"{lsh_family}_nbits{n_bits}_ncodes{n_codes}_k{k}_radius{radius}_iters{max_iters}_tunnel{tunneling_steps}_mode{mode}"
        gth_results = []
        
        for run in range(n_runs):
            try:
                result = run_gth_experiment_sprint8(
                    queries=queries,
                    base_embeddings=base_embeddings,
                    ground_truth=ground_truth,
                    lsh_family_name=lsh_family,
                    n_bits=n_bits,
                    n_codes=n_codes,
                    k=k,
                    hamming_radius=radius,
                    max_two_swap_iters=max_iters,
                    num_tunneling_steps=tunneling_steps,
                    mode=mode,
                    random_state=random_state + run,
                    # NEW (Sprint 9): Pass multi-radius and tunneling parameters
                    hamming_radii=hamming_radii,
                    radius_weights=radius_weights,
                    tunneling_on_stagnation=tunneling_on_stagnation,
                    tunneling_probability=tunneling_probability,
                    stagnation_window=stagnation_window,
                    stagnation_threshold=stagnation_threshold,
                )
                gth_results.append(result)
            except Exception as e:
                print(f"\nError in config {key}, run {run}: {e}")
                continue
        
        if not gth_results:
            continue
        
        # Average results
        avg_result = {
            "recall": float(np.mean([r["recall"] for r in gth_results])),
            "recall_std": float(np.std([r["recall"] for r in gth_results])),
            "build_time_s": float(np.mean([r["build_time_s"] for r in gth_results])),
            "search_time_s": float(np.mean([r["search_time_s"] for r in gth_results])),
            "avg_search_time_ms": float(np.mean([r["avg_search_time_ms"] for r in gth_results])),
            "j_phi_cost": float(np.mean([r["j_phi_cost"] for r in gth_results])),
            "j_phi_initial": float(np.mean([r["j_phi_initial"] for r in gth_results])),
            "j_phi_improvement": float(np.mean([r["j_phi_improvement"] for r in gth_results])),
            "hamming_ball_coverage": float(np.mean([r.get("hamming_ball_coverage", 0.0) for r in gth_results])),
            "avg_candidates_per_query": float(np.mean([r["avg_candidates_per_query"] for r in gth_results])),
        }
        avg_result.update(gth_results[0])  # Add metadata
        results["gth_sprint8"][key] = avg_result
    
    # Compute comparisons
    print("\nComputing comparisons...")
    for lsh_family in lsh_families:
        for n_bits in n_bits_list:
            for radius in hamming_radius_list:
                baseline_key = f"{lsh_family}_nbits{n_bits}_radius{radius}"
                baseline_recall = results["baselines"].get(baseline_key, {}).get("recall", 0.0)
                
                # Find best GTH result for this configuration
                best_gth_recall = 0.0
                best_gth_key = None
                
                for gth_key, gth_result in results["gth_sprint8"].items():
                    if (f"{lsh_family}_nbits{n_bits}" in gth_key and 
                        f"radius{radius}" in gth_key):
                        if gth_result["recall"] > best_gth_recall:
                            best_gth_recall = gth_result["recall"]
                            best_gth_key = gth_key
                
                if best_gth_key:
                    comparison_key = f"{lsh_family}_nbits{n_bits}_radius{radius}"
                    results["comparisons"][comparison_key] = {
                        "baseline_recall": baseline_recall,
                        "gth_recall": best_gth_recall,
                        "recall_improvement": best_gth_recall - baseline_recall,
                        "relative_improvement_pct": ((best_gth_recall - baseline_recall) / baseline_recall * 100) if baseline_recall > 0 else 0.0,
                        "is_better": best_gth_recall > baseline_recall,
                        "best_gth_config": best_gth_key,
                    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Sprint 8 Benchmark - GTH vs Baselines")
    parser.add_argument("--dataset", type=str, default="synthetic", help="Dataset name or 'synthetic'")
    parser.add_argument("--n-bits", type=str, default="6,8", help="Comma-separated list of n_bits")
    parser.add_argument("--n-codes", type=str, default="16,32", help="Comma-separated list of n_codes")
    parser.add_argument("--k", type=str, default="10", help="Comma-separated list of k values")
    parser.add_argument("--hamming-radius", type=str, default="1,2", help="Comma-separated list of radii")
    parser.add_argument("--max-iters", type=str, default="10,20", help="Comma-separated list of max_two_swap_iters")
    parser.add_argument("--tunneling-steps", type=str, default="0", help="Comma-separated list of num_tunneling_steps")
    parser.add_argument("--mode", type=str, default="two_swap_only", help="Comma-separated list of modes")
    parser.add_argument("--n-runs", type=int, default=3, help="Number of runs for averaging")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="experiments/real/results_sprint8_benchmark.json", help="Output JSON file")
    parser.add_argument("--quick", action="store_true", help="Quick mode (reduced configurations)")
    # NEW (Sprint 9): Multi-radius objective
    parser.add_argument("--hamming-radii", type=str, default=None, help="Comma-separated list of Hamming radii for multi-radius objective, e.g., '1,2,3'")
    # NEW (Sprint 9): Tunneling parameters
    parser.add_argument("--tunneling-on-stagnation", action="store_true", help="Enable tunneling when stagnation detected")
    parser.add_argument("--tunneling-probability", type=float, default=0.0, help="Base probability for probabilistic tunneling (0.0 to 1.0)")
    parser.add_argument("--stagnation-window", type=int, default=10, help="Number of iterations for stagnation detection")
    parser.add_argument("--stagnation-threshold", type=float, default=0.001, help="Relative improvement threshold for stagnation (default: 0.001 = 0.1%%)")
    
    args = parser.parse_args()
    
    # Parse list arguments
    n_bits_list = parse_list_arg(args.n_bits)
    n_codes_list = parse_list_arg(args.n_codes)
    k_list = parse_list_arg(args.k)
    hamming_radius_list = parse_list_arg(args.hamming_radius)
    max_iters_list = parse_list_arg(args.max_iters)
    tunneling_steps_list = parse_list_arg(args.tunneling_steps)
    mode_list = parse_list_arg(args.mode, dtype=str)
    
    # NEW (Sprint 9): Parse multi-radius argument
    hamming_radii = None
    if args.hamming_radii:
        hamming_radii = parse_list_arg(args.hamming_radii)
        # Validate and generate weights if needed
        from gray_tunneled_hashing.distribution.j_phi_objective import validate_radius_weights
        radius_weights = validate_radius_weights(hamming_radii)
    else:
        radius_weights = None
    
    if args.quick:
        # Reduce configurations for quick mode
        n_bits_list = n_bits_list[:2] if len(n_bits_list) > 2 else n_bits_list
        n_codes_list = n_codes_list[:2] if len(n_codes_list) > 2 else n_codes_list
        k_list = k_list[:1] if len(k_list) > 1 else k_list
        hamming_radius_list = hamming_radius_list[:2] if len(hamming_radius_list) > 2 else hamming_radius_list
        max_iters_list = max_iters_list[:2] if len(max_iters_list) > 2 else max_iters_list
        tunneling_steps_list = tunneling_steps_list[:1] if len(tunneling_steps_list) > 1 else tunneling_steps_list
        mode_list = mode_list[:1] if len(mode_list) > 1 else mode_list
    
    print(f"Configuration:")
    print(f"  n_bits: {n_bits_list}")
    print(f"  n_codes: {n_codes_list}")
    print(f"  k: {k_list}")
    print(f"  hamming_radius: {hamming_radius_list}")
    print(f"  max_iters: {max_iters_list}")
    print(f"  tunneling_steps: {tunneling_steps_list}")
    print(f"  mode: {mode_list}")
    print(f"  n_runs: {args.n_runs}")
    print()
    
    # Run benchmark
    results = run_benchmark(
        dataset_name=args.dataset,
        n_bits_list=n_bits_list,
        n_codes_list=n_codes_list,
        k_list=k_list,
        hamming_radius_list=hamming_radius_list,
        max_iters_list=max_iters_list,
        tunneling_steps_list=tunneling_steps_list,
        mode_list=mode_list,
        n_runs=args.n_runs,
        random_state=args.random_state,
        quick=args.quick,
        # NEW (Sprint 9): Pass multi-radius and tunneling parameters
        hamming_radii=hamming_radii,
        radius_weights=radius_weights,
        tunneling_on_stagnation=args.tunneling_on_stagnation,
        tunneling_probability=args.tunneling_probability,
        stagnation_window=args.stagnation_window,
        stagnation_threshold=args.stagnation_threshold,
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"\nBaselines: {len(results['baselines'])} configurations")
    print(f"GTH Sprint 8: {len(results['gth_sprint8'])} configurations")
    print(f"Comparisons: {len(results['comparisons'])}")
    
    if results["comparisons"]:
        print("\nBest Comparisons:")
        for comp_key, comp in list(results["comparisons"].items())[:5]:
            status = "✅" if comp["is_better"] else "❌"
            print(f"  {status} {comp_key}: Baseline={comp['baseline_recall']:.4f}, "
                  f"GTH={comp['gth_recall']:.4f}, "
                  f"Improvement={comp['relative_improvement_pct']:+.2f}%")


if __name__ == "__main__":
    main()

