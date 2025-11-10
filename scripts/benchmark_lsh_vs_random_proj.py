"""
Benchmark comparing LSH families vs. random projection with GTH.

This script compares:
- Hyperplane LSH + GTH
- p-stable LSH + GTH  
- Random projection + GTH
- Baselines (without GTH)

Measures recall@k and impact of Hamming ball radius.

Features:
- Parallel execution with multiprocessing
- Progress bar with tqdm
- Resume functionality (checkpoint/resume)
- Detailed execution time logging
- Real-time metrics display
"""

import numpy as np
import json
import time
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import hashlib
import sys

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gray_tunneled_hashing.binary.lsh_families import (
    HyperplaneLSH,
    PStableLSH,
    create_lsh_family,
)
from gray_tunneled_hashing.binary.baselines import (
    random_projection_binarize,
    apply_random_projection,
)
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.integrations.hamming_index import build_hamming_index
from gray_tunneled_hashing.evaluation.metrics import recall_at_k
from gray_tunneled_hashing.api.query_pipeline import query_with_hamming_ball


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    method: str
    run_idx: int
    n_bits: int
    n_codes: int
    k: int
    hamming_radius: int
    n_samples: int
    n_queries: int
    dim: int
    random_state: int
    block_size: int = 4
    num_tunneling_steps: int = 0
    mode: str = "two_swap_only"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for hashing."""
        return asdict(self)
    
    def to_hash(self) -> str:
        """Generate unique hash for this config."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()


def run_single_experiment(config_dict: Dict) -> Dict[str, Any]:
    """
    Run a single experiment (for parallel execution).
    
    Args:
        config_dict: Dictionary with experiment configuration
        
    Returns:
        Dictionary with results including execution times
    """
    config = ExperimentConfig(**config_dict)
    
    # Set random state
    np.random.seed(config.random_state)
    
    # Generate synthetic data
    base_embeddings = np.random.randn(config.n_samples, config.dim).astype(np.float32)
    queries = np.random.randn(config.n_queries, config.dim).astype(np.float32)
    
    # Generate ground truth using actual distances
    from sklearn.metrics.pairwise import euclidean_distances
    distances = euclidean_distances(queries, base_embeddings)
    ground_truth = np.argsort(distances, axis=1)[:, :config.k].astype(np.int32)
    
    dim = base_embeddings.shape[1]
    total_start = time.time()
    
    try:
        if config.method.startswith("baseline_"):
            # Baseline without GTH
            method_type = config.method.replace("baseline_", "")
            
            build_start = time.time()
            if method_type == "hyperplane":
                lsh = HyperplaneLSH(n_bits=config.n_bits, dim=dim, random_state=config.random_state)
                base_codes = lsh.hash(base_embeddings)
                query_codes = lsh.hash(queries)
            elif method_type == "p_stable":
                lsh = PStableLSH(n_bits=config.n_bits, dim=dim, w=1.0, random_state=config.random_state)
                base_codes = lsh.hash(base_embeddings)
                query_codes = lsh.hash(queries)
            elif method_type == "random_proj":
                base_codes, proj_matrix = random_projection_binarize(
                    base_embeddings, n_bits=config.n_bits, random_state=config.random_state
                )
                query_codes = apply_random_projection(queries, proj_matrix)
            else:
                raise ValueError(f"Unknown baseline method: {method_type}")
            
            # Build index and search
            index = build_hamming_index(base_codes, use_faiss=True)
            search_start = time.time()
            retrieved_indices, distances = index.search(query_codes, config.k)
            search_time = time.time() - search_start
            build_time = time.time() - build_start
            
            recall = recall_at_k(retrieved_indices, ground_truth, k=config.k)
            
            return {
                "config_hash": config.to_hash(),
                "method": config.method,
                "run": config.run_idx,
                "n_bits": config.n_bits,
                "n_codes": config.n_codes,
                "k": config.k,
                "hamming_radius": config.hamming_radius,
                "recall": float(recall),
                "build_time": build_time,
                "search_time": search_time,
                "total_time": time.time() - total_start,
                "success": True,
                "error": None,
            }
        
        else:
            # With GTH
            build_start = time.time()
            
            if config.method == "hyperplane":
                lsh = HyperplaneLSH(n_bits=config.n_bits, dim=dim, random_state=config.random_state)
                encoder = None
            elif config.method == "p_stable":
                lsh = PStableLSH(n_bits=config.n_bits, dim=dim, w=1.0, random_state=config.random_state)
                encoder = None
            elif config.method == "random_proj":
                lsh = None
                _, proj_matrix = random_projection_binarize(
                    base_embeddings, n_bits=config.n_bits, random_state=config.random_state
                )
                encoder = lambda emb: apply_random_projection(emb, proj_matrix)
            else:
                raise ValueError(f"Unknown method: {config.method}")
            
            # Build distribution-aware index
            index_obj = build_distribution_aware_index(
                base_embeddings=base_embeddings,
                queries=queries,
                ground_truth_neighbors=ground_truth,
                encoder=encoder,
                n_bits=config.n_bits,
                n_codes=config.n_codes,
                use_codebook=True,
                use_semantic_distances=False,
                lsh_family=lsh,
                block_size=config.block_size,
                max_two_swap_iters=20,
                num_tunneling_steps=config.num_tunneling_steps,
                mode=config.mode,
                random_state=config.random_state,
            )
            
            build_time = time.time() - build_start
            
            # Query with Hamming ball
            search_start = time.time()
            
            # Encode queries
            if lsh is not None:
                query_codes = lsh.hash(queries)
            else:
                query_codes = encoder(queries)
            
            # Map base embeddings to buckets using LSH codes
            if lsh is not None:
                base_codes_lsh = lsh.hash(base_embeddings)
            else:
                base_codes_lsh = encoder(base_embeddings)
            
            bucket_to_dataset_indices = {}
            for dataset_idx, code in enumerate(base_codes_lsh):
                code_tuple = tuple(code.astype(int).tolist())
                if code_tuple in index_obj.code_to_bucket:
                    bucket_idx = index_obj.code_to_bucket[code_tuple]
                    if bucket_idx not in bucket_to_dataset_indices:
                        bucket_to_dataset_indices[bucket_idx] = []
                    bucket_to_dataset_indices[bucket_idx].append(dataset_idx)
            
            # Apply GTH permutation and expand Hamming ball
            all_retrieved = []
            for query_code in query_codes:
                # bucket_to_embedding_idx: bucket i maps to embedding i (for first K buckets)
                # This is the default assumption when bucket_to_embedding_idx=None
                result = query_with_hamming_ball(
                    query_code=query_code,
                    permutation=index_obj.permutation,
                    code_to_bucket=index_obj.code_to_bucket,
                    bucket_to_code=index_obj.bucket_to_code,
                    n_bits=config.n_bits,
                    hamming_radius=config.hamming_radius,
                    bucket_to_embedding_idx=None,  # Default: bucket i -> embedding i
                )
                
                candidate_dataset_indices = []
                for bucket_idx in result.candidate_indices:
                    if bucket_idx in bucket_to_dataset_indices:
                        candidate_dataset_indices.extend(bucket_to_dataset_indices[bucket_idx])
                
                candidate_dataset_indices = list(dict.fromkeys(candidate_dataset_indices))[:config.k]
                
                if len(candidate_dataset_indices) < config.k:
                    candidate_dataset_indices.extend([-1] * (config.k - len(candidate_dataset_indices)))
                else:
                    candidate_dataset_indices = candidate_dataset_indices[:config.k]
                
                all_retrieved.append(candidate_dataset_indices)
            
            retrieved_indices = np.array(all_retrieved, dtype=np.int32)
            search_time = time.time() - search_start
            
            recall = recall_at_k(retrieved_indices, ground_truth, k=config.k)
            
            return {
                "config_hash": config.to_hash(),
                "method": config.method,
                "run": config.run_idx,
                "n_bits": config.n_bits,
                "n_codes": config.n_codes,
                "k": config.k,
                "hamming_radius": config.hamming_radius,
                "recall": float(recall),
                "build_time": build_time,
                "search_time": search_time,
                "total_time": time.time() - total_start,
                "success": True,
                "error": None,
            }
    
    except Exception as e:
        return {
            "config_hash": config.to_hash(),
            "method": config.method,
            "run": config.run_idx,
            "n_bits": config.n_bits,
            "n_codes": config.n_codes,
            "k": config.k,
            "hamming_radius": config.hamming_radius,
            "recall": 0.0,
            "build_time": 0.0,
            "search_time": 0.0,
            "total_time": time.time() - total_start,
            "success": False,
            "error": str(e),
        }


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Load checkpoint if it exists."""
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    return {"completed_configs": set(), "results": []}


def save_checkpoint(checkpoint_path: Path, completed_configs: set, results: List[Dict]):
    """Save checkpoint."""
    checkpoint_data = {
        "completed_configs": list(completed_configs),
        "results": results,
        "timestamp": time.time(),
    }
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LSH vs. random projection with GTH"
    )
    parser.add_argument(
        "--n-bits", type=int, default=8, help="Number of bits (default: 8)"
    )
    parser.add_argument(
        "--n-codes", type=int, default=32, help="Number of codebook codes (default: 32)"
    )
    parser.add_argument(
        "--k", type=int, default=5, help="Number of neighbors to retrieve (default: 5)"
    )
    parser.add_argument(
        "--hamming-radius", type=int, default=1, help="Hamming ball radius (default: 1)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=100, help="Number of base samples (default: 100)"
    )
    parser.add_argument(
        "--n-queries", type=int, default=20, help="Number of queries (default: 20)"
    )
    parser.add_argument(
        "--dim", type=int, default=16, help="Embedding dimension (default: 16)"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--n-runs", type=int, default=3, help="Number of runs for averaging (default: 3)"
    )
    parser.add_argument(
        "--n-workers", type=int, default=None,
        help="Number of parallel workers (default: auto-detect)"
    )
    parser.add_argument(
        "--output", type=str, default="benchmark_lsh_results.json", help="Output file"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Checkpoint file for resume (default: <output>.checkpoint.json)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint if available"
    )
    parser.add_argument(
        "--block-size", type=int, default=4,
        help="Block size for tunneling (default: 4)"
    )
    parser.add_argument(
        "--num-tunneling-steps", type=int, default=0,
        help="Number of tunneling steps (default: 0)"
    )
    parser.add_argument(
        "--mode", type=str, default="two_swap_only",
        choices=["trivial", "two_swap_only", "full"],
        help="GTH mode (default: two_swap_only)"
    )
    
    args = parser.parse_args()
    
    # Auto-detect number of workers
    if args.n_workers is None:
        args.n_workers = min(mp.cpu_count(), args.n_runs * 6)  # 6 methods
    
    # Setup checkpoint path
    output_path = Path(args.output)
    if args.checkpoint is None:
        checkpoint_path = output_path.parent / f"{output_path.stem}.checkpoint.json"
    else:
        checkpoint_path = Path(args.checkpoint)
    
    # Load checkpoint if resuming
    completed_configs = set()
    all_results = []
    if args.resume and checkpoint_path.exists():
        checkpoint_data = load_checkpoint(checkpoint_path)
        completed_configs = set(checkpoint_data.get("completed_configs", []))
        all_results = checkpoint_data.get("results", [])
        print(f"Resuming from checkpoint: {len(completed_configs)} experiments already completed")
    
    # Generate all experiment configurations
    methods = [
        "baseline_hyperplane",
        "baseline_p_stable",
        "baseline_random_proj",
        "hyperplane",
        "p_stable",
        "random_proj",
    ]
    
    configs = []
    for method in methods:
        for run_idx in range(args.n_runs):
            run_seed = args.random_state + run_idx
            config = ExperimentConfig(
                method=method,
                run_idx=run_idx,
                n_bits=args.n_bits,
                n_codes=args.n_codes,
                k=args.k,
                hamming_radius=args.hamming_radius,
                n_samples=args.n_samples,
                n_queries=args.n_queries,
                dim=args.dim,
                random_state=run_seed,
                block_size=args.block_size,
                num_tunneling_steps=args.num_tunneling_steps,
                mode=args.mode,
            )
            config_hash = config.to_hash()
            
            # Skip if already completed
            if config_hash not in completed_configs:
                configs.append(config.to_dict())
    
    if not configs:
        print("All experiments already completed!")
        return
    
    print(f"\n{'='*70}")
    print(f"Benchmark Configuration")
    print(f"{'='*70}")
    print(f"  Methods: {len(methods)}")
    print(f"  Runs per method: {args.n_runs}")
    print(f"  Total experiments: {len(methods) * args.n_runs}")
    print(f"  Remaining: {len(configs)}")
    print(f"  Workers: {args.n_workers}")
    print(f"  Output: {output_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'='*70}\n")
    
    # Run experiments in parallel
    start_time = time.time()
    new_results = []
    
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {executor.submit(run_single_experiment, config): config for config in configs}
        
        # Progress bar
        desc = f"Running {len(configs)} experiments"
        pbar = tqdm(total=len(configs), desc=desc, unit="exp") if HAS_TQDM else None
        
        metrics_summary = {
            "completed": 0,
            "successful": 0,
            "failed": 0,
            "total_recall": 0.0,
            "total_build_time": 0.0,
        }
        
        for future in as_completed(futures):
            try:
                result = future.result()
                new_results.append(result)
                config = futures[future]
                config_hash = ExperimentConfig(**config).to_hash()
                completed_configs.add(config_hash)
                
                # Update metrics
                metrics_summary["completed"] += 1
                if result["success"]:
                    metrics_summary["successful"] += 1
                    metrics_summary["total_recall"] += result["recall"]
                    metrics_summary["total_build_time"] += result["build_time"]
                else:
                    metrics_summary["failed"] += 1
                
                # Update progress bar
                if pbar:
                    avg_recall = metrics_summary["total_recall"] / max(metrics_summary["successful"], 1)
                    avg_build = metrics_summary["total_build_time"] / max(metrics_summary["successful"], 1)
                    pbar.set_postfix({
                        "recall": f"{avg_recall:.3f}",
                        "build": f"{avg_build:.1f}s",
                        "ok": metrics_summary["successful"],
                        "fail": metrics_summary["failed"],
                    })
                    pbar.update(1)
                else:
                    # Fallback: print progress
                    if metrics_summary["completed"] % 5 == 0:
                        avg_recall = metrics_summary["total_recall"] / max(metrics_summary["successful"], 1)
                        print(f"  Progress: {metrics_summary['completed']}/{len(configs)} | "
                              f"Recall: {avg_recall:.3f} | "
                              f"Success: {metrics_summary['successful']}/{metrics_summary['completed']}")
                
                # Save checkpoint periodically
                if metrics_summary["completed"] % 10 == 0:
                    save_checkpoint(checkpoint_path, completed_configs, all_results + new_results)
            
            except Exception as e:
                config = futures[future]
                print(f"\n✗ Error in experiment {config['method']} run {config['run_idx']}: {e}")
                import traceback
                traceback.print_exc()
        
        if pbar:
            pbar.close()
    
    # Combine with existing results
    all_results.extend(new_results)
    
    # Save final checkpoint
    save_checkpoint(checkpoint_path, completed_configs, all_results)
    
    # Group results by method and compute statistics
    method_results = {}
    for result in all_results:
        method = result["method"]
        if method not in method_results:
            method_results[method] = []
        method_results[method].append(result)
    
    # Compute statistics
    final_summaries = []
    for method, results in method_results.items():
        successful_results = [r for r in results if r["success"]]
        if not successful_results:
            continue
        
        recalls = [r["recall"] for r in successful_results]
        build_times = [r["build_time"] for r in successful_results]
        search_times = [r["search_time"] for r in successful_results]
        
        summary = {
            "method": method,
            "n_bits": args.n_bits,
            "n_codes": args.n_codes,
            "k": args.k,
            "hamming_radius": args.hamming_radius,
            "n_runs": len(successful_results),
            "recall_mean": float(np.mean(recalls)),
            "recall_std": float(np.std(recalls)),
            "recall_min": float(np.min(recalls)),
            "recall_max": float(np.max(recalls)),
            "build_time_mean": float(np.mean(build_times)),
            "build_time_std": float(np.std(build_times)),
            "search_time_mean": float(np.mean(search_times)) if search_times[0] > 0 else 0.0,
            "search_time_std": float(np.std(search_times)) if search_times[0] > 0 else 0.0,
            "runs": successful_results,
        }
        final_summaries.append(summary)
    
    # Save final results
    with open(output_path, "w") as f:
        json.dump(final_summaries, f, indent=2)
    
    total_time = time.time() - start_time
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Results Summary")
    print(f"{'='*70}")
    print(f"Total execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Experiments completed: {metrics_summary['completed']}")
    print(f"Successful: {metrics_summary['successful']}")
    print(f"Failed: {metrics_summary['failed']}")
    print(f"\nResults by method:")
    print(f"{'Method':<25} | {'Recall':<15} | {'Build Time':<12} | {'Search Time':<12}")
    print(f"{'-'*70}")
    
    for summary in sorted(final_summaries, key=lambda x: x["recall_mean"], reverse=True):
        print(
            f"{summary['method']:<25} | "
            f"{summary['recall_mean']:.4f} ± {summary['recall_std']:.4f} | "
            f"{summary['build_time_mean']:.2f} ± {summary['build_time_std']:.2f}s | "
            f"{summary['search_time_mean']:.4f} ± {summary['search_time_std']:.4f}s"
        )
    
    print(f"\n✓ Results saved to {output_path}")
    print(f"✓ Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
