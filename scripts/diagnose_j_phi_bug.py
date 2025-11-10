"""
Systematic diagnostic of the bug J(φ*) > J(φ₀).

Tests multiple hypotheses about the root causes:
1. Incorrect calculation of J(φ₀)
2. Incorrect calculation of J(φ*)
3. Permutation is not minimizing J(φ)
4. Incorrect bucket-to-code mapping
5. Problem in the construction of the D_weighted matrix
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gray_tunneled_hashing.binary.baselines import random_projection_binarize
from gray_tunneled_hashing.binary.codebooks import build_codebook_kmeans
from gray_tunneled_hashing.algorithms.gray_tunneled_hasher import GrayTunneledHasher
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.distribution.traffic_stats import (
    collect_traffic_stats,
    build_weighted_distance_matrix,
)
from gray_tunneled_hashing.data.synthetic_generators import generate_hypercube_vertices
from gray_tunneled_hashing.evaluation.metrics import hamming_distance
from gray_tunneled_hashing.algorithms.qap_objective import qap_cost, generate_hypercube_edges


# Colors for output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'


@dataclass
class HypothesisResult:
    """Result of testing a hypothesis."""
    hypothesis_id: str
    hypothesis_name: str
    status: str  # "PASS", "FAIL", "INCONCLUSIVE"
    evidence: Dict
    confidence: float  # 0.0 to 1.0
    details: str


def compute_j_phi_original_codes(
    pi: np.ndarray,
    w: np.ndarray,
    bucket_to_code: np.ndarray,
) -> float:
    """Compute J(φ₀) using original bucket codes directly."""
    K = len(pi)
    cost = 0.0
    
    for i in range(K):
        code_i = bucket_to_code[i]
        for j in range(K):
            code_j = bucket_to_code[j]
            d_h = hamming_distance(code_i[np.newaxis, :], code_j[np.newaxis, :])[0, 0]
            cost += pi[i] * w[i, j] * d_h
    
    return cost


def compute_j_phi_permuted(
    pi: np.ndarray,
    w: np.ndarray,
    bucket_to_code: np.ndarray,
    permutation: np.ndarray,
    n_bits: int,
) -> float:
    """Compute J(φ*) using permuted codes."""
    vertices = generate_hypercube_vertices(n_bits)
    N = len(vertices)
    K = len(pi)
    
    # Map: bucket_idx -> vertex_idx (which vertex is assigned to this bucket)
    bucket_to_vertex = {}
    for vertex_idx in range(N):
        bucket_idx = permutation[vertex_idx]
        if bucket_idx < K:
            if bucket_idx not in bucket_to_vertex:
                bucket_to_vertex[bucket_idx] = vertex_idx
    
    cost = 0.0
    for i in range(K):
        if i in bucket_to_vertex:
            vertex_i = bucket_to_vertex[i]
            code_i = vertices[vertex_i]
        else:
            # Fallback: use original code
            code_i = bucket_to_code[i]
        
        for j in range(K):
            if j in bucket_to_vertex:
                vertex_j = bucket_to_vertex[j]
                code_j = vertices[vertex_j]
            else:
                code_j = bucket_to_code[j]
            
            d_h = hamming_distance(code_i[np.newaxis, :], code_j[np.newaxis, :])[0, 0]
            cost += pi[i] * w[i, j] * d_h
    
    return cost


def test_hypothesis_1_incorrect_j_phi_0(
    pi: np.ndarray,
    w: np.ndarray,
    bucket_to_code: np.ndarray,
    permutation: np.ndarray,
    n_bits: int,
) -> HypothesisResult:
    """
    Hypothesis 1: J(φ₀) is being calculated incorrectly.
    
    Test: Compute J(φ₀) in multiple ways and check consistency.
    """
    # Method 1: Direct computation using original codes
    j_phi_0_method1 = compute_j_phi_original_codes(pi, w, bucket_to_code)
    
    # Method 2: Using identity permutation (should be equivalent if mapping is correct)
    identity_perm = np.arange(2 ** n_bits, dtype=np.int32)
    j_phi_0_method2 = compute_j_phi_permuted(pi, w, bucket_to_code, identity_perm, n_bits)
    
    # Check consistency
    diff = abs(j_phi_0_method1 - j_phi_0_method2)
    relative_diff = diff / max(j_phi_0_method1, 1e-10)
    
    status = "PASS" if relative_diff < 1e-6 else "FAIL"
    confidence = 1.0 - min(relative_diff, 1.0)
    
    return HypothesisResult(
        hypothesis_id="H1",
        hypothesis_name="J(φ₀) calculation incorrect",
        status=status,
        evidence={
            "j_phi_0_method1": float(j_phi_0_method1),
            "j_phi_0_method2": float(j_phi_0_method2),
            "difference": float(diff),
            "relative_difference": float(relative_diff),
        },
        confidence=confidence,
        details=f"Method 1 (direct): {j_phi_0_method1:.6f}, Method 2 (identity perm): {j_phi_0_method2:.6f}, Diff: {diff:.6f}",
    )


def test_hypothesis_2_incorrect_j_phi_star(
    pi: np.ndarray,
    w: np.ndarray,
    bucket_to_code: np.ndarray,
    permutation: np.ndarray,
    n_bits: int,
    j_phi_0: float,
) -> HypothesisResult:
    """
    Hypothesis 2: J(φ*) is being calculated incorrectly.
    
    Test: Verify that permutation actually maps buckets correctly.
    """
    j_phi_star = compute_j_phi_permuted(pi, w, bucket_to_code, permutation, n_bits)
    
    # Check if permutation is valid (each bucket appears at most once)
    K = len(pi)
    N = 2 ** n_bits
    bucket_counts = np.zeros(K, dtype=int)
    
    for vertex_idx in range(N):
        bucket_idx = permutation[vertex_idx]
        if bucket_idx < K:
            bucket_counts[bucket_idx] += 1
    
    # Check if any bucket is missing
    missing_buckets = np.where(bucket_counts == 0)[0]
    multiple_assignments = np.where(bucket_counts > 1)[0]
    
    # Sanity check: J(φ*) should be >= 0
    is_valid = j_phi_star >= 0 and len(missing_buckets) == 0
    
    status = "FAIL" if not is_valid else ("PASS" if j_phi_star <= j_phi_0 + 1e-6 else "FAIL")
    confidence = 0.9 if is_valid else 0.1
    
    return HypothesisResult(
        hypothesis_id="H2",
        hypothesis_name="J(φ*) calculation incorrect",
        status=status,
        evidence={
            "j_phi_star": float(j_phi_star),
            "j_phi_0": float(j_phi_0),
            "violation": float(j_phi_star - j_phi_0),
            "missing_buckets": int(len(missing_buckets)),
            "multiple_assignments": int(len(multiple_assignments)),
            "is_valid": bool(is_valid),
        },
        confidence=confidence,
        details=f"J(φ*): {j_phi_star:.6f}, J(φ₀): {j_phi_0:.6f}, Violation: {j_phi_star - j_phi_0:.6f}, Missing: {len(missing_buckets)}, Multiple: {len(multiple_assignments)}",
    )


def test_hypothesis_3_permutation_not_minimizing(
    pi: np.ndarray,
    w: np.ndarray,
    bucket_to_code: np.ndarray,
    permutation: np.ndarray,
    n_bits: int,
    D_weighted: np.ndarray,
) -> HypothesisResult:
    """
    Hypothesis 3: Permutation is not actually minimizing J(φ).
    
    Test: Check if permutation minimizes QAP cost on D_weighted.
    """
    edges = generate_hypercube_edges(n_bits)
    qap_cost_perm = qap_cost(permutation, D_weighted, edges)
    
    # Try identity permutation
    identity_perm = np.arange(2 ** n_bits, dtype=np.int32)
    qap_cost_identity = qap_cost(identity_perm, D_weighted, edges)
    
    # Try a few random permutations
    random_costs = []
    for _ in range(10):
        random_perm = np.random.permutation(2 ** n_bits).astype(np.int32)
        cost = qap_cost(random_perm, D_weighted, edges)
        random_costs.append(cost)
    
    min_random_cost = min(random_costs)
    is_better_than_identity = qap_cost_perm < qap_cost_identity
    is_better_than_random = qap_cost_perm < min_random_cost
    
    status = "PASS" if (is_better_than_identity and is_better_than_random) else "FAIL"
    confidence = 0.8 if is_better_than_identity else 0.3
    
    return HypothesisResult(
        hypothesis_id="H3",
        hypothesis_name="Permutation not minimizing J(φ)",
        status=status,
        evidence={
            "qap_cost_perm": float(qap_cost_perm),
            "qap_cost_identity": float(qap_cost_identity),
            "min_random_cost": float(min_random_cost),
            "is_better_than_identity": bool(is_better_than_identity),
            "is_better_than_random": bool(is_better_than_random),
        },
        confidence=confidence,
        details=f"QAP cost (perm): {qap_cost_perm:.6f}, (identity): {qap_cost_identity:.6f}, (min random): {min_random_cost:.6f}",
    )


def test_hypothesis_4_bucket_mapping_incorrect(
    pi: np.ndarray,
    w: np.ndarray,
    bucket_to_code: np.ndarray,
    permutation: np.ndarray,
    n_bits: int,
) -> HypothesisResult:
    """
    Hypothesis 4: Bucket to code mapping is incorrect.
    
    Test: Verify that bucket codes match what's expected.
    """
    vertices = generate_hypercube_vertices(n_bits)
    N = len(vertices)
    K = len(pi)
    
    # Check: Are all bucket codes valid hypercube vertices?
    invalid_codes = []
    for i in range(K):
        code = bucket_to_code[i]
        # Check if code matches any vertex
        matches = np.any(np.all(vertices == code, axis=1))
        if not matches:
            invalid_codes.append(i)
    
    # Check: Does permutation map buckets to valid vertices?
    bucket_to_vertex = {}
    invalid_mappings = []
    for vertex_idx in range(N):
        bucket_idx = permutation[vertex_idx]
        if bucket_idx < K:
            if bucket_idx not in bucket_to_vertex:
                bucket_to_vertex[bucket_idx] = vertex_idx
            else:
                invalid_mappings.append((bucket_idx, vertex_idx))
    
    status = "PASS" if (len(invalid_codes) == 0 and len(invalid_mappings) == 0) else "FAIL"
    confidence = 0.9 if len(invalid_codes) == 0 else 0.2
    
    return HypothesisResult(
        hypothesis_id="H4",
        hypothesis_name="Bucket to code mapping incorrect",
        status=status,
        evidence={
            "invalid_codes": len(invalid_codes),
            "invalid_mappings": len(invalid_mappings),
            "buckets_mapped": len(bucket_to_vertex),
            "total_buckets": K,
        },
        confidence=confidence,
        details=f"Invalid codes: {len(invalid_codes)}, Invalid mappings: {len(invalid_mappings)}, Mapped: {len(bucket_to_vertex)}/{K}",
    )


def test_hypothesis_5_d_weighted_construction(
    pi: np.ndarray,
    w: np.ndarray,
    bucket_embeddings: np.ndarray,
    use_semantic_distances: bool,
) -> HypothesisResult:
    """
    Hypothesis 5: D_weighted construction is incorrect.
    
    Test: Verify D_weighted has correct structure and values.
    """
    D_weighted = build_weighted_distance_matrix(
        pi=pi,
        w=w,
        bucket_embeddings=bucket_embeddings,
        use_semantic_distances=use_semantic_distances,
    )
    
    K = len(pi)
    
    # Check dimensions
    correct_shape = D_weighted.shape == (K, K)
    
    # Check non-negativity
    all_non_negative = np.all(D_weighted >= -1e-10)  # Small tolerance
    
    # Check symmetry (should be approximately symmetric for semantic distances)
    if use_semantic_distances:
        symmetry_diff = np.abs(D_weighted - D_weighted.T).max()
        is_symmetric = symmetry_diff < 1e-6
    else:
        is_symmetric = True  # Not required for pure traffic
    
    # Check diagonal (should be zero or very small)
    diagonal_max = np.abs(np.diag(D_weighted)).max()
    diagonal_ok = diagonal_max < 1e-6
    
    # Check that high-traffic buckets have larger weights
    high_traffic_indices = np.argsort(pi)[-min(5, K):]
    low_traffic_indices = np.argsort(pi)[:min(5, K)]
    
    high_traffic_mean = D_weighted[high_traffic_indices].mean()
    low_traffic_mean = D_weighted[low_traffic_indices].mean()
    
    status = "PASS" if (correct_shape and all_non_negative and diagonal_ok) else "FAIL"
    confidence = 0.8 if (correct_shape and all_non_negative) else 0.3
    
    return HypothesisResult(
        hypothesis_id="H5",
        hypothesis_name="D_weighted construction incorrect",
        status=status,
        evidence={
            "correct_shape": bool(correct_shape),
            "all_non_negative": bool(all_non_negative),
            "is_symmetric": bool(is_symmetric),
            "diagonal_max": float(diagonal_max),
            "diagonal_ok": bool(diagonal_ok),
            "high_traffic_mean": float(high_traffic_mean),
            "low_traffic_mean": float(low_traffic_mean),
        },
        confidence=confidence,
        details=f"Shape: {D_weighted.shape}, Non-neg: {all_non_negative}, Symmetric: {is_symmetric}, Diag max: {diagonal_max:.6f}",
    )


def run_single_diagnosis(args_tuple) -> Dict:
    """Run diagnosis on a single configuration (for parallel execution)."""
    (n_bits, n_codes, random_state) = args_tuple
    
    np.random.seed(random_state)
    
    # Generate synthetic data
    N = 500
    Q = 100
    dim = 32
    k = 5
    
    base_embeddings = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    ground_truth = np.random.randint(0, N, size=(Q, k), dtype=np.int32)
    
    # Create encoder
    _, proj_matrix = random_projection_binarize(
        base_embeddings,
        n_bits=n_bits,
        random_state=random_state,
    )
    
    def encoder_fn(emb):
        from gray_tunneled_hashing.binary.baselines import apply_random_projection
        proj = apply_random_projection(emb, proj_matrix)
        return (proj > 0).astype(bool)
    
    # Collect traffic stats
    traffic_stats = collect_traffic_stats(
        queries=queries,
        ground_truth_neighbors=ground_truth,
        base_embeddings=base_embeddings,
        encoder=encoder_fn,
    )
    
    pi = traffic_stats["pi"]
    w = traffic_stats["w"]
    bucket_to_code = traffic_stats["bucket_to_code"]
    K = traffic_stats["K"]
    
    # Build codebook for bucket embeddings
    centroids, _ = build_codebook_kmeans(
        embeddings=base_embeddings,
        n_codes=n_codes,
        random_state=random_state,
    )
    
    # Get bucket embeddings (simplified - use centroids)
    bucket_embeddings = centroids[:min(K, n_codes)]
    if K > n_codes:
        # Pad
        padding = np.tile(bucket_embeddings[-1:], (K - n_codes, 1))
        bucket_embeddings = np.vstack([bucket_embeddings, padding])
    
    # Build weighted distance matrix
    D_weighted = build_weighted_distance_matrix(
        pi=pi,
        w=w,
        bucket_embeddings=bucket_embeddings,
        use_semantic_distances=True,
    )
    
    # Pad D_weighted if needed
    N_hypercube = 2 ** n_bits
    if K < N_hypercube:
        D_padded = np.zeros((N_hypercube, N_hypercube), dtype=np.float64)
        D_padded[:K, :K] = D_weighted
        D_weighted = D_padded
        bucket_embeddings_padded = np.zeros((N_hypercube, bucket_embeddings.shape[1]), dtype=bucket_embeddings.dtype)
        bucket_embeddings_padded[:K] = bucket_embeddings
        bucket_embeddings_padded[K:] = bucket_embeddings[-1:]
        bucket_embeddings = bucket_embeddings_padded
    
    # Fit hasher
    hasher = GrayTunneledHasher(
        n_bits=n_bits,
        block_size=8,
        max_two_swap_iters=30,
        num_tunneling_steps=3,
        mode="two_swap_only",
        random_state=random_state,
    )
    
    hasher.fit(bucket_embeddings, D=D_weighted)
    permutation = hasher.get_assignment()
    
    # Compute J(φ₀) and J(φ*)
    j_phi_0 = compute_j_phi_original_codes(pi, w, bucket_to_code)
    j_phi_star = compute_j_phi_permuted(pi, w, bucket_to_code, permutation, n_bits)
    
    # Test all hypotheses
    results = []
    
    results.append(test_hypothesis_1_incorrect_j_phi_0(pi, w, bucket_to_code, permutation, n_bits))
    results.append(test_hypothesis_2_incorrect_j_phi_star(pi, w, bucket_to_code, permutation, n_bits, j_phi_0))
    results.append(test_hypothesis_3_permutation_not_minimizing(pi, w, bucket_to_code, permutation, n_bits, D_weighted))
    results.append(test_hypothesis_4_bucket_mapping_incorrect(pi, w, bucket_to_code, permutation, n_bits))
    results.append(test_hypothesis_5_d_weighted_construction(pi, w, bucket_embeddings[:K], True))
    
    return {
        "config": {"n_bits": n_bits, "n_codes": n_codes, "random_state": random_state},
        "j_phi_0": float(j_phi_0),
        "j_phi_star": float(j_phi_star),
        "violation": float(j_phi_star - j_phi_0),
        "hypotheses": [asdict(r) for r in results],
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose J(φ*) > J(φ₀) bug")
    parser.add_argument("--n-bits", type=int, default=10, help="Number of bits")
    parser.add_argument("--n-codes", type=int, default=32, help="Number of codebook vectors")
    parser.add_argument("--n-runs", type=int, default=10, help="Number of parallel runs")
    parser.add_argument("--n-workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--output", type=str, default="experiments/real/diagnosis_results.json")
    
    args = parser.parse_args()
    
    if args.n_workers is None:
        args.n_workers = min(args.n_runs, mp.cpu_count())
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}DIAGNOSIS: J(φ*) > J(φ₀) Bug Investigation{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}Configuration:{Colors.END}")
    print(f"  n_bits: {args.n_bits}")
    print(f"  n_codes: {args.n_codes}")
    print(f"  n_runs: {args.n_runs}")
    print(f"  n_workers: {args.n_workers}")
    print()
    
    # Prepare arguments for parallel execution
    configs = [
        (args.n_bits, args.n_codes, 42 + i)
        for i in range(args.n_runs)
    ]
    
    # Run diagnoses in parallel
    print(f"{Colors.YELLOW}Running {args.n_runs} diagnoses in parallel...{Colors.END}")
    all_results = []
    
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {executor.submit(run_single_diagnosis, config): config for config in configs}
        
        with tqdm(total=args.n_runs, desc="Diagnosing", colour="green") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    all_results.append(result)
                    pbar.update(1)
                except Exception as e:
                    config = futures[future]
                    print(f"{Colors.RED}Error in config {config}: {e}{Colors.END}")
                    import traceback
                    traceback.print_exc()
    
    # Aggregate results
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}AGGREGATED RESULTS{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    
    # Count violations
    violations = [r for r in all_results if r["violation"] > 1e-6]
    n_violations = len(violations)
    
    print(f"\n{Colors.BOLD}Violations:{Colors.END} {n_violations}/{args.n_runs} ({n_violations/args.n_runs*100:.1f}%)")
    
    if n_violations > 0:
        avg_violation = np.mean([r["violation"] for r in violations])
        print(f"{Colors.RED}Average violation: {avg_violation:.6f}{Colors.END}")
    
    # Analyze hypotheses
    print(f"\n{Colors.BOLD}{Colors.CYAN}HYPOTHESIS ANALYSIS{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    
    hypothesis_scores = {}
    for result in all_results:
        for hyp in result["hypotheses"]:
            hyp_id = hyp["hypothesis_id"]
            if hyp_id not in hypothesis_scores:
                hypothesis_scores[hyp_id] = {
                    "name": hyp["hypothesis_name"],
                    "passes": 0,
                    "fails": 0,
                    "confidences": [],
                }
            
            if hyp["status"] == "PASS":
                hypothesis_scores[hyp_id]["passes"] += 1
            elif hyp["status"] == "FAIL":
                hypothesis_scores[hyp_id]["fails"] += 1
            
            hypothesis_scores[hyp_id]["confidences"].append(hyp["confidence"])
    
    # Rank hypotheses by failure rate
    ranked = []
    for hyp_id, scores in hypothesis_scores.items():
        total = scores["passes"] + scores["fails"]
        fail_rate = scores["fails"] / total if total > 0 else 0
        avg_confidence = np.mean(scores["confidences"])
        ranked.append((hyp_id, fail_rate, avg_confidence, scores))
    
    ranked.sort(key=lambda x: (x[1], -x[2]), reverse=True)  # Sort by fail_rate desc, then confidence desc
    
    print(f"\n{Colors.BOLD}Ranked Hypotheses (by failure rate):{Colors.END}\n")
    for i, (hyp_id, fail_rate, avg_conf, scores) in enumerate(ranked, 1):
        color = Colors.RED if fail_rate > 0.5 else Colors.YELLOW if fail_rate > 0.2 else Colors.GREEN
        print(f"{i}. {Colors.BOLD}{hyp_id}: {scores['name']}{Colors.END}")
        print(f"   {color}Fail rate: {fail_rate*100:.1f}% ({scores['fails']}/{scores['passes']+scores['fails']}){Colors.END}")
        print(f"   {Colors.BLUE}Avg confidence: {avg_conf:.2f}{Colors.END}")
        print()
    
    # Detailed results for top failing hypothesis
    if ranked and ranked[0][1] > 0:
        top_hyp_id = ranked[0][0]
        print(f"{Colors.BOLD}{Colors.MAGENTA}DETAILED EVIDENCE: {top_hyp_id}{Colors.END}")
        print(f"{Colors.MAGENTA}{'='*70}{Colors.END}")
        
        for result in all_results[:3]:  # Show first 3
            for hyp in result["hypotheses"]:
                if hyp["hypothesis_id"] == top_hyp_id:
                    print(f"\n{Colors.CYAN}Run {result['config']['random_state']}:{Colors.END}")
                    print(f"  Status: {Colors.GREEN if hyp['status'] == 'PASS' else Colors.RED}{hyp['status']}{Colors.END}")
                    print(f"  Confidence: {hyp['confidence']:.2f}")
                    print(f"  Details: {hyp['details']}")
                    print(f"  Evidence: {json.dumps(hyp['evidence'], indent=4)}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump({
            "args": vars(args),
            "summary": {
                "n_runs": args.n_runs,
                "n_violations": n_violations,
                "violation_rate": n_violations / args.n_runs,
            },
            "hypothesis_ranking": [
                {
                    "hypothesis_id": hyp_id,
                    "name": scores["name"],
                    "fail_rate": fail_rate,
                    "avg_confidence": avg_conf,
                }
                for hyp_id, fail_rate, avg_conf, scores in ranked
            ],
            "detailed_results": all_results,
        }, f, indent=2)
    
    print(f"\n{Colors.GREEN}Results saved to: {output_path}{Colors.END}")


if __name__ == "__main__":
    main()

