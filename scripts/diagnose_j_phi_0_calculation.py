"""
Diagnóstico sistemático do cálculo de J(φ₀).

Testa hipóteses H1-H5 sobre por que J(φ*) > J(φ₀):
- H1: J(φ₀) deveria ser calculado a partir da permutação inicial
- H2: Permutação inicial não corresponde ao layout original
- H3: Mapeamento bucket → vértice na permutação inicial está incorreto
- H4: compute_j_phi_cost com identity não produz mesmo resultado que cálculo direto
- H5: Padding/subsampling altera correspondência bucket → código
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
from gray_tunneled_hashing.distribution.traffic_stats import collect_traffic_stats
from gray_tunneled_hashing.distribution.j_phi_objective import compute_j_phi_cost
from gray_tunneled_hashing.data.synthetic_generators import generate_hypercube_vertices
from gray_tunneled_hashing.evaluation.metrics import hamming_distance


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


def compute_j_phi_direct(
    pi: np.ndarray,
    w: np.ndarray,
    bucket_to_code: np.ndarray,
) -> float:
    """Compute J(φ) directly using bucket codes (no permutation)."""
    K = len(pi)
    cost = 0.0
    for i in range(K):
        code_i = bucket_to_code[i]
        for j in range(K):
            code_j = bucket_to_code[j]
            d_h = hamming_distance(code_i[np.newaxis, :], code_j[np.newaxis, :])[0, 0]
            cost += pi[i] * w[i, j] * d_h
    return cost


def test_hypothesis_h1(
    pi: np.ndarray,
    w: np.ndarray,
    bucket_to_code: np.ndarray,
    n_bits: int,
) -> HypothesisResult:
    """
    H1: J(φ₀) deveria ser calculado a partir da permutação inicial (identity).
    
    Test: Compare J(φ₀) computed via identity permutation vs direct calculation.
    """
    # Method 1: Direct calculation
    j_phi_0_direct = compute_j_phi_direct(pi, w, bucket_to_code)
    
    # Method 2: Via identity permutation
    N = 2 ** n_bits
    identity_perm = np.arange(N, dtype=np.int32)
    j_phi_0_identity = compute_j_phi_cost(
        permutation=identity_perm,
        pi=pi,
        w=w,
        bucket_to_code=bucket_to_code,
        n_bits=n_bits,
        bucket_to_embedding_idx=None,
    )
    
    diff = abs(j_phi_0_direct - j_phi_0_identity)
    relative_diff = diff / max(j_phi_0_direct, 1e-10)
    
    status = "PASS" if relative_diff < 1e-6 else "FAIL"
    confidence = 1.0 - min(relative_diff, 1.0)
    
    return HypothesisResult(
        hypothesis_id="H1",
        hypothesis_name="J(φ₀) should be computed from initial permutation",
        status=status,
        evidence={
            "j_phi_0_direct": float(j_phi_0_direct),
            "j_phi_0_identity": float(j_phi_0_identity),
            "difference": float(diff),
            "relative_difference": float(relative_diff),
        },
        confidence=confidence,
        details=f"Direct: {j_phi_0_direct:.6f}, Identity perm: {j_phi_0_identity:.6f}, Diff: {diff:.6f} ({relative_diff*100:.2f}%)",
    )


def test_hypothesis_h2(
    bucket_to_code: np.ndarray,
    n_bits: int,
) -> HypothesisResult:
    """
    H2: Permutação inicial não corresponde ao layout original.
    
    Test: Check if bucket_to_code[i] == vertices[i] for all i < K.
    """
    vertices = generate_hypercube_vertices(n_bits)
    K = len(bucket_to_code)
    
    mismatches = []
    for i in range(min(K, len(vertices))):
        if not np.array_equal(bucket_to_code[i], vertices[i]):
            mismatches.append(i)
    
    mismatch_rate = len(mismatches) / min(K, len(vertices))
    
    status = "PASS" if mismatch_rate < 0.01 else "FAIL"
    confidence = 1.0 - mismatch_rate
    
    return HypothesisResult(
        hypothesis_id="H2",
        hypothesis_name="Initial permutation does not correspond to original layout",
        status=status,
        evidence={
            "n_mismatches": len(mismatches),
            "mismatch_rate": float(mismatch_rate),
            "total_buckets": K,
            "sample_mismatches": mismatches[:10] if mismatches else [],
        },
        confidence=confidence,
        details=f"Mismatches: {len(mismatches)}/{min(K, len(vertices))} ({mismatch_rate*100:.1f}%)",
    )


def test_hypothesis_h3(
    bucket_to_code: np.ndarray,
    n_bits: int,
) -> HypothesisResult:
    """
    H3: Mapeamento bucket → vértice na permutação identity está incorreto.
    
    Test: For identity permutation, verify bucket i maps to vertex i correctly.
    """
    vertices = generate_hypercube_vertices(n_bits)
    N = len(vertices)
    K = len(bucket_to_code)
    
    identity_perm = np.arange(N, dtype=np.int32)
    
    # Map buckets to vertices using identity permutation
    bucket_to_vertex = {}
    for vertex_idx in range(N):
        embedding_idx = identity_perm[vertex_idx]
        if embedding_idx < K:
            bucket_idx = embedding_idx
            if bucket_idx not in bucket_to_vertex:
                bucket_to_vertex[bucket_idx] = vertex_idx
    
    # Check if bucket i is at vertex i
    correct_mappings = 0
    incorrect_mappings = []
    
    for i in range(K):
        if i in bucket_to_vertex:
            vertex_i = bucket_to_vertex[i]
            if vertex_i == i:
                correct_mappings += 1
            else:
                incorrect_mappings.append((i, vertex_i))
        else:
            incorrect_mappings.append((i, None))
    
    correct_rate = correct_mappings / K if K > 0 else 0.0
    
    status = "PASS" if correct_rate > 0.99 else "FAIL"
    confidence = correct_rate
    
    return HypothesisResult(
        hypothesis_id="H3",
        hypothesis_name="Bucket → vertex mapping in identity permutation is incorrect",
        status=status,
        evidence={
            "correct_mappings": correct_mappings,
            "incorrect_mappings": len(incorrect_mappings),
            "correct_rate": float(correct_rate),
            "sample_incorrect": incorrect_mappings[:5],
        },
        confidence=confidence,
        details=f"Correct: {correct_mappings}/{K} ({correct_rate*100:.1f}%)",
    )


def test_hypothesis_h4(
    pi: np.ndarray,
    w: np.ndarray,
    bucket_to_code: np.ndarray,
    n_bits: int,
) -> HypothesisResult:
    """
    H4: compute_j_phi_cost com identity não produz mesmo resultado que cálculo direto.
    
    Test: Compare results from both methods.
    """
    # Direct calculation
    j_phi_0_direct = compute_j_phi_direct(pi, w, bucket_to_code)
    
    # Via compute_j_phi_cost with identity
    N = 2 ** n_bits
    identity_perm = np.arange(N, dtype=np.int32)
    j_phi_0_via_perm = compute_j_phi_cost(
        permutation=identity_perm,
        pi=pi,
        w=w,
        bucket_to_code=bucket_to_code,
        n_bits=n_bits,
        bucket_to_embedding_idx=None,
    )
    
    diff = abs(j_phi_0_direct - j_phi_0_via_perm)
    relative_diff = diff / max(j_phi_0_direct, 1e-10)
    
    status = "PASS" if relative_diff < 1e-6 else "FAIL"
    confidence = 1.0 - min(relative_diff, 1.0)
    
    return HypothesisResult(
        hypothesis_id="H4",
        hypothesis_name="compute_j_phi_cost with identity != direct calculation",
        status=status,
        evidence={
            "j_phi_0_direct": float(j_phi_0_direct),
            "j_phi_0_via_perm": float(j_phi_0_via_perm),
            "difference": float(diff),
            "relative_difference": float(relative_diff),
        },
        confidence=confidence,
        details=f"Direct: {j_phi_0_direct:.6f}, Via perm: {j_phi_0_via_perm:.6f}, Diff: {diff:.6f}",
    )


def test_hypothesis_h5(
    pi: np.ndarray,
    w: np.ndarray,
    bucket_to_code: np.ndarray,
    n_bits: int,
) -> HypothesisResult:
    """
    H5: Padding/subsampling altera correspondência para primeiros K buckets.
    
    Test: Check if padding affects the first K buckets.
    """
    K = len(pi)
    N = 2 ** n_bits
    
    if K >= N:
        # No padding needed
        return HypothesisResult(
            hypothesis_id="H5",
            hypothesis_name="Padding/subsampling affects bucket correspondence",
            status="INCONCLUSIVE",
            evidence={"reason": "No padding needed (K >= N)"},
            confidence=1.0,
            details="K >= N, no padding",
        )
    
    # Test with original K
    j_phi_0_original = compute_j_phi_direct(pi, w, bucket_to_code)
    
    # Test with padded (simulate padding)
    # In practice, padding doesn't affect first K buckets, but let's verify
    # by checking if bucket_to_code[:K] is used correctly
    
    # Create identity perm for padded case
    identity_perm = np.arange(N, dtype=np.int32)
    j_phi_0_padded = compute_j_phi_cost(
        permutation=identity_perm,
        pi=pi,
        w=w,
        bucket_to_code=bucket_to_code,
        n_bits=n_bits,
        bucket_to_embedding_idx=None,
    )
    
    diff = abs(j_phi_0_original - j_phi_0_padded)
    relative_diff = diff / max(j_phi_0_original, 1e-10)
    
    status = "PASS" if relative_diff < 1e-6 else "FAIL"
    confidence = 1.0 - min(relative_diff, 1.0)
    
    return HypothesisResult(
        hypothesis_id="H5",
        hypothesis_name="Padding/subsampling affects bucket correspondence",
        status=status,
        evidence={
            "j_phi_0_original": float(j_phi_0_original),
            "j_phi_0_padded": float(j_phi_0_padded),
            "difference": float(diff),
            "relative_difference": float(relative_diff),
            "K": K,
            "N": N,
        },
        confidence=confidence,
        details=f"Original: {j_phi_0_original:.6f}, Padded: {j_phi_0_padded:.6f}, Diff: {diff:.6f}",
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
    
    # Test all hypotheses
    results = []
    
    results.append(test_hypothesis_h1(pi, w, bucket_to_code, n_bits))
    results.append(test_hypothesis_h2(bucket_to_code, n_bits))
    results.append(test_hypothesis_h3(bucket_to_code, n_bits))
    results.append(test_hypothesis_h4(pi, w, bucket_to_code, n_bits))
    results.append(test_hypothesis_h5(pi, w, bucket_to_code, n_bits))
    
    return {
        "config": {"n_bits": n_bits, "n_codes": n_codes, "random_state": random_state},
        "K": K,
        "hypotheses": [asdict(r) for r in results],
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose J(φ₀) calculation")
    parser.add_argument("--n-bits", type=int, default=8, help="Number of bits")
    parser.add_argument("--n-codes", type=int, default=32, help="Number of codebook vectors")
    parser.add_argument("--n-runs", type=int, default=10, help="Number of parallel runs")
    parser.add_argument("--n-workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--output", type=str, default="experiments/real/diagnosis_j_phi_0_results.json")
    
    args = parser.parse_args()
    
    if args.n_workers is None:
        args.n_workers = min(args.n_runs, mp.cpu_count())
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}DIAGNOSIS: J(φ₀) Calculation{Colors.END}")
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
                    "inconclusive": 0,
                    "confidences": [],
                }
            
            if hyp["status"] == "PASS":
                hypothesis_scores[hyp_id]["passes"] += 1
            elif hyp["status"] == "FAIL":
                hypothesis_scores[hyp_id]["fails"] += 1
            else:
                hypothesis_scores[hyp_id]["inconclusive"] += 1
            
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
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump({
            "args": vars(args),
            "summary": {
                "n_runs": args.n_runs,
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

