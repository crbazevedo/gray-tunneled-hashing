"""
Deep investigation of GTH low recall issue.

This script systematically tests hypotheses H1-H5 to identify the root cause
of why GTH methods show lower recall than baselines.

Hypotheses:
- H1: Bucket → Dataset mapping incorrect (code_to_bucket coverage)
- H2: Permutation reduces bucket coverage (vertex distribution)
- H3: Permutation applied incorrectly in query (order of operations)
- H4: Empty or sparsely populated buckets
- H5: Inconsistency between permutation and code_to_bucket
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import Counter, defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gray_tunneled_hashing.binary.lsh_families import HyperplaneLSH
from gray_tunneled_hashing.binary.baselines import random_projection_binarize, apply_random_projection
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.api.query_pipeline import query_with_hamming_ball
from gray_tunneled_hashing.evaluation.metrics import recall_at_k
from gray_tunneled_hashing.data.synthetic_generators import generate_hypercube_vertices
from sklearn.metrics.pairwise import euclidean_distances


def print_section(title: str, level: int = 1):
    """Print a formatted section header."""
    if level == 1:
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)
    elif level == 2:
        print("\n" + "-" * 70)
        print(f"  {title}")
        print("-" * 70)
    else:
        print(f"\n{title}")


def test_h1_bucket_coverage(
    base_embeddings: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    n_bits: int,
    n_codes: int,
    random_state: int = 42,
) -> Dict:
    """
    Test H1: Bucket → Dataset mapping coverage.
    
    Checks:
    - Are all base embedding LSH codes in code_to_bucket?
    - What percentage of base embeddings are mapped to buckets?
    - Are there base embeddings with codes not seen in queries?
    """
    print_section("H1: Testing Bucket → Dataset Mapping Coverage")
    
    results = {
        "base_codes_in_code_to_bucket": 0,
        "base_codes_total": 0,
        "coverage_rate": 0.0,
        "unmapped_embeddings": [],
        "query_codes_unique": 0,
        "base_codes_unique": 0,
        "code_overlap": 0.0,
        "issues": [],
    }
    
    # Build index
    lsh = HyperplaneLSH(n_bits=n_bits, dim=base_embeddings.shape[1], random_state=random_state)
    
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=None,
        n_bits=n_bits,
        n_codes=n_codes,
        use_codebook=True,
        use_semantic_distances=False,
        lsh_family=lsh,
        block_size=4,
        max_two_swap_iters=20,
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=random_state,
    )
    
    # Encode base embeddings and queries
    base_codes_lsh = lsh.hash(base_embeddings)
    query_codes_lsh = lsh.hash(queries)
    
    # Count unique codes
    base_codes_set = set(tuple(code.astype(int).tolist()) for code in base_codes_lsh)
    query_codes_set = set(tuple(code.astype(int).tolist()) for code in query_codes_lsh)
    
    results["base_codes_total"] = len(base_embeddings)
    results["base_codes_unique"] = len(base_codes_set)
    results["query_codes_unique"] = len(query_codes_set)
    results["code_overlap"] = len(base_codes_set & query_codes_set) / len(base_codes_set) if len(base_codes_set) > 0 else 0.0
    
    # Check coverage: how many base codes are in code_to_bucket?
    mapped_count = 0
    unmapped_indices = []
    
    for dataset_idx, code in enumerate(base_codes_lsh):
        code_tuple = tuple(code.astype(int).tolist())
        if code_tuple in index_obj.code_to_bucket:
            mapped_count += 1
        else:
            unmapped_indices.append(dataset_idx)
    
    results["base_codes_in_code_to_bucket"] = mapped_count
    results["coverage_rate"] = mapped_count / len(base_embeddings) if len(base_embeddings) > 0 else 0.0
    results["unmapped_embeddings"] = unmapped_indices
    
    # Print results
    print(f"Base embeddings total: {results['base_codes_total']}")
    print(f"Base codes unique: {results['base_codes_unique']}")
    print(f"Query codes unique: {results['query_codes_unique']}")
    print(f"Code overlap (base ∩ query): {results['code_overlap']:.1%}")
    print(f"Base codes in code_to_bucket: {mapped_count}/{len(base_embeddings)} ({results['coverage_rate']:.1%})")
    
    if results["coverage_rate"] < 1.0:
        results["issues"].append(
            f"Only {results['coverage_rate']:.1%} of base embeddings are mapped to buckets"
        )
        print(f"⚠️  {len(unmapped_indices)} base embeddings are NOT in code_to_bucket")
        print(f"   This means they cannot be retrieved by any query!")
    
    if results["code_overlap"] < 0.5:
        results["issues"].append(
            f"Low code overlap: only {results['code_overlap']:.1%} of base codes appear in queries"
        )
        print(f"⚠️  Low overlap between base and query codes")
    
    return results


def test_h2_permutation_coverage(
    base_embeddings: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    n_bits: int,
    n_codes: int,
    random_state: int = 42,
) -> Dict:
    """
    Test H2: Permutation coverage analysis.
    
    Checks:
    - How are vertices distributed across buckets?
    - Are there buckets without any vertices?
    - Are there vertices mapped to invalid buckets?
    - What's the distribution of vertices per bucket?
    """
    print_section("H2: Testing Permutation Coverage")
    
    results = {
        "total_vertices": 0,
        "total_buckets": 0,
        "buckets_with_vertices": 0,
        "buckets_without_vertices": [],
        "vertices_per_bucket": {},
        "max_vertices_per_bucket": 0,
        "min_vertices_per_bucket": 0,
        "avg_vertices_per_bucket": 0.0,
        "invalid_bucket_indices": [],
        "issues": [],
    }
    
    # Build index
    lsh = HyperplaneLSH(n_bits=n_bits, dim=base_embeddings.shape[1], random_state=random_state)
    
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=None,
        n_bits=n_bits,
        n_codes=n_codes,
        use_codebook=True,
        use_semantic_distances=False,
        lsh_family=lsh,
        block_size=4,
        max_two_swap_iters=20,
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=random_state,
    )
    
    permutation = index_obj.permutation
    K = index_obj.K
    N = len(permutation)  # Should be 2**n_bits
    
    results["total_vertices"] = N
    results["total_buckets"] = K
    
    # Analyze vertex distribution
    bucket_to_vertices = defaultdict(list)
    for vertex_idx in range(N):
        bucket_idx = permutation[vertex_idx]
        bucket_to_vertices[bucket_idx].append(vertex_idx)
        
        # Check for invalid bucket indices
        if bucket_idx >= K:
            results["invalid_bucket_indices"].append((vertex_idx, bucket_idx))
    
    results["buckets_with_vertices"] = len(bucket_to_vertices)
    results["buckets_without_vertices"] = [i for i in range(K) if i not in bucket_to_vertices]
    
    # Calculate statistics
    vertices_counts = [len(vertices) for vertices in bucket_to_vertices.values()]
    if vertices_counts:
        results["max_vertices_per_bucket"] = max(vertices_counts)
        results["min_vertices_per_bucket"] = min(vertices_counts)
        results["avg_vertices_per_bucket"] = np.mean(vertices_counts)
        results["vertices_per_bucket"] = dict(bucket_to_vertices)
    
    # Print results
    print(f"Total vertices (2^{n_bits}): {N}")
    print(f"Total buckets (K): {K}")
    print(f"Buckets with vertices: {results['buckets_with_vertices']}/{K}")
    print(f"Buckets without vertices: {len(results['buckets_without_vertices'])}")
    
    if vertices_counts:
        print(f"Vertices per bucket: min={results['min_vertices_per_bucket']}, "
              f"max={results['max_vertices_per_bucket']}, "
              f"avg={results['avg_vertices_per_bucket']:.1f}")
    
    if results["buckets_without_vertices"]:
        results["issues"].append(
            f"{len(results['buckets_without_vertices'])} buckets have no vertices mapped"
        )
        print(f"⚠️  Buckets without vertices: {results['buckets_without_vertices'][:10]}...")
        print(f"   Queries in these buckets cannot be expanded via Hamming ball!")
    
    if results["invalid_bucket_indices"]:
        results["issues"].append(
            f"{len(results['invalid_bucket_indices'])} vertices map to invalid bucket indices"
        )
        print(f"⚠️  Invalid bucket indices found: {len(results['invalid_bucket_indices'])}")
    
    # Check if some buckets have too many vertices (potential collision)
    if results["max_vertices_per_bucket"] > 10:
        results["issues"].append(
            f"Some buckets have {results['max_vertices_per_bucket']} vertices (potential collision)"
        )
        print(f"⚠️  High vertex collision: max {results['max_vertices_per_bucket']} vertices per bucket")
    
    return results


def test_h3_permutation_order(
    base_embeddings: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    n_bits: int,
    n_codes: int,
    k: int = 5,
    hamming_radius: int = 1,
    random_state: int = 42,
) -> Dict:
    """
    Test H3: Permutation application order.
    
    Compares:
    - Current: Expand Hamming ball → Apply permutation → Get buckets
    - Alternative: Apply permutation → Expand Hamming ball → Get buckets
    
    This tests if the order of operations affects recall.
    """
    print_section("H3: Testing Permutation Application Order")
    
    results = {
        "current_order_recall": 0.0,
        "alternative_order_recall": 0.0,
        "recall_difference": 0.0,
        "current_order_candidates": 0,
        "alternative_order_candidates": 0,
        "issues": [],
    }
    
    # Build index
    lsh = HyperplaneLSH(n_bits=n_bits, dim=base_embeddings.shape[1], random_state=random_state)
    
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=None,
        n_bits=n_bits,
        n_codes=n_codes,
        use_codebook=True,
        use_semantic_distances=False,
        lsh_family=lsh,
        block_size=4,
        max_two_swap_iters=20,
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=random_state,
    )
    
    # Build bucket → dataset mapping
    base_codes_lsh = lsh.hash(base_embeddings)
    query_codes_lsh = lsh.hash(queries)
    
    bucket_to_dataset_indices = {}
    for dataset_idx, code in enumerate(base_codes_lsh):
        code_tuple = tuple(code.astype(int).tolist())
        if code_tuple in index_obj.code_to_bucket:
            bucket_idx = index_obj.code_to_bucket[code_tuple]
            if bucket_idx not in bucket_to_dataset_indices:
                bucket_to_dataset_indices[bucket_idx] = []
            bucket_to_dataset_indices[bucket_idx].append(dataset_idx)
    
    # Current order: Expand → Permute
    all_retrieved_current = []
    total_candidates_current = 0
    
    for query_code in query_codes_lsh:
        result = query_with_hamming_ball(
            query_code=query_code,
            permutation=index_obj.permutation,
            code_to_bucket=index_obj.code_to_bucket,
            bucket_to_code=index_obj.bucket_to_code,
            n_bits=n_bits,
            hamming_radius=hamming_radius,
        )
        
        candidate_dataset_indices = []
        for bucket_idx in result.candidate_indices:
            if bucket_idx in bucket_to_dataset_indices:
                candidate_dataset_indices.extend(bucket_to_dataset_indices[bucket_idx])
        
        candidate_dataset_indices = list(dict.fromkeys(candidate_dataset_indices))[:k]
        if len(candidate_dataset_indices) < k:
            candidate_dataset_indices.extend([-1] * (k - len(candidate_dataset_indices)))
        else:
            candidate_dataset_indices = candidate_dataset_indices[:k]
        
        all_retrieved_current.append(candidate_dataset_indices)
        total_candidates_current += len(result.candidate_indices)
    
    retrieved_current = np.array(all_retrieved_current, dtype=np.int32)
    recall_current = recall_at_k(retrieved_current, ground_truth, k=k)
    results["current_order_recall"] = recall_current
    results["current_order_candidates"] = total_candidates_current / len(queries)
    
    # Alternative order: Permute → Expand (not implemented, but we can note it)
    # This would require a different implementation of query_with_hamming_ball
    # For now, we'll note that the current implementation expands first
    
    print(f"Current order (Expand → Permute) recall: {recall_current:.4f}")
    print(f"Average candidates per query: {results['current_order_candidates']:.1f}")
    print(f"Note: Alternative order (Permute → Expand) would require different implementation")
    
    if recall_current < 0.1:
        results["issues"].append(f"Very low recall with current order: {recall_current:.4f}")
    
    return results


def test_h4_empty_buckets(
    base_embeddings: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    n_bits: int,
    n_codes: int,
    random_state: int = 42,
) -> Dict:
    """
    Test H4: Empty or sparsely populated buckets.
    
    Checks:
    - How many base embeddings are in each bucket?
    - Are there buckets with zero embeddings?
    - What's the distribution of bucket sizes?
    - Are ground truth neighbors in populated buckets?
    """
    print_section("H4: Testing Empty/Sparse Buckets")
    
    results = {
        "total_buckets": 0,
        "populated_buckets": 0,
        "empty_buckets": [],
        "bucket_sizes": {},
        "min_bucket_size": 0,
        "max_bucket_size": 0,
        "avg_bucket_size": 0.0,
        "gt_neighbors_in_buckets": 0,
        "gt_neighbors_total": 0,
        "gt_coverage_rate": 0.0,
        "issues": [],
    }
    
    # Build index
    lsh = HyperplaneLSH(n_bits=n_bits, dim=base_embeddings.shape[1], random_state=random_state)
    
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=None,
        n_bits=n_bits,
        n_codes=n_codes,
        use_codebook=True,
        use_semantic_distances=False,
        lsh_family=lsh,
        block_size=4,
        max_two_swap_iters=20,
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=random_state,
    )
    
    # Build bucket → dataset mapping
    base_codes_lsh = lsh.hash(base_embeddings)
    
    bucket_to_dataset_indices = {}
    for dataset_idx, code in enumerate(base_codes_lsh):
        code_tuple = tuple(code.astype(int).tolist())
        if code_tuple in index_obj.code_to_bucket:
            bucket_idx = index_obj.code_to_bucket[code_tuple]
            if bucket_idx not in bucket_to_dataset_indices:
                bucket_to_dataset_indices[bucket_idx] = []
            bucket_to_dataset_indices[bucket_idx].append(dataset_idx)
    
    K = index_obj.K
    results["total_buckets"] = K
    results["populated_buckets"] = len(bucket_to_dataset_indices)
    results["empty_buckets"] = [i for i in range(K) if i not in bucket_to_dataset_indices]
    
    # Calculate bucket sizes
    bucket_sizes = {bucket_idx: len(indices) for bucket_idx, indices in bucket_to_dataset_indices.items()}
    results["bucket_sizes"] = bucket_sizes
    
    if bucket_sizes:
        results["min_bucket_size"] = min(bucket_sizes.values())
        results["max_bucket_size"] = max(bucket_sizes.values())
        results["avg_bucket_size"] = np.mean(list(bucket_sizes.values()))
    
    # Check ground truth coverage
    gt_set = set(ground_truth.flatten())
    mapped_set = set()
    for indices in bucket_to_dataset_indices.values():
        mapped_set.update(indices)
    
    gt_in_buckets = len(gt_set & mapped_set)
    results["gt_neighbors_total"] = len(gt_set)
    results["gt_neighbors_in_buckets"] = gt_in_buckets
    results["gt_coverage_rate"] = gt_in_buckets / len(gt_set) if len(gt_set) > 0 else 0.0
    
    # Print results
    print(f"Total buckets: {K}")
    print(f"Populated buckets: {results['populated_buckets']}/{K}")
    print(f"Empty buckets: {len(results['empty_buckets'])}")
    
    if bucket_sizes:
        print(f"Bucket sizes: min={results['min_bucket_size']}, "
              f"max={results['max_bucket_size']}, "
              f"avg={results['avg_bucket_size']:.1f}")
    
    print(f"Ground truth neighbors in buckets: {gt_in_buckets}/{len(gt_set)} ({results['gt_coverage_rate']:.1%})")
    
    if results["empty_buckets"]:
        results["issues"].append(
            f"{len(results['empty_buckets'])} buckets are empty (no base embeddings)"
        )
        print(f"⚠️  Empty buckets: {results['empty_buckets'][:10]}...")
    
    if results["gt_coverage_rate"] < 1.0:
        results["issues"].append(
            f"Only {results['gt_coverage_rate']:.1%} of ground truth neighbors are in buckets"
        )
        print(f"⚠️  Some ground truth neighbors are not in any bucket!")
    
    if results["min_bucket_size"] == 0 and results["populated_buckets"] < K:
        results["issues"].append("Some buckets have zero embeddings")
    
    return results


def test_h5_permutation_consistency(
    base_embeddings: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    n_bits: int,
    n_codes: int,
    random_state: int = 42,
) -> Dict:
    """
    Test H5: Consistency between permutation and code_to_bucket.
    
    Checks:
    - Do all bucket_idx from permutation exist in code_to_bucket?
    - Are there buckets in code_to_bucket that are never returned by permutation?
    - Is the mapping consistent?
    """
    print_section("H5: Testing Permutation-Code_to_Bucket Consistency")
    
    results = {
        "permutation_buckets": set(),
        "code_to_bucket_buckets": set(),
        "buckets_in_permutation_not_in_code": [],
        "buckets_in_code_not_in_permutation": [],
        "consistency_rate": 0.0,
        "issues": [],
    }
    
    # Build index
    lsh = HyperplaneLSH(n_bits=n_bits, dim=base_embeddings.shape[1], random_state=random_state)
    
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=ground_truth,
        encoder=None,
        n_bits=n_bits,
        n_codes=n_codes,
        use_codebook=True,
        use_semantic_distances=False,
        lsh_family=lsh,
        block_size=4,
        max_two_swap_iters=20,
        num_tunneling_steps=0,
        mode="two_swap_only",
        random_state=random_state,
    )
    
    permutation = index_obj.permutation
    code_to_bucket = index_obj.code_to_bucket
    
    # Get all buckets from permutation
    results["permutation_buckets"] = set(permutation)
    
    # Get all buckets from code_to_bucket
    results["code_to_bucket_buckets"] = set(code_to_bucket.values())
    
    # Find inconsistencies
    results["buckets_in_permutation_not_in_code"] = list(
        results["permutation_buckets"] - results["code_to_bucket_buckets"]
    )
    results["buckets_in_code_not_in_permutation"] = list(
        results["code_to_bucket_buckets"] - results["permutation_buckets"]
    )
    
    # Calculate consistency
    intersection = results["permutation_buckets"] & results["code_to_bucket_buckets"]
    union = results["permutation_buckets"] | results["code_to_bucket_buckets"]
    results["consistency_rate"] = len(intersection) / len(union) if len(union) > 0 else 0.0
    
    # Print results
    print(f"Buckets in permutation: {len(results['permutation_buckets'])}")
    print(f"Buckets in code_to_bucket: {len(results['code_to_bucket_buckets'])}")
    print(f"Intersection: {len(intersection)}")
    print(f"Consistency rate: {results['consistency_rate']:.1%}")
    
    if results["buckets_in_permutation_not_in_code"]:
        results["issues"].append(
            f"{len(results['buckets_in_permutation_not_in_code'])} buckets in permutation not in code_to_bucket"
        )
        print(f"⚠️  Buckets in permutation but not in code_to_bucket: {results['buckets_in_permutation_not_in_code'][:10]}...")
        print(f"   These buckets cannot be used for retrieval!")
    
    if results["buckets_in_code_not_in_permutation"]:
        results["issues"].append(
            f"{len(results['buckets_in_code_not_in_permutation'])} buckets in code_to_bucket not in permutation"
        )
        print(f"⚠️  Buckets in code_to_bucket but not in permutation: {results['buckets_in_code_not_in_permutation'][:10]}...")
        print(f"   These buckets exist but are never returned by queries!")
    
    if results["consistency_rate"] < 1.0:
        results["issues"].append(f"Inconsistency between permutation and code_to_bucket: {results['consistency_rate']:.1%}")
    
    return results


def main():
    """Run all diagnostic tests."""
    print("=" * 70)
    print("  Deep Investigation of GTH Low Recall Issue")
    print("=" * 70)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_queries = 20
    dim = 16
    n_bits = 6
    n_codes = 16
    k = 5
    hamming_radius = 1
    
    print(f"\nConfiguration:")
    print(f"  n_samples: {n_samples}")
    print(f"  n_queries: {n_queries}")
    print(f"  dim: {dim}")
    print(f"  n_bits: {n_bits}")
    print(f"  n_codes: {n_codes}")
    print(f"  k: {k}")
    print(f"  hamming_radius: {hamming_radius}")
    
    base_embeddings = np.random.randn(n_samples, dim).astype(np.float32)
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    
    # Generate ground truth
    distances = euclidean_distances(queries, base_embeddings)
    ground_truth = np.argsort(distances, axis=1)[:, :k].astype(np.int32)
    
    # Run all tests
    all_results = {}
    
    all_results["h1"] = test_h1_bucket_coverage(
        base_embeddings, queries, ground_truth, n_bits, n_codes
    )
    
    all_results["h2"] = test_h2_permutation_coverage(
        base_embeddings, queries, ground_truth, n_bits, n_codes
    )
    
    all_results["h3"] = test_h3_permutation_order(
        base_embeddings, queries, ground_truth, n_bits, n_codes, k, hamming_radius
    )
    
    all_results["h4"] = test_h4_empty_buckets(
        base_embeddings, queries, ground_truth, n_bits, n_codes
    )
    
    all_results["h5"] = test_h5_permutation_consistency(
        base_embeddings, queries, ground_truth, n_bits, n_codes
    )
    
    # Summary
    print_section("Summary of All Tests", level=1)
    
    total_issues = 0
    for test_name, result in all_results.items():
        issues = result.get("issues", [])
        total_issues += len(issues)
        
        print(f"\n{test_name.upper()}:")
        if issues:
            print(f"  ⚠️  {len(issues)} issue(s) found:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print(f"  ✓ No issues detected")
    
    # Overall assessment
    print(f"\n{'=' * 70}")
    if total_issues == 0:
        print("✓ All tests passed - no issues detected")
    else:
        print(f"⚠ {total_issues} total issue(s) detected across all tests")
        print("\nRecommended actions:")
        print("  1. Review issues above")
        print("  2. Prioritize fixes based on impact on recall")
        print("  3. Re-run tests after fixes")
    print("=" * 70)


if __name__ == "__main__":
    main()

