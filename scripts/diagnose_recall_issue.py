"""
Diagnostic script to investigate recall issues in GTH pipeline.

This script tests multiple hypotheses:
- H1: Problem in bucket → dataset index mapping
- H2: Ground truth incorrect or inconsistent
- H3: Permutation not applied correctly in query
- H4: Hamming ball expansion not returning correct candidates
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gray_tunneled_hashing.binary.lsh_families import HyperplaneLSH
from gray_tunneled_hashing.binary.baselines import random_projection_binarize, apply_random_projection
from gray_tunneled_hashing.distribution.pipeline import build_distribution_aware_index
from gray_tunneled_hashing.api.query_pipeline import query_with_hamming_ball
from gray_tunneled_hashing.binary.codebooks import build_codebook_kmeans, find_nearest_centroids
from gray_tunneled_hashing.evaluation.metrics import recall_at_k
from sklearn.metrics.pairwise import euclidean_distances


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_h1_bucket_mapping(
    base_embeddings: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    n_bits: int,
    n_codes: int,
    random_state: int = 42,
) -> Dict:
    """
    Test H1: Problem in bucket → dataset index mapping.
    
    Checks:
    - Are all base embeddings assigned to buckets?
    - Is bucket_to_dataset_indices mapping correct?
    - Do bucket assignments match codebook assignments?
    """
    print_section("H1: Testing Bucket → Dataset Mapping")
    
    results = {
        "all_embeddings_assigned": False,
        "bucket_coverage": 0.0,
        "mapping_consistency": False,
        "issues": [],
    }
    
    # Build codebook
    centroids, assignments = build_codebook_kmeans(
        embeddings=base_embeddings,
        n_codes=n_codes,
        random_state=random_state,
    )
    
    # Check all embeddings assigned
    if len(assignments) == len(base_embeddings):
        results["all_embeddings_assigned"] = True
        print(f"✓ All {len(base_embeddings)} embeddings assigned to buckets")
    else:
        results["issues"].append(f"Only {len(assignments)}/{len(base_embeddings)} embeddings assigned")
        print(f"✗ Assignment mismatch: {len(assignments)}/{len(base_embeddings)}")
    
    # Build bucket → dataset mapping
    bucket_to_dataset_indices = {}
    for dataset_idx, bucket_idx in enumerate(assignments):
        if bucket_idx not in bucket_to_dataset_indices:
            bucket_to_dataset_indices[bucket_idx] = []
        bucket_to_dataset_indices[bucket_idx].append(dataset_idx)
    
    # Check bucket coverage
    unique_buckets = len(bucket_to_dataset_indices)
    results["bucket_coverage"] = unique_buckets / n_codes
    print(f"✓ {unique_buckets}/{n_codes} buckets used ({results['bucket_coverage']:.1%})")
    
    # Check consistency: verify assignments match find_nearest_centroids
    test_assignments = find_nearest_centroids(base_embeddings, centroids)
    if np.array_equal(assignments, test_assignments):
        results["mapping_consistency"] = True
        print("✓ Codebook assignments consistent with find_nearest_centroids")
    else:
        diff_count = np.sum(assignments != test_assignments)
        results["issues"].append(f"{diff_count} inconsistent assignments")
        print(f"✗ {diff_count} inconsistent assignments")
    
    # Show bucket sizes
    bucket_sizes = [len(indices) for indices in bucket_to_dataset_indices.values()]
    print(f"  Bucket sizes: min={min(bucket_sizes)}, max={max(bucket_sizes)}, mean={np.mean(bucket_sizes):.1f}")
    
    return results


def test_h2_ground_truth(
    queries: np.ndarray,
    base_embeddings: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
) -> Dict:
    """
    Test H2: Ground truth correctness.
    
    Checks:
    - Are ground truth indices valid?
    - Do ground truth neighbors have smallest distances?
    - Is ground truth consistent with actual distances?
    """
    print_section("H2: Testing Ground Truth")
    
    results = {
        "valid_indices": False,
        "correct_neighbors": False,
        "distance_consistency": 0.0,
        "issues": [],
    }
    
    # Check validity of indices
    n_base = len(base_embeddings)
    n_queries = len(queries)
    
    if ground_truth.shape != (n_queries, k):
        results["issues"].append(f"Shape mismatch: expected ({n_queries}, {k}), got {ground_truth.shape}")
        print(f"✗ Shape mismatch: {ground_truth.shape}")
        return results
    
    if np.all((ground_truth >= 0) & (ground_truth < n_base)):
        results["valid_indices"] = True
        print(f"✓ All {n_queries * k} ground truth indices valid")
    else:
        invalid = np.sum((ground_truth < 0) | (ground_truth >= n_base))
        results["issues"].append(f"{invalid} invalid indices")
        print(f"✗ {invalid} invalid indices")
    
    # Verify ground truth matches actual distances
    distances = euclidean_distances(queries, base_embeddings)
    correct_count = 0
    total_count = 0
    
    for q_idx in range(n_queries):
        gt_neighbors = ground_truth[q_idx]
        actual_distances = distances[q_idx]
        sorted_indices = np.argsort(actual_distances)[:k]
        
        # Check if ground truth matches top-k
        gt_set = set(gt_neighbors)
        sorted_set = set(sorted_indices)
        
        if gt_set == sorted_set:
            correct_count += 1
        total_count += 1
    
    results["distance_consistency"] = correct_count / total_count
    print(f"✓ Ground truth matches actual distances: {correct_count}/{total_count} queries ({results['distance_consistency']:.1%})")
    
    if results["distance_consistency"] < 1.0:
        results["issues"].append(f"Only {results['distance_consistency']:.1%} queries have correct ground truth")
    
    return results


def test_h3_permutation_application(
    base_embeddings: np.ndarray,
    queries: np.ndarray,
    n_bits: int,
    n_codes: int,
    random_state: int = 42,
) -> Dict:
    """
    Test H3: Permutation application correctness.
    
    Checks:
    - Is permutation applied correctly to query codes?
    - Do permuted codes map to correct buckets?
    - Is permutation consistent with index_obj.permutation?
    """
    print_section("H3: Testing Permutation Application")
    
    results = {
        "permutation_applied": False,
        "bucket_mapping_correct": False,
        "consistency": 0.0,
        "issues": [],
    }
    
    # Build index
    lsh = HyperplaneLSH(n_bits=n_bits, dim=base_embeddings.shape[1], random_state=random_state)
    
    index_obj = build_distribution_aware_index(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth_neighbors=np.zeros((len(queries), 5), dtype=np.int32),  # Dummy
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
    
    # Encode queries
    query_codes = lsh.hash(queries)
    
    # Test permutation application
    consistent_count = 0
    total_count = len(queries)
    
    for i, query_code in enumerate(query_codes):
        # Get permuted code via query pipeline
        result = query_with_hamming_ball(
            query_code=query_code,
            permutation=index_obj.permutation,
            code_to_bucket=index_obj.code_to_bucket,
            bucket_to_code=index_obj.bucket_to_code,
            n_bits=n_bits,
            hamming_radius=0,  # Exact match only
        )
        
        # Check if permuted code is in code_to_bucket
        permuted_tuple = tuple(result.permuted_code.astype(int).tolist())
        if permuted_tuple in index_obj.code_to_bucket:
            consistent_count += 1
    
    results["consistency"] = consistent_count / total_count
    results["permutation_applied"] = results["consistency"] > 0.9
    print(f"✓ Permutation consistency: {consistent_count}/{total_count} queries ({results['consistency']:.1%})")
    
    if results["consistency"] < 1.0:
        results["issues"].append(f"Only {results['consistency']:.1%} queries have consistent permutation")
    
    return results


def test_h4_hamming_ball(
    base_embeddings: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    n_bits: int,
    n_codes: int,
    k: int,
    hamming_radius: int = 1,
    random_state: int = 42,
) -> Dict:
    """
    Test H4: Hamming ball expansion correctness.
    
    Checks:
    - Does Hamming ball return correct number of candidates?
    - Are candidates within Hamming radius?
    - Do candidates map to correct dataset indices?
    """
    print_section("H4: Testing Hamming Ball Expansion")
    
    results = {
        "ball_size_correct": False,
        "candidates_within_radius": False,
        "mapping_correct": False,
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
    centroids, assignments = build_codebook_kmeans(
        embeddings=base_embeddings,
        n_codes=n_codes,
        random_state=random_state,
    )
    
    bucket_to_dataset_indices = {}
    for dataset_idx, bucket_idx in enumerate(assignments):
        if bucket_idx not in bucket_to_dataset_indices:
            bucket_to_dataset_indices[bucket_idx] = []
        bucket_to_dataset_indices[bucket_idx].append(dataset_idx)
    
    # Encode queries
    query_codes = lsh.hash(queries)
    
    # Test Hamming ball expansion
    total_candidates = 0
    within_radius_count = 0
    mapping_correct_count = 0
    
    for i, query_code in enumerate(query_codes):
        result = query_with_hamming_ball(
            query_code=query_code,
            permutation=index_obj.permutation,
            code_to_bucket=index_obj.code_to_bucket,
            bucket_to_code=index_obj.bucket_to_code,
            n_bits=n_bits,
            hamming_radius=hamming_radius,
        )
        
        # Check Hamming distances
        from gray_tunneled_hashing.api.query_pipeline import hamming_distance
        distances = hamming_distance(
            result.permuted_code[np.newaxis, :],
            result.candidate_codes,
        )[0]
        
        if np.all(distances <= hamming_radius):
            within_radius_count += 1
        
        # Check mapping
        candidate_dataset_indices = []
        for bucket_idx in result.candidate_indices:
            if bucket_idx in bucket_to_dataset_indices:
                candidate_dataset_indices.extend(bucket_to_dataset_indices[bucket_idx])
        
        # Check if any ground truth neighbor is in candidates
        gt_neighbors = set(ground_truth[i])
        candidate_set = set(candidate_dataset_indices)
        
        if len(gt_neighbors & candidate_set) > 0:
            mapping_correct_count += 1
        
        total_candidates += len(result.candidate_indices)
    
    results["candidates_within_radius"] = within_radius_count == len(queries)
    results["mapping_correct"] = mapping_correct_count / len(queries)
    
    print(f"✓ Candidates within radius: {within_radius_count}/{len(queries)} queries")
    print(f"✓ Mapping correct: {mapping_correct_count}/{len(queries)} queries ({results['mapping_correct']:.1%})")
    print(f"  Average candidates per query: {total_candidates / len(queries):.1f}")
    
    if results["mapping_correct"] < 0.5:
        results["issues"].append(f"Only {results['mapping_correct']:.1%} queries have correct mapping")
    
    return results


def test_end_to_end_recall(
    base_embeddings: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    n_bits: int,
    n_codes: int,
    k: int,
    hamming_radius: int = 1,
    random_state: int = 42,
) -> Dict:
    """
    Test end-to-end recall calculation.
    
    Simulates the full pipeline and checks recall at each step.
    """
    print_section("End-to-End Recall Test")
    
    results = {
        "baseline_recall": 0.0,
        "gth_recall": 0.0,
        "improvement": 0.0,
        "issues": [],
    }
    
    # Baseline: LSH without GTH
    lsh = HyperplaneLSH(n_bits=n_bits, dim=base_embeddings.shape[1], random_state=random_state)
    base_codes = lsh.hash(base_embeddings)
    query_codes = lsh.hash(queries)
    
    from gray_tunneled_hashing.integrations.hamming_index import build_hamming_index
    baseline_index = build_hamming_index(base_codes, use_faiss=True)
    baseline_retrieved, _ = baseline_index.search(query_codes, k)
    baseline_recall = recall_at_k(baseline_retrieved, ground_truth, k=k)
    results["baseline_recall"] = baseline_recall
    
    print(f"Baseline recall: {baseline_recall:.4f}")
    
    # GTH: With permutation
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
    centroids, assignments = build_codebook_kmeans(
        embeddings=base_embeddings,
        n_codes=n_codes,
        random_state=random_state,
    )
    
    bucket_to_dataset_indices = {}
    for dataset_idx, bucket_idx in enumerate(assignments):
        if bucket_idx not in bucket_to_dataset_indices:
            bucket_to_dataset_indices[bucket_idx] = []
        bucket_to_dataset_indices[bucket_idx].append(dataset_idx)
    
    # Query with Hamming ball
    all_retrieved = []
    for query_code in query_codes:
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
        
        # Remove duplicates and limit to k
        candidate_dataset_indices = list(dict.fromkeys(candidate_dataset_indices))[:k]
        
        # Pad to k
        if len(candidate_dataset_indices) < k:
            candidate_dataset_indices.extend([-1] * (k - len(candidate_dataset_indices)))
        else:
            candidate_dataset_indices = candidate_dataset_indices[:k]
        
        all_retrieved.append(candidate_dataset_indices)
    
    retrieved_indices = np.array(all_retrieved, dtype=np.int32)
    gth_recall = recall_at_k(retrieved_indices, ground_truth, k=k)
    results["gth_recall"] = gth_recall
    results["improvement"] = gth_recall - baseline_recall
    
    print(f"GTH recall: {gth_recall:.4f}")
    print(f"Improvement: {results['improvement']:+.4f}")
    
    if results["improvement"] < 0:
        results["issues"].append(f"GTH recall ({gth_recall:.4f}) < baseline ({baseline_recall:.4f})")
    
    return results


def main():
    """Run all diagnostic tests."""
    print("=" * 70)
    print("  GTH Recall Issue Diagnostic")
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
    
    all_results["h1"] = test_h1_bucket_mapping(
        base_embeddings, queries, ground_truth, n_bits, n_codes
    )
    
    all_results["h2"] = test_h2_ground_truth(queries, base_embeddings, ground_truth, k)
    
    all_results["h3"] = test_h3_permutation_application(
        base_embeddings, queries, n_bits, n_codes
    )
    
    all_results["h4"] = test_h4_hamming_ball(
        base_embeddings, queries, ground_truth, n_bits, n_codes, k, hamming_radius
    )
    
    all_results["e2e"] = test_end_to_end_recall(
        base_embeddings, queries, ground_truth, n_bits, n_codes, k, hamming_radius
    )
    
    # Summary
    print_section("Summary")
    
    for test_name, result in all_results.items():
        print(f"\n{test_name.upper()}:")
        if result.get("issues"):
            print(f"  ✗ Issues found:")
            for issue in result["issues"]:
                print(f"    - {issue}")
        else:
            print(f"  ✓ No issues detected")
    
    # Overall assessment
    total_issues = sum(len(r.get("issues", [])) for r in all_results.values())
    if total_issues == 0:
        print("\n✓ All tests passed - no issues detected")
    else:
        print(f"\n⚠ {total_issues} issue(s) detected - review above")


if __name__ == "__main__":
    main()

