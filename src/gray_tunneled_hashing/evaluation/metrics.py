"""Evaluation metrics for hashing algorithms."""

import numpy as np
from typing import Optional


def hamming_preservation_score(
    original_distances: np.ndarray,
    hamming_distances: np.ndarray,
    metric: str = "correlation",
) -> float:
    """
    Compute a score measuring how well Hamming distances preserve original distances.
    
    This is a placeholder implementation. The full metric will be implemented
    in future sprints to measure the quality of Gray-Tunneled Hashing.
    
    Args:
        original_distances: Pairwise distances in original embedding space
        hamming_distances: Pairwise Hamming distances in binary code space
        metric: Type of preservation metric ('correlation', 'spearman', etc.)
        
    Returns:
        Score indicating how well distances are preserved (higher is better)
        
    Examples:
        >>> orig_dist = np.array([0.1, 0.5, 0.9])
        >>> hamm_dist = np.array([2, 5, 8])
        >>> score = hamming_preservation_score(orig_dist, hamm_dist)
        >>> isinstance(score, float)
        True
    """
    if original_distances.shape != hamming_distances.shape:
        raise ValueError(
            f"Distance arrays must have same shape: "
            f"{original_distances.shape} vs {hamming_distances.shape}"
        )
    
    if metric == "correlation":
        # Placeholder: return correlation coefficient
        if len(original_distances) < 2:
            return 0.0
        correlation = np.corrcoef(original_distances, hamming_distances)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0
    else:
        # Placeholder for other metrics
        return 0.0


def hamming_distance(codes1: np.ndarray, codes2: np.ndarray) -> np.ndarray:
    """
    Compute Hamming distances between binary codes.
    
    Args:
        codes1: Binary codes of shape (n_samples1, code_length)
        codes2: Binary codes of shape (n_samples2, code_length)
        
    Returns:
        Hamming distances of shape (n_samples1, n_samples2)
    """
    if codes1.ndim != 2 or codes2.ndim != 2:
        raise ValueError("codes must be 2D arrays")
    if codes1.shape[1] != codes2.shape[1]:
        raise ValueError("codes must have same code_length")
    
    # XOR to find differing bits, then sum
    # Expand dimensions for broadcasting
    codes1_expanded = codes1[:, np.newaxis, :]  # (n1, 1, code_length)
    codes2_expanded = codes2[np.newaxis, :, :]  # (1, n2, code_length)
    
    # XOR and sum
    differences = np.bitwise_xor(codes1_expanded, codes2_expanded)
    distances = np.sum(differences, axis=2)
    
    return distances


def recall_at_k(
    retrieved_indices: np.ndarray,
    ground_truth_indices: np.ndarray,
    k: Optional[int] = None,
) -> float:
    """
    Compute recall@k metric.
    
    Recall@k is the fraction of ground truth neighbors that appear
    in the top-k retrieved results.
    
    Args:
        retrieved_indices: Retrieved indices of shape (Q, k_retrieved)
                          where k_retrieved >= k
        ground_truth_indices: Ground truth indices of shape (Q, k_gt)
        k: Number of top results to consider (default: min(k_retrieved, k_gt))
        
    Returns:
        Recall@k score (0.0 to 1.0)
    """
    if retrieved_indices.ndim != 2 or ground_truth_indices.ndim != 2:
        raise ValueError("Both arrays must be 2D")
    
    Q_ret = retrieved_indices.shape[0]
    Q_gt = ground_truth_indices.shape[0]
    
    if Q_ret != Q_gt:
        raise ValueError(
            f"Number of queries must match: {Q_ret} vs {Q_gt}"
        )
    
    if k is None:
        k = min(retrieved_indices.shape[1], ground_truth_indices.shape[1])
    
    if k > retrieved_indices.shape[1]:
        raise ValueError(
            f"k={k} cannot exceed retrieved_indices.shape[1]={retrieved_indices.shape[1]}"
        )
    
    if k > ground_truth_indices.shape[1]:
        raise ValueError(
            f"k={k} cannot exceed ground_truth_indices.shape[1]={ground_truth_indices.shape[1]}"
        )
    
    # Slice to k
    retrieved_k = retrieved_indices[:, :k]
    ground_truth_k = ground_truth_indices[:, :k]
    
    # Compute recall for each query
    recalls = []
    for i in range(Q_ret):
        retrieved_set = set(retrieved_k[i])
        ground_truth_set = set(ground_truth_k[i])
        
        # Intersection size / ground truth size
        intersection = len(retrieved_set & ground_truth_set)
        recall = intersection / len(ground_truth_set) if len(ground_truth_set) > 0 else 0.0
        recalls.append(recall)
    
    return np.mean(recalls)

