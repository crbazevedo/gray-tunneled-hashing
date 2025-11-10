"""Métricas bem definidas para experimentos."""

import numpy as np
from typing import Optional

from gray_tunneled_hashing.evaluation.metrics import recall_at_k as recall_at_k_base
from gray_tunneled_hashing.api.query_pipeline import expand_hamming_ball


def compute_recall_at_k(
    retrieved_indices: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
) -> float:
    """
    Calcula recall@k.
    
    Definição: |retrieved ∩ ground_truth| / |ground_truth|
    
    Args:
        retrieved_indices: Índices recuperados, shape (n_queries, k)
        ground_truth: Índices verdadeiros, shape (n_queries, k)
        k: Número de vizinhos
        
    Returns:
        Recall@k (0.0 a 1.0)
    """
    return float(recall_at_k_base(retrieved_indices, ground_truth, k=k))


def compute_collision_preservation_rate(
    collisions_before: set[tuple[int, int]],
    collisions_after: set[tuple[int, int]],
) -> float:
    """
    Calcula taxa de preservação de colisões.
    
    Definição: % de pares que colidem antes e depois de GTH
    
    Args:
        collisions_before: Set de pares (i, j) que colidem antes
        collisions_after: Set de pares (i, j) que colidem depois
        
    Returns:
        Taxa de preservação (0.0 a 100.0)
    """
    if len(collisions_before) == 0:
        return 100.0  # No collisions to preserve
    
    preserved = len(collisions_before & collisions_after)
    return (preserved / len(collisions_before)) * 100.0


def compute_hamming_ball_coverage(
    center_code: np.ndarray,
    radius: int,
    n_bits: int,
) -> int:
    """
    Calcula cobertura do Hamming ball.
    
    Definição: Número de códigos no Hamming ball de radius r
    
    Args:
        center_code: Código central, shape (n_bits,)
        radius: Raio do Hamming ball
        n_bits: Número de bits
        
    Returns:
        Número de códigos no Hamming ball
    """
    codes = expand_hamming_ball(
        center_code=center_code,
        radius=radius,
        n_bits=n_bits,
    )
    return len(codes)


def compute_improvement_over_baseline(
    recall_gth: float,
    recall_baseline: float,
) -> float:
    """
    Calcula melhoria relativa sobre baseline.
    
    Definição: (recall_gth - recall_baseline) / recall_baseline * 100
    
    Args:
        recall_gth: Recall com GTH
        recall_baseline: Recall baseline
        
    Returns:
        Melhoria percentual (pode ser negativa)
    """
    if recall_baseline == 0.0:
        if recall_gth > 0.0:
            return float('inf')
        else:
            return 0.0
    
    return ((recall_gth - recall_baseline) / recall_baseline) * 100.0

