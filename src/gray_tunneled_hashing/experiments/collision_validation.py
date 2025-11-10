"""Validação de preservação de colisões."""

from dataclasses import dataclass
import numpy as np
from typing import Optional

from gray_tunneled_hashing.binary.lsh_families import LSHFamily
from gray_tunneled_hashing.distribution.pipeline import DistributionAwareIndex, apply_permutation


@dataclass
class CollisionPreservationResult:
    """Resultado de validação de preservação de colisões."""
    preservation_rate: float  # % de colisões preservadas
    total_collisions: int  # Número total de colisões antes
    preserved_collisions: int  # Número de colisões preservadas
    violated_pairs: list[tuple[int, int]]  # Pares que violam preservação


def validate_collision_preservation(
    embeddings: np.ndarray,
    lsh: LSHFamily,
    index_obj: DistributionAwareIndex,
) -> CollisionPreservationResult:
    """
    Valida que GTH preserva 100% das colisões LSH.
    
    Propriedade: Se c_i == c_j antes de GTH, então σ(c_i) == σ(c_j) depois.
    
    Args:
        embeddings: Embeddings originais, shape (N, dim)
        lsh: LSH family usada
        index_obj: Índice distribution-aware construído
        
    Returns:
        Resultado com estatísticas de preservação
    """
    # Hash embeddings antes de GTH
    codes_before = lsh.hash(embeddings)
    
    # Identificar colisões antes
    collisions_before = set()
    for i in range(len(codes_before)):
        for j in range(i + 1, len(codes_before)):
            if np.array_equal(codes_before[i], codes_before[j]):
                collisions_before.add((i, j))
    
    # Aplicar GTH permutation para obter códigos depois
    # A permutation é aplicada aos códigos dos buckets
    # Precisamos mapear embeddings -> buckets -> códigos permutados
    
    # Para cada embedding, encontrar seu bucket e código permutado
    # Como GTH preserva bucket membership, embeddings que colidem antes
    # devem estar no mesmo bucket e continuar colidindo depois
    
    # Obter códigos depois aplicando permutation
    codes_after = apply_permutation(
        codes=codes_before,
        bucket_to_code=index_obj.bucket_to_code,
        code_to_bucket=index_obj.code_to_bucket,
        permutation=index_obj.permutation,
        n_bits=index_obj.n_bits,
    )
    
    # Identificar colisões depois
    collisions_after = set()
    for i in range(len(codes_after)):
        for j in range(i + 1, len(codes_after)):
            if np.array_equal(codes_after[i], codes_after[j]):
                collisions_after.add((i, j))
    
    # Calcular preservação
    preserved = collisions_before & collisions_after
    violated = collisions_before - collisions_after
    
    preservation_rate = (
        (len(preserved) / len(collisions_before) * 100.0)
        if len(collisions_before) > 0
        else 100.0
    )
    
    return CollisionPreservationResult(
        preservation_rate=preservation_rate,
        total_collisions=len(collisions_before),
        preserved_collisions=len(preserved),
        violated_pairs=list(violated),
    )

