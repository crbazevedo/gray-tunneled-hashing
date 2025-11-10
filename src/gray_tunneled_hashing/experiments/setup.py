"""Experimental setup and data generation."""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from gray_tunneled_hashing.experiments.config import LSHExperimentConfig


@dataclass
class ExperimentalSetup:
    """Estrutura de dados para setup experimental."""
    base_embeddings: np.ndarray
    queries: np.ndarray
    ground_truth: np.ndarray
    config: LSHExperimentConfig


@dataclass
class ValidationResult:
    """Resultado de validação de setup."""
    is_valid: bool
    errors: list[str]
    warnings: list[str]


@dataclass
class SyntheticData:
    """Dados sintéticos gerados."""
    base_embeddings: np.ndarray
    queries: np.ndarray
    ground_truth: np.ndarray


def create_experimental_setup(config: LSHExperimentConfig) -> ExperimentalSetup:
    """
    Cria setup experimental completo.
    
    Args:
        config: Configuração experimental
        
    Returns:
        Setup experimental com embeddings, queries e ground truth
    """
    data = generate_synthetic_data(config)
    
    return ExperimentalSetup(
        base_embeddings=data.base_embeddings,
        queries=data.queries,
        ground_truth=data.ground_truth,
        config=config,
    )


def validate_setup(setup: ExperimentalSetup) -> ValidationResult:
    """
    Valida constraints do setup experimental.
    
    Args:
        setup: Setup experimental a validar
        
    Returns:
        Resultado de validação
    """
    errors = []
    warnings = []
    
    # Validar shapes
    if setup.base_embeddings.shape[0] != setup.config.n_samples:
        errors.append(
            f"base_embeddings shape[0] ({setup.base_embeddings.shape[0]}) "
            f"!= n_samples ({setup.config.n_samples})"
        )
    
    if setup.queries.shape[0] != setup.config.n_queries:
        errors.append(
            f"queries shape[0] ({setup.queries.shape[0]}) "
            f"!= n_queries ({setup.config.n_queries})"
        )
    
    if setup.base_embeddings.shape[1] != setup.config.dim:
        errors.append(
            f"base_embeddings shape[1] ({setup.base_embeddings.shape[1]}) "
            f"!= dim ({setup.config.dim})"
        )
    
    if setup.queries.shape[1] != setup.config.dim:
        errors.append(
            f"queries shape[1] ({setup.queries.shape[1]}) "
            f"!= dim ({setup.config.dim})"
        )
    
    if setup.ground_truth.shape[0] != setup.config.n_queries:
        errors.append(
            f"ground_truth shape[0] ({setup.ground_truth.shape[0]}) "
            f"!= n_queries ({setup.config.n_queries})"
        )
    
    if setup.ground_truth.shape[1] != setup.config.k:
        errors.append(
            f"ground_truth shape[1] ({setup.ground_truth.shape[1]}) "
            f"!= k ({setup.config.k})"
        )
    
    # Validar constraints
    config_errors = setup.config.validate()
    errors.extend(config_errors)
    
    # Warnings
    if setup.config.n_codes > setup.config.n_samples:
        warnings.append(
            f"n_codes ({setup.config.n_codes}) > n_samples ({setup.config.n_samples})"
        )
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def generate_synthetic_data(config: LSHExperimentConfig) -> SyntheticData:
    """
    Gera dados sintéticos para experimentos.
    
    Args:
        config: Configuração experimental
        
    Returns:
        Dados sintéticos (embeddings, queries, ground truth)
    """
    np.random.seed(config.random_state)
    
    # Gerar embeddings base
    base_embeddings = np.random.randn(
        config.n_samples, config.dim
    ).astype(np.float32)
    
    # Gerar queries
    queries = np.random.randn(
        config.n_queries, config.dim
    ).astype(np.float32)
    
    # Calcular ground truth k-NN
    distances = euclidean_distances(queries, base_embeddings)
    ground_truth = np.argsort(distances, axis=1)[:, :config.k].astype(np.int32)
    
    return SyntheticData(
        base_embeddings=base_embeddings,
        queries=queries,
        ground_truth=ground_truth,
    )

