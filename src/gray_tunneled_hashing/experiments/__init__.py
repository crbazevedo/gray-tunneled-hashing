"""Experimental configuration and setup for Sprint 5.1."""

from gray_tunneled_hashing.experiments.config import LSHExperimentConfig
from gray_tunneled_hashing.experiments.setup import (
    create_experimental_setup,
    validate_setup,
    generate_synthetic_data,
)
from gray_tunneled_hashing.experiments.metrics import (
    compute_recall_at_k,
    compute_collision_preservation_rate,
    compute_hamming_ball_coverage,
    compute_improvement_over_baseline,
)
from gray_tunneled_hashing.experiments.collision_validation import (
    validate_collision_preservation,
    CollisionPreservationResult,
)

__all__ = [
    "LSHExperimentConfig",
    "create_experimental_setup",
    "validate_setup",
    "generate_synthetic_data",
    "compute_recall_at_k",
    "compute_collision_preservation_rate",
    "compute_hamming_ball_coverage",
    "compute_improvement_over_baseline",
    "validate_collision_preservation",
    "CollisionPreservationResult",
]

