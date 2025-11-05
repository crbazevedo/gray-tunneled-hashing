"""Data generation and processing utilities."""

from gray_tunneled_hashing.data.synthetic_generators import (
    generate_synthetic_embeddings,
    generate_hypercube_vertices,
    generate_planted_phi,
    sample_noisy_embeddings,
    PlantedModelConfig,
)

__all__ = [
    "generate_synthetic_embeddings",
    "generate_hypercube_vertices",
    "generate_planted_phi",
    "sample_noisy_embeddings",
    "PlantedModelConfig",
]

