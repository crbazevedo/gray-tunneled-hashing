"""Data generation and processing utilities."""

from gray_tunneled_hashing.data.synthetic_generators import (
    generate_synthetic_embeddings,
    generate_hypercube_vertices,
    generate_planted_phi,
    sample_noisy_embeddings,
    PlantedModelConfig,
)
from gray_tunneled_hashing.data.real_datasets import (
    load_embeddings,
    load_queries_and_ground_truth,
    list_available_datasets,
)

__all__ = [
    "generate_synthetic_embeddings",
    "generate_hypercube_vertices",
    "generate_planted_phi",
    "sample_noisy_embeddings",
    "PlantedModelConfig",
    "load_embeddings",
    "load_queries_and_ground_truth",
    "list_available_datasets",
]

