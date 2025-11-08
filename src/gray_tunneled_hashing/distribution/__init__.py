"""Distribution-aware Gray-Tunneled Hashing module."""

from gray_tunneled_hashing.distribution.traffic_stats import (
    collect_traffic_stats,
    estimate_bucket_mass,
    estimate_neighbor_weights,
    build_weighted_distance_matrix,
)
from gray_tunneled_hashing.distribution.pipeline import (
    build_distribution_aware_index,
    apply_permutation,
    DistributionAwareIndex,
)

__all__ = [
    "collect_traffic_stats",
    "estimate_bucket_mass",
    "estimate_neighbor_weights",
    "build_weighted_distance_matrix",
    "build_distribution_aware_index",
    "apply_permutation",
    "DistributionAwareIndex",
]

