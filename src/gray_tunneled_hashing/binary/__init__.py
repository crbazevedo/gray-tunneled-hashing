"""Binary encoding and codebook utilities."""

from gray_tunneled_hashing.binary.baselines import (
    sign_binarize,
    random_projection_binarize,
    apply_random_projection,
)
from gray_tunneled_hashing.binary.codebooks import (
    build_codebook_kmeans,
    encode_with_codebook,
    find_nearest_centroids,
)
from gray_tunneled_hashing.binary.lsh_families import (
    LSHFamily,
    HyperplaneLSH,
    PStableLSH,
    create_lsh_family,
    validate_lsh_properties,
)

__all__ = [
    "sign_binarize",
    "random_projection_binarize",
    "apply_random_projection",
    "build_codebook_kmeans",
    "encode_with_codebook",
    "find_nearest_centroids",
    "LSHFamily",
    "HyperplaneLSH",
    "PStableLSH",
    "create_lsh_family",
    "validate_lsh_properties",
]

