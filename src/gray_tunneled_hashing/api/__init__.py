"""Unified API for building and searching binary indices."""

from gray_tunneled_hashing.api.index_builder import (
    BinaryIndex,
    build_binary_index,
    search_binary_index,
)

__all__ = [
    "BinaryIndex",
    "build_binary_index",
    "search_binary_index",
]

