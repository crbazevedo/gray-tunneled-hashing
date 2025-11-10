"""Unified API for building and searching binary indices."""

from gray_tunneled_hashing.api.index_builder import (
    BinaryIndex,
    build_binary_index,
    search_binary_index,
)
from gray_tunneled_hashing.api.query_pipeline import (
    QueryResult,
    expand_hamming_ball,
    query_with_hamming_ball,
    get_candidate_set,
    batch_query_with_hamming_ball,
    analyze_hamming_ball_coverage,
)

__all__ = [
    "BinaryIndex",
    "build_binary_index",
    "search_binary_index",
    "QueryResult",
    "expand_hamming_ball",
    "query_with_hamming_ball",
    "get_candidate_set",
    "batch_query_with_hamming_ball",
    "analyze_hamming_ball_coverage",
]

