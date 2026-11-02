"""Similarity and update-distance module for trust scoring."""

from trust.similarity.cosine_distance import (
    compute_coordinate_median,
    compute_cosine_similarity,
    compute_euclidean_distance,
    compute_parameter_delta,
    evaluate_similarity_and_distance,
    flatten_parameters,
    unflatten_parameters,
)

__all__ = [
    "compute_coordinate_median",
    "compute_cosine_similarity",
    "compute_euclidean_distance",
    "compute_parameter_delta",
    "evaluate_similarity_and_distance",
    "flatten_parameters",
    "unflatten_parameters",
]
