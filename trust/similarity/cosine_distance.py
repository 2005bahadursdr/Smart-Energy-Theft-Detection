"""
Cosine Directional Agreement and Update-Distance Metrics for Trust Evaluation.
Computes:
1. Coordinate-wise robust reference update vector.
2. Cosine similarity between client delta updates and reference.
3. Euclidean distance and norm outlier anomaly scores.
4. Robust edge-case sanitization (zero norm, identical updates, NaN/Inf).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


def flatten_parameters(params: List[np.ndarray]) -> np.ndarray:
    """
    Flatten a list of parameter arrays into a single contiguous 1D float32 vector.
    """
    if not params:
        return np.array([], dtype=np.float32)
    return np.concatenate([arr.flatten().astype(np.float32) for arr in params])


def unflatten_parameters(
    flat_vector: np.ndarray,
    shapes_and_dtypes: List[Tuple[Tuple[int, ...], np.dtype]],
) -> List[np.ndarray]:
    """
    Reconstruct parameter arrays from a flattened 1D vector.
    """
    arrays = []
    offset = 0
    for shape, dtype in shapes_and_dtypes:
        size = int(np.prod(shape))
        arr = flat_vector[offset : offset + size].reshape(shape).astype(dtype)
        arrays.append(arr)
        offset += size
    return arrays


def compute_parameter_delta(
    local_params: List[np.ndarray],
    global_params: List[np.ndarray],
) -> List[np.ndarray]:
    """
    Compute gradient-step parameter delta: Delta w = local - global.
    """
    return [l - g for l, g in zip(local_params, global_params)]


def compute_coordinate_median(vectors: List[np.ndarray]) -> np.ndarray:
    """
    Compute coordinate-wise median across a list of equal-length 1D vectors.
    Provides a Byzantine-robust empirical reference direction.
    """
    if not vectors:
        return np.array([], dtype=np.float32)
    stacked = np.stack(vectors, axis=0) # Shape: (N, D)
    return np.median(stacked, axis=0).astype(np.float32)


def compute_cosine_similarity(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Compute cosine similarity between two 1D vectors: (a . b) / (||a|| * ||b||).
    Handles zero norms and numerical bounds cleanly.
    """
    if vec_a.size == 0 or vec_b.size == 0:
        return 0.0

    # Sanitize inputs
    vec_a = np.nan_to_num(vec_a, nan=0.0, posinf=0.0, neginf=0.0)
    vec_b = np.nan_to_num(vec_b, nan=0.0, posinf=0.0, neginf=0.0)

    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))

    # Edge case: zero vectors (no parameter movement)
    if norm_a < eps or norm_b < eps:
        # If both are zero, they agree identically
        if norm_a < eps and norm_b < eps:
            return 1.0
        return 0.0

    dot_prod = float(np.dot(vec_a, vec_b))
    raw_sim = dot_prod / (norm_a * norm_b)
    # Clip to valid mathematical range [-1.0, 1.0]
    return float(np.clip(raw_sim, -1.0, 1.0))


def compute_euclidean_distance(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
) -> float:
    """
    Compute L2 Euclidean distance between two vectors: ||a - b||_2.
    """
    if vec_a.size == 0 or vec_b.size == 0:
        return 0.0
    vec_a = np.nan_to_num(vec_a, nan=0.0, posinf=0.0, neginf=0.0)
    vec_b = np.nan_to_num(vec_b, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.linalg.norm(vec_a - vec_b))


def evaluate_similarity_and_distance(
    client_deltas: Dict[str, List[np.ndarray]],
    reference_delta: Optional[List[np.ndarray]] = None,
    distance_scale: float = 2.0,
    eps: float = 1e-8,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate cosine similarity and distance-based anomaly metrics for all client updates.

    Args:
        client_deltas: Dictionary mapping client_id to list of delta arrays (local - global).
        reference_delta: Optional precomputed reference delta. If None, computes coordinate median.
        distance_scale: Exponential scaling factor for distance score mapping.
        eps: Small epsilon to prevent division by zero.

    Returns:
        Dictionary mapping client_id to metric dictionary containing:
        - 'raw_cosine': float in [-1.0, 1.0]
        - 'similarity_score': normalized non-negative float in [0.0, 1.0]
        - 'euclidean_distance': float >= 0.0
        - 'distance_score': normalized non-negative float in [0.0, 1.0]
        - 'norm': float >= 0.0
    """
    if not client_deltas:
        return {}

    client_ids = list(client_deltas.keys())
    flat_deltas = {cid: flatten_parameters(client_deltas[cid]) for cid in client_ids}

    # Determine reference vector (default: coordinate-wise median)
    if reference_delta is not None:
        flat_ref = flatten_parameters(reference_delta)
    else:
        flat_ref = compute_coordinate_median(list(flat_deltas.values()))

    # Compute raw distances and norms
    raw_cosines = {}
    euclidean_dists = {}
    norms = {}

    for cid in client_ids:
        vec = flat_deltas[cid]
        norms[cid] = float(np.linalg.norm(vec))
        raw_cosines[cid] = compute_cosine_similarity(vec, flat_ref, eps=eps)
        euclidean_dists[cid] = compute_euclidean_distance(vec, flat_ref)

    # Compute median absolute deviation (MAD) or IQR for distance scaling
    dist_values = np.array(list(euclidean_dists.values()))
    med_dist = float(np.median(dist_values))
    # Median Absolute Deviation (MAD) of Euclidean distances
    mad_dist = float(np.median(np.abs(dist_values - med_dist)))
    # Dynamic scale proportional to median deviation and spread
    scale = max(mad_dist, med_dist * 0.5, 1e-4) * distance_scale

    results = {}
    for cid in client_ids:
        raw_cos = raw_cosines[cid]
        # Normalized similarity: map [-1, 1] -> [0, 1], with negative values suppressed
        # If cos >= 0: linear ramp from 0.5 to 1.0; if cos < 0: decay from 0.5 to 0.0
        norm_sim = max(0.0, float((raw_cos + 1.0) / 2.0))
        # Severe penalty for strongly opposing updates (cosine < 0)
        if raw_cos < 0.0:
            norm_sim = max(0.0, float(norm_sim * (1.0 + raw_cos)))

        # Distance score: exponential penalty for deviations beyond median
        dist = euclidean_dists[cid]
        excess_dist = max(0.0, dist - med_dist)
        dist_score = float(np.exp(-excess_dist / max(scale, eps)))
        dist_score = float(np.clip(dist_score, 0.0, 1.0))

        results[cid] = {
            "raw_cosine": round(raw_cos, 4),
            "similarity_score": round(norm_sim, 4),
            "euclidean_distance": round(dist, 4),
            "distance_score": round(dist_score, 4),
            "norm": round(norms[cid], 4),
        }

    return results
