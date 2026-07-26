"""Construct the binary fingerprint queries used by MS-GPT."""
from __future__ import annotations

import numpy as np


def point_query(posterior: np.ndarray, threshold: float) -> np.ndarray:
    """Return one deterministic thresholded posterior query."""
    posterior = np.asarray(posterior, dtype=np.float32)
    if posterior.ndim != 1:
        raise ValueError(f"posterior must be one-dimensional, got {posterior.shape}")
    return (posterior > threshold).astype(np.float32)[None, :]


def density_calibrated_query_band(
    posterior: np.ndarray,
    num_queries: int,
    threshold_low: float,
    threshold_high: float,
) -> np.ndarray:
    """Interpolate the paper's density-calibrated threshold band."""
    posterior = np.asarray(posterior, dtype=np.float32)
    if posterior.ndim != 1:
        raise ValueError(f"posterior must be one-dimensional, got {posterior.shape}")
    thresholds = np.linspace(
        threshold_low, threshold_high, num_queries, dtype=np.float32
    )
    return np.stack(
        [(posterior > threshold).astype(np.float32) for threshold in thresholds]
    )


def construct_queries(
    method: str,
    posterior: np.ndarray,
    *,
    num_queries: int,
    threshold: float | None = None,
    threshold_low: float | None = None,
    threshold_high: float | None = None,
) -> np.ndarray:
    """Dispatch between the point-query ablation and the MS-GPT query band."""
    if method == "point":
        if threshold is None:
            raise ValueError("point query requires threshold")
        return point_query(posterior, threshold)
    if method == "density_calibrated_band":
        if threshold_low is None or threshold_high is None:
            raise ValueError("density-calibrated band requires both endpoints")
        return density_calibrated_query_band(
            posterior, num_queries, threshold_low, threshold_high
        )
    raise ValueError(f"unknown posterior-query method: {method}")
