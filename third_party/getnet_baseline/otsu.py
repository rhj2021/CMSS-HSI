from __future__ import annotations

import numpy as np


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Small dependency-free Otsu threshold implementation."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("Cannot compute Otsu threshold on empty/non-finite array")
    if np.min(x) == np.max(x):
        return float(np.min(x))
    hist, edges = np.histogram(x, bins=bins)
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * centers) / np.maximum(weight1, 1e-12)
    mean2 = (np.cumsum((hist * centers)[::-1]) / np.maximum(weight2[::-1], 1e-12))[::-1]
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    if variance12.size == 0:
        return float(np.mean(x))
    idx = int(np.argmax(variance12))
    return float(centers[:-1][idx])
