# Based on iMad.py by Mort Canty, Copyright (c) 2013.
# GPL-2.0-or-later; see THIRD_PARTY_NOTICES.md and third_party/licenses/GPL-2.0.txt.
"""Regularized IR-MAD change scores and Otsu thresholding."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy import linalg, stats
from sklearn.decomposition import PCA
from tqdm.auto import tqdm


@dataclass
class IRMADResult:
    score: np.ndarray
    mad_variates: np.ndarray | None
    threshold: float
    pred: np.ndarray
    metadata: dict[str, Any]


def otsu_threshold(values, bins=512):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return lo
    hist, edges = np.histogram(x, bins=bins, range=(lo, hi))
    hist = hist.astype(float)
    c = 0.5 * (edges[:-1] + edges[1:])
    w1 = np.cumsum(hist)
    w2 = np.cumsum(hist[::-1])[::-1]
    m1 = np.cumsum(hist * c) / np.maximum(w1, 1e-12)
    m2 = (np.cumsum((hist * c)[::-1]) / np.maximum(w2[::-1], 1e-12))[::-1]
    between = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
    return float(c[int(np.argmax(between))])


def _wcov(x, w):
    sw = max(float(w.sum()), 1e-12)
    mean = (x * w[:, None]).sum(0) / sw
    z = x - mean
    return mean, (z * w[:, None]).T @ z / sw


def _cca(s11, s22, s12, reg):
    d = len(s11)
    scale = max((np.trace(s11) + np.trace(s22)) / (2 * d), 1e-8)
    ridge = reg * scale
    a11 = (s11 + s11.T) / 2 + ridge * np.eye(d)
    a22 = (s22 + s22.T) / 2 + ridge * np.eye(d)
    inv22 = linalg.solve(a22, s12.T, assume_a="pos")
    c = s12 @ inv22
    c = (c + c.T) / 2
    ev, A = linalg.eigh(c, a11, check_finite=False)
    order = np.argsort(ev)[::-1]
    rho = np.sqrt(np.clip(ev[order], 0, 1 - 1e-8))
    A = A[:, order]
    B = inv22 @ A / np.maximum(rho[None, :], 1e-8)
    for j in range(d):
        A[:, j] /= np.sqrt(max(A[:, j].T @ a11 @ A[:, j], 1e-12))
        B[:, j] /= np.sqrt(max(B[:, j].T @ a22 @ B[:, j], 1e-12))
        if A[:, j].T @ s12 @ B[:, j] < 0:
            B[:, j] *= -1
    return A, B, rho


def run_irmad(
    t1,
    t2,
    pca_components=16,
    max_iter=30,
    tolerance=1e-3,
    regularization=1e-3,
    sample_size=50000,
    seed=0,
    return_mads=False,
    show_progress=False,
    progress_desc=None,
):
    """Estimate regularized IR-MAD scores in a shared PCA representation."""
    h, w, b = t1.shape
    x1 = np.asarray(t1, np.float32).reshape(-1, b)
    x2 = np.asarray(t2, np.float32).reshape(-1, b)
    valid = np.all(np.isfinite(x1), 1) & np.all(np.isfinite(x2), 1)
    idx = np.flatnonzero(valid)
    rng = np.random.default_rng(seed)
    fit = (
        idx if len(idx) <= sample_size else rng.choice(idx, sample_size, replace=False)
    )
    d = min(pca_components, b, max(1, len(fit) - 1))
    pca = PCA(n_components=d, svd_solver="randomized", random_state=seed).fit(
        np.concatenate([x1[fit], x2[fit]])
    )
    z1 = pca.transform(x1).astype(float)
    z2 = pca.transform(x2).astype(float)
    a = z1[fit]
    b2 = z2[fit]
    weights = np.ones(len(fit))
    old = np.zeros(d)
    hist = []
    iter_bar = tqdm(
        range(1, max_iter + 1),
        desc=progress_desc or "IRMAD",
        unit="iter",
        leave=False,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for it in iter_bar:
        m1, s11 = _wcov(a, weights)
        m2, s22 = _wcov(b2, weights)
        u = a - m1
        v = b2 - m2
        s12 = (u * weights[:, None]).T @ v / max(weights.sum(), 1e-12)
        A, B, rho = _cca(s11, s22, s12, regularization)
        m = u @ A - v @ B
        sig = np.maximum(2 * (1 - rho), 1e-6)
        chi = np.sum(m * m / sig[None, :], 1)
        weights = np.clip(1 - stats.chi2.cdf(chi, df=d), 1e-8, 1)
        delta = float(np.max(np.abs(rho - old)))
        hist.append(
            {
                "iteration": it,
                "delta": delta,
                "weight_mean": float(weights.mean()),
                "rho_min": float(rho.min()),
                "rho_max": float(rho.max()),
            }
        )
        iter_bar.set_postfix(delta=f"{delta:.2e}", wmean=f"{float(weights.mean()):.3f}")
        old = rho.copy()
        if delta < tolerance and it > 2:
            break
    m = (z1 - m1) @ A - (z2 - m2) @ B
    score = np.sum(m * m / sig[None, :], 1).reshape(h, w).astype(np.float32)
    thr = otsu_threshold(score)
    pred = (score >= thr).astype(np.uint8)
    meta = {
        "implementation": "numpy_scipy_adapter_of_user_provided_iMad",
        "input_bands": int(t1.shape[-1]),
        "pca_components": d,
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "iterations": len(hist),
        "converged_delta": hist[-1]["delta"],
        "threshold": thr,
        "rho": [float(v) for v in rho],
        "history": hist,
        "fit_pixels": len(fit),
    }
    return IRMADResult(
        score,
        m.reshape(h, w, d).astype(np.float32) if return_mads else None,
        thr,
        pred,
        meta,
    )
