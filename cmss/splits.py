"""Spatial block splits, guard regions, and class-balanced pixel indices."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy.ndimage import binary_erosion
from .utils import stable_hash_array


@dataclass
class SpatialSplit:
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    metadata: dict[str, Any]


def _counts(m, g):
    return int(np.sum(m & (g == 0))), int(np.sum(m & (g == 1)))


def make_spatial_block_split(
    gt: np.ndarray,
    seed: int,
    block_size: int = 24,
    train_fraction: float = 0.15,
    val_fraction: float = 0.10,
    guard: int = 3,
    max_trials: int = 512,
) -> SpatialSplit:
    valid = gt >= 0
    h, w = gt.shape
    rec = []
    for y0 in range(0, h, block_size):
        for x0 in range(0, w, block_size):
            y1, x1 = min(h, y0 + block_size), min(w, x0 + block_size)
            m = valid[y0:y1, x0:x1]
            if m.any():
                rec.append(
                    (
                        y0,
                        y1,
                        x0,
                        x1,
                        int(np.sum(gt[y0:y1, x0:x1] == 0)),
                        int(np.sum(gt[y0:y1, x0:x1] == 1)),
                    )
                )
    if len(rec) < 3:
        raise ValueError("Too few blocks")
    total = np.array([np.sum(gt == 0), np.sum(gt == 1)], float)
    target = np.vstack(
        [
            train_fraction * total,
            val_fraction * total,
            (1 - train_fraction - val_fraction) * total,
        ]
    )
    nb = len(rec)
    nt = max(1, round(train_fraction * nb))
    nv = max(1, round(val_fraction * nb))
    rng = np.random.default_rng(seed)
    best = None
    best_score = np.inf
    for _ in range(max_trials):
        p = rng.permutation(nb)
        a = np.full(nb, 2, np.int8)
        a[p[:nt]] = 0
        a[p[nt : nt + nv]] = 1
        c = np.zeros((3, 2), float)
        for i, r in enumerate(rec):
            c[a[i]] += r[4:6]
        if np.any(c < 1):
            continue
        score = float(np.sum(((c - target) / np.maximum(target, 1)) ** 2))
        if score < best_score:
            best_score = score
            best = a.copy()
    if best is None:
        raise RuntimeError("Could not make split")
    masks = [np.zeros((h, w), bool) for _ in range(3)]
    for a, r in zip(best, rec):
        masks[int(a)][r[0] : r[1], r[2] : r[3]] = True
    masks = [m & valid for m in masks]
    guard_used = guard
    if guard > 0:
        st = np.ones((2 * guard + 1, 2 * guard + 1), bool)
        er = [binary_erosion(m, structure=st, border_value=0) & valid for m in masks]
        if all(min(_counts(m, gt)) >= 8 for m in er):
            masks = er
        else:
            guard_used = 0
    tr, va, te = masks
    meta = {
        "seed": seed,
        "block_size": block_size,
        "guard_requested": guard,
        "guard_used": guard_used,
        "score": best_score,
        "train_unchanged": _counts(tr, gt)[0],
        "train_changed": _counts(tr, gt)[1],
        "val_unchanged": _counts(va, gt)[0],
        "val_changed": _counts(va, gt)[1],
        "test_unchanged": _counts(te, gt)[0],
        "test_changed": _counts(te, gt)[1],
        "train_hash": stable_hash_array(tr),
        "val_hash": stable_hash_array(va),
        "test_hash": stable_hash_array(te),
    }
    return SpatialSplit(tr, va, te, meta)


def balanced_indices(
    mask: np.ndarray, gt: np.ndarray, max_per_class: int | None, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for c in (0, 1):
        idx = np.flatnonzero(mask.ravel() & (gt.ravel() == c))
        if max_per_class and len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        out.append(idx)
    z = np.concatenate(out).astype(np.int64)
    rng.shuffle(z)
    return z
