from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np

from .otsu import otsu_threshold


@dataclass
class PseudoSummary:
    height: int
    width: int
    bands: int
    source: str
    threshold: float
    num_changed: int
    num_unchanged: int
    selected_changed: int
    selected_unchanged: int
    fraction: float
    max_per_class: int


def cva_magnitude(t1: np.ndarray, t2: np.ndarray, mode: str = "l2") -> np.ndarray:
    diff = t2.astype(np.float32) - t1.astype(np.float32)
    mode = mode.lower()
    if mode == "l2":
        return np.sqrt(np.sum(diff * diff, axis=-1))
    if mode == "l1":
        return np.sum(np.abs(diff), axis=-1)
    if mode == "sumabs":
        return np.abs(np.sum(diff, axis=-1))
    raise ValueError(f"Unknown CVA magnitude mode: {mode}")


def _binary_from_pseudo_map(pseudo_map: np.ndarray) -> np.ndarray:
    pseudo_map = np.asarray(pseudo_map).squeeze()
    vals = np.unique(pseudo_map[np.isfinite(pseudo_map)])
    if vals.size <= 2:
        return (pseudo_map > np.min(vals)).astype(np.uint8)
    return (pseudo_map != 0).astype(np.uint8)


def make_pseudo_training_set(
    t1: np.ndarray,
    t2: np.ndarray,
    pseudo_map: Optional[np.ndarray] = None,
    pseudo_source: str = "cva",
    fraction: float = 0.7,
    max_per_class: int = 20000,
    min_per_class: int = 16,
    seed: int = 0,
    cva_mode: str = "l2",
    high_confidence: bool = True,
) -> Tuple[np.ndarray, np.ndarray, PseudoSummary]:
    """Create or filter pseudo-labels for GETNET training.

    Use a supplied pseudo map or compute CVA with Otsu thresholding, then select
    high-confidence changed and unchanged pixels.
    """
    if t1.shape != t2.shape or t1.ndim != 3:
        raise ValueError(f"Expected t1/t2 as [H,W,B], got {t1.shape} and {t2.shape}")
    h, w, b = t1.shape
    mag = cva_magnitude(t1, t2, cva_mode)
    threshold = otsu_threshold(mag)
    source = pseudo_source
    if pseudo_map is not None:
        if pseudo_map.shape != (h, w):
            if pseudo_map.T.shape == (h, w):
                pseudo_map = pseudo_map.T
            else:
                raise ValueError(
                    f"Pseudo map shape {pseudo_map.shape} does not match {(h, w)}"
                )
        pseudo = _binary_from_pseudo_map(pseudo_map)
        source = "provided_map"
    else:
        pseudo = (mag > threshold).astype(np.uint8)
        source = "cva_otsu"

    flat_mag = mag.reshape(-1)
    flat_pseudo = pseudo.reshape(-1)
    changed = np.where(flat_pseudo == 1)[0]
    unchanged = np.where(flat_pseudo == 0)[0]

    rng = np.random.default_rng(seed)

    def select(indices: np.ndarray, changed_class: bool) -> np.ndarray:
        if indices.size == 0:
            return indices
        n = (
            int(np.ceil(indices.size * fraction))
            if 0 < fraction <= 1
            else int(fraction)
        )
        n = max(min_per_class, n)
        n = min(n, max_per_class, indices.size)
        scores = flat_mag[indices]
        # Stable high-confidence selection: high magnitude for changed, low magnitude for unchanged.
        order = np.argsort(scores)
        if changed_class:
            order = order[::-1]
        keep = indices[order[:n]]
        rng.shuffle(keep)
        return keep

    if high_confidence:
        pos_idx = select(changed, True)
        neg_idx = select(unchanged, False)
    else:
        npos = min(max_per_class, max(min_per_class, int(len(changed) * fraction)))
        nneg = min(max_per_class, max(min_per_class, int(len(unchanged) * fraction)))
        pos_idx = (
            rng.choice(changed, size=min(npos, len(changed)), replace=False)
            if len(changed)
            else np.array([], dtype=int)
        )
        neg_idx = (
            rng.choice(unchanged, size=min(nneg, len(unchanged)), replace=False)
            if len(unchanged)
            else np.array([], dtype=int)
        )

    indices = np.concatenate([pos_idx, neg_idx]).astype(np.int64)
    labels = np.concatenate(
        [np.ones(len(pos_idx), dtype=np.int64), np.zeros(len(neg_idx), dtype=np.int64)]
    )
    perm = rng.permutation(indices.size)
    indices = indices[perm]
    labels = labels[perm]

    summary = PseudoSummary(
        height=h,
        width=w,
        bands=b,
        source=source,
        threshold=float(threshold),
        num_changed=int(len(changed)),
        num_unchanged=int(len(unchanged)),
        selected_changed=int(len(pos_idx)),
        selected_unchanged=int(len(neg_idx)),
        fraction=float(fraction),
        max_per_class=int(max_per_class),
    )
    return indices, labels, summary
