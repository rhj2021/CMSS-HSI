"""Patch extraction, training-only normalization and shared CMSS data splits."""

from __future__ import annotations
from dataclasses import dataclass
import hashlib, json, zlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from cmss.data import load_dataset
from cmss.splits import make_spatial_block_split, balanced_indices
from cmss.transforms import oracle_realign
from experiments.benchmark.transforms import apply_extended


def array_hash(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def split_data(ds, seed, cfg):
    s = cfg["split"]
    sp = make_spatial_block_split(
        ds.gt,
        seed,
        int(s["block_size"]),
        float(s["train_fraction"]),
        float(s["val_fraction"]),
        int(s["guard"]),
    )
    train_idx = balanced_indices(
        sp.train_mask, ds.gt, int(s["max_train_per_class"]), seed
    )
    val_idx = balanced_indices(
        sp.val_mask, ds.gt, int(s["max_val_per_class"]), seed + 1000
    )
    return sp, train_idx, val_idx


def make_case(ds, spec, base, device):
    # Use the same deterministic perturbation as the benchmark.
    if spec["type"] == "identity":
        return ds.t2, np.ones(ds.gt.shape, bool)
    seed = int(
        zlib.crc32(f"{ds.name}:{spec['name']}:{base['correction']['seed']}".encode())
        & 0x7FFFFFFF
    )
    tr = apply_extended(ds.t2, spec, device, seed)
    _, ov = oracle_realign(tr.moving, tr.forward_flow, device=device)
    return tr.moving, tr.valid_reference & ov


@dataclass
class TrainMinMax:
    low: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, t1, t2, indices):
        # Fit per-band scaling to training centers, pooled across dates.
        a = t1.reshape(-1, t1.shape[-1])[indices]
        b = t2.reshape(-1, t2.shape[-1])[indices]
        lo = np.minimum(a.min(0), b.min(0)).astype(np.float32)
        hi = np.maximum(a.max(0), b.max(0)).astype(np.float32)
        return cls(lo, np.maximum(hi - lo, np.float32(1e-6)))

    def apply(self, x):
        return np.clip(
            (np.asarray(x, dtype=np.float32) - self.low) / self.scale, 0, 1
        ).astype(np.float32)

    def save(self, path):
        np.savez_compressed(path, low=self.low, scale=self.scale)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls(z["low"], z["scale"])


class PatchPair(Dataset):
    def __init__(self, t1, t2, indices, patch_size=25, labels=None):
        if patch_size != 25:
            raise ValueError("RCDIFO requires 25x25 patches.")
        if t1.shape != t2.shape or t1.ndim != 3:
            raise ValueError("Expected equal HxWxB cubes")
        self.h, self.w, self.b = t1.shape
        self.indices = np.asarray(indices, np.int64)
        if len(np.unique(self.indices)) != len(self.indices):
            raise ValueError("Duplicate patch centers")
        self.p = patch_size
        pad = patch_size // 2
        self.a = np.pad(t1, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        self.bimg = np.pad(t2, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        # Only center labels for the explicit train/validation subset are retained.
        self.labels = (
            None
            if labels is None
            else np.asarray(labels).reshape(-1)[self.indices].astype(np.int64)
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        r, c = divmod(int(self.indices[k]), self.w)
        p = self.p
        a = torch.from_numpy(self.a[r : r + p, c : c + p].transpose(2, 0, 1).copy())
        b = torch.from_numpy(self.bimg[r : r + p, c : c + p].transpose(2, 0, 1).copy())
        target = -1 if self.labels is None else int(self.labels[k])
        return (
            a,
            b,
            target,
            torch.tensor([r, c], dtype=torch.long),
            int(self.indices[k]),
        )
