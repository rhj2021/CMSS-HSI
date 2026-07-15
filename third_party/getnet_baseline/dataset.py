from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class GETNETPixelDataset(Dataset):
    """Pixel-pair dataset that creates GETNET spectral similarity planes on demand."""

    def __init__(self, t1: np.ndarray, t2: np.ndarray, indices: np.ndarray, labels: np.ndarray, plane_normalize: str = "maxabs"):
        if t1.shape != t2.shape or t1.ndim != 3:
            raise ValueError("t1/t2 must be [H,W,B] with matching shapes")
        self.h, self.w, self.b = t1.shape
        self.t1 = t1.reshape(-1, self.b).astype(np.float32)
        self.t2 = t2.reshape(-1, self.b).astype(np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.plane_normalize = plane_normalize
        if len(self.indices) != len(self.labels):
            raise ValueError("indices and labels must have same length")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        before = self.t1[idx]
        after = self.t2[idx]
        plane = after[None, :] - before[:, None]  # [B, B], same element-to-wise matrix as GETNET.
        if self.plane_normalize == "maxabs":
            denom = np.max(np.abs(plane)) + 1e-6
            plane = plane / denom
        elif self.plane_normalize == "max":
            denom = np.max(plane) + 1e-6
            plane = plane / denom
        elif self.plane_normalize in {"none", None}:
            pass
        else:
            raise ValueError(f"Unknown plane_normalize: {self.plane_normalize}")
        return torch.from_numpy(plane[None, :, :].astype(np.float32)), torch.tensor(self.labels[i], dtype=torch.long)


class GETNETFullImageIterator:
    """Batch iterator for full-image prediction without materializing all [B,B] planes."""

    def __init__(self, t1: np.ndarray, t2: np.ndarray, batch_size: int = 256, plane_normalize: str = "maxabs"):
        self.h, self.w, self.b = t1.shape
        self.t1 = t1.reshape(-1, self.b).astype(np.float32)
        self.t2 = t2.reshape(-1, self.b).astype(np.float32)
        self.batch_size = batch_size
        self.plane_normalize = plane_normalize

    def __iter__(self):
        n = self.t1.shape[0]
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            before = self.t1[start:end]
            after = self.t2[start:end]
            # [N, B, B]
            planes = after[:, None, :] - before[:, :, None]
            if self.plane_normalize == "maxabs":
                denom = np.max(np.abs(planes), axis=(1, 2), keepdims=True) + 1e-6
                planes = planes / denom
            elif self.plane_normalize == "max":
                denom = np.max(planes, axis=(1, 2), keepdims=True) + 1e-6
                planes = planes / denom
            elif self.plane_normalize in {"none", None}:
                pass
            else:
                raise ValueError(f"Unknown plane_normalize: {self.plane_normalize}")
            yield start, end, torch.from_numpy(planes[:, None, :, :].astype(np.float32))
