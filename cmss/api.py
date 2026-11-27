"""Public interface to the complete CMSS preprocessor."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

from cmss.corrections import CorrectionResult


def default_config() -> dict:
    """Return the correction settings used in the paper, including the selector."""
    path = Path(__file__).resolve().parents[1] / "configs/benchmark.yaml"
    return deepcopy(yaml.safe_load(path.read_text(encoding="utf-8"))["correction"])


def synchronize(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    device: str = "cpu",
    config: dict | None = None,
    progress: bool = False,
) -> CorrectionResult:
    """Align a moving H x W x B cube to a reference cube.

    Both inputs must be finite numeric arrays with identical shapes. Geometry is
    estimated from the two images; labels and trained detector weights are not
    inputs. The returned correction keeps all spectral bands.

    The flow contains sampling offsets in pixels, ordered (dy, dx): output[y, x]
    samples the moving image at (y + dy, x + dx). The full method compares the
    scanline candidate with global PCA translation and identity.

    ``config`` replaces the complete correction dictionary. Obtain a copy with
    ``default_config()`` before changing a setting for a separate experiment.
    """
    from experiments.benchmark.core import full_cmss

    cubes = []
    for name, value in [("reference", reference), ("moving", moving)]:
        arr = np.asarray(value)
        if arr.ndim != 3 or min(arr.shape[:2]) < 2 or arr.shape[2] < 1:
            raise ValueError(f"{name} must have shape H x W x B with H, W >= 2.")
        if not np.issubdtype(arr.dtype, np.number) or np.iscomplexobj(arr):
            raise ValueError(f"{name} must contain real numeric values.")
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN or infinite values.")
        cubes.append(arr)
    if cubes[0].shape != cubes[1].shape:
        raise ValueError("reference and moving must have identical H x W x B shapes.")
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("Use a CPU or CUDA device.")
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; select device='cpu' explicitly.")
        if target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(target)
    settings = default_config() if config is None else deepcopy(config)
    return full_cmss(*cubes, settings, str(target), progress=progress)
