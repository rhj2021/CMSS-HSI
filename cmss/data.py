"""Load benchmark MAT files as H x W x B cubes and mapped change labels."""

from __future__ import annotations
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any, Iterable
import h5py, numpy as np, scipy.io as sio
from .utils import stable_hash_array


@dataclass
class DatasetBundle:
    name: str
    t1: np.ndarray
    t2: np.ndarray
    gt: np.ndarray
    valid_mask: np.ndarray
    metadata: dict[str, Any]


DIRS = {
    "River": ["River", "river", "river_scene"],
    "Farmland": ["Farmland", "farmland", "farm", "farmland_scene"],
    "Hermiston": ["Hermiston", "hermiston", "hermiston_scene"],
}


def _find_dir(root: Path, names: Iterable[str]) -> Path:
    lower = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
    for n in names:
        p = root / n
        if p.is_dir():
            return p
        if n.lower() in lower:
            return lower[n.lower()]
    raise FileNotFoundError(f"Cannot find {list(names)} under {root}")


def _find_file(d: Path, names: Iterable[str]) -> Path:
    lower = {p.name.lower(): p for p in d.iterdir() if p.is_file()}
    for n in names:
        p = d / n
        if p.is_file():
            return p
        if n.lower() in lower:
            return lower[n.lower()]
    raise FileNotFoundError(f"Cannot find {list(names)} under {d}")


def _mat_candidates(path: Path) -> dict[str, np.ndarray]:
    try:
        m = sio.loadmat(path)
        return {
            str(k): np.asarray(v)
            for k, v in m.items()
            if not str(k).startswith("__") and np.asarray(v).dtype.kind in "biufc"
        }
    except (NotImplementedError, ValueError, OSError):
        out = {}
        with h5py.File(path, "r") as f:

            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    a = np.asarray(obj)
                    if a.dtype.kind in "biufc":
                        out[name] = a

            f.visititems(visit)
        return out


def load_mat_array(
    path: Path, keys: Iterable[str] | None = None, ndim: int | None = None
) -> tuple[np.ndarray, str]:
    c = _mat_candidates(path)
    if not c:
        raise ValueError(f"No numeric arrays in {path}")
    lower = {k.lower(): k for k in c}
    for key in keys or []:
        actual = key if key in c else lower.get(key.lower())
        if actual is not None:
            a = np.asarray(c[actual])
            if ndim is None or a.squeeze().ndim == ndim:
                return a, actual
    eligible = [
        (k, np.asarray(v))
        for k, v in c.items()
        if ndim is None or np.asarray(v).squeeze().ndim == ndim
    ]
    if not eligible:
        raise ValueError(
            f"No {ndim}D array in {path}; {[ (k,v.shape) for k,v in c.items() ]}"
        )
    k, a = max(eligible, key=lambda kv: kv[1].size)
    return a, k


def _orient(
    cube: np.ndarray, gt_shape: tuple[int, int]
) -> tuple[np.ndarray, tuple[int, int, int]]:
    cube = np.asarray(cube).squeeze()
    opts = []
    if cube.ndim != 3:
        raise ValueError(f"Cube must be 3D, got {cube.shape}")
    for perm in permutations(range(3)):
        shape = tuple(cube.shape[i] for i in perm)
        if shape[:2] == gt_shape:
            score = (10 if 4 <= shape[2] <= 512 else 0) + (
                2 if shape[2] not in gt_shape else 0
            )
            opts.append((score, perm))
    if not opts:
        raise ValueError(f"Cannot orient {cube.shape} to {gt_shape}")
    perm = max(opts, key=lambda x: x[0])[1]
    return np.transpose(cube, perm).astype(np.float32, copy=False), perm


def _sanitize(cube: np.ndarray) -> tuple[np.ndarray, int]:
    cube = np.asarray(cube, dtype=np.float32)
    bad = ~np.isfinite(cube)
    n = int(bad.sum())
    if n:
        flat = cube.reshape(-1, cube.shape[-1])
        med = np.nanmedian(np.where(np.isfinite(flat), flat, np.nan), axis=0)
        med = np.where(np.isfinite(med), med, 0).astype(np.float32)
        y, x, b = np.where(bad)
        cube = cube.copy()
        cube[y, x, b] = med[b]
    return cube, n


def _map_gt(
    raw: np.ndarray, mapping: str | dict[str, Any] = "auto"
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.asarray(raw).squeeze()
    finite = np.isfinite(raw)
    vals, counts = np.unique(raw[finite], return_counts=True)
    s = set(float(v) for v in vals)
    gt = np.full(raw.shape, -1, dtype=np.int8)
    if isinstance(mapping, dict):
        for v in mapping.get("unchanged", []):
            gt[raw == float(v)] = 0
        for v in mapping.get("changed", []):
            gt[raw == float(v)] = 1
        rule = "explicit"
    elif s == {0.0, 1.0}:
        gt[raw == 0] = 0
        gt[raw == 1] = 1
        rule = "0=unchanged,1=changed"
    elif s == {1.0, 2.0}:
        gt[raw == 1] = 0
        gt[raw == 2] = 1
        rule = "1=unchanged,2=changed"
    elif s == {0.0, 255.0}:
        gt[raw == 0] = 0
        gt[raw == 255] = 1
        rule = "0=unchanged,255=changed"
    elif s == {0.0, 1.0, 2.0}:
        gt[raw == 0] = -1
        gt[raw == 1] = 0
        gt[raw == 2] = 1
        rule = "0=ignore,1=unchanged,2=changed"
    elif len(vals) == 2:
        lo, hi = float(vals[0]), float(vals[1])
        gt[raw == lo] = 0
        gt[raw == hi] = 1
        rule = f"lower({lo})=unchanged,higher({hi})=changed"
    elif 0.0 in s:
        gt[raw == 0] = 0
        gt[(raw != 0) & finite] = 1
        rule = "0=unchanged,nonzero=changed"
    else:
        lo, hi = float(vals.min()), float(vals.max())
        gt[raw == lo] = 0
        gt[raw == hi] = 1
        rule = "extreme fallback"
    meta = {
        "raw_values": [float(v) for v in vals],
        "raw_counts": {str(float(v)): int(c) for v, c in zip(vals, counts)},
        "mapping_rule": rule,
        "unchanged_count": int((gt == 0).sum()),
        "changed_count": int((gt == 1).sum()),
        "ignored_count": int((gt < 0).sum()),
    }
    if not meta["unchanged_count"] or not meta["changed_count"]:
        raise ValueError(f"Empty GT class: {meta}")
    return gt, meta


def load_dataset(
    name: str, data_root: str | Path, dataset_config: dict[str, Any] | None = None
) -> DatasetBundle:
    dataset_config = dataset_config or {}
    n = name.strip().lower()
    canonical = {
        "river": "River",
        "farmland": "Farmland",
        "hermiston": "Hermiston",
    }.get(n)
    if canonical is None:
        raise ValueError(name)
    root = Path(data_root)
    d = _find_dir(root, DIRS[canonical])
    if canonical == "River":
        p1 = _find_file(d, ["river_before.mat", "before.mat"])
        p2 = _find_file(d, ["river_after.mat", "after.mat"])
        pg = _find_file(d, ["groundtruth.mat", "gt.mat", "label.mat"])
        g, gk = load_mat_array(pg, ["groundtruth", "gt", "GT", "label"], 2)
        a, k1 = load_mat_array(p1, ["river_before", "before", "T1", "im1"], 3)
        b, k2 = load_mat_array(p2, ["river_after", "after", "T2", "im2"], 3)
    elif canonical == "Farmland":
        p1 = _find_file(d, ["farm06.mat"])
        p2 = _find_file(d, ["farm07.mat"])
        pg = _find_file(d, ["label.mat", "groundtruth.mat", "gt.mat"])
        g, gk = load_mat_array(pg, ["label", "gt", "GT"], 2)
        a, k1 = load_mat_array(p1, ["imgh", "farm06", "T1", "im1"], 3)
        b, k2 = load_mat_array(p2, ["imghl", "farm07", "T2", "im2"], 3)
    else:
        p1 = p2 = pg = _find_file(d, ["Hermiston.mat", "hermiston.mat"])
        g, gk = load_mat_array(pg, ["hsi_gt", "gt", "GT"], 2)
        a, k1 = load_mat_array(p1, ["hsi_t1", "T1", "im1"], 3)
        b, k2 = load_mat_array(p2, ["hsi_t2", "T2", "im2"], 3)
    g = np.asarray(g).squeeze()
    oriented = None
    errors = []
    for gg, tr in [(g, False), (g.T, True)]:
        try:
            t1, perm1 = _orient(a, gg.shape)
            t2, perm2 = _orient(b, gg.shape)
            if t1.shape == t2.shape:
                oriented = (t1, t2, gg, perm1, perm2, tr)
                break
        except Exception as e:
            errors.append(str(e))
    if oriented is None:
        raise ValueError(errors)
    t1, t2, gg, perm1, perm2, gtr = oriented
    t1, bad1 = _sanitize(t1)
    t2, bad2 = _sanitize(t2)
    gt, gm = _map_gt(gg, dataset_config.get("gt_mapping", "auto"))
    meta = {
        "name": canonical,
        "directory": str(d),
        "t1_path": str(p1),
        "t2_path": str(p2),
        "gt_path": str(pg),
        "t1_key": k1,
        "t2_key": k2,
        "gt_key": gk,
        "raw_t1_shape": list(np.asarray(a).shape),
        "raw_t2_shape": list(np.asarray(b).shape),
        "raw_gt_shape": list(g.shape),
        "t1_permutation": list(perm1),
        "t2_permutation": list(perm2),
        "gt_transposed": bool(gtr),
        "shape_hwb": list(t1.shape),
        "nan_or_inf_replaced_t1": bad1,
        "nan_or_inf_replaced_t2": bad2,
        "gt": gm,
        "t1_hash": stable_hash_array(t1),
        "t2_hash": stable_hash_array(t2),
        "gt_hash": stable_hash_array(gt),
    }
    return DatasetBundle(canonical, t1, t2, gt, gt >= 0, meta)
