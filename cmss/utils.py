"""Configuration, random seeds, content hashes, and output utilities."""

from __future__ import annotations
import hashlib, json, os, random
from pathlib import Path
from typing import Any
import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch and select deterministic cuDNN settings."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str) -> torch.device:
    return (
        torch.device("cpu")
        if device.startswith("cuda") and not torch.cuda.is_available()
        else torch.device(device)
    )


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def stable_hash_array(a: np.ndarray, max_bytes: int = 2_000_000) -> str:
    """Hash the array shape, dtype, and a deterministic byte subsample."""
    a = np.ascontiguousarray(a)
    raw = memoryview(a).cast("B")
    step = max(1, len(raw) // max_bytes) if len(raw) > max_bytes else 1
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(bytes(raw[::step]))
    return h.hexdigest()


def stable_hash_dict(obj: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=_json_default).encode()
    ).hexdigest()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
