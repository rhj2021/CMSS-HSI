"""File, environment and random-state helpers for RCDIFO."""

from __future__ import annotations
import hashlib, importlib.metadata, json, os, platform, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + ".tmp")
    q.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
            default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
        ),
        encoding="utf-8",
    )
    os.replace(q, p)


def atomic_torch_save(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(".tmp.pt")
    torch.save(obj, q)
    os.replace(q, p)


def append_rows(path, rows, keys):
    if not rows:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if p.exists():
        df = pd.concat([pd.read_csv(p), df], ignore_index=True)
    df = df.drop_duplicates(keys, keep="last")
    q = p.with_suffix(".tmp.csv")
    df.to_csv(q, index=False)
    os.replace(q, p)


def stable_hash(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()


def environment(device):
    packages = {}
    for p in [
        "torch",
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "mamba-ssm",
        "causal-conv1d",
        "einops",
        "transformers",
        "triton",
    ]:
        try:
            packages[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            pass
    return dict(
        python=platform.python_version(),
        platform=platform.platform(),
        packages=packages,
        device=str(device),
        cuda=torch.version.cuda,
        torch_threads=torch.get_num_threads(),
        gpu=(
            torch.cuda.get_device_name(device)
            if str(device).startswith("cuda")
            else None
        ),
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    return dict(
        py=random.getstate(),
        np=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def restore_rng(state):
    random.setstate(state["py"])
    np.random.set_state(state["np"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


class ResultSource:
    """Read the matching benchmark output directory."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        if not self.path.is_dir():
            raise FileNotFoundError(self.path)

    def read(self, name):
        path = self.path / name
        return path.read_bytes() if path.is_file() else None

    def json(self, name):
        data = self.read(name)
        return json.loads(data) if data is not None else None
