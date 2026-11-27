"""Small integrity and output helpers for experiment runs."""

from __future__ import annotations


import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def sha_array(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def object_hash(x):
    # Use the same dictionary serialization as benchmark cache signatures.
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    return x


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def write_json(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(
        json.dumps(jsonable(x), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, p)


def write_csv(p, rows):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if len(df.columns) == 0:
        df = pd.DataFrame(columns=["status"])
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, p)


def write_npz(p, **arrays):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, p)


def beneath(p, root):
    try:
        Path(p).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def safe_child(root, rel):
    p = (Path(root) / str(rel)).resolve()
    require(beneath(p, root), f"Path escapes source directory: {rel}")
    return p


def close(a, b, atol=1e-5, rtol=1e-5):
    return np.isclose(float(a), float(b), atol=atol, rtol=rtol, equal_nan=True)


def log(msg):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)
