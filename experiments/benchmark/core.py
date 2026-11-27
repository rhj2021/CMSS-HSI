"""CMSS candidates and content-addressed correction caches."""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json
import os
import numpy as np
from cmss.corrections import (
    CorrectionResult,
    cmss_correct,
    global_pca_translation,
    select_safe_cmss,
    tvl1_dense_correction,
)
from cmss.utils import stable_hash_dict
from .registration import classical, learned_roma


def raw_cmss(t1, moving, cfg, device, k=None, weighted=True, progress=False):
    w = cfg["weighting"]
    return cmss_correct(
        t1,
        moving,
        n_components=int(k or cfg["pca_components"]),
        global_radius=int(cfg["global_radius"]),
        row_radius=int(cfg["row_radius"]),
        smooth_sigma=float(cfg["smooth_sigma"]),
        outer_iters=int(cfg["outer_iters"]),
        use_weights=weighted,
        irmad_pca_components=int(w["pca_components"]),
        irmad_sample_size=int(w["sample_size"]),
        irmad_epsilon=float(w["epsilon"]),
        seed=int(cfg["seed"]),
        device=device,
        show_progress=progress,
    )


def pca3(t1, moving, cfg, device):
    return global_pca_translation(
        t1,
        moving,
        n_components=3,
        global_radius=int(cfg["global_radius"]),
        seed=int(cfg["seed"]),
        device=device,
    )


def full_cmss(t1, moving, cfg, device, progress=False):
    """Complete, uncached preprocessor for valid end-to-end timing."""
    raw = raw_cmss(t1, moving, cfg, device, progress=progress)
    pca = pca3(t1, moving, cfg, device)
    return select_safe_cmss(
        t1,
        moving,
        raw,
        pca_result=pca,
        seed=int(cfg["seed"]),
        n_components=int(cfg["pca_components"]),
        cfg=cfg["safe_selector"],
    )


class CandidateStore:
    """Cache candidates by image, settings, method, and device.

    Full corrected cubes are stored to preserve values on resumed runs.
    """

    def __init__(
        self,
        root,
        t1,
        moving,
        cfg,
        external,
        device,
        case_signature,
        progress=False,
        use_disk=True,
    ):
        self.root = Path(root)
        self.t1 = t1
        self.moving = moving
        self.cfg = cfg
        self.external = external
        self.device = device
        self.case_signature = case_signature
        self.progress = progress
        self.use_disk = use_disk

    def get(self, method: str, selector_cfg: dict | None = None) -> CorrectionResult:
        external = deepcopy(self.external)
        if method == "roma_affine":
            import importlib.metadata

            external["installed_romatch_version"] = importlib.metadata.version(
                "romatch"
            )
        signature = stable_hash_dict(
            dict(
                case=self.case_signature,
                correction=self.cfg,
                external=external,
                method=method,
                selector=selector_cfg,
                device=self.device,
            )
        )
        path = self.root / signature
        if self.use_disk and (path / "metadata.json").exists():
            meta = json.loads((path / "metadata.json").read_text())
            if meta.get("_signature") != signature:
                raise RuntimeError("Corrupt candidate cache signature")
            weights = (
                np.load(path / "weights.npy", allow_pickle=False)
                if (path / "weights.npy").exists()
                else None
            )
            return CorrectionResult(
                np.load(path / "corrected.npy", allow_pickle=False),
                np.load(path / "flow.npy", allow_pickle=False),
                meta,
                weights,
            )
        c = self.cfg
        if method == "pca3_translation":
            result = pca3(self.t1, self.moving, c, self.device)
        elif method == "scanline_unweighted":
            result = raw_cmss(
                self.t1,
                self.moving,
                c,
                self.device,
                weighted=False,
                progress=self.progress,
            )
        elif method.startswith("cmss_raw_k"):
            k = int(method.removeprefix("cmss_raw_k"))
            result = raw_cmss(
                self.t1, self.moving, c, self.device, k=k, progress=self.progress
            )
            result.metadata.update(method=method, ablation_K=k, selector_enabled=False)
        elif method == "cmss" or method.startswith("selector_"):
            raw = self.get(f"cmss_raw_k{int(c['pca_components'])}")
            pca = self.get("pca3_translation")
            sc = deepcopy(
                selector_cfg if selector_cfg is not None else c["safe_selector"]
            )
            result = select_safe_cmss(
                self.t1,
                self.moving,
                raw,
                pca_result=pca,
                seed=int(c["seed"]),
                n_components=int(c["pca_components"]),
                cfg=sc,
            )
            result.metadata["requested_selector_config"] = sc
        elif method == "tvl1_dense":
            result = tvl1_dense_correction(
                self.t1,
                self.moving,
                device=self.device,
                seed=int(c["seed"]),
                **c["tvl1"],
            )
        elif method in {"phase_correlation", "ecc_affine"}:
            result = classical(
                self.t1, self.moving, method, self.device, int(c["seed"]), self.external
            )
        elif method == "roma_affine":
            result = learned_roma(
                self.t1, self.moving, self.device, int(c["seed"]), self.external
            )
        else:
            raise ValueError(f"Unknown correction method {method}")
        result.metadata = dict(result.metadata)
        result.metadata["_signature"] = signature
        if self.use_disk:
            path.mkdir(parents=True, exist_ok=True)
            # Metadata is the commit marker, written last. Interrupted array files
            # are recomputed rather than trusted on the next invocation.
            np.save(path / "corrected.npy", result.corrected.astype(np.float32))
            np.save(path / "flow.npy", result.flow.astype(np.float32))
            if result.weights is not None:
                np.save(path / "weights.npy", result.weights.astype(np.float32))
            tmp = path / "metadata.json.tmp"
            tmp.write_text(json.dumps(result.metadata, indent=2), encoding="utf-8")
            os.replace(tmp, path / "metadata.json")
        return result
