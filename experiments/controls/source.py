"""Read benchmark inputs, splits and frozen detectors for matched controls."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zlib

import numpy as np
import pandas as pd
import torch
import yaml

from .util import (
    require,
    read_json,
    sha_array,
    sha_file,
    object_hash,
    safe_child,
    close,
)


@dataclass
class Case:
    dataset: object
    transform: str
    moving: np.ndarray
    truth: np.ndarray
    valid: np.ndarray
    signature: str
    spec: dict
    tx: dict


class Source:
    def __init__(self, root, data_root=None, device=None, threads=4):
        self.root = Path(root).expanduser().resolve()
        for relative in (
            "manifest.json",
            "config.yaml",
            "metrics/conditions.csv",
            "metrics/geometry.csv",
        ):
            require(
                (self.root / relative).is_file(),
                f"Missing benchmark result {self.root / relative}",
            )
        self.cfg = yaml.safe_load(
            (self.root / "config.yaml").read_text(encoding="utf-8")
        )
        self.manifest = read_json(self.root / "manifest.json")
        self.protocol = self.manifest["protocol_hash"]
        self.df = pd.read_csv(self.root / "metrics/conditions.csv")
        self.geometry = pd.read_csv(self.root / "metrics/geometry.csv")
        self.suite = "baselines"
        self._validate_tables()
        self.correction = self.cfg["correction"]
        required = {
            "seed",
            "pca_components",
            "global_radius",
            "row_radius",
            "smooth_sigma",
            "outer_iters",
            "weighting",
            "safe_selector",
        }
        require(
            required <= set(self.correction),
            "Incomplete benchmark correction configuration",
        )
        selector = self.correction["safe_selector"]
        require(
            selector.get("enabled", True) and selector.get("use_change_weights", True),
            "The controls require the complete weighted selector",
        )
        require(
            close(
                selector.get("identity_min_rel_improvement", np.nan),
                0.05,
                atol=1e-12,
                rtol=0,
            ),
            "The controls use the benchmark 5% acceptance margin",
        )
        for key, expected in (
            ("max_iter", 30),
            ("tolerance", 0.001),
            ("regularization", 0.001),
        ):
            if key in self.correction["weighting"]:
                require(
                    close(self.correction["weighting"][key], expected),
                    f"Unsupported internal IR-MAD setting: {key}",
                )
        environment = self.manifest["environment"]
        original_device = str(environment["device"])
        self.device = str(device or original_device)
        require(
            self.device.split(":")[0] == original_device.split(":")[0],
            "Use the same CPU/CUDA resampling backend as the benchmark",
        )
        target = torch.device(self.device)
        require(target.type in {"cpu", "cuda"}, "Use a CPU or CUDA device")
        if target.type == "cuda":
            require(torch.cuda.is_available(), "CUDA is unavailable")
            if target.index is None:
                target = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.set_device(target)
            self.device = str(target)
        self.threads = int(threads)
        require(self.threads > 0, "Thread count must be positive")
        torch.set_num_threads(self.threads)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.data_root = (
            Path(data_root or self.cfg["project"]["data_root"]).expanduser().resolve()
        )
        require(self.data_root.is_dir(), f"Data directory not found: {self.data_root}")
        self._datasets = {}
        self._hashes = {}
        self._splits = {}
        self._checkpoints = {}
        self._models = {}
        from experiments.benchmark.transforms import apply_extended

        self.extended = apply_extended

    def case(self, name, transform):
        from cmss.transforms import oracle_realign

        ds = self.dataset(name)
        tx = read_json(self.root / "transforms" / name / f"{transform}.json")
        spec = tx["spec"]
        seed = int(
            zlib.crc32(f"{name}:{transform}:{self.correction['seed']}".encode())
            & 0x7FFFFFFF
        )
        signature = object_hash(
            dict(protocol=self.protocol, data=self._hashes[name], spec=spec, seed=seed)
        )
        require(
            signature == tx["signature"],
            f"Transform signature mismatch: {name}/{transform}",
        )
        tr = self.extended(ds.t2, spec, self.device, seed)
        require(
            tr.metadata["flow_hash"] == tx["metadata"]["flow_hash"],
            f"Regenerated displacement differs: {name}/{transform}",
        )
        _, oracle_valid = oracle_realign(tr.moving, tr.forward_flow, device=self.device)
        return Case(
            ds,
            transform,
            tr.moving,
            tr.forward_flow,
            tr.valid_reference & oracle_valid,
            signature,
            spec,
            tx,
        )

    def _validate_tables(self):
        keys = ["suite", "dataset", "transform", "baseline", "seed", "correction"]
        cols = set(
            keys + ["threshold", "F1", "IoU", "BoundaryF1", "N", "TP", "TN", "FP", "FN"]
        )
        require(cols <= set(self.df), f"conditions.csv lacks {cols-set(self.df)}")
        require(
            not self.df.duplicated(keys).any(),
            "Duplicate condition keys; refusing to average different records",
        )
        require(
            not self.geometry.duplicated(
                ["suite", "dataset", "transform", "correction"]
            ).any(),
            "Duplicate geometry keys",
        )
        require(
            self.suite in set(self.df.suite.astype(str)),
            f"Suite {self.suite} missing; choose explicitly from {sorted(self.df.suite.unique())}",
        )
        for col in ["seed", "F1", "IoU", "threshold", "TP", "TN", "FP", "FN", "N"]:
            self.df[col] = pd.to_numeric(self.df[col], errors="raise")
            require(
                np.isfinite(self.df[col]).all(), f"Nonfinite {col} in source conditions"
            )
        tp, fp, fn = self.df.TP, self.df.FP, self.df.FN
        require(
            np.allclose(
                self.df.F1, 2 * tp / np.maximum(2 * tp + fp + fn, 1), atol=1e-9, rtol=0
            ),
            "Source F1/confusion-count mismatch",
        )
        require(
            np.allclose(
                self.df.IoU, tp / np.maximum(tp + fp + fn, 1), atol=1e-9, rtol=0
            ),
            "Source IoU/confusion-count mismatch",
        )
        require(
            (self.df.N == self.df.TP + self.df.TN + self.df.FP + self.df.FN).all(),
            "Source N/confusion-count mismatch",
        )
        require(
            np.equal(self.df.seed, np.floor(self.df.seed)).all(),
            "Source seed indices must be integers",
        )
        self.df["seed"] = self.df.seed.astype(int)

    def dataset(self, name):
        from cmss.data import load_dataset

        if name not in self._datasets:
            options = self.cfg.get("dataset_options", {}).get(name, {})
            ds = load_dataset(name, self.data_root, options)
            hashes = {k: sha_array(getattr(ds, k)) for k in ("t1", "t2", "gt")}
            inv = read_json(self.root / "inventory" / f"{name}.json")
            require(
                hashes == inv.get("full_hashes"),
                f"Data hash mismatch for {name}; not the HSI/labels used by this result directory",
            )
            self._datasets[name] = ds
            self._hashes[name] = hashes
        return self._datasets[name]

    def release_dataset(self, name):
        self._datasets.pop(name, None)
        for k in list(self._splits):
            if k[0] == name:
                self._splits.pop(k)
        self._models.clear()

    def split(self, name, seed):
        from cmss.splits import make_spatial_block_split

        key = (name, seed)
        if key not in self._splits:
            ds = self.dataset(name)
            path = self.root / "splits" / name / f"seed{seed}.npz"
            require(
                path.is_file(),
                f"Missing saved spatial split {path}; do not generate a new training split",
            )
            sp = make_spatial_block_split(
                ds.gt,
                seed,
                **{
                    k: self.cfg["split"][k]
                    for k in ("block_size", "train_fraction", "val_fraction", "guard")
                },
            )
            with np.load(path, allow_pickle=False) as d:
                for n, m in (
                    ("train", sp.train_mask),
                    ("val", sp.val_mask),
                    ("test", sp.test_mask),
                ):
                    require(
                        n in d.files and np.array_equal(d[n].astype(bool), m),
                        f"Stored/reconstructed {n} split mismatch: {path}",
                    )
            self._splits[key] = sp
        return self._splits[key]

    def row(self, name, transform, detector, seed, correction, suite=None):
        d = self.df
        q = d[
            (d.suite == (suite or self.suite))
            & (d.dataset == name)
            & (d["transform"] == transform)
            & (d.baseline == detector)
            & (d.seed == seed)
            & (d.correction == correction)
        ]
        require(
            len(q) == 1,
            f"Missing/duplicate row: {suite or self.suite}/{name}/{transform}/{detector}/{seed}/{correction}",
        )
        return q.iloc[0]

    def support(self, case, detector, seed):
        from cmss.splits import balanced_indices

        ds = case.dataset
        mask = self.split(ds.name, seed).test_mask & case.valid & (ds.gt >= 0)
        idx = balanced_indices(mask, ds.gt, None, seed)
        for m in ("clean_reference", "misaligned", "cmss"):
            r = self.row(ds.name, case.transform, detector, seed, m)
            require(
                len(idx) == int(r.N),
                f"Support size mismatch for {ds.name}/{case.transform}/{detector}/{seed}/{m}",
            )
        return mask, idx

    def checkpoint(self, name, detector, seed):
        key = (name, detector, seed)
        if key not in self._checkpoints:
            self.dataset(name)
            path = self.root / "checkpoints" / name / f"seed{seed}" / detector
            cp = read_json(path / "checkpoint.json")
            require(
                cp["data_hashes"] == self._hashes[name]
                and int(cp["seed"]) == seed
                and cp["detector"] == detector,
                "Checkpoint provenance mismatch",
            )
            require(
                isinstance(cp.get("files"), dict)
                and {"best.pt", "normalizer.npz"} <= set(cp["files"]),
                f"Incomplete checkpoint manifest {path}",
            )
            for rel, digest in cp["files"].items():
                p = safe_child(path, rel)
                require(
                    p.is_file(),
                    f"Missing model artifact {p}. Run benchmark training first.",
                )
                require(sha_file(p) == digest, f"Model artifact hash mismatch: {p}")
            self._checkpoints[key] = (path, cp, object_hash(cp["files"]))
        return self._checkpoints[key]

    def diagnostic(self, name, transform, method="cmss", suite=None):
        return read_json(
            self.root
            / "diagnostics"
            / (suite or self.suite)
            / name
            / transform
            / f"{method}.json"
        )

    def geometry_row(self, name, transform, method="cmss", suite=None):
        g = self.geometry
        q = g[
            (g.suite == (suite or self.suite))
            & (g.dataset == name)
            & (g["transform"] == transform)
            & (g.correction == method)
        ]
        require(
            len(q) == 1,
            f"Missing/duplicate geometry row {name}/{transform}/{method}/{suite or self.suite}",
        )
        return q.iloc[0]

    def default_check(self, case, trace, suite=None):
        from cmss.corrections import flow_errors

        ds = case.dataset
        diag = self.diagnostic(ds.name, case.transform, "cmss", suite)
        m = trace.selected.metadata
        out = dict(dataset=ds.name, transform=case.transform, suite=suite or self.suite)
        require(
            m.get("selector_choice") == diag.get("selector_choice"),
            f"Default selector choice changed: {ds.name}/{case.transform}",
        )
        for k in ("selector_identity_loss", "selector_cmss_loss", "selector_pca3_loss"):
            require(
                k in diag and close(m[k], diag[k]),
                f"Default replay mismatch {ds.name}/{case.transform}/{k}: new={m.get(k)}, benchmark={diag.get(k)}",
            )
            out[k] = float(m[k])
        actual = flow_errors(trace.selected.flow, case.truth, case.valid)[
            "flow_epe_mean"
        ]
        old = float(
            self.geometry_row(ds.name, case.transform, "cmss", suite)["flow_epe_mean"]
        )
        require(
            close(actual, old),
            f"Default EPE changed: {ds.name}/{case.transform}: {actual} vs {old}",
        )
        out.update(
            EPE_replayed=actual, EPE_benchmark=old, selector_choice=m["selector_choice"]
        )
        cand = safe_child(self.root / "candidates", diag["_signature"])
        if (cand / "metadata.json").is_file():
            require(
                read_json(cand / "metadata.json").get("_signature")
                == diag["_signature"],
                "Candidate metadata signature mismatch",
            )
        for filename, array in (
            ("flow.npy", trace.selected.flow),
            ("corrected.npy", trace.selected.corrected),
        ):
            p = cand / filename
            out[filename + "_present"] = p.is_file()
            if p.is_file():
                a = np.load(p, allow_pickle=False, mmap_mode="r")
                require(a.shape == array.shape, f"Benchmark {filename} shape mismatch")
                # Chunked comparison limits temporary memory for HSI cubes.
                for y in range(0, len(a), 32):
                    require(
                        np.allclose(
                            a[y : y + 32],
                            array[y : y + 32],
                            atol=1e-5,
                            rtol=1e-5,
                            equal_nan=False,
                        ),
                        f"Default {filename} differs at row {y}: {ds.name}/{case.transform}",
                    )
        out["passed"] = True
        return out

    def raw_baseline_check(self, case, trace, method):
        from cmss.corrections import flow_errors

        r = self.geometry_row(case.dataset.name, case.transform, method)
        actual = flow_errors(trace.raw.flow, case.truth, case.valid)["flow_epe_mean"]
        require(
            close(actual, float(r["flow_epe_mean"])),
            f"{method} baseline EPE does not reproduce ({actual} vs {r['flow_epe_mean']})",
        )
        return dict(
            dataset=case.dataset.name,
            transform=case.transform,
            method=method,
            EPE_replayed=actual,
            EPE_benchmark=float(r["flow_epe_mean"]),
            passed=True,
        )

    def model(self, name, detector, seed):
        """Load tensor/primitive checkpoints after verifying file hashes."""
        import torch

        key = (name, detector, seed)
        if key in self._models:
            return self._models[key]
        ckdir, cp, _ = self.checkpoint(name, detector, seed)
        ck = torch.load(ckdir / "best.pt", map_location="cpu", weights_only=True)
        require(
            "threshold" in ck,
            "Checkpoint is missing the original validation-selected threshold; no threshold fitting is allowed",
        )
        device = torch.device(self.device)
        ds = self.dataset(name)
        if detector == "GETNET":
            from cmss.baselines import getnet_adapter as api

            model = api.build_model(
                "compact", int(ck["bands"]), int(ck.get("pool_size", 8))
            )
            with np.load(ckdir / "normalizer.npz", allow_pickle=False) as d:
                norm = api.Norm(
                    str(d["mode"].item()), d["offset"].copy(), d["scale"].copy()
                )
            a = norm.apply(
                api.spectral_bin_average(
                    ds.t1, int(ck.get("target_bands", ck["bands"]))
                )
            )
        else:
            from cmss.baselines import glaformer as api

            model = api.GLAFormer(
                int(ck["bands"]),
                int(ck["patch_size"]),
                int(ck["embed_dim"]),
                int(ck["depth"]),
                int(ck["heads"]),
                float(ck.get("drop_path", 0)),
            )
            with np.load(ckdir / "normalizer.npz", allow_pickle=False) as d:
                norm = api.Norm(d["mean"].copy(), d["std"].copy())
            a = norm.apply(ds.t1)
        model.load_state_dict(ck["model_state"], strict=True)
        model.to(device)
        model.eval()
        model.requires_grad_(False)
        out = dict(
            api=api,
            model=model,
            normalizer=norm,
            reference=a,
            checkpoint=ck,
            device=device,
        )
        self._models[key] = out
        return out

    def predict(self, case, cube, detector, seed, idx):
        m = self.model(case.dataset.name, detector, seed)
        api = m["api"]
        ck = m["checkpoint"]
        cfg = self.cfg["baselines"][detector]
        common = dict(
            progress_desc=f"Control inference {case.dataset.name}/{detector}/seed{seed}",
            show_progress=False,
        )
        batch = int(cfg.get("eval_batch_size", 128))
        workers = int(cfg.get("num_workers", 0))
        if detector == "GETNET":
            b = m["normalizer"].apply(
                api.spectral_bin_average(cube, int(ck.get("target_bands", ck["bands"])))
            )
            prob, oi = api._predict(
                m["model"],
                m["reference"],
                b,
                idx,
                batch,
                m["device"],
                str(ck.get("plane_normalize", "maxabs")),
                workers,
                **common,
            )
        else:
            b = m["normalizer"].apply(cube)
            prob, oi = api._predict(
                m["model"],
                m["reference"],
                b,
                case.dataset.gt,
                idx,
                int(ck["patch_size"]),
                batch,
                m["device"],
                workers,
                **common,
            )
        oi = np.asarray(oi, np.int64)
        prob = np.asarray(prob, np.float32)
        require(
            len(np.unique(oi)) == len(oi) and set(oi.tolist()) == set(idx.tolist()),
            "Predictor returned different pixel indices",
        )
        if not np.array_equal(oi, idx):
            p = {int(v): i for i, v in enumerate(oi)}
            prob = prob[[p[int(v)] for v in idx]]
        require(np.isfinite(prob).all(), "Nonfinite detector predictions")
        threshold = float(ck["threshold"])
        expected = float(
            self.row(
                case.dataset.name, case.transform, detector, seed, "misaligned"
            ).threshold
        )
        require(
            close(threshold, expected, atol=1e-10, rtol=0),
            "Checkpoint threshold does not match recorded threshold",
        )
        return prob, threshold

    def check_prediction(
        self, case, prob, threshold, idx, mask, detector, seed, method=None
    ):
        from cmss.metrics import binary_metrics

        require(
            prob.shape == idx.shape and np.isfinite(prob).all(),
            "Invalid probability array",
        )
        pm = np.full(case.dataset.gt.shape, np.nan, np.float32)
        pm.ravel()[idx] = prob
        m = binary_metrics(pm >= threshold, case.dataset.gt, mask=mask, prob=pm)
        if method is not None:
            r = self.row(case.dataset.name, case.transform, detector, seed, method)
            require(
                close(threshold, r.threshold, atol=1e-10, rtol=0),
                "Benchmark prediction threshold mismatch",
            )
            for k in ("F1", "IoU", "BoundaryF1", "N", "TP", "TN", "FP", "FN"):
                require(
                    close(m[k], r[k], atol=1e-8, rtol=0),
                    f"Benchmark prediction metric differs (cause not established): {case.dataset.name}/{case.transform}/{detector}/{seed}/{method}/{k}: {m[k]} vs {r[k]}",
                )
        return m

    def initial_preflight(self, datasets, transforms, seeds, detectors, detection=True):
        out = []
        for name in datasets:
            ds = self.dataset(name)
            for t in transforms:
                require(
                    (self.root / "transforms" / name / f"{t}.json").is_file(),
                    f"Missing saved transform {name}/{t}",
                )
                self.diagnostic(name, t)
                self.geometry_row(name, t)
                for d in detectors:
                    for seed in seeds:
                        for method in ("clean_reference", "misaligned", "cmss"):
                            self.row(name, t, d, seed, method)
            for seed in seeds:
                self.split(name, seed)
                if detection:
                    for d in detectors:
                        ck, cp, _ = self.checkpoint(name, d, seed)
                        out.append(
                            dict(
                                dataset=name,
                                detector=d,
                                seed=seed,
                                directory=str(ck),
                                hashes_verified=True,
                            )
                        )
            self.release_dataset(name)
        return out
