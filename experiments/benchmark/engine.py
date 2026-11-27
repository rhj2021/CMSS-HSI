"""Benchmark runner with recorded data, checkpoints, and experiment settings."""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import zlib
import numpy as np
import pandas as pd
import torch
import yaml
from cmss.baselines.getnet_adapter import predict_getnet, train_getnet_clean
from cmss.baselines.glaformer import predict_glaformer, train_glaformer
from cmss.corrections import flow_errors
from cmss.data import load_dataset
from cmss.metrics import binary_metrics
from cmss.splits import balanced_indices, make_spatial_block_split
from cmss.transforms import oracle_realign
from cmss.utils import stable_hash_dict
from .core import CandidateStore
from .transforms import apply_extended
from .registration import roma_model


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_array(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
        ),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def write_csv_rows(path, rows, keys):
    """Idempotent single-process merge; never duplicate already completed seeds."""
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    new = new.drop_duplicates(keys, keep="last")
    tmp = path.with_suffix(".csv.tmp")
    new.to_csv(tmp, index=False)
    os.replace(tmp, path)


def training_settings(cfg, detector):
    b = deepcopy(cfg["baselines"][detector])
    for k in ["enabled", "progress", "progress_desc", "eval_batch_size", "num_workers"]:
        b.pop(k, None)
    return dict(split=cfg["split"], baseline=b)


def validate_dataset(ds, expected_inventory):
    """Compare dataset shape, fingerprints and label counts with the configuration."""
    expected = expected_inventory.get(ds.name)
    if not expected:
        raise ValueError(f"Missing dataset specification for {ds.name}.")
    for key in ("shape_hwb", "t1_hash", "t2_hash", "gt_hash"):
        if ds.metadata[key] != expected[key]:
            raise ValueError(f"{ds.name}: {key} differs from the configured dataset.")
    for key in ("unchanged_count", "changed_count", "ignored_count"):
        if ds.metadata["gt"][key] != expected["gt"][key]:
            raise ValueError(
                f"{ds.name}: label count {key} differs from the configured dataset."
            )


class BenchmarkRunner:

    def __init__(
        self,
        config,
        project_root,
        device=None,
        datasets=None,
        seeds=None,
        detectors=None,
        data_root=None,
        output=None,
        include_deep=True,
        geometry_only=False,
        progress=None,
    ):
        self.project_root = Path(project_root).resolve()
        self.cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
        c = self.cfg
        self.benchmark = c["benchmark"]
        self.device = str(device or c["project"].get("device", "cpu"))
        if self.device.startswith("cuda") and (not torch.cuda.is_available()):
            raise RuntimeError(
                "CUDA was requested but is unavailable. Select --device cpu explicitly for CPU execution."
            )
        if self.device.startswith("cuda"):
            torch.cuda.set_device(torch.device(self.device))

        def resolve(value):
            path = Path(value).expanduser()
            return (path if path.is_absolute() else self.project_root / path).resolve()

        self.root = resolve(output or c["project"]["output_dir"])
        self.data_root = resolve(data_root or c["project"]["data_root"])
        c["project"]["output_dir"] = str(self.root)
        c["project"]["data_root"] = str(self.data_root)
        c["project"]["device"] = self.device
        if self.root == self.project_root or self.root in self.project_root.parents:
            raise ValueError("Output cannot be the project root or an ancestor.")
        for protected in [self.data_root] + [
            self.project_root / name
            for name in ("cmss", "experiments", "third_party", "configs")
        ]:
            if (
                self.root == protected
                or self.root in protected.parents
                or protected in self.root.parents
            ):
                raise ValueError(f"Output overlaps data or source code: {protected}")
        self.datasets = list(datasets or c["project"]["datasets"])
        self.seeds = [
            int(s) for s in (seeds if seeds is not None else c["project"]["seeds"])
        ]
        self.detectors = list(detectors or self.benchmark["detectors"])
        if not set(self.detectors) <= {"GETNET", "GLAFormer"}:
            raise ValueError("Supported detectors: GETNET and GLAFormer.")
        if not set(self.datasets) <= set(c["project"]["datasets"]) or not set(
            self.seeds
        ) <= set(c["project"]["seeds"]):
            raise ValueError(
                "Dataset and seed filters must be subsets of the configuration."
            )
        self.include_deep = include_deep
        self.geometry_only = geometry_only
        if int(self.benchmark.get("schema_version", 0)) != 1:
            raise ValueError("Unknown benchmark configuration schema.")
        if (
            float(c["correction"]["safe_selector"]["identity_min_rel_improvement"])
            != 0.05
        ):
            raise ValueError(
                "The default benchmark uses an acceptance margin of 0.05; use the selector stage for its sensitivity study."
            )
        self.progress = bool(
            c["project"].get("progress", True) if progress is None else progress
        )
        protocol = deepcopy(c)
        for key in ("output_dir", "data_root", "progress"):
            protocol["project"].pop(key, None)
        protocol["project"]["device"] = self.device
        packages = {}
        for name in [
            "numpy",
            "scipy",
            "scikit-learn",
            "pandas",
            "torch",
            "opencv-python-headless",
            "opencv-python",
            "scikit-image",
            "romatch",
        ]:
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        environment = dict(
            python=platform.python_version(),
            platform=platform.platform(),
            packages=packages,
            torch_threads=torch.get_num_threads(),
            device=self.device,
            cuda_name=(
                torch.cuda.get_device_name(torch.device(self.device))
                if self.device.startswith("cuda")
                else None
            ),
        )
        source_hashes = {
            str(path.relative_to(self.project_root)): sha_file(path)
            for folder in [
                self.project_root / "cmss",
                self.project_root / "third_party/getnet_baseline",
                Path(__file__).resolve().parent,
            ]
            for path in sorted(folder.rglob("*.py"))
        }
        stable_env = deepcopy(environment)
        stable_env["packages"].pop("romatch", None)
        self.protocol_hash = stable_hash_dict(
            dict(config=protocol, code=source_hashes, environment=stable_env)
        )
        manifest = self.root / "manifest.json"
        if manifest.exists():
            old = json.loads(manifest.read_text())
            if old["protocol_hash"] != self.protocol_hash:
                raise ValueError(
                    "Output belongs to different settings, code or environment. Use a new --output directory."
                )
        elif self.root.exists() and any(self.root.iterdir()):
            raise ValueError(
                "Output is nonempty and has no benchmark manifest. Use a new --output directory."
            )
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(
            manifest,
            dict(
                protocol_hash=self.protocol_hash,
                method_acceptance_margin=c["correction"]["safe_selector"][
                    "identity_min_rel_improvement"
                ],
                environment=environment,
                code_sha256=source_hashes,
            ),
        )
        (self.root / "config.yaml").write_text(
            yaml.safe_dump(c, sort_keys=False), encoding="utf-8"
        )
        write_json(self.root / "environment.json", environment)
        self._checkpoint_hashes = {}
        invpath = resolve(
            self.benchmark.get("expected_inventory", "configs/datasets.json")
        )
        self.expected_inventory = (
            json.loads(invpath.read_text()) if invpath.exists() else {}
        )
        self.lock = self.root / ".benchmark.lock"

    def __enter__(self):
        try:
            with self.lock.open("x") as f:
                f.write(f"pid={os.getpid()}\n")
        except FileExistsError:
            raise RuntimeError(
                "Another process may be writing this benchmark output. Stop it before restarting this stage."
            )
        return self

    def __exit__(self, *exc):
        self.lock.unlink(missing_ok=True)

    def dataset(self, name):
        ds = load_dataset(
            name, self.data_root, self.cfg.get("dataset_options", {}).get(name, {})
        )
        if self.benchmark.get("strict_inventory", True):
            validate_dataset(ds, self.expected_inventory)
        full_hashes = {k: sha_array(getattr(ds, k)) for k in ["t1", "t2", "gt"]}
        path = self.root / "inventory" / f"{name}.json"
        if (
            path.exists()
            and json.loads(path.read_text()).get("full_hashes") != full_hashes
        ):
            raise ValueError(
                "Dataset contents changed inside this output. Choose a new output directory."
            )
        ds.metadata["full_hashes"] = full_hashes
        write_json(path, ds.metadata)
        return ds

    def inspect(self):
        print(
            "Acceptance margin:",
            self.cfg["correction"]["safe_selector"]["identity_min_rel_improvement"],
        )
        print("Independent output:", self.root)
        for n in self.datasets:
            ds = self.dataset(n)
            print("[DATA OK]", n, ds.t1.shape)
            if not self.geometry_only:
                for s in self.seeds:
                    for det in self.detectors:
                        base = self.root / "checkpoints" / n / f"seed{s}" / det
                        state = (
                            "available"
                            if all(
                                (
                                    (base / f).exists()
                                    for f in ["best.pt", "normalizer.npz"]
                                )
                            )
                            else "prepare required"
                        )
                        print(f"  seed{s}/{det}: {state}")

    def split(self, ds, seed):
        path = self.root / "splits" / ds.name / f"seed{seed}.npz"
        if not path.exists():
            c = self.cfg["split"]
            sp = make_spatial_block_split(
                ds.gt,
                seed=seed,
                block_size=int(c["block_size"]),
                train_fraction=float(c["train_fraction"]),
                val_fraction=float(c["val_fraction"]),
                guard=int(c["guard"]),
            )
            arr = dict(train=sp.train_mask, val=sp.val_mask, test=sp.test_mask)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **arr)
        with np.load(path, allow_pickle=False) as f:
            return tuple((f[k].astype(bool) for k in ["train", "val", "test"]))

    def prepare(self):
        for name in self.datasets:
            ds = self.dataset(name)
            del ds
        for name in self.datasets:
            ds = self.dataset(name)
            for seed in self.seeds:
                train, val, _ = self.split(ds, seed)
                tridx = balanced_indices(
                    train,
                    ds.gt,
                    max_per_class=int(self.cfg["split"]["max_train_per_class"]),
                    seed=seed,
                )
                vidx = balanced_indices(
                    val,
                    ds.gt,
                    max_per_class=int(self.cfg["split"]["max_val_per_class"]),
                    seed=seed + 1000,
                )
                for det in self.detectors:
                    out = self.root / "checkpoints" / name / f"seed{seed}" / det
                    if (out / "checkpoint.json").exists():
                        self.checkpoint(ds, seed, det)
                        print("[MODEL CACHED]", name, seed, det)
                        continue
                    if out.exists():
                        raise RuntimeError(
                            f"Uncommitted model directory {out}; use a fresh output or restore a complete model. It will not be silently reused."
                        )
                    tmp = out.with_name(det + "_incomplete")
                    if tmp.exists():
                        raise RuntimeError(
                            f"Interrupted preparation in {tmp}. Remove this incomplete directory to restart training."
                        )
                    tmp.mkdir(parents=True, exist_ok=False)
                    bc = deepcopy(self.cfg["baselines"][det])
                    bc["progress"] = self.progress
                    bc["progress_desc"] = f"{name}/seed{seed}/{det}"
                    print("[TRAIN ONCE]", name, seed, det, flush=True)
                    if det == "GETNET":
                        train_getnet_clean(
                            ds.t1, ds.t2, ds.gt, vidx, tmp, bc, self.device, seed
                        )
                    else:
                        train_glaformer(
                            ds.t1, ds.t2, ds.gt, tridx, vidx, tmp, bc, self.device, seed
                        )
                    origin = "clean_training"
                    files = {
                        f: sha_file(tmp / f) for f in ["best.pt", "normalizer.npz"]
                    }
                    write_json(
                        tmp / "checkpoint.json",
                        dict(
                            origin=origin,
                            seed=seed,
                            detector=det,
                            training=training_settings(self.cfg, det),
                            data_hashes=ds.metadata["full_hashes"],
                            files=files,
                        ),
                    )
                    os.replace(tmp, out)
                    print("[MODEL READY]", name, seed, det, flush=True)

    def checkpoint(self, ds, seed, det):
        out = self.root / "checkpoints" / ds.name / f"seed{seed}" / det
        marker = out / "checkpoint.json"
        if not marker.exists():
            raise FileNotFoundError(f"Run --stage prepare first. Missing {marker}.")
        meta = json.loads(marker.read_text())
        if meta["data_hashes"] != ds.metadata["full_hashes"] or meta[
            "training"
        ] != training_settings(self.cfg, det):
            raise ValueError(
                "Checkpoint provenance does not match current data/training configuration"
            )
        key = str(out)
        if key not in self._checkpoint_hashes:
            current = {f: sha_file(out / f) for f in ["best.pt", "normalizer.npz"]}
            if current != meta["files"]:
                raise ValueError(f"Checkpoint was modified: {out}")
            self._checkpoint_hashes[key] = stable_hash_dict(current)
        return (out, self._checkpoint_hashes[key])

    def _case(self, ds, spec):
        seed = int(
            zlib.crc32(
                f"{ds.name}:{spec['name']}:{self.cfg['correction']['seed']}".encode()
            )
            & 2147483647
        )
        if spec["type"] == "identity":
            moving = ds.t2
            flow = np.zeros(ds.t1.shape[:2] + (2,), np.float32)
            valid = np.ones(ds.gt.shape, bool)
            oracle = ds.t2
            meta = {}
        else:
            tr = apply_extended(ds.t2, spec, self.device, seed)
            moving, flow = (tr.moving, tr.forward_flow)
            oracle, ov = oracle_realign(moving, flow, device=self.device)
            valid = tr.valid_reference & ov
            meta = tr.metadata
        signature = stable_hash_dict(
            dict(
                protocol=self.protocol_hash,
                data=ds.metadata["full_hashes"],
                spec=spec,
                seed=seed,
            )
        )
        write_json(
            self.root / "transforms" / ds.name / (spec["name"] + ".json"),
            dict(spec=spec, signature=signature, metadata=meta),
        )
        return (moving, flow, oracle, valid, signature)

    def methods(self, suite):
        c = self.cfg["correction"]
        rev = self.benchmark
        if suite == "clean":
            return [("cmss", None)]
        if suite in {"baselines", "geometry"}:
            ms = list(
                rev["baseline_methods" if suite == "baselines" else "geometry_methods"]
            )
            if self.include_deep and "roma_affine" not in ms:
                ms.append("roma_affine")
            return [(m, None) for m in ms]
        if suite == "k":
            if rev.get("k_selector_enabled", False):
                raise ValueError(
                    "This controlled K study requires selector disabled for ALL K values"
                )
            return [(f"cmss_raw_k{k}", None) for k in rev["k_values"]] + [
                ("cmss", None)
            ]
        if suite == "selector":
            out = [("cmss", None), (f"cmss_raw_k{c['pca_components']}", None)]
            for key, vals in [
                ("identity_min_rel_improvement", rev["acceptance_margins"]),
                ("pca_tie_rel", rev["pca_tie_margins"]),
            ]:
                for v in vals:
                    if float(v) == float(c["safe_selector"][key]):
                        continue
                    sc = deepcopy(c["safe_selector"])
                    sc[key] = float(v)
                    name = (
                        "selector_accept_"
                        if key.startswith("identity")
                        else "selector_tie_"
                    ) + str(v).replace(".", "p")
                    out.append((name, sc))
            return out
        raise ValueError(suite)

    def _prediction(self, ds, seed, det, cube, mask, key):
        ckpt, ckpt_hash = self.checkpoint(ds, seed, det)
        idx = balanced_indices(mask, ds.gt, max_per_class=None, seed=seed)
        if len(idx) == 0:
            raise ValueError("Empty held-out support")
        signature = stable_hash_dict(
            dict(
                case=key,
                checkpoint=ckpt_hash,
                indices=sha_array(idx),
                protocol=self.protocol_hash,
            )
        )
        path = self.root / "predictions" / signature[:2] / (signature + ".npz")
        if path.exists():
            with np.load(path, allow_pickle=False) as f:
                prob = f["prob"]
                thr = float(f["threshold"])
                saved_idx = f["indices"]
            if not np.array_equal(saved_idx, idx):
                raise ValueError("Prediction-cache index mismatch")
        else:
            bc = self.cfg["baselines"][det]
            bs = int(bc.get("eval_batch_size", 128))
            nw = int(bc.get("num_workers", 0))
            kwargs = dict(
                progress_desc=f"{ds.name}/seed{seed}/{det}", show_progress=self.progress
            )
            if det == "GETNET":
                prob, outidx, thr = predict_getnet(
                    ckpt / "best.pt",
                    ckpt / "normalizer.npz",
                    ds.t1,
                    cube,
                    idx,
                    bs,
                    self.device,
                    nw,
                    **kwargs,
                )
            else:
                prob, outidx, thr = predict_glaformer(
                    ckpt / "best.pt",
                    ckpt / "normalizer.npz",
                    ds.t1,
                    cube,
                    ds.gt,
                    idx,
                    bs,
                    self.device,
                    nw,
                    **kwargs,
                )
            if not np.array_equal(outidx, idx):
                lookup = {int(v): i for i, v in enumerate(outidx)}
                prob = np.asarray([prob[lookup[int(i)]] for i in idx], np.float32)
            if not np.isfinite(prob).all():
                raise RuntimeError("Nonfinite detector probabilities")
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, prob=prob, indices=idx, threshold=float(thr))
            os.replace(tmp, path)
        pm = np.full(ds.gt.shape, np.nan, np.float32)
        pm.ravel()[idx] = prob
        return (float(thr), binary_metrics(pm >= thr, ds.gt, mask=mask, prob=pm))

    def evaluate(self, suite):
        clean = dict(name="clean_no_harm", type="identity")
        specs = (
            [clean]
            if suite == "clean"
            else (
                self.benchmark["geometry_transforms"]
                if suite == "geometry"
                else self.cfg["transforms"]
            )
        )
        if suite in {"k", "selector", "baselines"}:
            specs = [clean] + list(specs)
        methods = self.methods(suite)
        if any((m == "roma_affine" for m, _ in methods)):
            roma_model(self.device)
        for name in self.datasets:
            ds = self.dataset(name)
            scene_methods = [
                (method, options)
                for method, options in methods
                if suite != "geometry" or method != "ecc_affine" or name == "Farmland"
            ]
            if not self.geometry_only:
                for s in self.seeds:
                    for det in self.detectors:
                        self.checkpoint(ds, s, det)
            for spec in specs:
                print(f"[{suite}] {name}/{spec['name']}", flush=True)
                moving, truth, oracle, valid, sig = self._case(ds, spec)
                external = deepcopy(self.benchmark["external"])
                try:
                    external["romatch_version"] = importlib.metadata.version("romatch")
                except importlib.metadata.PackageNotFoundError:
                    external["romatch_version"] = None
                store = CandidateStore(
                    self.root / "candidates",
                    ds.t1,
                    moving,
                    self.cfg["correction"],
                    external,
                    self.device,
                    sig,
                    self.progress,
                )
                items = [
                    ("clean_reference", None),
                    ("misaligned", None),
                    ("oracle", None),
                ] + scene_methods
                for method, scfg in items:
                    if method == "clean_reference":
                        cube = ds.t2
                        pred_key = sig + "_clean"
                    elif method == "misaligned":
                        cube = moving
                        pred_key = sig + "_mis"
                    elif method == "oracle":
                        cube = oracle
                        pred_key = sig + "_oracle"
                    else:
                        result = store.get(method, scfg)
                        cube = result.corrected
                        pred_key = result.metadata["_signature"]
                        geom = dict(
                            suite=suite,
                            dataset=name,
                            transform=spec["name"],
                            correction=method,
                            valid_fraction=float(valid.mean()),
                            **flow_errors(result.flow, truth, valid),
                            selector_choice=result.metadata.get(
                                "selector_choice", "not_applicable"
                            ),
                            registration_status=result.metadata.get(
                                "registration_status", "ok"
                            ),
                            failure_reason=result.metadata.get("failure_reason", ""),
                            unchanged_input_bitwise=bool(np.array_equal(cube, moving)),
                            maximum_input_abs_change=float(
                                np.max(np.abs(cube - moving))
                            ),
                        )
                        for k, v in result.metadata.items():
                            if k.startswith("selector_") and isinstance(
                                v, (str, int, float, bool)
                            ):
                                geom[k] = v
                        write_csv_rows(
                            self.root / "metrics" / "geometry.csv",
                            [geom],
                            ["suite", "dataset", "transform", "correction"],
                        )
                        write_json(
                            self.root
                            / "diagnostics"
                            / suite
                            / name
                            / spec["name"]
                            / (method + ".json"),
                            result.metadata,
                        )
                    if not self.geometry_only:
                        rows = []
                        for seed in self.seeds:
                            _, _, test = self.split(ds, seed)
                            mask = test & valid & (ds.gt >= 0)
                            for det in self.detectors:
                                threshold, metrics = self._prediction(
                                    ds, seed, det, cube, mask, pred_key
                                )
                                rows.append(
                                    dict(
                                        suite=suite,
                                        dataset=name,
                                        seed=seed,
                                        baseline=det,
                                        transform=spec["name"],
                                        correction=method,
                                        threshold=threshold,
                                        evaluation_fraction=float(mask.mean()),
                                        **metrics,
                                    )
                                )
                        write_csv_rows(
                            self.root / "metrics" / "conditions.csv",
                            rows,
                            [
                                "suite",
                                "dataset",
                                "seed",
                                "baseline",
                                "transform",
                                "correction",
                            ],
                        )
                    if method not in ["clean_reference", "misaligned", "oracle"]:
                        del result
                write_json(
                    self.root / "completed" / suite / name / (spec["name"] + ".json"),
                    dict(
                        methods=[m for m, _ in scene_methods],
                        seeds=self.seeds,
                        detectors=self.detectors,
                        geometry_only=self.geometry_only,
                        case_signature=sig,
                    ),
                )
