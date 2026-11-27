"""Train and evaluate the RCDIFO system with fixed clean training."""

from __future__ import annotations
from copy import deepcopy
import io, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from cmss.data import load_dataset
from cmss.metrics import binary_metrics
from .common import (
    ResultSource,
    append_rows,
    atomic_torch_save,
    environment,
    restore_rng,
    rng_state,
    set_seed,
    sha_file,
    stable_hash,
    write_json,
)
from .data import PatchPair, TrainMinMax, array_hash, make_case, split_data

CLEAN = {"name": "clean_no_harm", "type": "identity"}
MAIN_TRANSFORMS = {"shift_1", "shift_3", "row_shift_4"}


def classification_loss(logits, labels, name, settings):
    if name != "River":
        return F.cross_entropy(logits, labels)
    # Algebraic form of the supplied FocalLoss; avoids BCE underflow at p=0/1.
    ce = F.cross_entropy(logits, labels, reduction="none")
    pt = torch.exp(-ce)
    alpha = float(settings["river_focal_alpha"])
    gamma = float(settings["river_focal_gamma"])
    wt = torch.where(labels == 1, alpha, 1 - alpha)
    return (wt * (1 - pt).pow(gamma) * ce).mean()


def model_for(bands, device):
    from ._vendor.main_model import RSModel

    return RSModel(
        c_in_channels=bands,
        c_num_classes=2,
        r_in_channels=bands,
        r_num_features=6,
        r_patch_size=25,
        device=torch.device(device),
    ).to(device)


def check_model_result(logits, aligned, matrix):
    if not all(bool(torch.isfinite(t).all()) for t in [logits, aligned, matrix]):
        raise FloatingPointError("RCDIFO returned nonfinite values.")


def _worker_seed(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)


class Runner:
    def __init__(
        self,
        config,
        project_root,
        device=None,
        datasets=None,
        seeds=None,
        data_root=None,
        output=None,
        main_output=None,
    ):
        self.root = Path(project_root).resolve()
        self.cfg = yaml.safe_load(Path(config).read_text())
        self.base = yaml.safe_load((self.root / self.cfg["base_config"]).read_text())
        self.source = ResultSource(main_output) if main_output else None
        src_config = None
        default_device = self.cfg["project"]["device"]
        if self.source:
            raw = self.source.read("config.yaml")
            if raw is None:
                raise FileNotFoundError("Benchmark output must contain config.yaml.")
            src_config = yaml.safe_load(raw)
            if data_root is None:
                data_root = src_config["project"]["data_root"]
            self.base["dataset_options"] = deepcopy(
                src_config.get("dataset_options", {})
            )
            main_environment = self.source.json("environment.json") or {}
            default_device = main_environment.get(
                "device", src_config["project"].get("device", default_device)
            )
        self.device = str(device or default_device)
        if not self.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("RCDIFO requires a CUDA-enabled PyTorch environment.")
        torch.cuda.set_device(torch.device(self.device))
        torch.set_num_threads(int(self.cfg["project"]["torch_threads"]))
        self.datasets = list(datasets or self.cfg["project"]["datasets"])
        self.seeds = list(seeds if seeds is not None else self.cfg["project"]["seeds"])
        if not set(self.datasets) <= set(self.cfg["project"]["datasets"]) or not set(
            self.seeds
        ) <= set(self.cfg["project"]["seeds"]):
            raise ValueError(
                "Dataset and seed selections must be configuration subsets."
            )
        self.data_root = (
            Path(data_root or self.base["project"]["data_root"]).expanduser().resolve()
        )
        self.base["project"]["data_root"] = str(self.data_root)
        self.out = Path(output or self.cfg["project"]["output_dir"]).resolve()
        if self.source and (
            self.out == self.source.path
            or self.out in self.source.path.parents
            or self.source.path in self.out.parents
        ):
            raise ValueError(
                "RCDIFO and benchmark output directories must be separate."
            )
        self.mode = "clean_frozen"
        if int(self.cfg["model"]["patch_size"]) != 25:
            raise ValueError("RCDIFO requires patch_size=25.")
        if float(self.cfg["training"]["threshold"]) != 0.5:
            raise ValueError("The fixed classification threshold is 0.5.")
        vendor = self.root / "experiments/rcdifo/_vendor"
        vm = vendor / "manifest.json"
        if not vm.is_file():
            raise FileNotFoundError("Run prepare_rcdifo.py before training RCDIFO.")
        vmeta = json.loads(vm.read_text())
        expected = json.loads((self.root / "configs/rcdifo_sources.json").read_text())[
            "files"
        ]
        if set(vmeta["files"]) != set(expected):
            raise ValueError("Incomplete prepared RCDIFO source.")
        for rel, meta in vmeta["files"].items():
            path = vendor / rel.removeprefix("model/")
            if (
                meta["source_sha256"] != expected[rel]
                or sha_file(path) != meta["adapted_sha256"]
            ):
                raise ValueError("Prepared RCDIFO source changed: " + str(path))
        env = environment(self.device)
        method_cfg = deepcopy(self.cfg)
        method_cfg["project"].pop("output_dir", None)
        method_cfg["project"]["device"] = self.device
        files = list((self.root / "experiments/rcdifo").glob("*.py"))
        files += [
            self.root / p
            for p in [
                "cmss/data.py",
                "cmss/splits.py",
                "cmss/metrics.py",
                "cmss/transforms.py",
                "cmss/utils.py",
                "experiments/benchmark/transforms.py",
            ]
        ]
        code = {str(p.relative_to(self.root)): sha_file(p) for p in files}
        main_protocol = None
        if self.source:
            for key in ["split", "transforms"]:
                if src_config[key] != self.base[key]:
                    raise ValueError("Benchmark " + key + " mismatch.")
            if src_config["correction"]["seed"] != self.base["correction"]["seed"]:
                raise ValueError("Perturbation seed mismatch.")
            if (
                src_config["benchmark"]["geometry_transforms"]
                != self.base["benchmark"]["geometry_transforms"]
            ):
                raise ValueError("Extended perturbation settings differ.")
            main_protocol = self.source.json("manifest.json")["protocol_hash"]
        self.ph = stable_hash(
            dict(
                adapter=method_cfg,
                base_split=self.base["split"],
                base_transforms=self.base["transforms"],
                extended=self.base["benchmark"]["geometry_transforms"],
                code=code,
                vendor=vmeta,
                environment=env,
                main_protocol=main_protocol,
                geometric_seed=self.base["correction"]["seed"],
                dataset_options=self.base.get("dataset_options", {}),
            )
        )
        mp = self.out / "manifest.json"
        if mp.exists():
            prior = json.loads(mp.read_text())
            if prior["protocol_hash"] != self.ph:
                raise ValueError(
                    "Settings or code changed; use a new output directory."
                )
            if prior["main_output"] != (str(self.source.path) if self.source else None):
                raise ValueError("Keep --main-output unchanged when resuming.")
        elif self.out.exists() and any(self.out.iterdir()):
            raise ValueError("Output directory is nonempty and has no run manifest.")
        self.out.mkdir(parents=True, exist_ok=True)
        write_json(
            mp,
            dict(
                protocol_hash=self.ph,
                main_output=str(self.source.path) if self.source else None,
                training_protocol=self.mode,
                environment=env,
                vendor=vmeta,
            ),
        )
        (self.out / "config.yaml").write_text(yaml.safe_dump(self.cfg, sort_keys=False))
        (self.out / "benchmark_config.yaml").write_text(
            yaml.safe_dump(self.base, sort_keys=False)
        )
        write_json(self.out / "environment.json", env)
        self.lock = self.out / ".run.lock"
        self._source_df = None

    def __enter__(self):
        try:
            with self.lock.open("x") as handle:
                handle.write(f"pid={os.getpid()}\n")
        except FileExistsError:
            raise RuntimeError("Output is locked by another run: " + str(self.lock))
        return self

    def __exit__(self, *args):
        self.lock.unlink(missing_ok=True)

    def dataset(self, name):
        ds = load_dataset(
            name, self.data_root, self.base.get("dataset_options", {}).get(name, {})
        )
        expected_path = self.root / "configs/datasets.json"
        expected = (
            self.source.json(f"inventory/{name}.json")
            if self.source
            else (
                json.loads(expected_path.read_text()).get(name)
                if expected_path.exists()
                else None
            )
        )
        if expected:
            for k in ["t1_hash", "t2_hash", "gt_hash", "shape_hwb"]:
                if ds.metadata[k] != expected[k]:
                    raise ValueError(f"{name}: {k} differs from the benchmark data")
            if "full_hashes" in expected:
                for k in ["t1", "t2", "gt"]:
                    if array_hash(getattr(ds, k)) != expected["full_hashes"][k]:
                        raise ValueError("Full data hash mismatch")
        else:
            raise ValueError("Dataset inventory is unavailable")
        ds.metadata["full_hashes"] = {
            k: array_hash(getattr(ds, k)) for k in ["t1", "t2", "gt"]
        }
        p = self.out / "inventory" / f"{name}.json"
        if (
            p.exists()
            and json.loads(p.read_text())["full_hashes"] != ds.metadata["full_hashes"]
        ):
            raise ValueError("Dataset changed in this output")
        write_json(p, ds.metadata)
        return ds

    def split(self, ds, seed):
        sp, ti, vi = split_data(ds, seed, self.base)
        d = self.out / "splits" / ds.name
        d.mkdir(parents=True, exist_ok=True)
        arrays = dict(
            train=sp.train_mask,
            val=sp.val_mask,
            test=sp.test_mask,
            train_indices=ti,
            val_indices=vi,
        )
        p = d / f"seed{seed}.npz"
        if p.exists():
            with np.load(p, allow_pickle=False) as z:
                if any(not np.array_equal(z[k], v) for k, v in arrays.items()):
                    raise ValueError("Stored RCDIFO split mismatch")
        else:
            np.savez_compressed(p, **arrays)
        if self.source:
            raw = self.source.read(f"splits/{ds.name}/seed{seed}.npz")
            if raw is None:
                raise FileNotFoundError(
                    f"Benchmark split is missing: {ds.name}/seed{seed}"
                )
            if raw:
                with np.load(io.BytesIO(raw), allow_pickle=False) as z:
                    if any(
                        not np.array_equal(z[k].astype(bool), arrays[k])
                        for k in ["train", "val", "test"]
                    ):
                        raise ValueError("CMSS split mismatch")
        write_json(
            d / f"seed{seed}.json",
            dict(
                **sp.metadata,
                train_context_radius=12,
                strictly_disjoint_patch_contexts=False,
                test_batch_order="row-major",
            ),
        )
        return sp, ti, vi

    def specs(self, name):
        specs = [("clean", CLEAN)]
        main = [s for s in self.base["transforms"] if s["name"] in MAIN_TRANSFORMS]
        if {s["name"] for s in main} != MAIN_TRANSFORMS:
            raise ValueError("The three benchmark perturbations are required.")
        specs.extend(("baselines", s) for s in main)
        if name == "Farmland":
            rotation = [
                s
                for s in self.base["benchmark"]["geometry_transforms"]
                if s["name"] == "rotation_1deg"
            ]
            if len(rotation) != 1:
                raise ValueError("Farmland rotation_1deg is required.")
            specs.append(("geometry", rotation[0]))
        return specs

    def checkpoint_dir(self, ds, seed):
        return self.out / "checkpoints" / ds.name / f"seed{seed}"

    def loader(self, pair, training=False, epoch=0):
        bs = int(self.cfg["model"]["batch_size" if training else "eval_batch_size"])
        # The released KDE is batch-dependent. Batch size/order are protocol.
        generator = torch.Generator().manual_seed(100003 + epoch)
        return DataLoader(
            pair,
            batch_size=bs,
            shuffle=training,
            num_workers=int(self.cfg["training"]["num_workers"]),
            pin_memory=self.device.startswith("cuda"),
            worker_init_fn=_worker_seed,
            generator=generator,
            drop_last=False,
        )

    def predict(self, model, t1, t2, idx, shape):
        pair = PatchPair(t1, t2, idx, labels=None)
        prob = np.full(shape, np.nan, np.float32)
        matrices_changed = 0
        total = 0
        model.eval()
        start = time.perf_counter()
        with torch.no_grad():
            for a, b, _, pos, ids in self.loader(pair):
                a = a.to(self.device)
                b = b.to(self.device)
                pos = pos.to(self.device).float()
                # Inference uses zero feedback.
                y = torch.zeros(len(a), device=self.device)
                logits, _, aligned, mat = model(a, b, pos, y)
                check_model_result(logits, aligned, mat)
                p = torch.softmax(logits, 1)[:, 1].cpu().numpy()
                prob.ravel()[ids.numpy()] = p
                eye = torch.eye(2, 3, device=self.device).expand_as(mat)
                matrices_changed += int(
                    (mat - eye).abs().flatten(1).amax(1).gt(1e-6).sum()
                )
                total += len(a)
        if not np.isfinite(prob.ravel()[idx]).all():
            raise RuntimeError("Incomplete prediction coverage")
        return prob, dict(
            seconds=time.perf_counter() - start,
            affine_nonidentity_n=matrices_changed,
            n=total,
        )

    def _train_one(self, ds, seed):
        spec = CLEAN
        sp, ti, vi = self.split(ds, seed)
        folder = self.checkpoint_dir(ds, seed)
        folder.mkdir(parents=True, exist_ok=True)
        done = folder / "checkpoint.json"
        if done.exists():
            info = json.loads(done.read_text())
            if (
                sha_file(folder / "best.pt") != info["best_sha256"]
                or sha_file(folder / "normalizer.npz") != info["normalizer_sha256"]
            ):
                raise ValueError("Completed model artifact modified")
            print("[MODEL CACHED]", ds.name, seed, spec["name"])
            return True
        moving, valid = make_case(ds, spec, self.base, self.device)
        ti = ti[valid.ravel()[ti]]
        vi = vi[valid.ravel()[vi]]
        if (
            len(np.unique(ds.gt.ravel()[ti])) < 2
            or len(np.unique(ds.gt.ravel()[vi])) < 2
        ):
            raise ValueError("Training/validation lost a class")
        normal = TrainMinMax.fit(ds.t1, ds.t2, ti)
        normal.save(folder / "normalizer.npz")
        a = normal.apply(ds.t1)
        b = normal.apply(moving)
        pair = PatchPair(a, b, ti, labels=ds.gt)
        # Validation batching depends on coordinates, not the order of class labels.
        vi = np.sort(vi)
        vm = np.zeros(ds.gt.shape, bool)
        vm.ravel()[vi] = True
        set_seed(seed)
        model = model_for(ds.t1.shape[-1], self.device)
        cfg = self.cfg["training"]
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg["scheduler_T_max"]),
            eta_min=float(cfg["scheduler_eta_min"]),
        )
        feedback = np.zeros(ds.gt.size, np.uint8)
        start_epoch = 0
        best = -1.0
        best_epoch = -1
        last = folder / "last.pt"
        iterations = int(cfg["outer_iterations"])
        ep_per = int(cfg["epochs_per_iteration"])
        total = iterations * ep_per
        if last.exists():
            # Load checkpoints generated locally by this adapter.
            state = torch.load(last, map_location="cpu", weights_only=False)
            if state["protocol_hash"] != self.ph:
                raise ValueError("Checkpoint protocol mismatch")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            for st in optimizer.state.values():
                for k, v in st.items():
                    if torch.is_tensor(v):
                        st[k] = v.to(self.device)
            feedback = state["feedback"]
            start_epoch = state["next_epoch"]
            best = state["best"]
            best_epoch = state["best_epoch"]
            restore_rng(state["rng"])
            print(
                "[RESUME]", ds.name, seed, spec["name"], "next epoch", start_epoch + 1
            )
        for ep in range(start_epoch, total):
            model.train()
            t0 = time.perf_counter()
            n = 0
            lc = 0.0
            lr = 0.0
            reg_active = 0
            has_reg_grad = 0
            next_feedback = feedback.copy()
            bar = tqdm(
                self.loader(pair, training=True, epoch=seed * 100000 + ep),
                desc=f'{ds.name}/s{seed}/{spec["name"]} {ep+1}/{total}',
                leave=False,
            )
            for x1, x2, target, pos, ids in bar:
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                target = target.to(self.device).long()
                pos = pos.to(self.device).float()
                y = torch.from_numpy(feedback[ids.numpy()].astype(np.float32)).to(
                    self.device
                )
                logits, _, aligned, mat = model(x1, x2, pos, y)
                check_model_result(logits, aligned, mat)
                c_loss = classification_loss(logits, target, ds.name, cfg)
                unchanged = target == 0
                r_loss = (
                    F.l1_loss(aligned[unchanged], x1[unchanged])
                    if unchanged.any()
                    else logits.sum() * 0
                )
                loss = c_loss + float(cfg["lambda_registration"]) * r_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Nonfinite joint loss; no valid result generated"
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grads = [
                    p.grad
                    for p in model.r_model.unet.parameters()
                    if p.grad is not None
                ]
                if grads:
                    if not all(bool(torch.isfinite(g).all()) for g in grads):
                        raise FloatingPointError("Nonfinite registration gradient")
                    has_reg_grad += int(any(bool(g.abs().max() > 0) for g in grads))
                optimizer.step()
                nn = len(target)
                n += nn
                lc += float(c_loss.detach()) * nn
                lr += float(r_loss.detach()) * nn
                eye = torch.eye(2, 3, device=self.device).expand_as(mat)
                reg_active += int((mat - eye).abs().flatten(1).amax(1).gt(1e-6).sum())
                next_feedback[ids.numpy()] = (
                    logits.detach().argmax(1).cpu().numpy().astype(np.uint8)
                )
                bar.set_postfix(loss=f"{float(loss.detach()):.4f}")
            scheduler.step()
            # Update feedback after each complete training round.
            if (ep + 1) % ep_per == 0:
                feedback = next_feedback
            prob, _ = self.predict(model, a, b, vi, ds.gt.shape)
            m = binary_metrics(prob > 0.5, ds.gt, mask=vm, prob=prob)
            if m["F1"] > best:
                best = m["F1"]
                best_epoch = ep + 1
                atomic_torch_save(
                    folder / "best.pt",
                    dict(
                        model=model.state_dict(),
                        protocol_hash=self.ph,
                        epoch=ep + 1,
                        best_val_F1=best,
                        bands=ds.t1.shape[-1],
                        threshold=0.5,
                        training_condition=spec,
                    ),
                )
            row = dict(
                epoch=ep + 1,
                outer_iteration=ep // ep_per + 1,
                loss_cls=lc / max(n, 1),
                loss_reg=lr / max(n, 1),
                val_F1=m["F1"],
                best_val_F1=best,
                best_epoch=best_epoch,
                registration_active_n=reg_active,
                training_n=n,
                registration_gradient_batches=has_reg_grad,
                lr=optimizer.param_groups[0]["lr"],
                seconds=time.perf_counter() - t0,
            )
            append_rows(folder / "training_log.csv", [row], ["epoch"])
            atomic_torch_save(
                last,
                dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    protocol_hash=self.ph,
                    feedback=feedback,
                    next_epoch=ep + 1,
                    best=best,
                    best_epoch=best_epoch,
                    rng=rng_state(),
                ),
            )
            print(
                f'[EPOCH] {ds.name}/s{seed}/{spec["name"]} {ep+1}/{total} valF1={m["F1"]:.4f} active={reg_active}/{n} time={row["seconds"]:.1f}s',
                flush=True,
            )
        write_json(
            done,
            dict(
                protocol_hash=self.ph,
                best_epoch=best_epoch,
                best_val_F1=best,
                epochs=total,
                best_sha256=sha_file(folder / "best.pt"),
                normalizer_sha256=sha_file(folder / "normalizer.npz"),
                training_protocol=self.mode,
                training_condition=spec,
            ),
        )
        return True

    def train(self):
        for name in self.datasets:
            self.dataset(name)
        for name in self.datasets:
            ds = self.dataset(name)
            for seed in self.seeds:
                self._train_one(ds, seed)

    def _source_rows(self, name, seed, suite, transform, N):
        if not self.source:
            return []
        if self._source_df is None:
            raw = self.source.read("metrics/conditions.csv")
            if raw is None:
                raise ValueError("Comparison source lacks metrics/conditions.csv")
            self._source_df = pd.read_csv(io.BytesIO(raw))
        df = self._source_df
        target_suite = "clean" if suite == "clean" else suite
        q = df[
            (df.dataset == name)
            & (df.seed == seed)
            & (df.suite == target_suite)
            & (df["transform"] == transform)
            & df.baseline.isin(self.cfg["evaluation"]["reference_systems"])
            & df.correction.isin(["cmss", "misaligned"])
        ]
        expected_groups = {
            (baseline, correction)
            for baseline in self.cfg["evaluation"]["reference_systems"]
            for correction in ["cmss", "misaligned"]
        }
        if (
            len(q) != len(expected_groups)
            or set(zip(q.baseline, q.correction)) != expected_groups
        ):
            raise ValueError(
                "Matching benchmark reference rows are incomplete or duplicated"
            )
        if not bool((q.N == N).all()):
            raise ValueError(
                f"Held-out support count differs from completed CMSS for {name}/{transform}/{seed}"
            )
        return [
            dict(
                system=f"{r.correction}+{r.baseline}",
                dataset=name,
                seed=seed,
                suite=suite,
                transform=transform,
                F1=r.F1,
                IoU=r.IoU,
                BoundaryF1=r.BoundaryF1,
                N=int(r.N),
                source="benchmark",
                protocol=self.mode,
                training_condition="clean",
                threshold=r.threshold,
            )
            for r in q.itertuples()
        ]

    def evaluate(self):
        rows = []
        diag = []
        for name in self.datasets:
            ds = self.dataset(name)
            for seed in self.seeds:
                sp, _, _ = self.split(ds, seed)
                model = None
                loaded = None
                for suite, spec in self.specs(name):
                    folder = self.checkpoint_dir(ds, seed)
                    if not (folder / "checkpoint.json").exists():
                        raise FileNotFoundError(f"Training incomplete: {folder}.")
                    complete = json.loads((folder / "checkpoint.json").read_text())
                    if (
                        sha_file(folder / "best.pt") != complete["best_sha256"]
                        or sha_file(folder / "normalizer.npz")
                        != complete["normalizer_sha256"]
                    ):
                        raise ValueError("Model modified")
                    ck = folder / "best.pt"
                    ch = sha_file(ck)
                    moving, valid = make_case(ds, spec, self.base, self.device)
                    mask = sp.test_mask & valid & (ds.gt >= 0)
                    idx = np.flatnonzero(
                        mask.ravel()
                    )  # Label-independent test batch membership.
                    pred_dir = self.out / "predictions" / name / f"seed{seed}"
                    pred_dir.mkdir(parents=True, exist_ok=True)
                    pred_path = pred_dir / (spec["name"] + ".npz")
                    if pred_path.exists():
                        with np.load(pred_path, allow_pickle=False) as z:
                            if str(z["checkpoint_hash"]) != ch or not np.array_equal(
                                z["mask"], mask
                            ):
                                raise ValueError("Prediction cache mismatch")
                            prob = z["prob"]
                            meta = json.loads(str(z["stats_json"]))
                    else:
                        if loaded != str(ck):
                            if model is not None:
                                del model
                            model = model_for(ds.t1.shape[-1], self.device)
                            state = torch.load(
                                ck, map_location="cpu", weights_only=True
                            )
                            if state["protocol_hash"] != self.ph:
                                raise ValueError("Best checkpoint protocol mismatch")
                            model.load_state_dict(state["model"])
                            loaded = str(ck)
                        normal = TrainMinMax.load(folder / "normalizer.npz")
                        prob, meta = self.predict(
                            model,
                            normal.apply(ds.t1),
                            normal.apply(moving),
                            idx,
                            ds.gt.shape,
                        )
                        temp = pred_path.with_suffix(".tmp.npz")
                        np.savez_compressed(
                            temp,
                            prob=prob,
                            mask=mask,
                            checkpoint_hash=ch,
                            stats_json=json.dumps(meta),
                        )
                        os.replace(temp, pred_path)
                    m = binary_metrics(prob > 0.5, ds.gt, mask=mask, prob=prob)
                    row = dict(
                        system="RCDIFO-adapted",
                        dataset=name,
                        seed=seed,
                        suite=suite,
                        transform=spec["name"],
                        protocol=self.mode,
                        training_condition="clean",
                        threshold=0.5,
                        source="RCDIFO-adapted",
                        **m,
                    )
                    rows.append(row)
                    rows += self._source_rows(name, seed, suite, spec["name"], m["N"])
                    diag.append(
                        dict(
                            dataset=name,
                            seed=seed,
                            suite=suite,
                            transform=spec["name"],
                            checkpoint_sha256=ch,
                            mask_sha256=array_hash(mask),
                            **meta,
                        )
                    )
                    append_rows(
                        self.out / "metrics/conditions.csv",
                        rows,
                        ["system", "dataset", "seed", "suite", "transform"],
                    )
                    append_rows(
                        self.out / "metrics/registration_activity.csv",
                        diag,
                        ["dataset", "seed", "suite", "transform"],
                    )
                    rows = []
                    diag = []
                    print(
                        "[EVAL]",
                        name,
                        seed,
                        spec["name"],
                        f'F1={m["F1"]:.4f}',
                        flush=True,
                    )

    def summarize(self):
        summarize(self.out)


def summarize(output):
    output = Path(output)
    path = output / "metrics/conditions.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    metrics = ["F1", "IoU", "BoundaryF1"]
    groups = ["system", "dataset", "suite", "transform", "protocol"]
    rows = []
    for key, group in df.groupby(groups, dropna=False):
        row = dict(zip(groups, key))
        row["n_seeds"] = group.seed.nunique()
        for metric in metrics:
            row[metric + "_mean"] = group[metric].mean()
            row[metric + "_std"] = group[metric].std(ddof=1)
        rows.append(row)
    append_rows(output / "summary/per_condition_mean_std.csv", rows, groups)
    main = df[
        (df.suite == "baselines")
        & df.system.isin(["RCDIFO-adapted", "cmss+GETNET", "cmss+GLAFormer"])
    ]
    if not main.empty:
        per_seed = main.groupby(["system", "dataset", "seed"], as_index=False)[
            metrics
        ].mean()
        rows = []
        for (system, dataset), group in per_seed.groupby(["system", "dataset"]):
            row = dict(system=system, dataset=dataset, n_seeds=group.seed.nunique())
            for metric in metrics:
                row[metric + "_mean"] = group[metric].mean()
                row[metric + "_std"] = group[metric].std(ddof=1)
            rows.append(row)
        append_rows(output / "summary/main_mean_std.csv", rows, ["system", "dataset"])
    activity = []
    for log in (output / "checkpoints").rglob("training_log.csv"):
        group = pd.read_csv(log)
        activity.append(
            dict(
                model=str(log.parent.relative_to(output)),
                epochs=len(group),
                registration_gradient_batches=int(
                    group.registration_gradient_batches.sum()
                ),
                active_training_patches=int(group.registration_active_n.sum()),
                total_training_patches=int(group.training_n.sum()),
            )
        )
    if activity:
        append_rows(output / "summary/registration_activity.csv", activity, ["model"])
