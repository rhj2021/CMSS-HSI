"""GETNET preprocessing, training, and fixed-checkpoint inference."""

from __future__ import annotations
import csv, json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np, torch
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from third_party.getnet_baseline.dataset import GETNETPixelDataset
from third_party.getnet_baseline.model import build_model
from third_party.getnet_baseline.pseudo import make_pseudo_training_set
from ..metrics import find_best_f1_threshold
from ..utils import parameter_count, resolve_device, set_seed


def spectral_bin_average(cube, target):
    cube = np.asarray(cube, np.float32)
    b = cube.shape[-1]
    if not target or target <= 0 or target >= b:
        return cube
    edges = np.linspace(0, b, target + 1)
    out = []
    for i in range(target):
        lo = int(np.floor(edges[i]))
        hi = max(lo + 1, int(np.floor(edges[i + 1])))
        out.append(cube[..., lo : min(hi, b)].mean(-1))
    return np.stack(out, -1).astype(np.float32)


@dataclass
class Norm:
    mode: str
    offset: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, a, b, mode="channel_minmax"):
        x = np.concatenate([a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])])
        if mode == "channel_minmax":
            lo = np.percentile(x, 1, axis=0).astype(np.float32)
            hi = np.percentile(x, 99, axis=0).astype(np.float32)
            return cls(mode, lo, np.maximum(hi - lo, 1e-6))
        mean = x.mean(0).astype(np.float32)
        return cls(mode, mean, np.maximum(x.std(0), 1e-6).astype(np.float32))

    def apply(self, x):
        z = (np.asarray(x, np.float32) - self.offset) / self.scale
        return (
            np.clip(z, 0, 1).astype(np.float32)
            if "minmax" in self.mode
            else np.clip(z, -8, 8).astype(np.float32)
        )

    def save(self, p):
        np.savez(p, mode=np.array(self.mode), offset=self.offset, scale=self.scale)

    @classmethod
    def load(cls, p):
        d = np.load(p, allow_pickle=True)
        return cls(str(d["mode"].item()), d["offset"], d["scale"])


class IndexDataset(Dataset):
    def __init__(self, a, b, indices, plane="maxabs"):
        self.base = GETNETPixelDataset(
            a, b, np.asarray(indices, np.int64), np.zeros(len(indices), np.int64), plane
        )
        self.idx = np.asarray(indices, np.int64)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        x, _ = self.base[i]
        return x, int(self.idx[i])


def _predict(
    model,
    a,
    b,
    indices,
    batch,
    device,
    plane,
    workers=0,
    progress_desc=None,
    show_progress=True,
):
    dl = DataLoader(
        IndexDataset(a, b, indices, plane),
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    ps = []
    ids = []
    model.eval()
    iterator = tqdm(
        dl,
        desc=progress_desc or "predict GETNET",
        total=len(dl),
        unit="batch",
        leave=False,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    with torch.no_grad():
        for x, idx in iterator:
            ps.append(torch.softmax(model(x.to(device)), 1)[:, 1].cpu().numpy())
            ids.append(idx.numpy())
    return np.concatenate(ps), np.concatenate(ids)


def train_getnet_clean(t1, t2, gt, val_indices, output_dir, config, device_name, seed):
    set_seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    show_progress = bool(config.get("progress", True))
    progress_desc = str(config.get("progress_desc", f"train GETNET seed{seed}"))

    target = int(config.get("target_bands", 64))
    a0 = spectral_bin_average(t1, target)
    b0 = spectral_bin_average(t2, target)
    norm = Norm.fit(a0, b0, str(config.get("normalize", "channel_minmax")))
    norm.save(out / "normalizer.npz")
    a, b = norm.apply(a0), norm.apply(b0)
    idx, lab, pseudo = make_pseudo_training_set(
        a,
        b,
        fraction=float(config.get("pseudo_fraction", 0.7)),
        max_per_class=int(config.get("max_pseudo_per_class", 6000)),
        min_per_class=int(config.get("min_pseudo_per_class", 32)),
        seed=seed,
        cva_mode=str(config.get("cva_mode", "l2")),
        high_confidence=True,
    )
    ds = GETNETPixelDataset(
        a, b, idx, lab, str(config.get("plane_normalize", "maxabs"))
    )
    nv = max(1, int(len(ds) * float(config.get("pseudo_val_fraction", 0.1))))
    tr, va = random_split(
        ds, [len(ds) - nv, nv], generator=torch.Generator().manual_seed(seed)
    )
    workers = int(config.get("num_workers", 0))
    device = resolve_device(device_name)
    batch_size = int(config.get("batch_size", 64))
    dl = DataLoader(
        tr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=(len(tr) > batch_size),
    )
    vl = DataLoader(
        va,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    model = build_model(
        "compact", bands=a.shape[-1], pool_size=int(config.get("pool_size", 8))
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("lr", 2e-4)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )
    epochs = int(config.get("epochs", 40))
    patience = int(config.get("patience", 8))
    left = patience
    best = -1.0
    hist = []
    path = out / "best.pt"

    epoch_bar = tqdm(
        range(1, epochs + 1),
        desc=progress_desc,
        unit="epoch",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for ep in epoch_bar:
        model.train()
        total = 0
        correct = 0
        ls = 0.0
        train_iter = tqdm(
            dl,
            desc=f"{progress_desc} ep{ep}",
            total=len(dl),
            unit="batch",
            leave=False,
            dynamic_ncols=True,
            disable=not show_progress,
        )
        for x, y in train_iter:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ls += float(loss.detach().item()) * len(y)
            total += len(y)
            correct += int((logits.argmax(1) == y).sum())
            train_iter.set_postfix(
                loss=f"{ls / max(total, 1):.4f}", acc=f"{correct / max(total, 1):.3f}"
            )
        model.eval()
        vt = 0
        vc = 0
        val_iter = tqdm(
            vl,
            desc=f"{progress_desc} val",
            total=len(vl),
            unit="batch",
            leave=False,
            dynamic_ncols=True,
            disable=not show_progress,
        )
        with torch.no_grad():
            for x, y in val_iter:
                y = y.to(device)
                z = model(x.to(device))
                vt += len(y)
                vc += int((z.argmax(1) == y).sum())
        vacc = vc / max(vt, 1)
        row = {
            "epoch": ep,
            "train_loss": ls / max(total, 1),
            "train_acc": correct / max(total, 1),
            "pseudo_val_acc": vacc,
        }
        hist.append(row)
        if vacc > best + 1e-6:
            best = vacc
            left = patience
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "bands": a.shape[-1],
                    "target_bands": target,
                    "pool_size": int(config.get("pool_size", 8)),
                    "plane_normalize": str(config.get("plane_normalize", "maxabs")),
                    "parameter_count": parameter_count(model),
                    "seed": seed,
                },
                path,
            )
        else:
            left -= 1
            if patience and left <= 0:
                epoch_bar.set_postfix(best=f"{best:.4f}", early_stop=ep)
                break
        epoch_bar.set_postfix(
            loss=f'{row["train_loss"]:.4f}',
            pseudo_val=f"{vacc:.4f}",
            best=f"{best:.4f}",
        )

    ck = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["model_state"])
    pv, oi = _predict(
        model,
        a,
        b,
        val_indices,
        int(config.get("eval_batch_size", 128)),
        device,
        ck["plane_normalize"],
        workers,
        progress_desc=f"{progress_desc} threshold",
        show_progress=show_progress,
    )
    pmap = np.full(gt.shape, np.nan, np.float32)
    pmap.ravel()[oi] = pv
    vm = np.zeros(gt.shape, bool)
    vm.ravel()[oi] = True
    thr, vf = find_best_f1_threshold(pmap, gt, vm)
    ck["threshold"] = thr
    torch.save(ck, path)
    with open(out / "train_history.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=hist[0].keys())
        wr.writeheader()
        wr.writerows(hist)
    summary = {
        "best_path": str(path),
        "normalizer": str(out / "normalizer.npz"),
        "best_pseudo_val_acc": best,
        "threshold": thr,
        "clean_spatial_val_f1": vf,
        "target_bands": target,
        "parameter_count": ck["parameter_count"],
        "pseudo_summary": pseudo.__dict__,
        "implementation": "user_provided_GETNET_compact_PyTorch_package",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def predict_getnet(
    checkpoint,
    normalizer_path,
    t1,
    t2,
    indices,
    batch_size,
    device_name,
    num_workers=0,
    progress_desc=None,
    show_progress=True,
):
    ck = torch.load(checkpoint, map_location="cpu")
    target = int(ck.get("target_bands", ck["bands"]))
    a0 = spectral_bin_average(t1, target)
    b0 = spectral_bin_average(t2, target)
    norm = Norm.load(normalizer_path)
    a, b = norm.apply(a0), norm.apply(b0)
    model = build_model("compact", int(ck["bands"]), int(ck.get("pool_size", 8)))
    model.load_state_dict(ck["model_state"])
    device = resolve_device(device_name)
    model.to(device)
    p, idx = _predict(
        model,
        a,
        b,
        indices,
        batch_size,
        device,
        str(ck.get("plane_normalize", "maxabs")),
        num_workers,
        progress_desc=progress_desc,
        show_progress=show_progress,
    )
    return p, idx, float(ck.get("threshold", 0.5))
