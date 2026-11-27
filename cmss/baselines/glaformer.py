"""GLAFormer training and inference with saved normalization and thresholds."""

from __future__ import annotations
import csv, json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np, torch
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from ..metrics import find_best_f1_threshold
from ..utils import parameter_count, resolve_device, set_seed


class _NoOpScaler:
    def scale(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass


def _make_grad_scaler(enabled: bool):
    if not enabled:
        return _NoOpScaler()
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=True)
        except TypeError:
            return torch.amp.GradScaler(enabled=True)
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler(enabled=True)
    return _NoOpScaler()


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda", enabled=True)
        if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
            return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        r = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        r.floor_()
        return x.div(keep) * r


class PixelEmbedding(nn.Module):
    def __init__(self, bands, dim, patch):
        super().__init__()
        self.fc = nn.Linear(bands, dim)
        self.pos = nn.Parameter(torch.randn(1, patch * patch, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        return self.fc(x) + self.pos


class GLA(nn.Module):
    def __init__(self, dim, heads=8, ws=3, alpha=0.5):
        super().__init__()
        assert dim % heads == 0
        hd = dim // heads
        self.ws = ws
        self.lh = int(heads * alpha)
        self.gh = heads - self.lh
        self.ld = self.lh * hd
        self.gd = self.gh * hd
        self.scale = hd**-0.5
        if self.lh:
            self.lqkv = nn.Linear(dim, self.ld * 3)
            self.lproj = nn.Linear(self.ld, self.ld)
        if self.gh:
            self.pool = nn.AvgPool2d(ws, ws) if ws > 1 else nn.Identity()
            self.gq = nn.Linear(dim, self.gd)
            self.gkv = nn.Linear(dim, self.gd * 2)
            self.gproj = nn.Linear(self.gd, self.gd)

    def local(self, x):
        b, h, w, c = x.shape
        hg, wg = h // self.ws, w // self.ws
        groups = hg * wg
        z = x.reshape(b, hg, self.ws, wg, self.ws, c).transpose(2, 3)
        qkv = (
            self.lqkv(z)
            .reshape(b, groups, self.ws * self.ws, 3, self.lh, self.ld // self.lh)
            .permute(3, 0, 1, 4, 2, 5)
        )
        q, k, v = qkv
        a = (q @ k.transpose(-2, -1)) * self.scale
        a = a.softmax(-1)
        o = (
            (a @ v)
            .transpose(2, 3)
            .reshape(b, hg, wg, self.ws, self.ws, self.ld)
            .transpose(2, 3)
            .reshape(b, h, w, self.ld)
        )
        return self.lproj(o)

    def global_(self, x):
        b, h, w, c = x.shape
        q = (
            self.gq(x)
            .reshape(b, h * w, self.gh, self.gd // self.gh)
            .permute(0, 2, 1, 3)
        )
        r = self.pool(x.permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        kv = (
            self.gkv(r)
            .reshape(b, -1, 2, self.gh, self.gd // self.gh)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv
        a = (q @ k.transpose(-2, -1)) * self.scale
        a = a.softmax(-1)
        o = (a @ v).transpose(1, 2).reshape(b, h, w, self.gd)
        return self.gproj(o)

    def forward(self, x, h, w):
        b, n, c = x.shape
        z = x.reshape(b, h, w, c)
        pieces = []
        if self.lh:
            pieces.append(self.local(z))
        if self.gh:
            pieces.append(self.global_(z))
        return (pieces[0] if len(pieces) == 1 else torch.cat(pieces, -1)).reshape(
            b, n, c
        )


class CrossFFN(nn.Module):
    def __init__(self, dim, patch, exp=2.66):
        super().__init__()
        hidden = int(dim * exp)
        self.p = patch
        self.pin = nn.Conv2d(dim, hidden * 2, 1)
        self.dw = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2)
        self.pout = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        b, _, c = x.shape
        z = x.reshape(b, self.p, self.p, c).permute(0, 3, 1, 2)
        a, b2 = self.dw(self.pin(z)).chunk(2, 1)
        z = F.gelu(b2) * a + F.gelu(a) * b2
        return self.pout(z).permute(0, 2, 3, 1).reshape(x.shape[0], self.p * self.p, c)


class Block(nn.Module):
    def __init__(self, dim, heads, patch, dp=0.0):
        super().__init__()
        self.p = patch
        self.n1 = nn.LayerNorm(dim, eps=1e-6)
        self.att = GLA(dim, heads)
        self.dp = DropPath(dp) if dp else nn.Identity()
        self.n2 = nn.LayerNorm(dim, eps=1e-6)
        self.ff = CrossFFN(dim, patch)

    def forward(self, x):
        x = x + self.dp(self.att(self.n1(x), self.p, self.p))
        return x + self.dp(self.ff(self.n2(x)))


class GLAFormer(nn.Module):
    def __init__(
        self, bands, patch_size=6, embed_dim=128, depth=4, heads=8, drop_path=0.0
    ):
        super().__init__()
        assert patch_size % 3 == 0
        self.p = patch_size
        self.e1 = PixelEmbedding(bands, embed_dim, patch_size)
        self.e2 = PixelEmbedding(bands, embed_dim, patch_size)
        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.b1 = nn.Sequential(
            *[Block(embed_dim, heads, patch_size, dpr[i]) for i in range(depth)]
        )
        self.b2 = nn.Sequential(
            *[Block(embed_dim, heads, patch_size, dpr[i]) for i in range(depth)]
        )
        self.n1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.n2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.c1 = nn.Sequential(nn.Conv2d(embed_dim, 16, 3, padding=1), nn.ReLU())
        self.c2 = nn.Conv2d(16, 1, 3, padding=1)
        self.fc = nn.Sequential(
            nn.Linear(patch_size * patch_size, 20), nn.ReLU(), nn.Linear(20, 2)
        )
        self.apply(self.init)

    @staticmethod
    def init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias) if m.bias is not None else None
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            nn.init.zeros_(m.bias) if m.bias is not None else None
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x1, x2):
        a = self.n1(self.b1(self.e1(x1)))
        b = self.n2(self.b2(self.e2(x2)))
        z = torch.abs(a - b)
        n, _, c = z.shape
        z = z.reshape(n, self.p, self.p, c).permute(0, 3, 1, 2)
        return self.fc(self.c2(self.c1(z)).reshape(n, -1))


class PairPatchDataset(Dataset):
    def __init__(self, t1, t2, gt, indices, patch):
        self.t1 = np.asarray(t1, np.float32)
        self.t2 = np.asarray(t2, np.float32)
        self.gt = gt
        self.idx = np.asarray(indices, np.int64)
        self.h, self.w, self.b = self.t1.shape
        self.p = patch
        pre = patch // 2
        post = patch - pre - 1
        self.a = np.pad(self.t1, ((pre, post), (pre, post), (0, 0)), mode="reflect")
        self.b2 = np.pad(self.t2, ((pre, post), (pre, post), (0, 0)), mode="reflect")

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        idx = int(self.idx[i])
        y, x = divmod(idx, self.w)
        p = self.p
        a = self.a[y : y + p, x : x + p].reshape(p * p, self.b)
        b = self.b2[y : y + p, x : x + p].reshape(p * p, self.b)
        return (
            torch.from_numpy(a),
            torch.from_numpy(b),
            torch.tensor(int(self.gt[y, x]), dtype=torch.long),
            idx,
        )


@dataclass
class Norm:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, a, b, sample=100000, seed=0):
        x = np.concatenate([a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])])
        rng = np.random.default_rng(seed)
        x = x[rng.choice(len(x), min(sample, len(x)), replace=False)]
        return cls(
            x.mean(0).astype(np.float32), np.maximum(x.std(0), 1e-5).astype(np.float32)
        )

    def apply(self, x):
        return np.clip(
            (np.asarray(x, np.float32) - self.mean) / self.std, -8, 8
        ).astype(np.float32)

    def save(self, p):
        np.savez(p, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, p):
        d = np.load(p)
        return cls(d["mean"], d["std"])


def _predict(
    model,
    t1,
    t2,
    gt,
    indices,
    patch,
    batch,
    device,
    workers=0,
    progress_desc=None,
    show_progress=True,
):
    ds = PairPatchDataset(t1, t2, gt, indices, patch)
    dl = DataLoader(
        ds,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    probs = []
    ids = []
    model.eval()
    iterator = tqdm(
        dl,
        desc=progress_desc or "predict GLAFormer",
        total=len(dl),
        unit="batch",
        leave=False,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    with torch.no_grad():
        for a, b, _, idx in iterator:
            probs.append(
                torch.softmax(model(a.to(device), b.to(device)), 1)[:, 1].cpu().numpy()
            )
            ids.append(idx.numpy())
    return np.concatenate(probs), np.concatenate(ids)


def train_glaformer(
    t1, t2, gt, train_indices, val_indices, output_dir, config, device_name, seed
):
    set_seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    show_progress = bool(config.get("progress", True))
    progress_desc = str(config.get("progress_desc", f"train GLAFormer seed{seed}"))

    norm = Norm.fit(t1, t2, int(config.get("normalizer_sample_size", 100000)), seed)
    norm.save(out / "normalizer.npz")
    a, b = norm.apply(t1), norm.apply(t2)
    p = int(config.get("patch_size", 6))
    tr = PairPatchDataset(a, b, gt, train_indices, p)
    va = PairPatchDataset(a, b, gt, val_indices, p)
    dl = DataLoader(
        tr,
        batch_size=int(config.get("batch_size", 64)),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    vl = DataLoader(
        va,
        batch_size=int(config.get("eval_batch_size", 128)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    model = GLAFormer(
        t1.shape[-1],
        p,
        int(config.get("embed_dim", 128)),
        int(config.get("depth", 4)),
        int(config.get("heads", 8)),
        float(config.get("drop_path", 0)),
    ).to(device)
    counts = np.bincount(gt.ravel()[train_indices].astype(int), minlength=2)
    weights = torch.tensor(
        counts.sum() / np.maximum(2 * counts, 1), dtype=torch.float32, device=device
    )
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("lr", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    epochs = int(config.get("epochs", 60))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    patience = int(config.get("patience", 12))
    amp = bool(config.get("amp", True)) and device.type == "cuda"
    scaler = _make_grad_scaler(amp)
    hist = []
    best = -1.0
    left = patience
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
        loss_sum = 0.0
        correct = 0
        total = 0
        train_iter = tqdm(
            dl,
            desc=f"{progress_desc} ep{ep}",
            total=len(dl),
            unit="batch",
            leave=False,
            dynamic_ncols=True,
            disable=not show_progress,
        )
        for x1, x2, y, _ in train_iter:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with _autocast_context(device, amp):
                logits = model(x1, x2)
                loss = F.cross_entropy(logits, y, weight=weights)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += float(loss.detach().item()) * len(y)
            total += len(y)
            correct += int((logits.argmax(1) == y).sum())
            train_iter.set_postfix(
                loss=f"{loss_sum / max(total, 1):.4f}",
                acc=f"{correct / max(total, 1):.3f}",
            )
        sched.step()

        vp = []
        vy = []
        model.eval()
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
            for x1, x2, y, _ in val_iter:
                vp.append(
                    torch.softmax(model(x1.to(device), x2.to(device)), 1)[:, 1]
                    .cpu()
                    .numpy()
                )
                vy.append(y.numpy())
        ps = np.concatenate(vp)
        ys = np.concatenate(vy)
        thr, vf = find_best_f1_threshold(
            ps[:, None], ys[:, None], np.ones((len(ys), 1), bool)
        )
        row = {
            "epoch": ep,
            "train_loss": loss_sum / max(total, 1),
            "train_acc": correct / max(total, 1),
            "val_f1": vf,
            "val_threshold": thr,
            "lr": opt.param_groups[0]["lr"],
        }
        hist.append(row)
        if vf > best + 1e-6:
            best = vf
            left = patience
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "bands": t1.shape[-1],
                    "patch_size": p,
                    "embed_dim": int(config.get("embed_dim", 128)),
                    "depth": int(config.get("depth", 4)),
                    "heads": int(config.get("heads", 8)),
                    "drop_path": float(config.get("drop_path", 0)),
                    "threshold": thr,
                    "seed": seed,
                    "parameter_count": parameter_count(model),
                },
                path,
            )
        else:
            left -= 1
            if patience and left <= 0:
                epoch_bar.set_postfix(best=f"{best:.4f}", early_stop=ep)
                break
        epoch_bar.set_postfix(
            loss=f'{row["train_loss"]:.4f}', val_f1=f"{vf:.4f}", best=f"{best:.4f}"
        )

    with open(out / "train_history.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=hist[0].keys())
        wr.writeheader()
        wr.writerows(hist)
    ck = torch.load(path, map_location="cpu")
    summary = {
        "best_path": str(path),
        "best_val_f1": best,
        "threshold": float(ck["threshold"]),
        "parameter_count": parameter_count(model),
        "normalizer": str(out / "normalizer.npz"),
        "implementation": "adapted_user_provided_GLAFormer_notebook",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def predict_glaformer(
    checkpoint,
    normalizer_path,
    t1,
    t2,
    gt,
    indices,
    batch_size,
    device_name,
    num_workers=0,
    progress_desc=None,
    show_progress=True,
):
    ck = torch.load(checkpoint, map_location="cpu")
    model = GLAFormer(
        int(ck["bands"]),
        int(ck["patch_size"]),
        int(ck["embed_dim"]),
        int(ck["depth"]),
        int(ck["heads"]),
        float(ck.get("drop_path", 0)),
    )
    model.load_state_dict(ck["model_state"])
    device = resolve_device(device_name)
    model.to(device)
    norm = Norm.load(normalizer_path)
    p, idx = _predict(
        model,
        norm.apply(t1),
        norm.apply(t2),
        gt,
        indices,
        int(ck["patch_size"]),
        batch_size,
        device,
        num_workers,
        progress_desc=progress_desc,
        show_progress=show_progress,
    )
    return p, idx, float(ck.get("threshold", 0.5))
