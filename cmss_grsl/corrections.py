from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from .baselines.irmad import run_irmad
from .transforms import sample_hwc


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    mu = np.nanmean(x, axis=(0, 1), keepdims=True)
    sd = np.nanstd(x, axis=(0, 1), keepdims=True)
    return (x - mu) / np.maximum(sd, 1e-6)


def joint_pca_features(t1: np.ndarray, t2: np.ndarray, n_components: int = 3, sample_size: int = 50000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    h, w, b = t1.shape
    a = _normalize(t1).reshape(-1, b)
    c = _normalize(t2).reshape(-1, b)
    both = np.concatenate([a, c], axis=0)
    valid = np.all(np.isfinite(both), axis=1)
    idx = np.flatnonzero(valid)
    rng = np.random.default_rng(seed)
    fit = idx if len(idx) <= sample_size else rng.choice(idx, sample_size, replace=False)
    d = int(min(n_components, b, max(1, len(fit) - 1)))
    pca = PCA(n_components=d, svd_solver='randomized', random_state=seed).fit(both[fit])
    f1 = pca.transform(a).reshape(h, w, d).astype(np.float32)
    f2 = pca.transform(c).reshape(h, w, d).astype(np.float32)
    meta = {"components": d, "explained_variance": float(pca.explained_variance_ratio_.sum())}
    return f1, f2, meta


def weighted_shift_score(f1: np.ndarray, f2: np.ndarray, dy: int, dx: int, weights: np.ndarray | None = None) -> float:
    h, w, c = f1.shape
    y0a, y1a = max(0, dy), min(h, h + dy)
    y0b, y1b = max(0, -dy), min(h, h - dy)
    x0a, x1a = max(0, dx), min(w, w + dx)
    x0b, x1b = max(0, -dx), min(w, w - dx)
    if y1a <= y0a or x1a <= x0a:
        return -1e18
    a = f1[y0a:y1a, x0a:x1a]
    b = f2[y0b:y1b, x0b:x1b]
    if weights is None:
        wgt = 1.0
    else:
        wgt = weights[y0a:y1a, x0a:x1a][..., None]
    diff = a - b
    num = np.sum(wgt * diff * diff)
    den = np.sum(wgt)
    return -float(num / max(den, 1e-6))


def search_global_shift(f1: np.ndarray, f2: np.ndarray, max_shift: int = 6, weights: np.ndarray | None = None) -> tuple[int, int, float]:
    best = (-1e18, 0, 0)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            s = weighted_shift_score(f1, f2, dy, dx, weights)
            if s > best[0]:
                best = (s, dy, dx)
    return int(best[1]), int(best[2]), float(best[0])


def estimate_row_shifts(
    f1: np.ndarray,
    f2: np.ndarray,
    global_dy: int,
    global_dx: int,
    weights: np.ndarray | None = None,
    residual_radius: int = 3,
    smooth_sigma: float = 6.0,
    max_abs_row_shift: float = 6.0,
) -> np.ndarray:
    h, w, c = f1.shape
    rows = np.zeros(h, np.float32)
    # Match each reference row to the shifted moving row.
    for y in range(h):
        y2 = y - global_dy
        if y2 < 0 or y2 >= h:
            continue
        a = f1[y]
        b = f2[y2]
        wrow = None if weights is None else weights[y].astype(np.float32)
        best_score = -1e18
        best_dx = 0
        for ddx in range(-residual_radius, residual_radius + 1):
            dx = global_dx + ddx
            x0a, x1a = max(0, dx), min(w, w + dx)
            x0b, x1b = max(0, -dx), min(w, w - dx)
            if x1a <= x0a:
                continue
            aa = a[x0a:x1a]
            bb = b[x0b:x1b]
            if wrow is None:
                ww = 1.0
            else:
                ww = wrow[x0a:x1a][:, None]
            diff = aa - bb
            score = -float(np.sum(ww * diff * diff) / max(np.sum(ww), 1e-6))
            if score > best_score:
                best_score = score
                best_dx = dx
        rows[y] = float(best_dx)
    rows = gaussian_filter1d(rows, sigma=smooth_sigma, mode='nearest')
    rows = np.clip(rows, -max_abs_row_shift, max_abs_row_shift)
    return rows.astype(np.float32)


def warp_with_scanline(moving: np.ndarray, global_dy: float, row_dx: np.ndarray, device: str = 'cpu') -> tuple[np.ndarray, np.ndarray]:
    h, w, _ = moving.shape
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing='ij')
    qy = yy + float(global_dy)
    qx = xx + row_dx[:, None].astype(np.float32)
    valid = (qy >= 0) & (qy <= h - 1) & (qx >= 0) & (qx <= w - 1)
    warped = sample_hwc(moving, qy, qx, device=device)
    flow = np.stack([qy - yy, qx - xx], axis=-1).astype(np.float32)
    return warped, flow


def flow_errors(pred_flow: np.ndarray, true_flow: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    m = mask & np.all(np.isfinite(pred_flow), axis=-1) & np.all(np.isfinite(true_flow), axis=-1)
    if not np.any(m):
        return {"flow_epe_mean": float('nan'), "flow_epe_median": float('nan'), "dx_rmse": float('nan'), "dy_rmse": float('nan')}
    diff = pred_flow[m] - true_flow[m]
    epe = np.sqrt(np.sum(diff * diff, axis=-1))
    return {
        "flow_epe_mean": float(np.mean(epe)),
        "flow_epe_median": float(np.median(epe)),
        "dx_rmse": float(np.sqrt(np.mean(diff[:, 1] ** 2))),
        "dy_rmse": float(np.sqrt(np.mean(diff[:, 0] ** 2))),
    }


@dataclass
class CorrectionResult:
    corrected: np.ndarray
    flow: np.ndarray
    metadata: dict[str, Any]


def cmss_correct(
    t1: np.ndarray,
    moving: np.ndarray,
    true_flow: np.ndarray | None = None,
    device: str = 'cpu',
    seed: int = 0,
    n_components: int = 3,
    global_radius: int = 6,
    row_radius: int = 3,
    smooth_sigma: float = 6.0,
    outer_iters: int = 3,
    weight_mode: str = 'irmad',
    use_weights: bool = True,
    show_progress: bool = True,
    progress_desc: str | None = None,
) -> CorrectionResult:
    current = np.asarray(moving, np.float32)
    global_dy = 0.0
    row_dx = np.zeros(t1.shape[0], np.float32)
    history: list[dict[str, Any]] = []
    weights = np.ones(t1.shape[:2], np.float32)
    iter_bar = tqdm(
        range(outer_iters),
        desc=progress_desc or ('CMSS' if use_weights else 'ScanlineSync-unweighted'),
        unit='iter',
        leave=False,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for it in iter_bar:
        f1, f2, pmeta = joint_pca_features(t1, current, n_components=n_components, seed=seed + it)
        gd_y, gd_x, score = search_global_shift(f1, f2, max_shift=global_radius, weights=weights if use_weights else None)
        rows = estimate_row_shifts(
            f1,
            f2,
            global_dy=gd_y,
            global_dx=gd_x,
            weights=weights if use_weights else None,
            residual_radius=row_radius,
            smooth_sigma=smooth_sigma,
            max_abs_row_shift=global_radius + row_radius,
        )
        # Convert the search displacement to the sampling flow.
        apply_dy = -float(gd_y)
        apply_rows = -rows.astype(np.float32)
        # Accumulate the update and resample from the original moving image.
        global_dy += apply_dy
        row_dx += apply_rows
        current, flow = warp_with_scanline(moving, global_dy=global_dy, row_dx=row_dx, device=device)
        if use_weights and weight_mode == 'irmad':
            irm = run_irmad(t1, current, pca_components=min(16, t1.shape[-1]), sample_size=40000, seed=seed + it)
            score_map = irm.score.astype(np.float32)
            scale = max(float(np.nanpercentile(score_map, 75) - np.nanpercentile(score_map, 25)), 1e-6)
            logits = np.clip((score_map - float(irm.threshold)) / scale, -60.0, 60.0)
            weights = 1.0 / (1.0 + np.exp(logits))
            weights = 0.05 + 0.95 * weights
        history.append({
            "iter": it + 1,
            "estimated_global_dy_step": float(gd_y),
            "estimated_global_dx_center": float(gd_x),
            "applied_global_dy_step": float(apply_dy),
            "applied_row_dx_mean_step": float(np.mean(apply_rows)),
            "search_score": float(score),
            "row_dx_mean": float(np.mean(row_dx)),
            "row_dx_std": float(np.std(row_dx)),
            **pmeta,
        })
        iter_bar.set_postfix(
            dy=f'{global_dy:.2f}',
            row_dx=f'{float(np.mean(row_dx)):.2f}±{float(np.std(row_dx)):.2f}',
        )
    meta = {
        "method": "CMSS" if use_weights else "ScanlineSync-unweighted",
        "global_dy": float(global_dy),
        "row_dx_mean": float(np.mean(row_dx)),
        "row_dx_std": float(np.std(row_dx)),
        "row_dx_min": float(np.min(row_dx)),
        "row_dx_max": float(np.max(row_dx)),
        "outer_iters": int(outer_iters),
        "history": history,
    }
    if true_flow is not None:
        valid = np.ones(t1.shape[:2], dtype=bool)
        meta.update(flow_errors(flow, true_flow, valid))
    return CorrectionResult(current, flow, meta)



def alignment_mse(
    t1: np.ndarray,
    t2: np.ndarray,
    n_components: int = 3,
    seed: int = 0,
    sample_size: int = 50000,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute a lightweight PCA-space synchronization loss at zero displacement.

    This is used only as an unsupervised accept/reject signal for the safe CMSS
    selector.  Lower values mean that the two dates are more consistent after a
    candidate correction.  The loss is intentionally simple and detector-free.
    """
    f1, f2, meta = joint_pca_features(t1, t2, n_components=n_components, sample_size=sample_size, seed=seed)
    valid = np.all(np.isfinite(f1), axis=-1) & np.all(np.isfinite(f2), axis=-1)
    if weights is not None:
        valid &= np.isfinite(weights)
    if not np.any(valid):
        out = {"sync_loss": float('nan'), "sync_valid_fraction": 0.0}
        out.update({f"sync_{k}": v for k, v in meta.items()})
        return out
    diff2 = np.sum((f1 - f2) ** 2, axis=-1)
    if weights is None:
        loss = float(np.mean(diff2[valid]))
    else:
        w = np.asarray(weights, np.float32)[valid]
        loss = float(np.sum(w * diff2[valid]) / max(float(np.sum(w)), 1e-6))
    out = {"sync_loss": loss, "sync_valid_fraction": float(np.mean(valid))}
    out.update({f"sync_{k}": v for k, v in meta.items()})
    return out


def identity_correction(
    moving: np.ndarray,
    true_flow: np.ndarray | None = None,
) -> CorrectionResult:
    h, w = moving.shape[:2]
    flow = np.zeros((h, w, 2), dtype=np.float32)
    meta: dict[str, Any] = {
        "method": "identity",
        "global_dy": 0.0,
        "row_dx_mean": 0.0,
        "row_dx_std": 0.0,
        "row_dx_min": 0.0,
        "row_dx_max": 0.0,
        "selector_choice": "identity",
    }
    if true_flow is not None:
        valid = np.ones((h, w), dtype=bool)
        meta.update(flow_errors(flow, true_flow, valid))
    return CorrectionResult(np.asarray(moving, np.float32), flow, meta)


def _flow_energy(meta: dict[str, Any], flow: np.ndarray | None = None) -> float:
    if flow is not None and flow.size:
        e = np.sqrt(np.sum(np.asarray(flow, np.float32) ** 2, axis=-1))
        return float(np.nanmedian(e))
    vals = []
    for k in ("global_dy", "global_dx", "row_dx_mean", "row_dx_std"):
        if k in meta and np.isfinite(meta[k]):
            vals.append(float(meta[k]) ** 2)
    return float(np.sqrt(sum(vals))) if vals else 0.0


def select_safe_cmss(
    t1: np.ndarray,
    moving: np.ndarray,
    cmss_result: CorrectionResult,
    pca_result: CorrectionResult | None = None,
    true_flow: np.ndarray | None = None,
    seed: int = 0,
    n_components: int = 3,
    cfg: dict[str, Any] | None = None,
) -> CorrectionResult:
    """Apply detector-free no-harm and PCA3 fallback gates to full CMSS.

    The selector has two purposes:
    1. Preserve already registered pairs by returning identity when neither PCA3
       nor CMSS yields a meaningful synchronization-loss improvement.
    2. Avoid over-correcting pure global translations by falling back to PCA3 when
       the estimated scanline residual is essentially flat and PCA3 is at least as
       good as CMSS under the same detector-free synchronization loss.

    It does not use ground-truth labels or downstream detector scores.
    """
    cfg = dict(cfg or {})
    enabled = bool(cfg.get('enabled', True))
    if not enabled:
        cmss_result.metadata = dict(cmss_result.metadata)
        cmss_result.metadata['safe_selector_enabled'] = False
        return cmss_result

    sample_size = int(cfg.get('sample_size', 50000))
    min_abs = float(cfg.get('identity_min_abs_improvement', 1e-4))
    min_rel = float(cfg.get('identity_min_rel_improvement', 0.005))
    identity_global_thr = float(cfg.get('identity_global_shift_threshold', 0.25))
    identity_cmss_rel = float(cfg.get('identity_cmss_rel_improvement_threshold', 0.01))
    pca_row_std_thr = float(cfg.get('pca_fallback_row_std_threshold', 0.75))
    pca_loss_margin_rel = float(cfg.get('pca_fallback_loss_margin_rel', 0.001))
    max_reasonable_row_std = float(cfg.get('max_reasonable_row_std', 4.5))

    identity = identity_correction(moving, true_flow=true_flow)
    id_loss = alignment_mse(t1, moving, n_components=n_components, seed=seed, sample_size=sample_size)
    cmss_loss = alignment_mse(t1, cmss_result.corrected, n_components=n_components, seed=seed + 101, sample_size=sample_size)
    pca_loss = None
    if pca_result is not None:
        pca_loss = alignment_mse(t1, pca_result.corrected, n_components=n_components, seed=seed + 202, sample_size=sample_size)

    id_l = float(id_loss['sync_loss'])
    cmss_l = float(cmss_loss['sync_loss'])
    pca_l = float(pca_loss['sync_loss']) if pca_loss is not None else float('inf')

    best_l = min(cmss_l, pca_l)
    improvement = id_l - best_l
    required = max(min_abs, min_rel * max(abs(id_l), 1e-6))

    selected = cmss_result
    selected_name = 'cmss'
    reason = 'cmss_kept'

    pca_energy = _flow_energy(pca_result.metadata, pca_result.flow) if pca_result is not None else float('inf')
    cmss_row_std = float(cmss_result.metadata.get('row_dx_std', np.nan))

    # Keep identity when correction does not improve the image-space score.
    if not np.isfinite(best_l) or improvement <= required:
        selected = identity
        selected_name = 'identity'
        reason = 'no_meaningful_sync_loss_improvement'
    else:
        # Reject weak corrections for nearly aligned pairs.
        cmss_improvement_rel = (id_l - cmss_l) / max(abs(id_l), 1e-6) if np.isfinite(cmss_l) else -np.inf
        if pca_energy <= identity_global_thr and cmss_improvement_rel <= identity_cmss_rel:
            selected = identity
            selected_name = 'identity'
            reason = 'near_identity_pca_and_small_cmss_gain'
        # Prefer PCA3 when the row field is effectively global.
        elif pca_result is not None and np.isfinite(pca_l) and np.isfinite(cmss_l):
            pca_not_worse = pca_l <= cmss_l * (1.0 + pca_loss_margin_rel)
            flat_row_field = np.isfinite(cmss_row_std) and cmss_row_std <= pca_row_std_thr
            unreasonable_row_field = np.isfinite(cmss_row_std) and cmss_row_std > max_reasonable_row_std
            if (flat_row_field and pca_not_worse) or (unreasonable_row_field and pca_l < cmss_l):
                selected = pca_result
                selected_name = 'pca3_translation'
                reason = 'pca3_fallback_flat_or_unreasonable_row_field'

    meta = dict(selected.metadata)
    meta.update({
        'method': 'CMSS-safe' if selected_name != 'cmss' else str(cmss_result.metadata.get('method', 'CMSS')),
        'safe_selector_enabled': True,
        'selector_choice': selected_name,
        'selector_reason': reason,
        'selector_identity_loss': id_l,
        'selector_cmss_loss': cmss_l,
        'selector_pca3_loss': pca_l if np.isfinite(pca_l) else float('nan'),
        'selector_improvement': float(improvement) if np.isfinite(improvement) else float('nan'),
        'selector_required_improvement': float(required),
        'selector_pca3_flow_energy': float(pca_energy) if np.isfinite(pca_energy) else float('nan'),
        'selector_cmss_row_std_raw': float(cmss_row_std) if np.isfinite(cmss_row_std) else float('nan'),
    })
    # Report flow errors for the selected candidate.
    if true_flow is not None:
        valid = np.ones(t1.shape[:2], dtype=bool)
        meta.update(flow_errors(selected.flow, true_flow, valid))
    return CorrectionResult(selected.corrected, selected.flow, meta)

def global_pca_translation(
    t1: np.ndarray,
    moving: np.ndarray,
    true_flow: np.ndarray | None = None,
    device: str = 'cpu',
    seed: int = 0,
    n_components: int = 3,
    global_radius: int = 6,
) -> CorrectionResult:
    f1, f2, pmeta = joint_pca_features(t1, moving, n_components=n_components, seed=seed)
    dy, dx, score = search_global_shift(f1, f2, max_shift=global_radius, weights=None)
    # Convert the search displacement to the sampling flow.
    apply_dy = -float(dy)
    apply_dx = -float(dx)
    row_dx = np.full(t1.shape[0], apply_dx, np.float32)
    corrected, flow = warp_with_scanline(moving, global_dy=apply_dy, row_dx=row_dx, device=device)
    meta = {
        "method": f"PCA{n_components}_translation",
        "estimated_global_dy": float(dy),
        "estimated_global_dx": float(dx),
        "global_dy": apply_dy,
        "global_dx": apply_dx,
        "search_score": float(score),
        **pmeta,
    }
    if true_flow is not None:
        valid = np.ones(t1.shape[:2], dtype=bool)
        meta.update(flow_errors(flow, true_flow, valid))
    return CorrectionResult(corrected, flow, meta)
