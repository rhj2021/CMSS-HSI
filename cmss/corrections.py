"""CMSS scanline fitting and image-only correction selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from .baselines.irmad import run_irmad
from .transforms import sample_hwc


def _channel_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, np.float32)
    mu = np.nanmean(x, axis=(0, 1), keepdims=True).astype(np.float32)
    sd = np.nanstd(x, axis=(0, 1), keepdims=True).astype(np.float32)
    return mu, np.maximum(sd, 1e-6).astype(np.float32)


def _normalize_with_stats(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((np.asarray(x, np.float32) - mu) / np.maximum(sd, 1e-6)).astype(np.float32)


def _normalize(x: np.ndarray) -> np.ndarray:
    mu, sd = _channel_stats(x)
    return _normalize_with_stats(x, mu, sd)


def _fit_pca_from_two_arrays(
    a: np.ndarray,
    b: np.ndarray,
    n_components: int,
    sample_size: int,
    seed: int,
) -> PCA:
    both = np.concatenate([a, b], axis=0)
    valid = np.all(np.isfinite(both), axis=1)
    idx = np.flatnonzero(valid)
    if idx.size < 2:
        raise ValueError("Not enough finite pixels to fit PCA")
    rng = np.random.default_rng(seed)
    fit = (
        idx if len(idx) <= sample_size else rng.choice(idx, sample_size, replace=False)
    )
    d = int(min(n_components, both.shape[1], max(1, len(fit) - 1)))
    return PCA(n_components=d, svd_solver="randomized", random_state=seed).fit(
        both[fit]
    )


def joint_pca_features(
    t1: np.ndarray,
    t2: np.ndarray,
    n_components: int = 3,
    sample_size: int = 50000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit one PCA basis to separately standardized bi-temporal pixels."""
    h, w, bands = t1.shape
    a = _normalize(t1).reshape(-1, bands)
    b = _normalize(t2).reshape(-1, bands)
    pca = _fit_pca_from_two_arrays(a, b, n_components, sample_size, seed)
    f1 = pca.transform(a).reshape(h, w, -1).astype(np.float32)
    f2 = pca.transform(b).reshape(h, w, -1).astype(np.float32)
    meta = {
        "components": int(pca.n_components_),
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
    }
    return f1, f2, meta


def weighted_shift_score(
    f1: np.ndarray,
    f2: np.ndarray,
    dy: int,
    dx: int,
    weights: np.ndarray | None = None,
) -> float:
    """Negative mean squared matching error on the valid overlap.

    The spatial normalization is identical for weighted and unweighted matching.
    This prevents large candidate shifts from being preferred merely because they
    contain fewer overlapping pixels.
    """
    h, w, _ = f1.shape
    y0a, y1a = max(0, dy), min(h, h + dy)
    y0b, y1b = max(0, -dy), min(h, h - dy)
    x0a, x1a = max(0, dx), min(w, w + dx)
    x0b, x1b = max(0, -dx), min(w, w - dx)
    if y1a <= y0a or x1a <= x0a:
        return -1e18

    a = f1[y0a:y1a, x0a:x1a]
    b = f2[y0b:y1b, x0b:x1b]
    diff2 = np.sum((a - b) ** 2, axis=-1)
    finite = np.isfinite(diff2)
    if weights is None:
        return -float(np.mean(diff2[finite])) if np.any(finite) else -1e18

    wgt = np.asarray(weights[y0a:y1a, x0a:x1a], np.float32)
    valid = finite & np.isfinite(wgt) & (wgt > 0)
    if not np.any(valid):
        return -1e18
    return -float(
        np.sum(wgt[valid] * diff2[valid]) / max(float(np.sum(wgt[valid])), 1e-6)
    )


def search_global_shift(
    f1: np.ndarray,
    f2: np.ndarray,
    max_shift: int = 6,
    weights: np.ndarray | None = None,
) -> tuple[int, int, float]:
    best = (-1e18, 0, 0)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            score = weighted_shift_score(f1, f2, dy, dx, weights)
            if score > best[0]:
                best = (score, dy, dx)
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
    """Search each row near the global offset, then smooth and clip the row field."""
    h, w, _ = f1.shape
    rows = np.zeros(h, np.float32)
    for y in range(h):
        y2 = y - global_dy
        if y2 < 0 or y2 >= h:
            rows[y] = float(global_dx)
            continue
        a = f1[y]
        b = f2[y2]
        wrow = None if weights is None else np.asarray(weights[y], np.float32)
        best_score = -1e18
        best_dx = global_dx
        for ddx in range(-residual_radius, residual_radius + 1):
            dx = global_dx + ddx
            x0a, x1a = max(0, dx), min(w, w + dx)
            x0b, x1b = max(0, -dx), min(w, w - dx)
            if x1a <= x0a:
                continue
            diff2 = np.sum((a[x0a:x1a] - b[x0b:x1b]) ** 2, axis=-1)
            finite = np.isfinite(diff2)
            if wrow is None:
                score = -float(np.mean(diff2[finite])) if np.any(finite) else -1e18
            else:
                ww = wrow[x0a:x1a]
                valid = finite & np.isfinite(ww) & (ww > 0)
                score = (
                    -float(
                        np.sum(ww[valid] * diff2[valid])
                        / max(float(np.sum(ww[valid])), 1e-6)
                    )
                    if np.any(valid)
                    else -1e18
                )
            if score > best_score:
                best_score = score
                best_dx = dx
        rows[y] = float(best_dx)

    rows = gaussian_filter1d(rows, sigma=smooth_sigma, mode="nearest")
    rows = np.clip(rows, -max_abs_row_shift, max_abs_row_shift)
    return rows.astype(np.float32)


def flow_valid_mask(flow: np.ndarray) -> np.ndarray:
    """Pixels whose sampling coordinates remain inside the moving image."""
    h, w = flow.shape[:2]
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    qy = yy + np.asarray(flow[..., 0], np.float32)
    qx = xx + np.asarray(flow[..., 1], np.float32)
    return (
        np.isfinite(qy)
        & np.isfinite(qx)
        & (qy >= 0)
        & (qy <= h - 1)
        & (qx >= 0)
        & (qx <= w - 1)
    )


def warp_with_scanline(
    moving: np.ndarray,
    global_dy: float,
    row_dx: np.ndarray,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the moving cube with a global vertical and rowwise horizontal offset."""
    h, w, _ = moving.shape
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    qy = yy + float(global_dy)
    qx = xx + row_dx[:, None].astype(np.float32)
    warped = sample_hwc(moving, qy, qx, device=device)
    flow = np.stack([qy - yy, qx - xx], axis=-1).astype(np.float32)
    return warped, flow


def flow_errors(
    pred_flow: np.ndarray,
    true_flow: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Measure displacement error on the optional evaluation mask."""
    valid = np.all(np.isfinite(pred_flow), axis=-1) & np.all(
        np.isfinite(true_flow), axis=-1
    )
    if mask is not None:
        valid &= np.asarray(mask, bool)
    if not np.any(valid):
        return {
            "flow_epe_mean": float("nan"),
            "flow_epe_median": float("nan"),
            "dx_rmse": float("nan"),
            "dy_rmse": float("nan"),
        }
    diff = (
        np.asarray(pred_flow, np.float32)[valid]
        - np.asarray(true_flow, np.float32)[valid]
    )
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
    weights: np.ndarray | None = None


def cmss_correct(
    t1: np.ndarray,
    moving: np.ndarray,
    true_flow: np.ndarray | None = None,
    flow_eval_mask: np.ndarray | None = None,
    device: str = "cpu",
    seed: int = 0,
    n_components: int = 3,
    global_radius: int = 6,
    row_radius: int = 3,
    smooth_sigma: float = 6.0,
    outer_iters: int = 3,
    weight_mode: str = "irmad",
    use_weights: bool = True,
    irmad_pca_components: int = 16,
    irmad_sample_size: int = 40000,
    irmad_epsilon: float = 0.05,
    show_progress: bool = True,
    progress_desc: str | None = None,
) -> CorrectionResult:
    """Estimate the raw CMSS candidate before image-only selection.

    Each update samples the original moving cube with the accumulated displacement.
    Ground-truth flow is used only for optional diagnostic metrics.
    """
    current = np.asarray(moving, np.float32)
    global_dy = 0.0
    row_dx = np.zeros(t1.shape[0], np.float32)
    history: list[dict[str, Any]] = []
    weights = np.ones(t1.shape[:2], np.float32)

    iter_bar = tqdm(
        range(outer_iters),
        desc=progress_desc or ("CMSS" if use_weights else "ScanlineSync-unweighted"),
        unit="iter",
        leave=False,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for it in iter_bar:
        f1, f2, pmeta = joint_pca_features(
            t1, current, n_components=n_components, seed=seed + it
        )
        active_weights = weights if use_weights else None
        gd_y, gd_x, score = search_global_shift(
            f1,
            f2,
            max_shift=global_radius,
            weights=active_weights,
        )
        rows = estimate_row_shifts(
            f1,
            f2,
            global_dy=gd_y,
            global_dx=gd_x,
            weights=active_weights,
            residual_radius=row_radius,
            smooth_sigma=smooth_sigma,
            max_abs_row_shift=global_radius + row_radius,
        )

        # Matching reports how the moving feature map should be shifted for overlap;
        # the sampling flow that pulls pixels from the moving image has the opposite sign.
        apply_dy = -float(gd_y)
        apply_rows = -rows.astype(np.float32)
        global_dy += apply_dy
        row_dx += apply_rows
        current, flow = warp_with_scanline(
            moving, global_dy=global_dy, row_dx=row_dx, device=device
        )

        if use_weights and weight_mode == "irmad":
            irm = run_irmad(
                t1,
                current,
                pca_components=min(int(irmad_pca_components), t1.shape[-1]),
                sample_size=int(irmad_sample_size),
                seed=seed + it,
                show_progress=False,
            )
            score_map = irm.score.astype(np.float32)
            scale = max(
                float(
                    np.nanpercentile(score_map, 75) - np.nanpercentile(score_map, 25)
                ),
                1e-6,
            )
            logits = np.clip((score_map - float(irm.threshold)) / scale, -60.0, 60.0)
            weights = 1.0 / (1.0 + np.exp(logits))
            weights = float(irmad_epsilon) + (1.0 - float(irmad_epsilon)) * weights
            weights = weights.astype(np.float32)

        history.append(
            {
                "iter": it + 1,
                "estimated_global_dy_step": float(gd_y),
                "estimated_global_dx_center": float(gd_x),
                "applied_global_dy_step": float(apply_dy),
                "applied_row_dx_mean_step": float(np.mean(apply_rows)),
                "search_score": float(score),
                "row_dx_mean": float(np.mean(row_dx)),
                "row_dx_std": float(np.std(row_dx)),
                **pmeta,
            }
        )
        iter_bar.set_postfix(
            dy=f"{global_dy:.2f}",
            row_dx=f"{float(np.mean(row_dx)):.2f}±{float(np.std(row_dx)):.2f}",
        )

    meta: dict[str, Any] = {
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
        meta.update(flow_errors(flow, true_flow, flow_eval_mask))
    return CorrectionResult(current, flow, meta, weights if use_weights else None)


@dataclass
class CommonPCAMetric:
    ref_mu: np.ndarray
    ref_sd: np.ndarray
    moving_mu: np.ndarray
    moving_sd: np.ndarray
    pca: PCA
    reference_features: np.ndarray
    metadata: dict[str, Any]

    def project_moving(self, cube: np.ndarray) -> np.ndarray:
        h, w, bands = cube.shape
        x = _normalize_with_stats(cube, self.moving_mu, self.moving_sd).reshape(
            -1, bands
        )
        return self.pca.transform(x).reshape(h, w, -1).astype(np.float32)

    def loss(
        self,
        cube: np.ndarray,
        valid_mask: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> dict[str, float]:
        f2 = self.project_moving(cube)
        diff2 = np.sum((self.reference_features - f2) ** 2, axis=-1)
        valid = np.asarray(valid_mask, bool) & np.isfinite(diff2)
        if weights is not None:
            valid &= np.isfinite(weights) & (np.asarray(weights) > 0)
        if not np.any(valid):
            return {"sync_loss": float("nan"), "sync_valid_fraction": 0.0}
        if weights is None:
            loss = float(np.mean(diff2[valid]))
        else:
            w = np.asarray(weights, np.float32)[valid]
            loss = float(np.sum(w * diff2[valid]) / max(float(np.sum(w)), 1e-6))
        return {"sync_loss": loss, "sync_valid_fraction": float(np.mean(valid))}


def fit_common_pca_metric(
    t1: np.ndarray,
    moving: np.ndarray,
    n_components: int = 3,
    sample_size: int = 50000,
    seed: int = 0,
) -> CommonPCAMetric:
    """Fit one fixed normalization and PCA basis for all selector candidates."""
    h, w, bands = t1.shape
    ref_mu, ref_sd = _channel_stats(t1)
    mov_mu, mov_sd = _channel_stats(moving)
    a = _normalize_with_stats(t1, ref_mu, ref_sd).reshape(-1, bands)
    b = _normalize_with_stats(moving, mov_mu, mov_sd).reshape(-1, bands)
    pca = _fit_pca_from_two_arrays(a, b, n_components, sample_size, seed)
    f1 = pca.transform(a).reshape(h, w, -1).astype(np.float32)
    return CommonPCAMetric(
        ref_mu=ref_mu,
        ref_sd=ref_sd,
        moving_mu=mov_mu,
        moving_sd=mov_sd,
        pca=pca,
        reference_features=f1,
        metadata={
            "components": int(pca.n_components_),
            "explained_variance": float(pca.explained_variance_ratio_.sum()),
            "sample_size": int(min(sample_size, a.shape[0] + b.shape[0])),
            "seed": int(seed),
        },
    )


def alignment_mse(
    t1: np.ndarray,
    t2: np.ndarray,
    n_components: int = 3,
    seed: int = 0,
    sample_size: int = 50000,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Backward-compatible pairwise PCA-space discrepancy helper."""
    metric = fit_common_pca_metric(t1, t2, n_components, sample_size, seed)
    valid = np.ones(t1.shape[:2], dtype=bool)
    out = metric.loss(t2, valid, weights)
    out.update({f"sync_{k}": v for k, v in metric.metadata.items()})
    return out


def identity_correction(
    moving: np.ndarray,
    true_flow: np.ndarray | None = None,
    flow_eval_mask: np.ndarray | None = None,
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
        meta.update(flow_errors(flow, true_flow, flow_eval_mask))
    return CorrectionResult(np.asarray(moving, np.float32), flow, meta, None)


def select_safe_cmss(
    t1: np.ndarray,
    moving: np.ndarray,
    cmss_result: CorrectionResult,
    pca_result: CorrectionResult | None = None,
    true_flow: np.ndarray | None = None,
    flow_eval_mask: np.ndarray | None = None,
    seed: int = 0,
    n_components: int = 3,
    cfg: dict[str, Any] | None = None,
) -> CorrectionResult:
    """Select identity, translation, or CMSS with one fixed image-only metric.

    Identity is selected when the best improvement does not exceed the configured
    margin. PCA translation is preferred when its loss is within the configured
    tolerance of the CMSS loss.
    """
    cfg = dict(cfg or {})
    if not bool(cfg.get("enabled", True)):
        meta = dict(cmss_result.metadata)
        meta["safe_selector_enabled"] = False
        return CorrectionResult(
            cmss_result.corrected, cmss_result.flow, meta, cmss_result.weights
        )

    sample_size = int(cfg.get("sample_size", 50000))
    min_abs = float(cfg.get("identity_min_abs_improvement", 1e-4))
    min_rel = float(cfg.get("identity_min_rel_improvement", 0.005))
    pca_tie_abs = float(cfg.get("pca_tie_abs", 1e-4))
    pca_tie_rel = float(cfg.get("pca_tie_rel", 0.001))
    use_change_weights = bool(cfg.get("use_change_weights", True))

    identity = identity_correction(
        moving, true_flow=true_flow, flow_eval_mask=flow_eval_mask
    )
    metric = fit_common_pca_metric(
        t1,
        moving,
        n_components=n_components,
        sample_size=sample_size,
        seed=seed,
    )

    common_valid = flow_valid_mask(cmss_result.flow)
    if pca_result is not None:
        common_valid &= flow_valid_mask(pca_result.flow)
    common_valid &= np.all(np.isfinite(t1), axis=-1) & np.all(
        np.isfinite(moving), axis=-1
    )
    weights = (
        cmss_result.weights
        if (use_change_weights and cmss_result.weights is not None)
        else None
    )

    id_loss = metric.loss(moving, common_valid, weights)
    cmss_loss = metric.loss(cmss_result.corrected, common_valid, weights)
    pca_loss = (
        metric.loss(pca_result.corrected, common_valid, weights)
        if pca_result is not None
        else None
    )

    id_l = float(id_loss["sync_loss"])
    cmss_l = float(cmss_loss["sync_loss"])
    pca_l = float(pca_loss["sync_loss"]) if pca_loss is not None else float("inf")
    best_l = min(cmss_l, pca_l)
    improvement = id_l - best_l
    identity_required = max(min_abs, min_rel * max(abs(id_l), 1e-6))
    pca_tolerance = max(pca_tie_abs, pca_tie_rel * max(abs(cmss_l), 1e-6))

    if not np.isfinite(best_l) or improvement <= identity_required:
        selected = identity
        selected_name = "identity"
        reason = "no_meaningful_common_metric_improvement"
    elif (
        pca_result is not None
        and np.isfinite(pca_l)
        and pca_l <= cmss_l + pca_tolerance
    ):
        selected = pca_result
        selected_name = "pca3_translation"
        reason = "simpler_pca3_within_common_metric_tolerance"
    else:
        selected = cmss_result
        selected_name = "cmss"
        reason = "cmss_clear_common_metric_gain"

    meta = dict(selected.metadata)
    meta.update(
        {
            "method": (
                "CMSS-safe"
                if selected_name != "cmss"
                else str(cmss_result.metadata.get("method", "CMSS"))
            ),
            "safe_selector_enabled": True,
            "selector_choice": selected_name,
            "selector_reason": reason,
            "selector_identity_loss": id_l,
            "selector_cmss_loss": cmss_l,
            "selector_pca3_loss": pca_l if np.isfinite(pca_l) else float("nan"),
            "selector_improvement": (
                float(improvement) if np.isfinite(improvement) else float("nan")
            ),
            "selector_required_improvement": float(identity_required),
            "selector_pca3_tolerance": float(pca_tolerance),
            "selector_common_valid_fraction": float(np.mean(common_valid)),
            "selector_weighted": bool(weights is not None),
            "selector_metric_components": int(metric.metadata["components"]),
            "selector_metric_explained_variance": float(
                metric.metadata["explained_variance"]
            ),
            "selector_cmss_row_std_raw": float(
                cmss_result.metadata.get("row_dx_std", np.nan)
            ),
        }
    )
    if true_flow is not None:
        meta.update(flow_errors(selected.flow, true_flow, flow_eval_mask))
    return CorrectionResult(selected.corrected, selected.flow, meta, selected.weights)


def global_pca_translation(
    t1: np.ndarray,
    moving: np.ndarray,
    true_flow: np.ndarray | None = None,
    flow_eval_mask: np.ndarray | None = None,
    device: str = "cpu",
    seed: int = 0,
    n_components: int = 3,
    global_radius: int = 6,
) -> CorrectionResult:
    f1, f2, pmeta = joint_pca_features(t1, moving, n_components=n_components, seed=seed)
    dy, dx, score = search_global_shift(f1, f2, max_shift=global_radius, weights=None)
    apply_dy = -float(dy)
    apply_dx = -float(dx)
    row_dx = np.full(t1.shape[0], apply_dx, np.float32)
    corrected, flow = warp_with_scanline(
        moving, global_dy=apply_dy, row_dx=row_dx, device=device
    )
    meta: dict[str, Any] = {
        "method": f"PCA{n_components}_translation",
        "estimated_global_dy": float(dy),
        "estimated_global_dx": float(dx),
        "global_dy": apply_dy,
        "global_dx": apply_dx,
        "search_score": float(score),
        **pmeta,
    }
    if true_flow is not None:
        meta.update(flow_errors(flow, true_flow, flow_eval_mask))
    return CorrectionResult(corrected, flow, meta, None)


def _robust_unit_scale(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.nanpercentile(x[finite], [1, 99])
    return np.clip((x - float(lo)) / max(float(hi - lo), 1e-6), 0.0, 1.0).astype(
        np.float32
    )


def tvl1_dense_correction(
    t1: np.ndarray,
    moving: np.ndarray,
    true_flow: np.ndarray | None = None,
    flow_eval_mask: np.ndarray | None = None,
    device: str = "cpu",
    seed: int = 0,
    attachment: float = 15.0,
    tightness: float = 0.3,
    num_warp: int = 5,
    num_iter: int = 10,
    tol: float = 1e-4,
    prefilter: bool = False,
    max_abs_displacement: float | None = 9.0,
) -> CorrectionResult:
    """Classical dense TV-L1 optical flow on the first shared PCA component."""
    try:
        from skimage.registration import optical_flow_tvl1
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "tvl1_dense requires scikit-image; install requirements.txt"
        ) from exc

    f1, f2, pmeta = joint_pca_features(t1, moving, n_components=1, seed=seed)
    ref = _robust_unit_scale(f1[..., 0])
    mov = _robust_unit_scale(f2[..., 0])
    v, u = optical_flow_tvl1(
        ref,
        mov,
        attachment=float(attachment),
        tightness=float(tightness),
        num_warp=int(num_warp),
        num_iter=int(num_iter),
        tol=float(tol),
        prefilter=bool(prefilter),
        dtype=np.float32,
    )
    flow = np.stack([v, u], axis=-1).astype(np.float32)
    if max_abs_displacement is not None:
        max_abs = float(max_abs_displacement)
        flow = np.clip(flow, -max_abs, max_abs).astype(np.float32)

    h, w = ref.shape
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    corrected = sample_hwc(moving, yy + flow[..., 0], xx + flow[..., 1], device=device)
    mag = np.sqrt(np.sum(flow * flow, axis=-1))
    meta: dict[str, Any] = {
        "method": "TVL1-dense-PC1",
        "global_dy": float(np.median(flow[..., 0])),
        "row_dx_mean": float(np.mean(flow[..., 1])),
        "row_dx_std": float(np.std(flow[..., 1])),
        "flow_magnitude_median": float(np.median(mag)),
        "flow_magnitude_max": float(np.max(mag)),
        "attachment": float(attachment),
        "tightness": float(tightness),
        "num_warp": int(num_warp),
        "num_iter": int(num_iter),
        **pmeta,
    }
    if true_flow is not None:
        meta.update(flow_errors(flow, true_flow, flow_eval_mask))
    return CorrectionResult(corrected, flow, meta, None)
