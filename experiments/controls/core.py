"""Instrumented CMSS for weight and representation experiments.

Reuses the search, smoothing, resampling, IR-MAD and selector functions from
cmss.corrections. Geometry estimation uses image inputs only.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import zlib
import numpy as np
from .util import require, sha_array


@dataclass(frozen=True)
class Variant:
    name: str
    experiment: str
    representation: str = "pca3"
    weighting: str = "irmad"
    fraction: float = 0.0
    intervention_seed: int = -1
    block_size: int = 24
    selector: bool = True

    @property
    def group(self):
        if self.weighting in {"block_flip"}:
            return f"{self.weighting}_{self.fraction:g}"
        return self.name


@dataclass
class Trace:
    raw: Any
    selected: Any
    arrays: dict
    stages: list
    intervention: dict
    representation: str


def block_area_mask(domain, fraction, block_size, seed):
    """Select a fixed fraction of valid overlap in spatial blocks.

    Randomly order the blocks, filling each until the requested pixel count is
    reached. The final block uses a contiguous scan prefix. The same seed gives
    nested masks across area fractions.
    """
    domain = np.asarray(domain, bool)
    require(domain.ndim == 2, "Area intervention requires a two-dimensional domain")
    require(0 < fraction <= 1 and block_size >= 1, "Invalid fraction/block size")
    n = int(domain.sum())
    require(
        n > 0, "Empty finite first-update overlap: cannot apply an area intervention"
    )
    target = min(n, max(1, int(np.floor(float(fraction) * n + 0.5))))
    h, w = domain.shape
    blocks = [
        (y, min(y + block_size, h), x, min(x + block_size, w))
        for y in range(0, h, block_size)
        for x in range(0, w, block_size)
        if domain[y : y + block_size, x : x + block_size].any()
    ]
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(blocks))
    chosen = np.zeros_like(domain)
    remaining = target
    full = partial = used = 0
    for j in order:
        if remaining == 0:
            break
        y0, y1, x0, x1 = blocks[int(j)]
        yy, xx = np.where(domain[y0:y1, x0:x1])
        # Independent of dose and direction, so the last-block prefixes nest.
        transpose, flip_y, flip_x = rng.integers(0, 2, size=3).astype(bool)
        ykey = (y1 - y0 - 1 - yy) if flip_y else yy
        xkey = (x1 - x0 - 1 - xx) if flip_x else xx
        pos = np.lexsort((ykey, xkey)) if transpose else np.lexsort((xkey, ykey))
        take = min(remaining, len(pos))
        picked = pos[:take]
        chosen[y0 + yy[picked], x0 + xx[picked]] = True
        used += 1
        full += int(take == len(pos))
        partial += int(take < len(pos))
        remaining -= take
    require(
        remaining == 0 and int(chosen.sum()) == target,
        "Internal block-dose allocation error",
    )
    meta = dict(
        intervention_protocol="fixed_area_opposite_confidence",
        requested_fraction=float(fraction),
        block_size=int(block_size),
        mask_seed=int(seed),
        dose_domain_pixels=n,
        eligible_count=n,
        target_pixel_count=target,
        overridden_pixel_count=target,
        achieved_fraction_of_domain=target / n,
        selected_block_count=used,
        full_selected_blocks=full,
        partial_selected_blocks=partial,
        mask_sha256=sha_array(chosen),
        denominator="finite first-update geometric overlap, not predicted-change count or true FNR/FPR",
        reference="fixed spatial domain after first unperturbed uniform-weight update",
        pixel_count_rounding="nearest integer, minimum 1 for a positive dose",
        last_block_rule="contiguous scan prefix; at most one partial block",
    )
    return chosen, meta


def opposite_weight_targets(reference_weights, epsilon):
    """Freeze opposite-endpoint targets from the initial soft weights.

    Weights above (1 + epsilon)/2 map to epsilon; others map to 1. Each selected
    weight changes by at least (1 - epsilon)/2 at the first intervention.
    """
    w = np.asarray(reference_weights, np.float32)
    require(
        np.isfinite(w).all()
        and float(w.min()) >= epsilon - 1e-6
        and float(w.max()) <= 1 + 1e-6,
        "Reference soft weights must be finite and in [epsilon,1]",
    )
    return np.where(w > (1 + epsilon) / 2, epsilon, 1.0).astype(np.float32)


def inject_area_weights(weights, override, targets, epsilon, effect_tolerance=1e-7):
    """Apply fixed targets and record the affected area and weight changes."""
    w = np.asarray(weights, np.float32)
    mask = np.asarray(override, bool)
    targets = np.asarray(targets, np.float32)
    require(
        w.shape == mask.shape == targets.shape and mask.any(),
        "Empty or incompatible intervention mask/targets",
    )
    require(
        np.all(
            np.isclose(targets[mask], epsilon, atol=1e-7, rtol=0)
            | np.isclose(targets[mask], 1.0, atol=1e-7, rtol=0)
        ),
        "Frozen targets must equal epsilon or 1",
    )
    result = w.copy()
    result[mask] = targets[mask]
    diff = result.astype(np.float64) - w.astype(np.float64)
    n = int(mask.sum())
    effective = int(np.sum(mask & (np.abs(diff) > effect_tolerance)))
    effect = dict(
        selected_pixels=n,
        effective_changed_pixels=effective,
        effective_fraction_of_selected=effective / n,
        effective_weight_tolerance=effect_tolerance,
        increased_weight_pixels=int(np.sum(mask & (diff > effect_tolerance))),
        decreased_weight_pixels=int(np.sum(mask & (diff < -effect_tolerance))),
        mean_abs_weight_change_selected=float(np.abs(diff[mask]).mean()),
        min_abs_weight_change_selected=float(np.abs(diff[mask]).min()),
        max_abs_weight_change=float(np.abs(diff).max()),
        signed_weight_change_sum=float(diff[mask].sum()),
        absolute_weight_change_sum=float(np.abs(diff[mask]).sum()),
    )
    require(
        np.array_equal(result[~mask], w[~mask]),
        "Weights changed outside fixed intervention mask",
    )
    return result, effect


def matching_features(t1, current, representation, seed):
    from cmss.corrections import joint_pca_features, _normalize

    if representation in {"pca1", "pca3"}:
        return joint_pca_features(
            t1, current, n_components=int(representation[-1]), seed=seed
        )
    a, b = _normalize(t1), _normalize(current)
    require(representation == "full", f"Unknown representation {representation}")
    return (
        a.astype(np.float32),
        b.astype(np.float32),
        dict(
            components=int(a.shape[-1]),
            representation=representation,
            explained_variance=None,
            note="Separate per-date channel standardization; no PCA fit or whitening.",
        ),
    )


def estimate(t1, moving, cfg, device, variant, case_key, collect=True):
    from cmss.corrections import (
        CorrectionResult,
        search_global_shift,
        estimate_row_shifts,
        warp_with_scanline,
        flow_valid_mask,
    )
    from cmss.baselines.irmad import run_irmad

    seed = int(cfg["seed"])
    outer = int(cfg["outer_iters"])
    wc = cfg["weighting"]
    eps = float(wc["epsilon"])
    require(outer >= 1 and 0 < eps < 1, "Invalid CMSS settings")
    current = np.asarray(moving, np.float32)
    weights = np.ones(t1.shape[:2], np.float32)
    global_dy = 0.0
    row_dx = np.zeros(t1.shape[0], np.float32)
    arrays = {}
    stages = []
    history = []
    override = None
    fixed_targets = None
    intervention = {}
    for it in range(outer):
        if collect:
            arrays[f"weights_used_{it+1}"] = weights.copy()
        f1, f2, pmeta = matching_features(
            t1, current, variant.representation, seed + it
        )
        active = None if variant.weighting == "uniform" else weights
        gdy, gdx, score = search_global_shift(
            f1, f2, max_shift=int(cfg["global_radius"]), weights=active
        )
        rows = estimate_row_shifts(
            f1,
            f2,
            global_dy=gdy,
            global_dx=gdx,
            weights=active,
            residual_radius=int(cfg["row_radius"]),
            smooth_sigma=float(cfg["smooth_sigma"]),
            max_abs_row_shift=int(cfg["global_radius"]) + int(cfg["row_radius"]),
        )
        # Preserve the core sign convention and additive flow composition.
        apply_dy = -float(gdy)
        apply_rows = -rows.astype(np.float32)
        global_dy += apply_dy
        row_dx += apply_rows
        current, flow = warp_with_scanline(
            moving, global_dy=global_dy, row_dx=row_dx, device=device
        )
        h = dict(
            iter=it + 1,
            estimated_global_dy_step=float(gdy),
            estimated_global_dx_center=float(gdx),
            applied_global_dy_step=apply_dy,
            applied_row_dx_mean_step=float(np.mean(apply_rows)),
            search_score=float(score),
            row_dx_mean=float(np.mean(row_dx)),
            row_dx_std=float(np.std(row_dx)),
            **pmeta,
        )
        history.append(h)
        if variant.weighting != "uniform":
            # Match the IR-MAD arguments used by cmss_correct.
            irm = run_irmad(
                t1,
                current,
                pca_components=min(int(wc["pca_components"]), t1.shape[-1]),
                sample_size=int(wc["sample_size"]),
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
            weights = (eps + (1.0 - eps) * weights).astype(np.float32)
            clean_weights = weights.copy() if collect else weights
            if variant.weighting in {"block_flip"}:
                if override is None:
                    # Freeze the valid finite overlap after the first update.
                    domain = (
                        flow_valid_mask(flow)
                        & np.isfinite(score_map)
                        & np.all(np.isfinite(t1), axis=-1)
                        & np.all(np.isfinite(current), axis=-1)
                    )
                    ref = (score_map > float(irm.threshold)) & domain
                    mask_seed = int(
                        zlib.crc32(
                            f"{case_key}:area_block_v2:{variant.intervention_seed}".encode()
                        )
                        & 0x7FFFFFFF
                    )
                    override, intervention = block_area_mask(
                        domain, variant.fraction, variant.block_size, mask_seed
                    )
                    fixed_targets = opposite_weight_targets(weights, eps)
                    intervention.update(
                        kind=variant.weighting,
                        reference_predicted_change_count=int(ref.sum()),
                        reference_predicted_change_count_is_not_dose_denominator=True,
                        selected_reference_predicted_change_count=int(
                            (ref & override).sum()
                        ),
                        frozen_target_rule="reference w > (1+epsilon)/2 -> epsilon; otherwise -> 1",
                        target_high_weight_pixels=int(
                            np.sum(override & (fixed_targets == 1))
                        ),
                        target_low_weight_pixels=int(
                            np.sum(override & (fixed_targets != 1))
                        ),
                    )
                    if collect:
                        arrays["override_mask"] = override
                        arrays["override_targets"] = fixed_targets.copy()
                        arrays["reference_predicted_change"] = ref
                        arrays["corruption_domain"] = domain
                weights, effect = inject_area_weights(
                    weights, override, fixed_targets, eps
                )
                if it == 0:
                    intervention.update(
                        initial_effective_changed_pixels=effect[
                            "effective_changed_pixels"
                        ],
                        initial_mean_abs_weight_change_selected=effect[
                            "mean_abs_weight_change_selected"
                        ],
                        initial_absolute_weight_change_sum=effect[
                            "absolute_weight_change_sum"
                        ],
                        initial_min_abs_weight_change_selected=effect[
                            "min_abs_weight_change_selected"
                        ],
                    )
                    require(
                        effect["effective_changed_pixels"] == int(override.sum())
                        and effect["min_abs_weight_change_selected"]
                        >= (1 - eps) / 2 - 1e-6,
                        f"NO_EFFECT_INTERVENTION {case_key}/{variant.name}: first injection did not reverse every selected soft-weight decision. "
                        "This case is not labelled complete. No automatic reseeding or target substitution is performed.",
                    )
            stage = dict(
                iteration=it + 1,
                irmad_threshold=float(irm.threshold),
                irmad_scale=scale,
                next_usage=(
                    f"matching_update_{it+2}" if it + 1 < outer else "selector_only"
                ),
                weights_sha256=sha_array(weights) if collect else None,
                match_components=int(pmeta["components"]),
                match_cumulative_explained_variance=pmeta.get("explained_variance"),
            )
            if variant.weighting in {"block_flip"}:
                stage.update({"intervention_" + k: v for k, v in effect.items()})
            if collect:
                arrays[f"weights_clean_{it+1}"] = clean_weights
                arrays[f"weights_injected_{it+1}"] = weights.copy()
                arrays[f"irmad_score_{it+1}"] = score_map
        else:
            stage = dict(
                iteration=it + 1,
                next_usage=(
                    f"matching_update_{it+2}" if it + 1 < outer else "selector_only"
                ),
                match_components=int(pmeta["components"]),
                match_cumulative_explained_variance=pmeta.get("explained_variance"),
            )
        if collect:
            arrays[f"flow_update_{it+1}"] = flow.copy()
            stages.append(stage)
    if variant.weighting in {"block_flip"} and collect:
        intervention["effective_weight_updates"] = sum(
            s.get("intervention_effective_changed_pixels", 0) > 0 for s in stages
        )
        intervention["effective_matching_updates"] = sum(
            s.get("intervention_effective_changed_pixels", 0) > 0
            for s in stages
            if s["iteration"] < outer
        )
        intervention["intervention_verified"] = bool(
            intervention["initial_effective_changed_pixels"]
            == intervention["overridden_pixel_count"]
            > 0
            and intervention["effective_matching_updates"] > 0
        )
    meta = dict(
        method="CMSS" if variant.weighting != "uniform" else "ScanlineSync-unweighted",
        global_dy=float(global_dy),
        row_dx_mean=float(np.mean(row_dx)),
        row_dx_std=float(np.std(row_dx)),
        row_dx_min=float(np.min(row_dx)),
        row_dx_max=float(np.max(row_dx)),
        outer_iters=outer,
        history=history,
    )
    raw = CorrectionResult(
        current, flow, meta, weights if variant.weighting != "uniform" else None
    )
    arrays["raw_row_dx"] = row_dx.copy()
    arrays["raw_global_dy"] = np.array(global_dy, dtype=np.float64)
    arrays["flow_pre"] = flow.copy()
    if raw.weights is not None:
        arrays["final_weights"] = raw.weights.copy()
    return raw, arrays, stages, intervention


def run(t1, moving, cfg, device, variant, case_key, pca=None, collect=True):
    from cmss.corrections import select_safe_cmss, global_pca_translation

    raw, arrays, stages, intervention = estimate(
        t1, moving, cfg, device, variant, case_key, collect
    )
    if variant.selector:
        if pca is None:
            pca = global_pca_translation(
                t1,
                moving,
                device=device,
                seed=int(cfg["seed"]),
                n_components=int(cfg["pca_components"]),
                global_radius=int(cfg["global_radius"]),
            )
        selected = select_safe_cmss(
            t1,
            moving,
            raw,
            pca_result=pca,
            seed=int(cfg["seed"]),
            n_components=int(cfg["pca_components"]),
            cfg=cfg["safe_selector"],
        )
    else:
        selected = raw
    arrays["flow_post"] = selected.flow.copy()
    return Trace(raw, selected, arrays, stages, intervention, variant.representation)


def restore(t1, moving, cfg, device, meta, arrays, pca=None):
    from cmss.corrections import (
        CorrectionResult,
        warp_with_scanline,
        global_pca_translation,
    )

    cube, flow = warp_with_scanline(
        moving, float(arrays["raw_global_dy"]), arrays["raw_row_dx"], device
    )
    require(
        np.array_equal(flow, arrays["flow_pre"]),
        "Restored raw flow differs from saved flow",
    )
    require(
        sha_array(cube) == meta["raw_cube_sha256"],
        "Restored cube changed (backend/code difference). Use the recorded environment.",
    )
    raw = CorrectionResult(
        cube, flow, meta["raw_metadata"], arrays.get("final_weights")
    )
    choice = meta["selected_metadata"].get("selector_choice", "cmss")
    if not meta["variant"]["selector"] or choice == "cmss":
        selected_cube = cube
    elif choice == "identity":
        selected_cube = np.asarray(moving, np.float32)
    elif choice == "pca3_translation":
        if pca is None:
            pca = global_pca_translation(
                t1,
                moving,
                device=device,
                seed=int(cfg["seed"]),
                n_components=int(cfg["pca_components"]),
                global_radius=int(cfg["global_radius"]),
            )
        selected_cube = pca.corrected
    else:
        raise RuntimeError(f"Unexpected selector choice {choice}")
    require(
        sha_array(selected_cube) == meta["selected_cube_sha256"],
        "Restored selected cube differs",
    )
    selected = CorrectionResult(
        selected_cube, arrays["flow_post"], meta["selected_metadata"], None
    )
    return Trace(
        raw,
        selected,
        arrays,
        meta["stages"],
        meta["intervention"],
        meta["variant"]["representation"],
    )
