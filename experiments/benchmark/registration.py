"""Registration comparisons using shared PC1 features.

Phase correlation and ECC use a common percentile mapping. RoMa receives PC1
replicated across RGB channels, then an affine RANSAC model is fitted. Numerical
registration failures return identity with a failed_identity status; missing
packages or weights raise an error.
"""

from __future__ import annotations
from functools import lru_cache
from typing import Any
import time
import numpy as np
import torch
from cmss.corrections import (
    CorrectionResult,
    joint_pca_features,
    identity_correction,
)
from cmss.transforms import sample_hwc


def cv_module():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Install the dependencies in requirements.txt for OpenCV baselines"
        ) from exc
    return cv2


def shared_pc1(
    t1: np.ndarray, moving: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    a, b, _ = joint_pca_features(t1, moving, n_components=1, seed=seed)
    a, b = a[..., 0], b[..., 0]
    low, high = np.percentile(np.concatenate([a.ravel(), b.ravel()]), [1, 99])
    scale = max(float(high - low), 1e-6)
    return tuple(
        np.ascontiguousarray(np.clip((x - low) / scale, 0, 1), dtype=np.float32)
        for x in (a, b)
    )


def affine_flow(matrix: np.ndarray, height: int, width: int) -> np.ndarray:
    """matrix maps reference (x,y,1) to a moving sampling location."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (2, 3) or not np.isfinite(m).all():
        raise ValueError("Expected finite 2x3 reference-to-moving matrix")
    yy, xx = np.indices((height, width), dtype=np.float32)
    qx = m[0, 0] * xx + m[0, 1] * yy + m[0, 2]
    qy = m[1, 0] * xx + m[1, 1] * yy + m[1, 2]
    return np.stack([qy - yy, qx - xx], axis=-1).astype(np.float32)


def apply_matrix(
    moving: np.ndarray, matrix: np.ndarray, device: str, metadata: dict[str, Any]
) -> CorrectionResult:
    h, w = moving.shape[:2]
    flow = affine_flow(matrix, h, w)
    yy, xx = np.indices((h, w), dtype=np.float32)
    corrected = sample_hwc(moving, yy + flow[..., 0], xx + flow[..., 1], device=device)
    return CorrectionResult(
        corrected,
        flow,
        {
            **metadata,
            "matrix_reference_to_moving_xy": np.asarray(matrix).tolist(),
            "registration_status": "ok",
        },
    )


def failed(moving: np.ndarray, method: str, reason: str) -> CorrectionResult:
    r = identity_correction(moving)
    r.metadata.update(
        method=method, registration_status="failed_identity", failure_reason=reason
    )
    return r


def phase_estimate(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    cv2 = cv_module()
    h, w = a.shape
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    # Copies protect against OpenCV implementations that apply windows in place.
    (dx, dy), response = cv2.phaseCorrelate(a.copy(), b.copy(), window)
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], np.float32), float(response)


def classical(
    t1: np.ndarray,
    moving: np.ndarray,
    method: str,
    device: str,
    seed: int,
    cfg: dict[str, Any],
) -> CorrectionResult:
    cv2 = cv_module()
    a, b = shared_pc1(t1, moving, seed)
    if float(np.std(a)) < 1e-7 or float(np.std(b)) < 1e-7:
        return failed(moving, method, "constant_registration_image")
    matrix, response = phase_estimate(a, b)
    if not np.isfinite(matrix).all():
        return failed(moving, method, "nonfinite_phase_correlation")
    if method == "phase_correlation":
        return apply_matrix(
            moving, matrix, device, dict(method=method, phase_response=response)
        )
    if method != "ecc_affine":
        raise ValueError(method)
    motion = cv2.MOTION_AFFINE
    criteria = (
        cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
        int(cfg.get("ecc_max_iterations", 200)),
        float(cfg.get("ecc_epsilon", 1e-6)),
    )
    candidates, errors = [], []
    # A fixed two-start policy prevents a poor phase initialization from silently
    # crippling ECC on changed scenes. Both starts are image-only and both are timed.
    starts = [np.eye(2, 3, dtype=np.float32), matrix]
    for init in starts:
        try:
            rho, warp = cv2.findTransformECC(
                a,
                b,
                init.copy(),
                motion,
                criteria,
                None,
                int(cfg.get("ecc_gaussian_size", 5)),
            )
            if np.isfinite(rho) and np.isfinite(warp).all():
                candidates.append((float(rho), warp.copy()))
        except cv2.error as exc:
            errors.append(str(exc).splitlines()[-1])
    if not candidates:
        return failed(moving, method, "ECC nonconvergence: " + "; ".join(errors))
    rho, warp = max(candidates, key=lambda x: x[0])
    return apply_matrix(
        moving,
        warp,
        device,
        dict(
            method=method,
            ecc_score=rho,
            phase_response=response,
            ecc_successful_starts=len(candidates),
            ecc_attempted_starts=len(starts),
        ),
    )


@lru_cache(maxsize=2)
def roma_model(device: str):
    try:
        from romatch import roma_outdoor
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "RoMa is unavailable. Install requirements.txt with compatible "
            "PyTorch and torchvision builds."
        ) from exc
    start = time.perf_counter()
    # This uses the official published architecture/checkpoint; no HSI fine-tuning.
    model = roma_outdoor(device=torch.device(device))
    model.eval()
    return model, time.perf_counter() - start


def learned_roma(
    t1: np.ndarray, moving: np.ndarray, device: str, seed: int, cfg: dict[str, Any]
) -> CorrectionResult:
    from PIL import Image

    cv2 = cv_module()
    model, load_seconds = roma_model(device)
    a, b = shared_pc1(t1, moving, seed)
    # RoMa accepts PIL images; avoid temporary-image file I/O in the timed path.
    ia, ib = [
        Image.fromarray(
            np.repeat(np.round(x * 255).astype(np.uint8)[..., None], 3, axis=-1)
        )
        for x in (a, b)
    ]
    h, w = a.shape
    cv2.setRNGSeed(int(seed) % (2**31 - 1))
    # fork_rng prevents the baseline from changing subsequent detector RNG states.
    devices = [torch.device(device).index or 0] if device.startswith("cuda") else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        with torch.inference_mode():
            warp, certainty = model.match(ia, ib, device=torch.device(device))
            matches, _ = model.sample(
                warp, certainty, num=int(cfg.get("roma_matches", 5000))
            )
            p, q = model.to_pixel_coordinates(matches, h, w, h, w)
    # Official RoMa returns edge-origin pixel coordinates (centers at 0.5).
    # This repository/OpenCV uses zero-based pixel centers. Convert BOTH dates.
    p, q = p.detach().cpu().numpy() - 0.5, q.detach().cpu().numpy() - 0.5
    valid = np.isfinite(p).all(1) & np.isfinite(q).all(1)
    p, q = np.asarray(p[valid], np.float32), np.asarray(q[valid], np.float32)
    if len(p) < 3:
        return failed(moving, "roma_affine", f"Only {len(p)} finite matches")
    matrix, inliers = cv2.estimateAffine2D(
        p,
        q,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(cfg.get("ransac_reprojection_px", 2.0)),
        maxIters=int(cfg.get("ransac_iterations", 10000)),
        confidence=float(cfg.get("ransac_confidence", 0.999)),
        refineIters=10,
    )
    if (
        matrix is None
        or not np.isfinite(matrix).all()
        or inliers is None
        or int(inliers.sum()) < 3
    ):
        return failed(moving, "roma_affine", "RANSAC failed to obtain an affine model")
    return apply_matrix(
        moving,
        matrix,
        device,
        dict(
            method="RoMa-outdoor + RANSAC affine",
            input_representation="shared_PC1_replicated_RGB",
            matches=int(len(p)),
            inliers=int(inliers.sum()),
            roma_model_load_seconds=load_seconds,
            pretrained=True,
            hsi_finetuned=False,
        ),
    )
