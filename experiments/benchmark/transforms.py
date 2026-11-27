"""Additional small residuals using the original forward-flow/warping convention.

flow[..., 0:2] = (dy, dx), and clean point p moves to p + flow(p).
The same field is the reference-to-moving sampling field used for correction.
No ground-truth change labels are accessed here.
"""

from __future__ import annotations
from typing import Any
import numpy as np
from scipy.ndimage import gaussian_filter1d
from cmss.transforms import (
    TransformResult,
    apply_transform,
    warp_image_from_forward_flow,
)
from cmss.utils import stable_hash_array, stable_hash_dict


def make_extended_flow(
    spec: dict[str, Any], height: int, width: int, seed: int
) -> np.ndarray:
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    flow = np.zeros((height, width, 2), dtype=np.float32)
    kind = spec["type"]
    if kind == "random_row":
        rng = np.random.default_rng(seed)
        sigma = float(spec.get("sigma", 24.0))
        if sigma <= 0:
            raise ValueError("random_row sigma must be positive")
        for channel, key in [(1, "amplitude_x"), (0, "amplitude_y")]:
            # Draw both vectors even for amplitude=0, to keep seeds comparable.
            row = gaussian_filter1d(rng.normal(size=height), sigma, mode="reflect")
            row -= row.mean()
            amplitude = float(spec.get(key, 0.0))
            if amplitude < 0:
                raise ValueError("Amplitudes must be nonnegative")
            row *= amplitude / max(float(np.max(np.abs(row))), 1e-12)
            flow[..., channel] = row[:, None]
        flow[..., 0] += float(spec.get("dy", 0.0))
        flow[..., 1] += float(spec.get("dx", 0.0))
        # Keep the synthetic forward map locally invertible.
        if height > 1 and np.any(1.0 + np.diff(flow[:, 0, 0]) <= 0):
            raise ValueError("Vertical row drift is not invertible; reduce amplitude")
    elif kind == "affine_extended":
        angle = np.deg2rad(float(spec.get("rotation_deg", 0.0)))
        rot = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=float,
        )
        shear = np.array(
            [
                [1.0, float(spec.get("shear_xy", 0.0))],
                [float(spec.get("shear_yx", 0.0)), 1.0],
            ],
            dtype=float,
        )
        matrix = float(spec.get("scale", 1.0)) * rot @ shear
        if np.linalg.det(matrix) <= 0:
            raise ValueError("Affine transform must preserve orientation")
        cx, cy = (width - 1) / 2, (height - 1) / 2
        x0, y0 = xx - cx, yy - cy
        flow[..., 1] = (
            matrix[0, 0] * x0 + matrix[0, 1] * y0 - x0 + float(spec.get("dx", 0))
        )
        flow[..., 0] = (
            matrix[1, 0] * x0 + matrix[1, 1] * y0 - y0 + float(spec.get("dy", 0))
        )
    else:
        raise ValueError(f"Unknown additional transform: {kind}")
    return flow


def apply_extended(
    img: np.ndarray, spec: dict[str, Any], device: str, seed: int
) -> TransformResult:
    if spec["type"] not in {"random_row", "affine_extended"}:
        return apply_transform(img, spec, device=device, seed=seed)
    h, w = img.shape[:2]
    flow = make_extended_flow(spec, h, w, seed)
    moving, _, stats = warp_image_from_forward_flow(
        img,
        flow,
        device=device,
        inverse_iterations=int(spec.get("inverse_iterations", 24)),
    )
    yy, xx = np.indices((h, w), dtype=np.float32)
    qy, qx = yy + flow[..., 0], xx + flow[..., 1]
    valid = (qy >= 0) & (qy <= h - 1) & (qx >= 0) & (qx <= w - 1)
    mag = np.linalg.norm(flow, axis=-1)
    meta = dict(
        name=spec["name"],
        spec=dict(spec),
        spec_hash=stable_hash_dict(spec),
        flow_hash=stable_hash_array(flow),
        max_displacement=float(mag.max()),
        mean_displacement=float(mag.mean()),
        valid_reference_fraction=float(valid.mean()),
        **stats,
    )
    if stats["inverse_residual_max"] > 0.02:
        raise ValueError(
            f"Synthetic inverse did not converge: {stats['inverse_residual_max']} px"
        )
    return TransformResult(spec["name"], moving, flow, valid, meta)
