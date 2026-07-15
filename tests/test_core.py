from __future__ import annotations

import numpy as np

from cmss_grsl.corrections import (
    CorrectionResult,
    global_pca_translation,
    identity_correction,
    select_safe_cmss,
)
from cmss_grsl.metrics import boundary_f1
from cmss_grsl.transforms import apply_transform


def _toy_cube(height: int = 24, width: int = 24, bands: int = 5) -> np.ndarray:
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, height, dtype=np.float32),
        np.linspace(-1.0, 1.0, width, dtype=np.float32),
        indexing="ij",
    )
    channels = [
        xx,
        yy,
        xx * yy,
        np.exp(-4.0 * (xx**2 + yy**2)),
        np.sin(3.0 * xx) + np.cos(2.0 * yy),
    ]
    return np.stack(channels[:bands], axis=-1).astype(np.float32)


def test_boundary_f1_identical_masks() -> None:
    gt = np.zeros((20, 20), dtype=np.uint8)
    gt[5:15, 7:14] = 1
    assert boundary_f1(gt, gt, tolerance=2) == 1.0


def test_pca_translation_output() -> None:
    reference = _toy_cube()
    transformed = apply_transform(
        reference,
        {"name": "shift", "type": "translation", "dx": 1.0, "dy": 0.0},
        device="cpu",
        seed=0,
    )
    result = global_pca_translation(
        reference,
        transformed.moving,
        true_flow=transformed.forward_flow,
        device="cpu",
        seed=7,
        n_components=3,
        global_radius=2,
    )
    assert result.corrected.shape == reference.shape
    assert result.flow.shape == reference.shape[:2] + (2,)
    assert np.isfinite(result.corrected).all()


def test_safe_selector_identity() -> None:
    reference = _toy_cube()
    identity = identity_correction(reference)
    raw = CorrectionResult(identity.corrected.copy(), identity.flow.copy(), dict(identity.metadata))
    selected = select_safe_cmss(
        reference,
        reference,
        raw,
        pca_result=identity,
        seed=3,
        n_components=3,
        cfg={"enabled": True, "sample_size": 2000},
    )
    assert selected.metadata["selector_choice"] == "identity"
    np.testing.assert_allclose(selected.corrected, reference)
