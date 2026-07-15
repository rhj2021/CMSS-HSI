from __future__ import annotations

from pathlib import Path
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter


def make_pair(h: int = 30, w: int = 36, bands: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    latent = gaussian_filter(rng.normal(size=(h, w, 3)).astype(np.float32), sigma=(2, 2, 0))
    mix = rng.normal(size=(3, bands)).astype(np.float32)
    t1 = latent @ mix + 0.04 * rng.normal(size=(h, w, bands))
    t1 = (t1 - t1.min()) / (t1.max() - t1.min() + 1e-6)
    t2 = t1 + 0.01 * rng.normal(size=t1.shape)
    gt = np.zeros((h, w), dtype=np.uint8)
    gt[7:21, 11:28] = 1
    t2[gt == 1] += np.linspace(-0.20, 0.30, bands, dtype=np.float32)
    return t1.astype(np.float32), t2.astype(np.float32), gt


def main() -> None:
    root = Path("tests/toy_data")
    t1, t2, gt = make_pair()

    (root / "River").mkdir(parents=True, exist_ok=True)
    sio.savemat(root / "River" / "river_before.mat", {"river_before": t1})
    sio.savemat(root / "River" / "river_after.mat", {"river_after": t2})
    sio.savemat(root / "River" / "groundtruth.mat", {"groundtruth": gt})

    (root / "Farmland").mkdir(parents=True, exist_ok=True)
    sio.savemat(root / "Farmland" / "farm06.mat", {"imgh": t1})
    sio.savemat(root / "Farmland" / "farm07.mat", {"imghl": t2})
    sio.savemat(root / "Farmland" / "label.mat", {"label": gt + 1})

    (root / "Hermiston").mkdir(parents=True, exist_ok=True)
    sio.savemat(
        root / "Hermiston" / "Hermiston.mat",
        {
            "hsi_t1": np.transpose(t1, (2, 0, 1)),
            "hsi_t2": np.transpose(t2, (2, 0, 1)),
            "hsi_gt": gt,
            "Imgname": np.arange(9, dtype=np.uint16).reshape(9, 1),
        },
    )


if __name__ == "__main__":
    main()
