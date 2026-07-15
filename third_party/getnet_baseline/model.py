from __future__ import annotations

import torch
import torch.nn as nn


class GETNETCompact(nn.Module):
    """Memory-safe GETNET variant: 2D CNN over spectral similarity plane.

    The original Keras code flattened the full BxB feature map, which becomes very
    large for 200+ bands. This compact variant keeps GETNET's spectral-similarity
    2D CNN core and adds adaptive pooling before the MLP head.
    """

    def __init__(self, bands: int, num_classes: int = 2, pool_size: int = 8):
        super().__init__()
        self.bands = bands
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.AdaptiveAvgPool2d((pool_size, pool_size)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * pool_size * pool_size, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.Tanh(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


class GETNETLegacyFlatten(nn.Module):
    """Closer PyTorch translation of the original GETNET Keras head.

    Warning: parameter count grows with B^2 and can be very large for 200+ bands.
    """

    def __init__(self, bands: int, num_classes: int = 2):
        super().__init__()
        if bands < 5:
            raise ValueError("GETNET needs at least 5 bands for two 3x3 convolutions")
        self.bands = bands
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.Tanh(),
        )
        flat = 64 * (bands - 4) * (bands - 4)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.Tanh(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def build_model(architecture: str, bands: int, pool_size: int = 8) -> nn.Module:
    architecture = architecture.lower()
    if architecture == "compact":
        return GETNETCompact(bands=bands, pool_size=pool_size)
    if architecture == "legacy":
        return GETNETLegacyFlatten(bands=bands)
    raise ValueError(f"Unknown architecture: {architecture}")
