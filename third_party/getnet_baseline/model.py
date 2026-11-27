from __future__ import annotations

import torch
import torch.nn as nn


class GETNETCompact(nn.Module):
    """Spectral-similarity CNN with adaptive pooling and an MLP head."""

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




def build_model(architecture: str, bands: int, pool_size: int = 8) -> nn.Module:
    architecture = architecture.lower()
    if architecture == "compact":
        return GETNETCompact(bands=bands, pool_size=pool_size)
    raise ValueError(f"Unknown architecture: {architecture}")
