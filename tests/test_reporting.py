from __future__ import annotations

import numpy as np

from cmss_grsl.reporting import finite_pair_rate, finite_rate


def test_finite_rate_ignores_nan() -> None:
    values = [1.0, 0.5, np.nan, -0.2]
    assert finite_rate(values, lambda a: a > 0) == 2 / 3


def test_finite_pair_rate_uses_shared_valid_rows() -> None:
    left = [1.0, np.nan, 0.5, -0.2]
    right = [0.0, 0.0, np.nan, -0.3]
    assert finite_pair_rate(left, right, lambda a, b: a > b) == 1.0
