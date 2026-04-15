"""Normalization utilities that remain independent of modality-specific science."""

from __future__ import annotations

import numpy as np


def normalize_max_abs(signal: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalize by the maximum absolute amplitude."""

    scale = float(np.max(np.abs(signal)))
    if scale == 0.0:
        return signal.copy(), 1.0
    return signal / scale, scale
