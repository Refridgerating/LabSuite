"""Smoothing utilities that stay generic across modalities."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def savgol_smooth(
    signal: np.ndarray,
    window_length: int,
    polyorder: int,
) -> tuple[np.ndarray, int]:
    """Apply a Savitzky-Golay smoother while resolving a valid odd window length."""

    resolved_window = _resolve_window_length(signal.size, window_length, polyorder)
    return savgol_filter(
        signal, window_length=resolved_window, polyorder=polyorder
    ), resolved_window


def _resolve_window_length(size: int, requested: int, polyorder: int) -> int:
    if size <= polyorder + 2:
        raise ValueError("Signal is too short for Savitzky-Golay smoothing")

    resolved = min(requested, size if size % 2 == 1 else size - 1)
    if resolved % 2 == 0:
        resolved -= 1

    minimum = polyorder + 2
    if minimum % 2 == 0:
        minimum += 1

    if resolved < minimum:
        resolved = minimum

    if resolved > size:
        resolved = size if size % 2 == 1 else size - 1

    if resolved <= polyorder:
        raise ValueError("Unable to resolve a valid Savitzky-Golay window length")

    return resolved
