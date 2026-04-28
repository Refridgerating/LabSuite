"""Baseline utilities that are modality-agnostic."""

from __future__ import annotations

import numpy as np


def subtract_linear_edge_baseline(
    field: np.ndarray,
    signal: np.ndarray,
    edge_points: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Subtract a linear baseline estimated from the trace edges."""

    if signal.size < edge_points * 2:
        raise ValueError("Signal is too short for the requested edge baseline window")

    edge_field = np.concatenate((field[:edge_points], field[-edge_points:]))
    edge_signal = np.concatenate((signal[:edge_points], signal[-edge_points:]))
    slope, intercept = np.polyfit(edge_field, edge_signal, deg=1)
    baseline = slope * field + intercept
    corrected = signal - baseline
    return (
        corrected,
        baseline,
        {"edge_points": float(edge_points), "slope": float(slope), "intercept": float(intercept)},
    )


def subtract_edge_baseline(signal: np.ndarray, edge_points: int) -> tuple[np.ndarray, float]:
    """Subtract the mean of the first and last edge windows."""

    if signal.size < edge_points * 2:
        raise ValueError("Signal is too short for the requested edge baseline window")

    baseline_window = np.concatenate((signal[:edge_points], signal[-edge_points:]))
    baseline = float(np.mean(baseline_window))
    return signal - baseline, baseline
