"""Explicit preprocessing for standardized FMR traces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from labsuite.core.preprocessing import savgol_smooth, subtract_linear_edge_baseline
from labsuite.core.recipes import FmrRecipe
from labsuite.plugins.fmr.models import FmrTraceDataset


@dataclass(slots=True)
class FmrProcessedTrace:
    """One FMR trace after explicit preprocessing."""

    trace_id: str
    field_mT: np.ndarray
    signal: np.ndarray
    steps: list[dict[str, object]]
    baseline_summary: dict[str, float] | None = None


def apply_fmr_preprocessing(trace: FmrTraceDataset, recipe: FmrRecipe) -> FmrProcessedTrace:
    """Apply explicit baseline subtraction and optional smoothing to one FMR trace."""

    signal = np.asarray(trace.signal, dtype=float).copy()
    steps: list[dict[str, object]] = []
    baseline_summary: dict[str, float] | None = None

    if recipe.baseline_enabled:
        corrected, _baseline_curve, baseline = subtract_linear_edge_baseline(
            trace.field_mT,
            signal,
            edge_points=recipe.baseline_edge_points,
        )
        signal = np.asarray(corrected, dtype=float)
        baseline_summary = {
            "edge_points": int(recipe.baseline_edge_points),
            "slope": float(baseline["slope"]),
            "intercept": float(baseline["intercept"]),
        }
        steps.append(
            {
                "name": "linear_edge_baseline_subtraction",
                "parameters": dict(baseline_summary),
            }
        )

    if recipe.smoothing_enabled:
        smoothed, resolved_window = savgol_smooth(
            signal,
            window_length=recipe.smoothing_window,
            polyorder=recipe.smoothing_polyorder,
        )
        signal = np.asarray(smoothed, dtype=float)
        steps.append(
            {
                "name": "savgol_smoothing",
                "parameters": {
                    "window_length": int(resolved_window),
                    "polyorder": int(recipe.smoothing_polyorder),
                },
            }
        )

    if not steps:
        steps.append({"name": "identity", "parameters": {}})

    return FmrProcessedTrace(
        trace_id=trace.trace_id,
        field_mT=np.asarray(trace.field_mT, dtype=float).copy(),
        signal=signal,
        steps=steps,
        baseline_summary=baseline_summary,
    )
