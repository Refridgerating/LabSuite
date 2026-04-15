"""Shared preprocessing primitives."""

from labsuite.core.preprocessing.baseline import subtract_edge_baseline, subtract_linear_edge_baseline
from labsuite.core.preprocessing.integration import cumulative_integral, scalar_integral
from labsuite.core.preprocessing.normalization import normalize_max_abs
from labsuite.core.preprocessing.smoothing import savgol_smooth

__all__ = [
    "cumulative_integral",
    "normalize_max_abs",
    "savgol_smooth",
    "subtract_edge_baseline",
    "subtract_linear_edge_baseline",
    "scalar_integral",
]
