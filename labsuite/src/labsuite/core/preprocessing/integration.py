"""Integration utilities for derivative-like spectra."""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid


def cumulative_integral(field: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """Integrate a trace over field using cumulative trapezoidal integration."""

    return np.asarray(cumulative_trapezoid(signal, field, initial=0.0), dtype=float)


def scalar_integral(field: np.ndarray, signal: np.ndarray) -> float:
    """Return the scalar integral of a trace over field."""

    return float(np.trapezoid(signal, field))
