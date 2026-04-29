"""Branch-level FMR linewidth fitting."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import curve_fit

from labsuite.plugins.fmr.models import FmrModelFitSummary


def fit_linewidth_branch(
    frequency_GHz: np.ndarray,
    linewidth_mT: np.ndarray,
    *,
    gamma_over_2pi_GHz_per_T: float | None,
    min_points: int = 4,
) -> FmrModelFitSummary:
    """Fit mu0_deltaH(f) and derive alpha_eff when gamma is available."""

    frequency = np.asarray(frequency_GHz, dtype=float)
    linewidth = np.asarray(linewidth_mT, dtype=float)
    warnings: list[str] = []
    if frequency.size < min_points:
        warnings.append(f"linewidth_fit_diagnostic_insufficient_points:{frequency.size}<{min_points}")
    try:
        params, covariance = curve_fit(
            _line_width_model,
            frequency,
            linewidth,
            p0=(float(np.nanmin(linewidth)), 0.1),
            maxfev=20000,
        )
    except RuntimeError as exc:
        return FmrModelFitSummary(
            model_name="linewidth_vs_frequency_linear",
            success=False,
            message=str(exc),
            x=frequency.tolist(),
            y=linewidth.tolist(),
            warnings=warnings,
        )
    fitted = _line_width_model(frequency, *params)
    delta_h0_t = float(params[0] / 1000.0)
    slope_t_per_ghz = float(params[1] / 1000.0)
    parameters: dict[str, float | None] = {
        "DeltaH0_mT": float(params[0]),
        "mu0_deltaH0_T": delta_h0_t,
        "slope_mT_per_GHz": float(params[1]),
        "linewidth_slope_T_per_GHz": slope_t_per_ghz,
        "alpha_eff": None,
    }
    if gamma_over_2pi_GHz_per_T is not None:
        gamma_rad_per_s_t = 2.0 * math.pi * gamma_over_2pi_GHz_per_T * 1e9
        slope_t_per_hz = slope_t_per_ghz / 1e9
        parameters["alpha_eff"] = float(slope_t_per_hz * gamma_rad_per_s_t / (4.0 * math.pi))
    else:
        warnings.append("alpha_eff_requires_gamma")
    return FmrModelFitSummary(
        model_name="linewidth_vs_frequency_linear",
        success=True,
        message="fit converged",
        parameters=parameters,
        stderr=_stderr_dict(["DeltaH0_mT", "slope_mT_per_GHz"], covariance),
        metrics=_fit_metrics(linewidth, fitted, len(params)),
        x=frequency.tolist(),
        y=linewidth.tolist(),
        fitted_y=np.asarray(fitted, dtype=float).tolist(),
        warnings=warnings,
    )


def _line_width_model(
    frequency_GHz: np.ndarray, DeltaH0_mT: float, slope_mT_per_GHz: float
) -> np.ndarray:
    return DeltaH0_mT + slope_mT_per_GHz * frequency_GHz


def _stderr_dict(names: list[str], covariance: np.ndarray | None) -> dict[str, float | None]:
    if covariance is None:
        return {name: None for name in names}
    diagonal = np.sqrt(np.diag(np.asarray(covariance, dtype=float)))
    return {name: float(diagonal[index]) for index, name in enumerate(names)}


def _fit_metrics(y_true: np.ndarray, y_fit: np.ndarray, n_parameters: int) -> dict[str, float]:
    residual = y_true - y_fit
    rss = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "rss": rss,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "red_chi2": rss / max(y_true.size - n_parameters, 1),
        "r_squared": 1.0 if ss_tot == 0.0 else 1.0 - (rss / ss_tot),
    }
