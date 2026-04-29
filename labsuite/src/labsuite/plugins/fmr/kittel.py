"""Branch-level Kittel fits for FMR resonance series."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import curve_fit

from labsuite.plugins.fmr.models import FmrModelFitSummary

GAMMA_OVER_2PI_PER_G = 13.99624555


def gamma_over_2pi_from_g(g_value: float) -> float:
    return float(g_value * GAMMA_OVER_2PI_PER_G)


def g_from_gamma_over_2pi(gamma_over_2pi_GHz_per_T: float) -> float:
    return float(gamma_over_2pi_GHz_per_T / GAMMA_OVER_2PI_PER_G)


def fit_kittel_branch(
    frequency_GHz: np.ndarray,
    resonance_field_mT: np.ndarray,
    *,
    model: str,
    g_locked: float | None,
    gamma_locked_GHz_per_T: float | None,
    fit_g: bool,
    Hk_locked_mT: float | None = None,
    fit_Hk: bool = False,
) -> FmrModelFitSummary:
    """Fit one branch with a selectable Kittel model."""

    h_t = np.asarray(resonance_field_mT, dtype=float) / 1000.0
    frequency = np.asarray(frequency_GHz, dtype=float)
    normalized_model = "ip_simple" if model == "ip_field_swept_kittel" else model
    warnings: list[str] = []
    if normalized_model not in {"ip_simple", "ip_with_Hk", "oop_simple"}:
        warnings.append(f"unknown_kittel_model_using_ip_simple:{model}")
        normalized_model = "ip_simple"

    gamma_locked = gamma_locked_GHz_per_T
    if gamma_locked is None and g_locked is not None:
        gamma_locked = gamma_over_2pi_from_g(g_locked)
    floating_gamma = fit_g or gamma_locked is None
    if floating_gamma:
        warnings.append("floating_g_fit_is_diagnostic")
    if floating_gamma and fit_Hk:
        warnings.append("g_Meff_Hk_all_floating_is_weakly_constrained")

    try:
        names, p0, lower, upper, model_fn = _fit_spec(
            normalized_model,
            h_t,
            gamma_locked=gamma_locked,
            floating_gamma=floating_gamma,
            Hk_locked_mT=Hk_locked_mT,
            fit_Hk=fit_Hk,
        )
        params, covariance = curve_fit(
            model_fn,
            h_t,
            frequency,
            p0=p0,
            bounds=(lower, upper),
            maxfev=40000,
        )
    except (RuntimeError, ValueError) as exc:
        return FmrModelFitSummary(
            model_name=normalized_model,
            success=False,
            message=str(exc),
            x=resonance_field_mT.tolist(),
            y=frequency_GHz.tolist(),
            warnings=warnings,
        )

    values = {name: float(value) for name, value in zip(names, params, strict=True)}
    gamma = values.get("gamma_over_2pi_GHz_per_T", gamma_locked)
    if gamma is None:
        gamma = gamma_over_2pi_from_g(2.0)
    h_k_t = values.get("mu0_Hk_T", 0.0 if Hk_locked_mT is None else Hk_locked_mT / 1000.0)
    fitted = _kittel_frequency(
        h_t,
        gamma,
        values["mu0_Meff_T"],
        h_k_t,
        normalized_model,
    )
    g_fit = g_from_gamma_over_2pi(gamma)
    g_locked_value = None if gamma_locked is None else g_from_gamma_over_2pi(gamma_locked)
    parameters = {
        "gamma_over_2pi_GHz_per_T": float(gamma),
        "gamma_GHz_per_T": float(gamma),
        "gamma_over_2pi_locked_GHz_per_T": gamma_locked,
        "g": float(g_fit),
        "g_fit": float(g_fit) if floating_gamma else None,
        "g_locked": g_locked_value,
        "M_eff_T": float(values["mu0_Meff_T"]),
        "mu0_Meff_T": float(values["mu0_Meff_T"]),
        "mu0_Ms_apparent_T": float(values["mu0_Meff_T"]),
        "mu0_Hk_T": float(h_k_t),
    }
    if g_locked_value is not None:
        parameters["delta_g"] = float(g_fit - g_locked_value)
        parameters["percent_delta_g"] = float(100.0 * (g_fit - g_locked_value) / g_locked_value)
        parameters["delta_gamma_over_2pi"] = float(gamma - gamma_locked)
        if abs(parameters["percent_delta_g"]) > 2.0:
            warnings.append("g_fit_differs_from_locked_g_by_more_than_threshold")
    return FmrModelFitSummary(
        model_name=normalized_model + ("_floating_g" if floating_gamma else "_locked_g"),
        success=True,
        message="fit converged",
        parameters=parameters,
        stderr=_stderr_dict(names, covariance),
        metrics=_fit_metrics(frequency, fitted, len(params)),
        x=resonance_field_mT.tolist(),
        y=frequency.tolist(),
        fitted_y=np.asarray(fitted, dtype=float).tolist(),
        warnings=warnings,
    )


def _fit_spec(
    model: str,
    h_t: np.ndarray,
    *,
    gamma_locked: float | None,
    floating_gamma: bool,
    Hk_locked_mT: float | None,
    fit_Hk: bool,
):
    names: list[str] = []
    p0: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    if floating_gamma:
        names.append("gamma_over_2pi_GHz_per_T")
        p0.append(gamma_locked or gamma_over_2pi_from_g(2.0))
        lower.append(1.0)
        upper.append(80.0)
    names.append("mu0_Meff_T")
    p0.append(max(float(np.nanmax(np.abs(h_t))), 0.1))
    lower.append(-10.0 if model == "oop_simple" else 0.0)
    upper.append(10.0)
    if model == "ip_with_Hk" and fit_Hk:
        names.append("mu0_Hk_T")
        p0.append(0.0 if Hk_locked_mT is None else Hk_locked_mT / 1000.0)
        lower.append(-2.0)
        upper.append(2.0)

    def model_fn(field_t: np.ndarray, *values: float) -> np.ndarray:
        params = dict(zip(names, values, strict=True))
        gamma = params.get("gamma_over_2pi_GHz_per_T", gamma_locked)
        if gamma is None:
            gamma = gamma_over_2pi_from_g(2.0)
        h_k_t = params.get("mu0_Hk_T", 0.0 if Hk_locked_mT is None else Hk_locked_mT / 1000.0)
        return _kittel_frequency(field_t, gamma, params["mu0_Meff_T"], h_k_t, model)

    return names, tuple(p0), tuple(lower), tuple(upper), model_fn


def _kittel_frequency(
    h_t: np.ndarray,
    gamma_over_2pi_GHz_per_T: float,
    mu0_Meff_T: float,
    mu0_Hk_T: float,
    model: str,
) -> np.ndarray:
    field = np.asarray(h_t, dtype=float)
    if model == "oop_simple":
        return gamma_over_2pi_GHz_per_T * (field - mu0_Meff_T)
    shifted = field + (mu0_Hk_T if model == "ip_with_Hk" else 0.0)
    return gamma_over_2pi_GHz_per_T * np.sqrt(np.maximum(shifted * (shifted + mu0_Meff_T), 0.0))


def _stderr_dict(names: list[str], covariance: np.ndarray | None) -> dict[str, float | None]:
    if covariance is None:
        return {name: None for name in names}
    diagonal = np.sqrt(np.diag(np.asarray(covariance, dtype=float)))
    return {name: float(diagonal[index]) for index, name in enumerate(names)}


def _fit_metrics(y_true: np.ndarray, y_fit: np.ndarray, n_parameters: int) -> dict[str, float]:
    residual = y_true - y_fit
    rss = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    variance = max(rss / max(y_true.size, 1), 1e-300)
    return {
        "rss": rss,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "red_chi2": rss / max(y_true.size - n_parameters, 1),
        "r_squared": 1.0 if ss_tot == 0.0 else 1.0 - (rss / ss_tot),
        "AIC": float(y_true.size * math.log(variance) + 2 * n_parameters),
        "BIC": float(y_true.size * math.log(variance) + n_parameters * math.log(y_true.size)),
    }
