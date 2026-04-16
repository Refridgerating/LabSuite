"""Series assembly and higher-level physics fits for FMR trace results."""

from __future__ import annotations

import math

import numpy as np
from scipy.constants import physical_constants
from scipy.optimize import curve_fit

from labsuite.core.recipes import FmrRecipe
from labsuite.plugins.fmr.models import (
    FmrModelFitSummary,
    FmrPhysicsCollectionResult,
    FmrPhysicsResult,
    FmrSeriesCollectionResult,
    FmrSeriesResult,
    FmrTraceFitResult,
)

_HBAR = physical_constants["Planck constant over 2 pi"][0]
_MU_B = physical_constants["Bohr magneton"][0]


def build_fmr_series(trace_fit_results: list[FmrTraceFitResult], *, measurement_mode: str | None = None) -> FmrSeriesCollectionResult:
    ordered = sorted(trace_fit_results, key=lambda item: (float(item.frequency_GHz), float("inf") if item.angle_deg is None else float(item.angle_deg), item.trace_id))
    first = ordered[0] if ordered else None
    sample_name = first.sample_name if first is not None else "unknown"
    angle_deg = first.angle_deg if first is not None else None
    nominal_temperature_K = first.temperature_K if first is not None else None
    grouped: dict[str, list[tuple[FmrTraceFitResult, object]]] = {"single_unassigned": [], "mode_1": [], "mode_2": []}
    mode_counts = {"single": 0, "double": 0, "partial_double": 0}
    warnings: list[str] = []
    excluded_trace_ids = [item.trace_id for item in ordered if not item.accepted]
    for item in ordered:
        if item.selected_mode == "single":
            mode_counts["single"] += 1
        elif item.selected_mode == "double":
            mode_counts["double"] += 1
            if item.partial_component_qc:
                mode_counts["partial_double"] += 1
        for component in item.selected_components:
            if component.accepted:
                grouped.setdefault(component.component_label, []).append((item, component))
    series_by_label: dict[str, FmrSeriesResult] = {}
    for label, entries in grouped.items():
        if not entries:
            continue
        series_by_label[label] = _build_single_series(label, entries, measurement_mode=measurement_mode, excluded_trace_ids=excluded_trace_ids)
    if not series_by_label:
        warnings.append("no_accepted_trace_components")
    return FmrSeriesCollectionResult(sample_name=sample_name, angle_deg=angle_deg, nominal_temperature_K=nominal_temperature_K, measurement_mode=measurement_mode, series_by_label=series_by_label, warnings=warnings, metadata={"trace_count": len(ordered), "excluded_trace_count": len(excluded_trace_ids), "mode_counts": mode_counts})


def fit_fmr_physics(series_collection: FmrSeriesCollectionResult, recipe: FmrRecipe) -> FmrPhysicsCollectionResult:
    physics_by_label: dict[str, FmrPhysicsResult] = {}
    warnings = list(series_collection.warnings)
    for label, series in series_collection.series_by_label.items():
        physics_by_label[label] = _fit_single_series_physics(series, recipe)
        warnings.extend(f"{label}:{warning}" for warning in physics_by_label[label].warnings)
    return FmrPhysicsCollectionResult(sample_name=series_collection.sample_name, angle_deg=series_collection.angle_deg, nominal_temperature_K=series_collection.nominal_temperature_K, measurement_mode=series_collection.measurement_mode, physics_by_label=physics_by_label, warnings=warnings, metadata={"series_labels": sorted(physics_by_label), "series_count": len(physics_by_label)})


def _build_single_series(series_label: str, entries: list[tuple[FmrTraceFitResult, object]], *, measurement_mode: str | None, excluded_trace_ids: list[str]) -> FmrSeriesResult:
    ordered = sorted(entries, key=lambda item: (float(item[0].frequency_GHz), item[0].trace_id))
    first_trace = ordered[0][0]
    return FmrSeriesResult(
        series_label=series_label,
        sample_name=first_trace.sample_name,
        angle_deg=first_trace.angle_deg,
        nominal_temperature_K=first_trace.temperature_K,
        measurement_mode=measurement_mode,
        frequency_GHz=np.asarray([trace.frequency_GHz for trace, _component in ordered], dtype=float),
        resonance_field_mT=np.asarray([component.H_res_mT for _trace, component in ordered], dtype=float),
        linewidth_mT=np.asarray([component.DeltaH_mT for _trace, component in ordered], dtype=float),
        amplitude_symmetric=np.asarray([component.amplitude_symmetric for _trace, component in ordered], dtype=float),
        amplitude_antisymmetric=np.asarray([component.amplitude_antisymmetric for _trace, component in ordered], dtype=float),
        resonance_field_stderr_mT=np.asarray([_stderr_or_nan(component, "H_res_mT") for _trace, component in ordered], dtype=float),
        linewidth_stderr_mT=np.asarray([_stderr_or_nan(component, "DeltaH_mT") for _trace, component in ordered], dtype=float),
        included_trace_ids=[trace.trace_id for trace, _component in ordered],
        included_component_ids=[component.component_id for _trace, component in ordered],
        excluded_trace_ids=list(excluded_trace_ids),
        warnings=[],
        metadata={"accepted_component_count": len(ordered), "accepted_trace_count": len({trace.trace_id for trace, _component in ordered})},
    )


def _fit_single_series_physics(series: FmrSeriesResult, recipe: FmrRecipe) -> FmrPhysicsResult:
    warnings = list(series.warnings)
    kittel_fit = None
    linewidth_fit = None
    derived_parameters: dict[str, float | None] = {"gamma_GHz_per_T": None, "gamma_rad_per_s_T": None, "g": None, "M_eff_mT": None, "M_eff_T": None, "alpha": None, "DeltaH0_mT": None}
    if series.frequency_GHz.size >= recipe.kittel_min_points:
        kittel_fit = _fit_kittel(series.frequency_GHz, series.resonance_field_mT)
        if kittel_fit.success:
            gamma = kittel_fit.parameters["gamma_GHz_per_T"]
            gamma_rad = 2.0 * math.pi * gamma * 1e9
            derived_parameters["gamma_GHz_per_T"] = gamma
            derived_parameters["gamma_rad_per_s_T"] = gamma_rad
            derived_parameters["g"] = gamma_rad * _HBAR / _MU_B
            derived_parameters["M_eff_T"] = kittel_fit.parameters["M_eff_T"]
            derived_parameters["M_eff_mT"] = kittel_fit.parameters["M_eff_T"] * 1_000.0
        else:
            warnings.append("kittel_fit_failed")
    else:
        warnings.append(f"kittel_fit_insufficient_points:{series.frequency_GHz.size}<{recipe.kittel_min_points}")
    if series.frequency_GHz.size >= recipe.linewidth_min_points:
        linewidth_fit = _fit_linewidth(series.frequency_GHz, series.linewidth_mT)
        if linewidth_fit.success:
            derived_parameters["DeltaH0_mT"] = linewidth_fit.parameters["DeltaH0_mT"]
            if derived_parameters["gamma_rad_per_s_T"] is not None:
                slope_t_per_hz = linewidth_fit.parameters["slope_mT_per_GHz"] * 1e-12
                derived_parameters["alpha"] = slope_t_per_hz * float(derived_parameters["gamma_rad_per_s_T"]) / (4.0 * math.pi)
            else:
                warnings.append("alpha_requires_kittel_gamma")
        else:
            warnings.append("linewidth_fit_failed")
    else:
        warnings.append(f"linewidth_fit_insufficient_points:{series.frequency_GHz.size}<{recipe.linewidth_min_points}")
    return FmrPhysicsResult(sample_name=series.sample_name, angle_deg=series.angle_deg, nominal_temperature_K=series.nominal_temperature_K, measurement_mode=series.measurement_mode, kittel_fit=kittel_fit, linewidth_fit=linewidth_fit, derived_parameters=derived_parameters, warnings=warnings, metadata={"series_label": series.series_label, "accepted_component_count": int(series.frequency_GHz.size)})


def _fit_kittel(frequency_GHz: np.ndarray, resonance_field_mT: np.ndarray) -> FmrModelFitSummary:
    resonance_field_T = np.asarray(resonance_field_mT, dtype=float) / 1_000.0
    try:
        params, covariance = curve_fit(_ip_field_swept_kittel, resonance_field_T, np.asarray(frequency_GHz, dtype=float), p0=(28.0, max(float(np.nanmax(resonance_field_T)), 0.1)), bounds=((1.0, 0.0), (80.0, 10.0)), maxfev=20000)
    except RuntimeError as exc:
        return FmrModelFitSummary(model_name="ip_field_swept_kittel", success=False, message=str(exc), x=resonance_field_mT.tolist(), y=frequency_GHz.tolist())
    fitted_y = _ip_field_swept_kittel(resonance_field_T, *params)
    return FmrModelFitSummary(model_name="ip_field_swept_kittel", success=True, message="fit converged", parameters={"gamma_GHz_per_T": float(params[0]), "M_eff_T": float(params[1])}, stderr=_stderr_dict(["gamma_GHz_per_T", "M_eff_T"], covariance), metrics=_fit_metrics(np.asarray(frequency_GHz, dtype=float), np.asarray(fitted_y, dtype=float)), x=resonance_field_mT.tolist(), y=frequency_GHz.tolist(), fitted_y=np.asarray(fitted_y, dtype=float).tolist())


def _fit_linewidth(frequency_GHz: np.ndarray, linewidth_mT: np.ndarray) -> FmrModelFitSummary:
    try:
        params, covariance = curve_fit(_line_width_model, np.asarray(frequency_GHz, dtype=float), np.asarray(linewidth_mT, dtype=float), p0=(float(np.nanmin(linewidth_mT)), 0.1), maxfev=20000)
    except RuntimeError as exc:
        return FmrModelFitSummary(model_name="linewidth_vs_frequency_linear", success=False, message=str(exc), x=frequency_GHz.tolist(), y=linewidth_mT.tolist())
    fitted_y = _line_width_model(np.asarray(frequency_GHz, dtype=float), *params)
    return FmrModelFitSummary(model_name="linewidth_vs_frequency_linear", success=True, message="fit converged", parameters={"DeltaH0_mT": float(params[0]), "slope_mT_per_GHz": float(params[1])}, stderr=_stderr_dict(["DeltaH0_mT", "slope_mT_per_GHz"], covariance), metrics=_fit_metrics(np.asarray(linewidth_mT, dtype=float), np.asarray(fitted_y, dtype=float)), x=frequency_GHz.tolist(), y=linewidth_mT.tolist(), fitted_y=np.asarray(fitted_y, dtype=float).tolist())


def _ip_field_swept_kittel(resonance_field_T: np.ndarray, gamma_GHz_per_T: float, M_eff_T: float) -> np.ndarray:
    return gamma_GHz_per_T * np.sqrt(resonance_field_T * (resonance_field_T + M_eff_T))


def _line_width_model(frequency_GHz: np.ndarray, DeltaH0_mT: float, slope_mT_per_GHz: float) -> np.ndarray:
    return DeltaH0_mT + slope_mT_per_GHz * frequency_GHz


def _stderr_or_nan(component, parameter_name: str) -> float:
    diagnostic = component.parameter_diagnostics.get(parameter_name)
    return float("nan") if diagnostic is None or diagnostic.stderr is None else float(diagnostic.stderr)


def _stderr_dict(names: list[str], covariance: np.ndarray | None) -> dict[str, float | None]:
    if covariance is None:
        return {name: None for name in names}
    diagonal = np.sqrt(np.diag(np.asarray(covariance, dtype=float)))
    return {name: float(diagonal[index]) for index, name in enumerate(names)}


def _fit_metrics(y_true: np.ndarray, y_fit: np.ndarray) -> dict[str, float]:
    residual = y_true - y_fit
    rss = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {"rss": rss, "rmse": float(np.sqrt(np.mean(residual**2))), "r_squared": 1.0 if ss_tot == 0.0 else 1.0 - (rss / ss_tot)}
