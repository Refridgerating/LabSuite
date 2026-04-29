"""Multi-peak FMR spectral fitting and model selection."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares

from labsuite.core.recipes import FmrRecipe
from labsuite.core.types import ConvergenceSummary, ParameterDiagnostic, ResidualSummary
from labsuite.plugins.fmr.candidate_generation import (
    candidate_guesses_for_n,
    detect_candidate_windows,
    detect_feature,
    median_step,
)
from labsuite.plugins.fmr.models import FmrComponentFitResult, FmrTraceModelResult
from labsuite.plugins.fmr.spectral_models import (
    absorption_lorentzian_component,
    background_signal,
    derivative_lorentzian_component,
    multi_component_derivative_lorentzian,
)


def fit_candidate_models(
    field: np.ndarray,
    signal: np.ndarray,
    recipe: FmrRecipe,
) -> tuple[
    FmrTraceModelResult,
    dict[int, FmrTraceModelResult],
    str,
    float | None,
    list,
    dict[str, float | None],
]:
    """Fit candidate N-peak models and return the selected result."""

    field = np.asarray(field, dtype=float)
    signal = np.asarray(signal, dtype=float)
    windows = detect_candidate_windows(field, signal, recipe)
    max_requested = _requested_peak_count(recipe)
    single = fit_n_peak_model(field, signal, 1, recipe)
    results: dict[int, FmrTraceModelResult] = {1: single}
    residual_for_guesses = single.residual
    for n_peaks in range(2, max_requested + 1):
        results[n_peaks] = fit_n_peak_model(
            field,
            signal,
            n_peaks,
            recipe,
            residual_for_guesses=residual_for_guesses,
        )
    selected, reason = select_best_model(results, recipe)
    improvement = _improvement_ratio(results.get(1), results.get(2))
    diagnostics = {
        "single_to_double_improvement_ratio": improvement,
        "best_aic": selected.fit_aic,
        "best_bic": selected.fit_bic,
        "residual_structure_score": selected.residual_structure_score,
    }
    return selected, results, reason, improvement, windows, diagnostics


def fit_n_peak_model(
    field: np.ndarray,
    signal: np.ndarray,
    n_peaks: int,
    recipe: FmrRecipe,
    *,
    residual_for_guesses: np.ndarray | None = None,
) -> FmrTraceModelResult:
    """Fit a fixed-N mixed derivative Lorentzian model."""

    step = median_step(field)
    guesses = candidate_guesses_for_n(
        field,
        signal,
        n_peaks,
        recipe,
        residual=residual_for_guesses,
    )
    lower, upper, names = _bounds(field, signal, n_peaks, recipe)
    initial = _initial_vector(field, signal, guesses, recipe)
    initial = np.minimum(np.maximum(initial, lower + 1e-12), upper - 1e-12)

    def residual_fn(values: np.ndarray) -> np.ndarray:
        return signal - _vector_model(field, values, n_peaks, recipe.background_model)

    try:
        fit = least_squares(
            residual_fn,
            initial,
            bounds=(lower, upper),
            max_nfev=30000,
        )
        params = {name: float(value) for name, value in zip(names, fit.x, strict=True)}
        fitted = _vector_model(field, fit.x, n_peaks, recipe.background_model)
        residual = signal - fitted
        covariance = _estimate_covariance(fit, residual)
        diagnostics = _build_diagnostics(names, fit.x, covariance, lower, upper)
        success = bool(fit.success)
        message = str(fit.message)
    except ValueError as exc:
        params = {name: float(value) for name, value in zip(names, initial, strict=True)}
        fitted = _vector_model(field, initial, n_peaks, recipe.background_model)
        residual = signal - fitted
        covariance = None
        diagnostics = _build_diagnostics(names, initial, None, lower, upper)
        success = False
        message = str(exc)

    components = _components_from_parameters(
        field,
        params,
        diagnostics,
        _build_bound_hits(params, lower, upper, names),
        n_peaks,
        recipe,
        signal,
    )
    metrics = _compute_fit_metrics(signal, residual, len(names))
    bg = background_signal(
        field,
        params.get("baseline_offset", 0.0),
        params.get("baseline_slope", 0.0),
        params.get("baseline_quadratic", 0.0),
        model=recipe.background_model,
    )
    return FmrTraceModelResult(
        model_name=_model_name(n_peaks, recipe.background_model),
        success=success,
        parameters=params,
        parameter_diagnostics=diagnostics,
        convergence=ConvergenceSummary(
            success=success,
            message=message,
            nfev=None if "fit" not in locals() else int(fit.nfev),
            nvarys=len(names),
            errorbars=covariance is not None,
        ),
        residual_summary=_build_residual_summary(residual),
        metrics=metrics,
        bound_hits=_build_bound_hits(params, lower, upper, names),
        covariance=None if covariance is None else covariance.tolist(),
        fitted_signal=np.asarray(fitted, dtype=float),
        residual=np.asarray(residual, dtype=float),
        components=components,
        warnings=[],
        n_peaks=n_peaks,
        background_model=recipe.background_model,
        fit_aic=metrics["aic"],
        fit_bic=metrics["bic"],
        fit_red_chi2=metrics["reduced_chi_square"],
        residual_rms=metrics["rmse"],
        residual_structure_score=_residual_structure_score(residual, signal, step),
        background_signal=np.asarray(bg, dtype=float),
    )


def select_best_model(
    results: dict[int, FmrTraceModelResult], recipe: FmrRecipe
) -> tuple[FmrTraceModelResult, str]:
    """Select the simplest physically acceptable model that improves diagnostics."""

    if recipe.n_peaks == "1":
        return results[1], "maximum one peak was requested"
    if recipe.fit_mode == "single":
        return results[1], "single mode was explicitly requested"
    if recipe.fit_mode == "double":
        result = results.get(2) or results[1]
        return result, "double mode was explicitly requested"

    selected = results[1]
    criterion_name = recipe.multi_peak_selection
    for n_peaks in sorted(key for key in results if key > 1):
        candidate = results[n_peaks]
        if not candidate.success:
            continue
        if not _extra_components_are_physical(candidate, recipe):
            candidate.warnings.append("extra_peak_rejected_unphysical")
            continue
        if _model_improves(selected, candidate, criterion_name, recipe):
            selected = candidate
            continue
        break
    if selected.n_peaks == 1:
        if recipe.n_peaks in {"2", "3"}:
            return (
                selected,
                f"maximum {recipe.n_peaks} peaks requested; extra peaks were not justified",
            )
        return selected, "auto mode kept the single fit because extra peaks were not justified"
    if recipe.n_peaks in {"2", "3"}:
        return (
            selected,
            f"maximum {recipe.n_peaks} peaks requested; selected {selected.n_peaks} peaks",
        )
    return (
        selected,
        f"auto mode selected {selected.n_peaks} peaks using {criterion_name} model selection",
    )


def _requested_peak_count(recipe: FmrRecipe) -> int:
    if recipe.n_peaks in {"1", "2", "3"}:
        return int(recipe.n_peaks)
    if recipe.fit_mode == "single":
        return 1
    if recipe.fit_mode == "double":
        return 2
    return max(1, min(3, int(recipe.max_resonance_count)))


def _initial_vector(
    field: np.ndarray,
    signal: np.ndarray,
    guesses: list[dict[str, float]],
    recipe: FmrRecipe,
) -> np.ndarray:
    values: list[float] = []
    for guess in guesses:
        values.extend([guess["center"], guess["linewidth"], guess["sym"], guess["asym"]])
    values.extend([float(np.median(signal)), 0.0])
    if recipe.background_model == "quadratic":
        values.append(0.0)
    return np.asarray(values, dtype=float)


def _bounds(
    field: np.ndarray, signal: np.ndarray, n_peaks: int, recipe: FmrRecipe
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    step = median_step(field)
    span = abs(float(np.max(field) - np.min(field)))
    min_linewidth = max(step * 0.5, recipe.min_linewidth_mT or 1e-6)
    max_linewidth = max(
        min_linewidth * 1.01,
        recipe.max_linewidth_mT or recipe.linewidth_max_sweep_fraction * span,
    )
    lower: list[float] = []
    upper: list[float] = []
    names: list[str] = []
    for index in range(1, n_peaks + 1):
        suffix = "" if n_peaks == 1 else f"_{index}"
        names.extend(
            [
                f"H_res{suffix}_mT",
                f"DeltaH{suffix}_mT",
                f"amplitude_symmetric{suffix}",
                f"amplitude_antisymmetric{suffix}",
            ]
        )
        lower.extend([float(np.min(field)), min_linewidth, -np.inf, -np.inf])
        upper.extend([float(np.max(field)), max_linewidth, np.inf, np.inf])
    names.extend(["baseline_offset", "baseline_slope"])
    lower.extend([-np.inf, -np.inf])
    upper.extend([np.inf, np.inf])
    if recipe.background_model == "quadratic":
        names.append("baseline_quadratic")
        lower.append(-np.inf)
        upper.append(np.inf)
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float), names


def _vector_model(
    field: np.ndarray, values: np.ndarray, n_peaks: int, background_model: str
) -> np.ndarray:
    components: list[dict[str, float]] = []
    offset = n_peaks * 4
    for index in range(n_peaks):
        base = index * 4
        components.append(
            {
                "H_res_mT": float(values[base]),
                "DeltaH_mT": float(values[base + 1]),
                "amplitude_symmetric": float(values[base + 2]),
                "amplitude_antisymmetric": float(values[base + 3]),
            }
        )
    baseline_quadratic = float(values[offset + 2]) if background_model == "quadratic" else 0.0
    return multi_component_derivative_lorentzian(
        field,
        components,
        baseline_offset=float(values[offset]),
        baseline_slope=float(values[offset + 1]),
        baseline_quadratic=baseline_quadratic,
        background_model=background_model,
    )


def _components_from_parameters(
    field: np.ndarray,
    params: dict[str, float],
    diagnostics: dict[str, ParameterDiagnostic],
    hits: dict[str, bool],
    n_peaks: int,
    recipe: FmrRecipe,
    signal: np.ndarray,
) -> list[FmrComponentFitResult]:
    feature = detect_feature(field, signal, recipe.shape_pair_prominence_ratio)
    components: list[FmrComponentFitResult] = []
    for index in range(1, n_peaks + 1):
        suffix = "" if n_peaks == 1 else f"_{index}"
        label = "single_unassigned" if n_peaks == 1 else f"mode_{index}"
        h_name = f"H_res{suffix}_mT"
        linewidth_name = f"DeltaH{suffix}_mT"
        sym_name = f"amplitude_symmetric{suffix}"
        asym_name = f"amplitude_antisymmetric{suffix}"
        h_res = params[h_name]
        delta_h = params[linewidth_name]
        sym = params[sym_name]
        asym = params[asym_name]
        component_signal = derivative_lorentzian_component(field, h_res, delta_h, sym, asym)
        component = FmrComponentFitResult(
            component_id="",
            component_label=label,
            H_res_mT=h_res,
            DeltaH_mT=delta_h,
            amplitude_symmetric=sym,
            amplitude_antisymmetric=asym,
            field_mT=np.asarray(field, dtype=float).copy(),
            component_signal=np.asarray(component_signal, dtype=float),
            absorption_signal=absorption_lorentzian_component(field, h_res, delta_h, sym, asym),
            peak_index=index,
            confidence="unassigned",
            parameter_diagnostics={
                "H_res_mT": diagnostics[h_name],
                "DeltaH_mT": diagnostics[linewidth_name],
                "amplitude_symmetric": diagnostics[sym_name],
                "amplitude_antisymmetric": diagnostics[asym_name],
            },
            bound_hits={
                "H_res_mT": bool(hits.get(h_name, False)),
                "DeltaH_mT": bool(hits.get(linewidth_name, False)),
                "amplitude_symmetric": bool(hits.get(sym_name, False)),
                "amplitude_antisymmetric": bool(hits.get(asym_name, False)),
            },
            feature_center_mT=feature["feature_center_mT"] if n_peaks == 1 else h_res,
            feature_peak_to_peak_mT=feature["feature_peak_to_peak_mT"] if n_peaks == 1 else None,
            metadata={"candidate_window_label": None},
        )
        components.append(component)
    components.sort(key=lambda item: item.H_res_mT)
    for new_index, component in enumerate(components, start=1):
        component.peak_index = new_index
        if n_peaks > 1:
            component.component_label = f"mode_{new_index}"
    return components


def _model_improves(
    current: FmrTraceModelResult,
    candidate: FmrTraceModelResult,
    criterion_name: str,
    recipe: FmrRecipe,
) -> bool:
    improvement = _improvement_ratio(current, candidate)
    if improvement is None or improvement < recipe.double_fit_min_improvement_ratio:
        return False
    if criterion_name == "residual":
        current_structure = current.residual_structure_score or math.inf
        candidate_structure = candidate.residual_structure_score or math.inf
        return candidate_structure <= current_structure * 0.65
    key = "fit_aic" if criterion_name == "aic" else "fit_bic"
    current_value = getattr(current, key)
    candidate_value = getattr(candidate, key)
    if current_value is None or candidate_value is None:
        return False
    current_structure = current.residual_structure_score or math.inf
    candidate_structure = candidate.residual_structure_score or math.inf
    return (
        float(current_value) - float(candidate_value) >= 10.0
        and candidate_structure <= current_structure * 0.65
    )


def _extra_components_are_physical(result: FmrTraceModelResult, recipe: FmrRecipe) -> bool:
    components = sorted(result.components, key=lambda item: item.H_res_mT)
    for component in components:
        if recipe.min_linewidth_mT is not None and component.DeltaH_mT < recipe.min_linewidth_mT:
            component.rejection_reason = "linewidth_below_minimum"
            return False
        if recipe.max_linewidth_mT is not None and component.DeltaH_mT > recipe.max_linewidth_mT:
            component.rejection_reason = "linewidth_above_maximum"
            return False
    for left, right in zip(components, components[1:], strict=False):
        if abs(right.H_res_mT - left.H_res_mT) < recipe.min_peak_separation_mT:
            right.rejection_reason = "peak_separation_below_minimum"
            return False
    return True


def _improvement_ratio(
    current: FmrTraceModelResult | None, candidate: FmrTraceModelResult | None
) -> float | None:
    if current is None or candidate is None:
        return None
    current_ss = current.metrics.get("sum_squared_residuals")
    candidate_ss = candidate.metrics.get("sum_squared_residuals")
    if current_ss is None or candidate_ss is None or current_ss <= 0.0:
        return None
    return max(0.0, float((current_ss - candidate_ss) / current_ss))


def _compute_fit_metrics(
    signal: np.ndarray, residual: np.ndarray, n_parameters: int
) -> dict[str, float]:
    n_points = max(int(signal.size), 1)
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((signal - np.mean(signal)) ** 2))
    variance = max(ss_res / n_points, 1e-300)
    return {
        "chi_square": ss_res,
        "reduced_chi_square": ss_res / max(n_points - n_parameters, 1),
        "r_squared": 1.0 if ss_tot == 0.0 else 1.0 - (ss_res / ss_tot),
        "sum_squared_residuals": ss_res,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "aic": float(n_points * math.log(variance) + 2 * n_parameters),
        "bic": float(n_points * math.log(variance) + n_parameters * math.log(n_points)),
    }


def _build_residual_summary(residual: np.ndarray) -> ResidualSummary:
    return ResidualSummary(
        rss=float(np.sum(residual**2)),
        rmse=float(np.sqrt(np.mean(residual**2))),
        mae=float(np.mean(np.abs(residual))),
        max_abs=float(np.max(np.abs(residual))),
        mean=float(np.mean(residual)),
        std=float(np.std(residual)),
    )


def _residual_structure_score(residual: np.ndarray, signal: np.ndarray, step: float) -> float:
    if residual.size < 5:
        return 0.0
    kernel_width = max(3, min(9, int(round(3.0 / max(step, 1e-9)))))
    if kernel_width % 2 == 0:
        kernel_width += 1
    kernel = np.ones(kernel_width, dtype=float) / kernel_width
    smoothed = np.convolve(residual, kernel, mode="same")
    signal_scale = max(float(np.max(np.abs(signal))), 1e-12)
    return float(np.max(np.abs(smoothed)) / signal_scale)


def _estimate_covariance(fit, residual: np.ndarray) -> np.ndarray | None:
    if fit.jac is None:
        return None
    jac = np.asarray(fit.jac, dtype=float)
    if jac.size == 0 or jac.shape[0] <= jac.shape[1]:
        return None
    try:
        _, singular_values, vt = np.linalg.svd(jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(jac.shape) * singular_values[0]
        singular_values = singular_values[singular_values > threshold]
        vt = vt[: singular_values.size]
        covariance = (vt.T / singular_values**2) @ vt
        covariance *= float(np.sum(residual**2) / max(jac.shape[0] - jac.shape[1], 1))
        return np.asarray(covariance, dtype=float)
    except np.linalg.LinAlgError:
        return None


def _build_diagnostics(
    names: list[str],
    values: np.ndarray,
    covariance: np.ndarray | None,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, ParameterDiagnostic]:
    stderr = None if covariance is None else np.sqrt(np.diag(covariance))
    diagnostics: dict[str, ParameterDiagnostic] = {}
    for index, name in enumerate(names):
        value = float(values[index])
        err = None if stderr is None else float(stderr[index])
        rel = None if err is None or value == 0.0 else float(abs(err / value))
        min_bound = None if np.isneginf(lower[index]) else float(lower[index])
        max_bound = None if np.isposinf(upper[index]) else float(upper[index])
        diagnostics[name] = ParameterDiagnostic(
            value=value,
            stderr=err,
            relative_stderr=rel,
            stderr_missing=err is None,
            min_bound=min_bound,
            max_bound=max_bound,
            hit_min_bound=_is_at_bound(value, min_bound),
            hit_max_bound=_is_at_bound(value, max_bound),
        )
    return diagnostics


def _build_bound_hits(
    params: dict[str, float], lower: np.ndarray, upper: np.ndarray, names: list[str]
) -> dict[str, bool]:
    hits: dict[str, bool] = {}
    for index, name in enumerate(names):
        hits[name] = _is_at_bound(params[name], float(lower[index])) or _is_at_bound(
            params[name], float(upper[index])
        )
    return hits


def _is_at_bound(value: float, bound: float | None) -> bool:
    if bound is None or np.isinf(bound):
        return False
    return abs(value - bound) <= max(1e-9, abs(bound) * 1e-6)


def _model_name(n_peaks: int, background_model: str) -> str:
    if n_peaks == 1:
        return f"mixed_derivative_lorentzian_{background_model}_background"
    return f"{n_peaks}_peak_mixed_derivative_lorentzian_{background_model}_background"
