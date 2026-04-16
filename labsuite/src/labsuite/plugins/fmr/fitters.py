"""Trace fitting and QC helpers for field-swept FMR data."""

from __future__ import annotations

import math

import numpy as np
from lmfit import Model
from scipy.signal import find_peaks

from labsuite.core.recipes import FmrRecipe
from labsuite.core.types import ConvergenceSummary, ParameterDiagnostic, ResidualSummary
from labsuite.plugins.fmr.models import (
    FmrCandidateWindow,
    FmrComponentFitResult,
    FmrTraceDataset,
    FmrTraceFitResult,
    FmrTraceModelResult,
)
from labsuite.plugins.fmr.preprocess import FmrProcessedTrace


def mixed_derivative_lorentzian(field_mT: np.ndarray, H_res_mT: float, DeltaH_mT: float, amplitude_symmetric: float, amplitude_antisymmetric: float, baseline_offset: float, baseline_slope: float) -> np.ndarray:
    return _component(field_mT, H_res_mT, DeltaH_mT, amplitude_symmetric, amplitude_antisymmetric) + baseline_offset + baseline_slope * field_mT


def double_mixed_derivative_lorentzian(field_mT: np.ndarray, H_res_1_mT: float, DeltaH_1_mT: float, amplitude_symmetric_1: float, amplitude_antisymmetric_1: float, H_res_2_mT: float, DeltaH_2_mT: float, amplitude_symmetric_2: float, amplitude_antisymmetric_2: float, baseline_offset: float, baseline_slope: float) -> np.ndarray:
    return _component(field_mT, H_res_1_mT, DeltaH_1_mT, amplitude_symmetric_1, amplitude_antisymmetric_1) + _component(field_mT, H_res_2_mT, DeltaH_2_mT, amplitude_symmetric_2, amplitude_antisymmetric_2) + baseline_offset + baseline_slope * field_mT


def fit_fmr_trace(raw_trace: FmrTraceDataset, processed_trace: FmrProcessedTrace, recipe: FmrRecipe) -> FmrTraceFitResult:
    field = np.asarray(processed_trace.field_mT, dtype=float)
    signal = np.asarray(processed_trace.signal, dtype=float)
    trace_feature = _detect_feature(field, signal, recipe.shape_pair_prominence_ratio)
    candidate_windows = detect_candidate_windows(field, signal, recipe)
    single_fit = _fit_single(field, signal, recipe, trace_feature)
    double_fit = _fit_double(field, signal, candidate_windows, recipe)
    selected_mode, selected_fit, selection_reason, improvement_ratio = _select_mode(recipe.fit_mode, single_fit, double_fit, candidate_windows, recipe.double_fit_min_improvement_ratio)
    selected_components = [_clone_component(component, raw_trace.trace_id) for component in selected_fit.components]
    return FmrTraceFitResult(
        trace_id=raw_trace.trace_id,
        source_file=raw_trace.source_file,
        sample_name=raw_trace.sample_name,
        frequency_GHz=raw_trace.frequency_GHz,
        angle_deg=raw_trace.angle_deg,
        temperature_K=raw_trace.temperature_K,
        model_name=selected_fit.model_name,
        signal_channel=str(raw_trace.metadata.get("selected_signal_channel", recipe.signal_channel)),
        field_mT=field.copy(),
        processed_signal=signal.copy(),
        fitted_signal=np.asarray(selected_fit.fitted_signal, dtype=float),
        residual=np.asarray(selected_fit.residual, dtype=float),
        parameters=dict(selected_fit.parameters),
        parameter_diagnostics=dict(selected_fit.parameter_diagnostics),
        convergence=selected_fit.convergence,
        residual_summary=selected_fit.residual_summary,
        metrics=dict(selected_fit.metrics),
        bound_hits=dict(selected_fit.bound_hits),
        covariance=None if selected_fit.covariance is None else [list(row) for row in selected_fit.covariance],
        success=bool(selected_fit.success),
        accepted=False,
        rejection_reason=None,
        requested_mode=recipe.fit_mode,
        selected_mode=selected_mode,
        selection_reason=selection_reason,
        candidate_window_count=len(candidate_windows),
        double_fit_improvement_ratio=improvement_ratio,
        double_fit_threshold=recipe.double_fit_min_improvement_ratio,
        candidate_windows=candidate_windows,
        single_fit=single_fit,
        double_fit=double_fit,
        selected_components=selected_components,
        partial_component_qc=False,
        signal_max_abs=None,
        residual_rmse_fraction=None,
        amplitude_snr=None,
        feature_center_mT=trace_feature["feature_center_mT"],
        feature_peak_to_peak_mT=trace_feature["feature_peak_to_peak_mT"],
        center_feature_disagreement_mT=None,
        critical_bound_hit_names=[],
        acceptance_checks={},
        warnings=[],
        preprocessing_steps=processed_trace.steps,
        baseline_summary=processed_trace.baseline_summary,
        metadata={**dict(raw_trace.metadata), "feature_positive_extremum_mT": trace_feature["positive_extremum_mT"], "feature_negative_extremum_mT": trace_feature["negative_extremum_mT"]},
    )


def detect_candidate_windows(field: np.ndarray, signal: np.ndarray, recipe: FmrRecipe) -> list[FmrCandidateWindow]:
    field = np.asarray(field, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if field.size < 3 or signal.size < 3:
        return []
    step = _median_step(field)
    dist = max(1, int(round(recipe.peak_min_distance_mT / max(step, 1e-9))))
    prom = max(float(np.max(np.abs(signal))) * recipe.peak_min_prominence_ratio, 1e-9)
    pos_i, pos_p = find_peaks(signal, prominence=prom, distance=dist)
    neg_i, neg_p = find_peaks(-signal, prominence=prom, distance=dist)
    if pos_i.size == 0 or neg_i.size == 0:
        return []
    candidates: dict[tuple[int, int], FmrCandidateWindow] = {}
    for source_i, source_p, target_i, target_p, source_kind in ((pos_i, pos_p["prominences"], neg_i, neg_p["prominences"], "positive"), (neg_i, neg_p["prominences"], pos_i, pos_p["prominences"], "negative")):
        for idx, prominence in zip(source_i, source_p, strict=True):
            nearest = int(np.argmin(np.abs(field[target_i] - field[idx])))
            other = int(target_i[nearest])
            width = abs(float(field[other] - field[idx]))
            if width < recipe.peak_min_pair_width_mT:
                continue
            peak_idx, trough_idx = (int(idx), other) if source_kind == "positive" else (other, int(idx))
            left, right = min(peak_idx, trough_idx), max(peak_idx, trough_idx)
            padding = max(4, int(round((width * recipe.candidate_window_padding_width_multiplier) / max(step, 1e-9))))
            start = max(0, left - padding)
            end = min(signal.size - 1, right + padding)
            key = (left, right)
            item = FmrCandidateWindow("", start, end, float(field[start]), float(field[end]), peak_idx, trough_idx, float(field[peak_idx]), float(field[trough_idx]), width, float(min(prominence, target_p[nearest])), float((field[peak_idx] + field[trough_idx]) / 2.0))
            prev = candidates.get(key)
            if prev is None or item.prominence > prev.prominence:
                candidates[key] = item
    ranked = sorted(candidates.values(), key=lambda item: item.prominence, reverse=True)
    selected: list[FmrCandidateWindow] = []
    for item in ranked:
        overlaps = any(not (item.end_index < current.start_index or item.start_index > current.end_index) for current in selected)
        if overlaps:
            continue
        selected.append(item)
        if len(selected) == recipe.max_resonance_count:
            break
    selected.sort(key=lambda item: item.candidate_center_mT)
    return [FmrCandidateWindow(f"candidate_{index}", item.start_index, item.end_index, item.start_field_mT, item.end_field_mT, item.peak_index, item.trough_index, item.peak_field_mT, item.trough_field_mT, item.width_mT, item.prominence, item.candidate_center_mT) for index, item in enumerate(selected, start=1)]


def assess_trace_fit_quality(fit_result: FmrTraceFitResult, *, recipe: FmrRecipe) -> tuple[bool, str | None, list[str]]:
    warnings = list(fit_result.warnings)
    signal_max_abs = float(np.max(np.abs(fit_result.processed_signal))) if fit_result.processed_signal.size else 0.0
    rmse = fit_result.residual_summary.rmse
    residual_rmse_fraction = 0.0 if signal_max_abs <= 1e-12 and rmse <= 1e-12 else (float("inf") if signal_max_abs <= 1e-12 else float(rmse / signal_max_abs))
    residual_std = fit_result.residual_summary.std
    strongest = max((max(abs(component.amplitude_symmetric), abs(component.amplitude_antisymmetric)) for component in fit_result.selected_components), default=0.0)
    fit_result.signal_max_abs = signal_max_abs
    fit_result.residual_rmse_fraction = residual_rmse_fraction
    fit_result.amplitude_snr = float("inf") if residual_std <= 1e-12 else float(strongest / residual_std)
    fit_result.acceptance_checks = {"fit_converged": bool(fit_result.convergence.success and fit_result.success), "has_selected_components": bool(fit_result.selected_components)}
    if not fit_result.acceptance_checks["fit_converged"]:
        fit_result.accepted = False
        fit_result.rejection_reason = "fit_did_not_converge"
        return False, fit_result.rejection_reason, warnings
    if not fit_result.acceptance_checks["has_selected_components"]:
        fit_result.accepted = False
        fit_result.rejection_reason = "selected_fit_has_no_components"
        return False, fit_result.rejection_reason, warnings
    critical_hits: list[str] = []
    reasons: list[str] = []
    accepted_components = 0
    for component in fit_result.selected_components:
        accepted, reason, component_warnings = _assess_component(component, fit_result.field_mT, signal_max_abs, residual_rmse_fraction, residual_std, recipe)
        component.accepted = accepted
        component.rejection_reason = reason
        component.warnings.extend(component_warnings)
        warnings.extend(f"{component.component_label}:{warning}" for warning in component_warnings)
        critical_hits.extend(f"{component.component_label}:{name}" for name in component.critical_bound_hit_names)
        if accepted:
            accepted_components += 1
        elif reason is not None:
            reasons.append(f"{component.component_label}={reason}")
    fit_result.critical_bound_hit_names = critical_hits
    fit_result.partial_component_qc = fit_result.selected_mode == "double" and 0 < accepted_components < len(fit_result.selected_components)
    if fit_result.partial_component_qc:
        warnings.append("partial_double_fit:" + "|".join(component.component_label for component in fit_result.selected_components if not component.accepted))
    fit_result.accepted = accepted_components > 0
    fit_result.rejection_reason = None if fit_result.accepted else ("all_components_rejected:" + "|".join(reasons) if reasons else "no_components_accepted")
    return fit_result.accepted, fit_result.rejection_reason, warnings


def _fit_single(field: np.ndarray, signal: np.ndarray, recipe: FmrRecipe, trace_feature: dict[str, float | None]) -> FmrTraceModelResult:
    center, linewidth, asym_s, asym_a = _single_guesses(field, signal, recipe)
    step = _median_step(field)
    max_linewidth = max(step * 2.0, recipe.linewidth_max_sweep_fraction * abs(float(field[-1] - field[0])))
    model = Model(mixed_derivative_lorentzian, independent_vars=["field_mT"])
    params = model.make_params(H_res_mT=center, DeltaH_mT=linewidth, amplitude_symmetric=asym_s, amplitude_antisymmetric=asym_a, baseline_offset=float(np.median(signal)), baseline_slope=0.0)
    params["H_res_mT"].set(min=float(np.min(field)), max=float(np.max(field)))
    params["DeltaH_mT"].set(min=max(step * 0.5, 1e-6), max=max_linewidth)
    fit = model.fit(signal, params, field_mT=field)
    diagnostics = _build_parameter_diagnostics(fit.params)
    hits = _build_bound_hits(fit.params)
    component = _component_result("single_unassigned", field, fit.params, diagnostics, hits, {"H_res_mT": "H_res_mT", "DeltaH_mT": "DeltaH_mT", "amplitude_symmetric": "amplitude_symmetric", "amplitude_antisymmetric": "amplitude_antisymmetric"}, trace_feature, {"candidate_window_label": None})
    return _model_result(fit, "mixed_derivative_lorentzian", [component])


def _fit_double(field: np.ndarray, signal: np.ndarray, candidate_windows: list[FmrCandidateWindow], recipe: FmrRecipe) -> FmrTraceModelResult | None:
    if len(candidate_windows) < 2:
        return None
    first, second = candidate_windows[:2]
    step = _median_step(field)
    max_linewidth = max(step * 2.0, recipe.linewidth_max_sweep_fraction * abs(float(field[-1] - field[0])))
    guess_1 = _window_guess(field, signal, first, step)
    guess_2 = _window_guess(field, signal, second, step)
    model = Model(double_mixed_derivative_lorentzian, independent_vars=["field_mT"])
    params = model.make_params(H_res_1_mT=guess_1["center"], DeltaH_1_mT=guess_1["linewidth"], amplitude_symmetric_1=guess_1["sym"], amplitude_antisymmetric_1=guess_1["asym"], H_res_2_mT=guess_2["center"], DeltaH_2_mT=guess_2["linewidth"], amplitude_symmetric_2=guess_2["sym"], amplitude_antisymmetric_2=guess_2["asym"], baseline_offset=float(np.median(signal)), baseline_slope=0.0)
    params["H_res_1_mT"].set(min=max(float(np.min(field)), first.start_field_mT), max=min(float(np.max(field)), first.end_field_mT))
    params["H_res_2_mT"].set(min=max(float(np.min(field)), second.start_field_mT), max=min(float(np.max(field)), second.end_field_mT))
    params["DeltaH_1_mT"].set(min=max(step * 0.5, 1e-6), max=max_linewidth)
    params["DeltaH_2_mT"].set(min=max(step * 0.5, 1e-6), max=max_linewidth)
    fit = model.fit(signal, params, field_mT=field)
    diagnostics = _build_parameter_diagnostics(fit.params)
    hits = _build_bound_hits(fit.params)
    components = [
        _component_result("component_1", field, fit.params, diagnostics, hits, {"H_res_mT": "H_res_1_mT", "DeltaH_mT": "DeltaH_1_mT", "amplitude_symmetric": "amplitude_symmetric_1", "amplitude_antisymmetric": "amplitude_antisymmetric_1"}, {"feature_center_mT": first.candidate_center_mT, "feature_peak_to_peak_mT": first.width_mT, "positive_extremum_mT": first.peak_field_mT, "negative_extremum_mT": first.trough_field_mT}, {"candidate_window_label": first.label}),
        _component_result("component_2", field, fit.params, diagnostics, hits, {"H_res_mT": "H_res_2_mT", "DeltaH_mT": "DeltaH_2_mT", "amplitude_symmetric": "amplitude_symmetric_2", "amplitude_antisymmetric": "amplitude_antisymmetric_2"}, {"feature_center_mT": second.candidate_center_mT, "feature_peak_to_peak_mT": second.width_mT, "positive_extremum_mT": second.peak_field_mT, "negative_extremum_mT": second.trough_field_mT}, {"candidate_window_label": second.label}),
    ]
    components.sort(key=lambda item: item.H_res_mT)
    for component, label in zip(components, ("mode_1", "mode_2"), strict=True):
        component.component_label = label
    return _model_result(fit, "double_mixed_derivative_lorentzian", components)


def _select_mode(requested_mode: str, single_fit: FmrTraceModelResult, double_fit: FmrTraceModelResult | None, candidate_windows: list[FmrCandidateWindow], threshold: float) -> tuple[str, FmrTraceModelResult, str, float | None]:
    improvement: float | None = None
    single_ss = single_fit.metrics.get("sum_squared_residuals")
    double_ss = None if double_fit is None else double_fit.metrics.get("sum_squared_residuals")
    if double_fit is not None and single_ss is not None and double_ss is not None and single_ss > 0.0:
        improvement = max(0.0, float((single_ss - double_ss) / single_ss))
    if requested_mode == "single":
        return "single", single_fit, "single mode was explicitly requested", improvement
    if requested_mode == "double":
        if double_fit is not None and len(candidate_windows) >= 2 and double_fit.success:
            return "double", double_fit, "double mode was explicitly requested", improvement
        return "single", single_fit, "double mode requested but no valid double fit was available", improvement
    if double_fit is None or len(candidate_windows) < 2 or not double_fit.success:
        return "single", single_fit, "auto mode found fewer than two valid double-fit candidate windows", improvement
    if improvement is not None and improvement >= threshold:
        return "double", double_fit, "auto mode selected double because residual improvement cleared threshold", improvement
    return "single", single_fit, "auto mode kept the single fit because double improvement was below threshold", improvement


def _assess_component(component: FmrComponentFitResult, field: np.ndarray, signal_max_abs: float, residual_rmse_fraction: float, residual_std: float, recipe: FmrRecipe) -> tuple[bool, str | None, list[str]]:
    warnings: list[str] = []
    sweep = abs(float(field[-1] - field[0]))
    guard = max(_median_step(field) * 2.0, recipe.field_guard_fraction * sweep)
    amplitude_snr = float("inf") if residual_std <= 1e-12 else float(max(abs(component.amplitude_symmetric), abs(component.amplitude_antisymmetric)) / residual_std)
    critical_hits = [name for name in ("H_res_mT", "DeltaH_mT") if component.bound_hits.get(name)]
    disagreement = None if component.feature_center_mT is None else float(abs(component.H_res_mT - component.feature_center_mT))
    component.signal_max_abs = signal_max_abs
    component.residual_rmse_fraction = residual_rmse_fraction
    component.amplitude_snr = amplitude_snr
    component.center_feature_disagreement_mT = disagreement
    component.critical_bound_hit_names = critical_hits
    component.acceptance_checks = {"center_inside_guard": bool(float(np.min(field)) + guard <= component.H_res_mT <= float(np.max(field)) - guard), "linewidth_positive": bool(component.DeltaH_mT > 0.0), "linewidth_within_fraction_limit": bool(component.DeltaH_mT <= recipe.linewidth_max_sweep_fraction * sweep), "residual_rmse_within_limit": bool(residual_rmse_fraction <= recipe.residual_rmse_max_signal_fraction), "amplitude_snr_within_limit": bool(amplitude_snr >= recipe.amplitude_snr_min), "critical_bound_hit_clear": not critical_hits}
    if component.feature_center_mT is None:
        component.acceptance_checks["shape_center_consistent"] = True
        warnings.append("shape_feature_reference_missing")
    else:
        tolerance = max(recipe.shape_center_tolerance_min_mT, recipe.shape_center_tolerance_linewidth_fraction * max(component.DeltaH_mT, 0.0))
        component.acceptance_checks["shape_center_consistent"] = bool(disagreement is not None and disagreement <= tolerance)
    if critical_hits and recipe.critical_bound_hit_policy == "warn":
        warnings.append(f"critical_parameter_hit_bound:{'|'.join(critical_hits)}")
    if not component.acceptance_checks["shape_center_consistent"] and recipe.shape_consistency_policy == "warn":
        warnings.append("shape_center_inconsistent_with_feature_reference")
    if not component.acceptance_checks["center_inside_guard"]:
        return False, "resonance_field_outside_guard", warnings
    if not component.acceptance_checks["linewidth_positive"]:
        return False, "linewidth_not_positive", warnings
    if not component.acceptance_checks["linewidth_within_fraction_limit"]:
        return False, "linewidth_exceeds_sweep_fraction_limit", warnings
    if not component.acceptance_checks["residual_rmse_within_limit"]:
        return False, "residual_rmse_exceeds_signal_fraction_limit", warnings
    if not component.acceptance_checks["amplitude_snr_within_limit"]:
        return False, "amplitude_snr_below_threshold", warnings
    if critical_hits and recipe.critical_bound_hit_policy == "reject":
        return False, f"critical_parameter_hit_bound:{'|'.join(critical_hits)}", warnings
    if not component.acceptance_checks["shape_center_consistent"] and recipe.shape_consistency_policy == "reject":
        return False, "shape_center_inconsistent_with_feature_reference", warnings
    return True, None, warnings


def _single_guesses(field: np.ndarray, signal: np.ndarray, recipe: FmrRecipe) -> tuple[float, float, float, float]:
    windows = detect_candidate_windows(field, signal, recipe)
    if windows:
        guess = _window_guess(field, signal, windows[0], _median_step(field))
        return guess["center"], guess["linewidth"], guess["sym"], guess["asym"]
    feature = _detect_feature(field, signal, recipe.shape_pair_prominence_ratio)
    center = float(field[int(np.argmax(np.abs(signal)))]) if feature["feature_center_mT"] is None else float(feature["feature_center_mT"])
    width = 0.0 if feature["feature_peak_to_peak_mT"] is None else float(feature["feature_peak_to_peak_mT"])
    step = _median_step(field)
    amplitude = max(float(np.max(np.abs(signal))), 1e-6)
    max_i = int(np.argmax(signal))
    min_i = int(np.argmin(signal))
    sign = 1.0 if field[max_i] < field[min_i] else -1.0
    return center, max(width * math.sqrt(3.0) / 2.0, step * 2.0, 1e-4), sign * amplitude, 0.1 * amplitude


def _window_guess(field: np.ndarray, signal: np.ndarray, window: FmrCandidateWindow, step: float) -> dict[str, float]:
    sub_signal = signal[window.start_index : window.end_index + 1]
    sub_field = field[window.start_index : window.end_index + 1]
    max_i = int(np.argmax(sub_signal))
    min_i = int(np.argmin(sub_signal))
    width = abs(float(sub_field[max_i] - sub_field[min_i]))
    amplitude = max(float(np.max(np.abs(sub_signal))), 1e-6)
    sign = 1.0 if sub_field[max_i] < sub_field[min_i] else -1.0
    return {"center": window.candidate_center_mT, "linewidth": max(width * math.sqrt(3.0) / 2.0, window.width_mT, step * 2.0, 1e-4), "sym": sign * amplitude, "asym": 0.1 * amplitude}


def _component(field_mT: np.ndarray, H_res_mT: float, DeltaH_mT: float, amplitude_symmetric: float, amplitude_antisymmetric: float) -> np.ndarray:
    delta = field_mT - H_res_mT
    denominator = 4.0 * delta**2 + DeltaH_mT**2
    squared = denominator**2
    symmetric = (4.0 * DeltaH_mT * delta) / squared
    antisymmetric = (DeltaH_mT**2 - 4.0 * delta**2) / squared
    return amplitude_symmetric * symmetric - amplitude_antisymmetric * antisymmetric


def _detect_feature(field: np.ndarray, signal: np.ndarray, prominence_ratio: float) -> dict[str, float | None]:
    field = np.asarray(field, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if field.size == 0 or signal.size == 0:
        return {"feature_center_mT": None, "feature_peak_to_peak_mT": None, "positive_extremum_mT": None, "negative_extremum_mT": None}
    scale = float(np.max(np.abs(signal[np.isfinite(signal)]))) if np.any(np.isfinite(signal)) else 0.0
    prominence = max(scale * prominence_ratio, 1e-12)
    pos, _ = find_peaks(signal, prominence=prominence)
    neg, _ = find_peaks(-signal, prominence=prominence)
    if pos.size == 0:
        pos = np.asarray([int(np.argmax(signal))], dtype=int)
    if neg.size == 0:
        neg = np.asarray([int(np.argmin(signal))], dtype=int)
    best_pair: tuple[int, int] | None = None
    best_score = -np.inf
    for pos_i in pos:
        for neg_i in neg:
            if pos_i == neg_i:
                continue
            score = abs(float(signal[pos_i])) + abs(float(signal[neg_i]))
            if score > best_score:
                best_pair = (int(pos_i), int(neg_i))
                best_score = score
    if best_pair is None:
        return {"feature_center_mT": None, "feature_peak_to_peak_mT": None, "positive_extremum_mT": None, "negative_extremum_mT": None}
    pos_field = float(field[best_pair[0]])
    neg_field = float(field[best_pair[1]])
    return {"feature_center_mT": float((pos_field + neg_field) / 2.0), "feature_peak_to_peak_mT": float(abs(pos_field - neg_field)), "positive_extremum_mT": pos_field, "negative_extremum_mT": neg_field}


def _model_result(fit, model_name: str, components: list[FmrComponentFitResult]) -> FmrTraceModelResult:
    fitted = np.asarray(fit.best_fit, dtype=float)
    residual = np.asarray(fit.data - fitted, dtype=float)
    return FmrTraceModelResult(model_name=model_name, success=bool(fit.success), parameters={name: float(parameter.value) for name, parameter in fit.params.items()}, parameter_diagnostics=_build_parameter_diagnostics(fit.params), convergence=ConvergenceSummary(success=bool(fit.success), message=str(fit.message), nfev=None if fit.nfev is None else int(fit.nfev), nvarys=None if fit.nvarys is None else int(fit.nvarys), errorbars=bool(fit.errorbars)), residual_summary=_build_residual_summary(residual), metrics=_compute_fit_metrics(np.asarray(fit.data, dtype=float), residual), bound_hits=_build_bound_hits(fit.params), covariance=None if fit.covar is None else np.asarray(fit.covar, dtype=float).tolist(), fitted_signal=fitted, residual=residual, components=components, warnings=[])


def _component_result(component_label: str, field: np.ndarray, params, diagnostics: dict[str, ParameterDiagnostic], bound_hits: dict[str, bool], names: dict[str, str], feature_reference: dict[str, float | None], metadata: dict[str, object]) -> FmrComponentFitResult:
    return FmrComponentFitResult(component_id="", component_label=component_label, H_res_mT=float(params[names["H_res_mT"]].value), DeltaH_mT=float(params[names["DeltaH_mT"]].value), amplitude_symmetric=float(params[names["amplitude_symmetric"]].value), amplitude_antisymmetric=float(params[names["amplitude_antisymmetric"]].value), field_mT=np.asarray(field, dtype=float).copy(), component_signal=np.asarray(_component(field, float(params[names["H_res_mT"]].value), float(params[names["DeltaH_mT"]].value), float(params[names["amplitude_symmetric"]].value), float(params[names["amplitude_antisymmetric"]].value)), dtype=float), parameter_diagnostics={key: diagnostics[value] for key, value in names.items()}, bound_hits={key: bool(bound_hits.get(value, False)) for key, value in names.items()}, feature_center_mT=feature_reference.get("feature_center_mT"), feature_peak_to_peak_mT=feature_reference.get("feature_peak_to_peak_mT"), metadata=dict(metadata))


def _clone_component(component: FmrComponentFitResult, trace_id: str) -> FmrComponentFitResult:
    return FmrComponentFitResult(component_id=f"{trace_id}:{component.component_label}", component_label=component.component_label, H_res_mT=component.H_res_mT, DeltaH_mT=component.DeltaH_mT, amplitude_symmetric=component.amplitude_symmetric, amplitude_antisymmetric=component.amplitude_antisymmetric, field_mT=np.asarray(component.field_mT, dtype=float).copy(), component_signal=np.asarray(component.component_signal, dtype=float).copy(), parameter_diagnostics=dict(component.parameter_diagnostics), bound_hits=dict(component.bound_hits), feature_center_mT=component.feature_center_mT, feature_peak_to_peak_mT=component.feature_peak_to_peak_mT, metadata=dict(component.metadata))


def _compute_fit_metrics(signal: np.ndarray, residual: np.ndarray) -> dict[str, float]:
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((signal - np.mean(signal)) ** 2))
    return {"chi_square": ss_res, "reduced_chi_square": ss_res / max(signal.size - 1, 1), "r_squared": 1.0 if ss_tot == 0.0 else 1.0 - (ss_res / ss_tot), "sum_squared_residuals": ss_res}


def _build_parameter_diagnostics(params) -> dict[str, ParameterDiagnostic]:
    diagnostics: dict[str, ParameterDiagnostic] = {}
    for name, parameter in params.items():
        stderr = None if parameter.stderr is None else float(parameter.stderr)
        rel = None if stderr is None or parameter.value in (None, 0.0) else float(abs(stderr / float(parameter.value)))
        min_bound = None if parameter.min in (-np.inf, None) else float(parameter.min)
        max_bound = None if parameter.max in (np.inf, None) else float(parameter.max)
        diagnostics[name] = ParameterDiagnostic(value=float(parameter.value), stderr=stderr, relative_stderr=rel, stderr_missing=stderr is None, min_bound=min_bound, max_bound=max_bound, hit_min_bound=_is_at_bound(float(parameter.value), min_bound), hit_max_bound=_is_at_bound(float(parameter.value), max_bound))
    return diagnostics


def _build_bound_hits(params) -> dict[str, bool]:
    hits: dict[str, bool] = {}
    for name, parameter in params.items():
        min_bound = None if parameter.min in (-np.inf, None) else float(parameter.min)
        max_bound = None if parameter.max in (np.inf, None) else float(parameter.max)
        hits[name] = _is_at_bound(float(parameter.value), min_bound) or _is_at_bound(float(parameter.value), max_bound)
    return hits


def _build_residual_summary(residual: np.ndarray) -> ResidualSummary:
    return ResidualSummary(rss=float(np.sum(residual**2)), rmse=float(np.sqrt(np.mean(residual**2))), mae=float(np.mean(np.abs(residual))), max_abs=float(np.max(np.abs(residual))), mean=float(np.mean(residual)), std=float(np.std(residual)))


def _median_step(field: np.ndarray) -> float:
    diffs = np.diff(field)
    nz = np.abs(diffs[diffs != 0.0])
    return 1.0 if nz.size == 0 else float(np.median(nz))


def _is_at_bound(value: float, bound: float | None) -> bool:
    if bound is None:
        return False
    return abs(value - bound) <= max(1e-9, abs(bound) * 1e-6)
