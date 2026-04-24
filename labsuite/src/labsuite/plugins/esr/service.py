"""Service layer for the ESR plugin."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from labsuite.core.preprocessing import cumulative_integral, scalar_integral
from labsuite.core.recipes import EsrPreprocessingRecipe, load_esr_recipe
from labsuite.core.resonance_metrics import (
    ResonanceMetricsConfig,
    ResonanceModeMetrics,
    compute_absorption_mode_metrics,
)
from labsuite.core.sample_registry import AnalysisSampleContext
from labsuite.core.types import (
    AnalysisResult,
    FitAttemptRecord,
    FitIntegratedCurves,
    FitResult,
    IntegralSummary,
    PeakFitResult,
    PeakWindow,
    PrimaryIntegratedCurves,
    ProcessedTrace,
)
from labsuite.plugins.esr.fitters import (
    absorption_lorentzian,
    build_split_fit,
    derivative_lorentzian,
    detect_peak_windows,
    fit_derivative_lorentzian,
    fit_derivative_lorentzian_in_window,
    fit_peak_windows,
    select_fit_mode,
)
from labsuite.plugins.esr.parser import parse_esr_file
from labsuite.plugins.esr.preprocess import (
    apply_esr_preprocessing,
    evaluate_local_resonance_diagnostic_reason,
    integrate_esr_trace,
    integrate_local_resonance_with_curves,
)


def analyze_esr_file(
    source_path: Path,
    recipe_path: Path,
    fit_mode: Literal["auto", "single", "split"] | None = None,
    resonance_metrics_config: ResonanceMetricsConfig | None = None,
    sample_context: AnalysisSampleContext | None = None,
) -> AnalysisResult:
    """Run the ESR parse, preprocess, and fit stages for one source file."""

    dataset = parse_esr_file(source_path.resolve())
    if sample_context is not None:
        dataset.metadata = {
            **dataset.metadata,
            "sample_registry": sample_context.to_dict(),
            "sample_id": sample_context.sample_id,
            "measurement_id": sample_context.measurement_id,
            "registry_geometry": sample_context.geometry,
            "g_mode": sample_context.g_mode,
            "g_value": sample_context.g_value,
        }
    recipe = load_esr_recipe(recipe_path)
    metrics_config = resonance_metrics_config or ResonanceMetricsConfig()
    requested_fit_mode = fit_mode or recipe.fit_mode
    processed, derivative_baseline = apply_esr_preprocessing(dataset, recipe)
    integrated, absorption_baseline = integrate_esr_trace(processed, recipe)
    peak_windows = detect_peak_windows(processed, recipe)
    fallback_window = _primary_detected_window(peak_windows)

    single_fit_attempts = _build_single_fit_attempts(processed, recipe, fallback_window)
    single_candidate_attempt = single_fit_attempts[-1]
    single_fit = single_candidate_attempt.fit

    peak_fit_candidates = fit_peak_windows(processed, peak_windows)
    candidate_peak_fits: list[PeakFitResult] = []
    candidate_peak_integrals: list[IntegralSummary] = []
    candidate_peak_curves: list[FitIntegratedCurves] = []
    candidate_peak_fit_local_integrals: list[IntegralSummary] = []
    candidate_peak_fit_local_curves: list[PrimaryIntegratedCurves] = []
    candidate_peak_local_integrals: list[IntegralSummary] = []
    candidate_peak_local_curves: list[PrimaryIntegratedCurves] = []
    for peak_fit in peak_fit_candidates:
        attempt = _create_fit_attempt(
            fit=peak_fit.fit,
            scope="peak_window_local",
            source_window=peak_fit.window,
            trace=processed,
            recipe=recipe,
        )
        peak_fit.attempts = [attempt]
        if not attempt.accepted:
            continue
        local_diagnostic_reason = evaluate_local_resonance_diagnostic_reason(
            processed,
            recipe,
            center_mT=peak_fit.fit.parameters["center_mT"],
            gamma_mT=peak_fit.fit.parameters["gamma_mT"],
            peak_window=peak_fit.window,
            exclude_windows=peak_windows,
        )
        peak_local_integral, peak_local_curves = integrate_local_resonance_with_curves(
            processed,
            recipe,
            label=peak_fit.label,
            center_mT=peak_fit.fit.parameters["center_mT"],
            gamma_mT=peak_fit.fit.parameters["gamma_mT"],
            peak_window=peak_fit.window,
            exclude_windows=peak_windows,
        )
        if peak_local_integral.area_integral is None:
            local_diagnostic_reason = local_diagnostic_reason or "insufficient_isolated_baseline"
        peak_integral = _build_fit_integral_summary(
            label=peak_fit.label,
            trace=processed,
            fit=peak_fit.fit,
            local_summary=peak_local_integral,
        )
        peak_curves = _build_fit_integrated_curves(processed, peak_fit.fit)
        if peak_local_integral.area_integral is None or peak_local_curves is None:
            peak_fit_local_integral = _unavailable_fit_local_integral(peak_local_integral, label=peak_fit.label)
            peak_fit_local_curves = None
            disagreement_ratio = None
            disagreement_flag = True
            disagreement_reason = f"split_local_diagnostic_unavailable:{local_diagnostic_reason}"
        else:
            peak_fit_local_integral, peak_fit_local_curves = _build_window_matched_fit_curves_and_integral(
                trace=processed,
                fit=peak_fit.fit,
                local_summary=peak_local_integral,
                label=peak_fit.label,
            )
            disagreement_ratio, disagreement_flag, disagreement_reason = _compute_fit_local_disagreement(
                fit_local_area_integral=peak_fit_local_integral.area_integral,
                local_area_integral=peak_local_integral.area_integral,
                threshold=recipe.fit_local_disagreement_ratio_threshold,
            )
        _attach_primary_intensity(
            peak_fit.fit,
            peak_integral=peak_integral,
            fit_local_integral=peak_fit_local_integral,
            local_integral=peak_local_integral,
            local_diagnostic_reason=local_diagnostic_reason,
            disagreement_ratio=disagreement_ratio,
            disagreement_flag=disagreement_flag,
            disagreement_reason=disagreement_reason,
        )
        candidate_peak_fits.append(peak_fit)
        candidate_peak_integrals.append(peak_integral)
        candidate_peak_fit_local_integrals.append(peak_fit_local_integral)
        candidate_peak_local_integrals.append(peak_local_integral)
        if peak_curves is not None:
            candidate_peak_curves.append(peak_curves)
        if peak_fit_local_curves is not None:
            candidate_peak_fit_local_curves.append(peak_fit_local_curves)
        if peak_local_curves is not None:
            candidate_peak_local_curves.append(peak_local_curves)

    split_fit = build_split_fit(processed, candidate_peak_fits) if len(candidate_peak_fits) >= 2 else None
    if split_fit is not None:
        split_fit.derived["fit_valid"] = bool(split_fit.success)
        split_fit.derived["fit_scope"] = "split_selected"
        split_fit.derived["fit_rejection_reason"] = None
        split_fit.derived["selected_for_primary"] = False
        split_fit.derived["integration_window_clipped_by_detected_window"] = any(
            integral.integration_window_clipped_by_detected_window for integral in candidate_peak_integrals
        )

    selected_mode, fit_decision = select_fit_mode(
        requested_mode=requested_fit_mode,
        single_fit=single_fit,
        split_fit=split_fit,
        peak_windows=peak_windows,
        split_threshold=recipe.split_min_improvement_ratio,
    )

    selected_fit_signal = _full_axis_fit_signal(processed, single_fit)
    selected_residual = processed.signal - selected_fit_signal
    selected_peak_fits: list[PeakFitResult] = []
    selected_peak_integrals: list[IntegralSummary] = []
    selected_peak_fit_local_integrals: list[IntegralSummary] = []
    selected_peak_local_integrals: list[IntegralSummary] = []
    (
        total_integral,
        primary_integrated,
        fit_local_total_integral,
        fit_local_integrated,
        local_total_integral,
        local_integrated,
    ) = _selected_single_integral(
        processed,
        recipe,
        single_candidate_attempt,
    )
    fit_local_disagreement_ratio, fit_local_disagreement_flag, fit_local_disagreement_reason = (
        _result_fit_local_disagreement(single_fit)
    )

    if selected_mode == "single":
        _mark_selected_attempt(single_candidate_attempt)
    elif selected_mode == "split" and split_fit is not None:
        selected_fit_signal = split_fit.fitted_signal
        selected_residual = split_fit.residual
        selected_peak_fits = candidate_peak_fits
        selected_peak_integrals = candidate_peak_integrals
        selected_peak_fit_local_integrals = candidate_peak_fit_local_integrals
        selected_peak_local_integrals = candidate_peak_local_integrals
        total_integral = _combine_integral_summaries(
            "total",
            selected_peak_integrals,
            integration_kind="primary_fit_model",
        )
        fit_local_total_integral = _combine_integral_summaries(
            "fit_local_total",
            selected_peak_fit_local_integrals,
            integration_kind="fit_local_windowed_model",
        )
        local_total_integral = _combine_integral_summaries(
            "local_total",
            selected_peak_local_integrals,
            integration_kind="primary_local_window",
        )
        primary_integrated = (
            _combine_fit_integrated_curves(candidate_peak_curves)
            if len(candidate_peak_curves) == len(candidate_peak_integrals)
            else None
        )
        fit_local_integrated = (
            _combine_windowed_integrated_curves(candidate_peak_fit_local_curves)
            if len(candidate_peak_fit_local_curves) == len(candidate_peak_fit_local_integrals)
            else None
        )
        local_integrated = (
            _combine_windowed_integrated_curves(candidate_peak_local_curves)
            if len(candidate_peak_local_curves) == len(candidate_peak_local_integrals)
            else None
        )
        split_local_reason = _split_local_diagnostic_reason(selected_peak_fits)
        if split_local_reason is not None:
            fit_local_disagreement_ratio = None
            fit_local_disagreement_flag = True
            fit_local_disagreement_reason = split_local_reason
        else:
            fit_local_disagreement_ratio, fit_local_disagreement_flag, fit_local_disagreement_reason = (
                _compute_fit_local_disagreement(
                    fit_local_area_integral=fit_local_total_integral.area_integral,
                    local_area_integral=local_total_integral.area_integral,
                    threshold=recipe.fit_local_disagreement_ratio_threshold,
                )
            )
        if split_fit is not None:
            split_fit.derived["selected_for_primary"] = True
        for peak_fit in selected_peak_fits:
            for attempt in peak_fit.attempts:
                _mark_selected_attempt(attempt)

    diagnostic_total_integral = _build_diagnostic_integral_summary(
        label="diagnostic_total",
        field=processed.field_mT,
        diagnostic_absorption_signal=integrated.absorption_signal,
        diagnostic_area_signal=integrated.area_signal,
    )
    resonance_metrics = _compute_selected_resonance_metrics(
        trace=processed,
        selected_mode=selected_mode,
        single_fit=single_fit,
        peak_fits=selected_peak_fits,
        local_total_integral=local_total_integral,
        local_peak_integrals=selected_peak_local_integrals,
        config=metrics_config,
    )

    return AnalysisResult(
        dataset=dataset,
        processed=processed,
        integrated=integrated,
        primary_integrated=primary_integrated,
        fit_local_integrated=fit_local_integrated,
        local_integrated=local_integrated,
        derivative_baseline=derivative_baseline,
        absorption_baseline=absorption_baseline,
        selected_mode=selected_mode,
        fit_decision=fit_decision,
        single_fit=single_fit,
        single_fit_attempts=single_fit_attempts,
        peak_fits=selected_peak_fits,
        selected_fit_signal=selected_fit_signal,
        selected_residual=selected_residual,
        total_integral=total_integral,
        fit_local_total_integral=fit_local_total_integral,
        local_total_integral=local_total_integral,
        diagnostic_total_integral=diagnostic_total_integral,
        peak_integrals=selected_peak_integrals,
        fit_local_peak_integrals=selected_peak_fit_local_integrals,
        local_peak_integrals=selected_peak_local_integrals,
        fit_local_disagreement_ratio=fit_local_disagreement_ratio,
        fit_local_disagreement_flag=fit_local_disagreement_flag,
        fit_local_disagreement_reason=fit_local_disagreement_reason,
        recipe_name=recipe.name,
        recipe_config=recipe.to_dict(),
        resonance_metrics_config=metrics_config.to_dict(),
        resonance_metrics=resonance_metrics,
    )


def _build_single_fit_attempts(
    trace: ProcessedTrace,
    recipe: EsrPreprocessingRecipe,
    fallback_window: PeakWindow | None,
) -> list[FitAttemptRecord]:
    attempts: list[FitAttemptRecord] = []
    global_fit = fit_derivative_lorentzian(trace)
    global_attempt = _create_fit_attempt(
        fit=global_fit,
        scope="global_full_trace",
        source_window=fallback_window,
        trace=trace,
        recipe=recipe,
    )
    attempts.append(global_attempt)
    if global_attempt.accepted or fallback_window is None:
        return attempts

    fallback_fit = fit_derivative_lorentzian_in_window(trace, fallback_window)
    fallback_attempt = _create_fit_attempt(
        fit=fallback_fit,
        scope="detected_window_fallback",
        source_window=fallback_window,
        trace=trace,
        recipe=recipe,
    )
    attempts.append(fallback_attempt)
    return attempts


def _create_fit_attempt(
    *,
    fit: FitResult,
    scope: Literal["global_full_trace", "detected_window_fallback", "peak_window_local"],
    source_window: PeakWindow | None,
    trace: ProcessedTrace,
    recipe: EsrPreprocessingRecipe,
) -> FitAttemptRecord:
    rejection_reason = _validate_fit_result(fit, trace=trace, recipe=recipe, source_window=source_window)
    accepted = rejection_reason is None
    fit.success = fit.success and accepted
    fit.derived["fit_scope"] = scope
    fit.derived["fit_valid"] = accepted
    fit.derived["fit_rejection_reason"] = rejection_reason
    fit.derived["selected_for_primary"] = False
    fit.derived["integration_window_clipped_by_detected_window"] = False
    return FitAttemptRecord(
        scope=scope,
        fit=fit,
        source_window=source_window,
        accepted=accepted,
        rejection_reason=rejection_reason,
        selected_for_primary=False,
    )


def _validate_fit_result(
    fit: FitResult,
    *,
    trace: ProcessedTrace,
    recipe: EsrPreprocessingRecipe,
    source_window: PeakWindow | None,
) -> str | None:
    if not fit.convergence.success or not fit.success:
        return "fit_did_not_converge"
    if fit.bound_hits.get("center_mT"):
        return "center_hit_bound"
    if fit.bound_hits.get("gamma_mT"):
        return "gamma_hit_bound"

    sweep_width = float(trace.field_mT[-1] - trace.field_mT[0])
    gamma_mT = fit.parameters.get("gamma_mT")
    if gamma_mT is None:
        return "missing_gamma_parameter"
    if gamma_mT > recipe.fit_max_gamma_as_sweep_fraction * sweep_width:
        return "gamma_exceeds_sweep_fraction_limit"

    feature_summary = fit.feature_summary
    if feature_summary is None:
        return "missing_feature_summary"
    if source_window is None:
        return None

    guard_start, guard_end = _guard_bounds(source_window, recipe)
    center_mT = feature_summary.zero_crossing_field_mT
    if not guard_start <= center_mT <= guard_end:
        return "center_outside_detected_window_guard"

    guard_width = guard_end - guard_start
    if feature_summary.peak_to_peak_separation_mT > guard_width:
        return "peak_to_peak_separation_exceeds_detected_window_guard"
    return None


def _primary_detected_window(peak_windows: list[PeakWindow]) -> PeakWindow | None:
    if not peak_windows:
        return None
    return max(peak_windows, key=lambda window: window.prominence)


def _guard_bounds(window: PeakWindow, recipe: EsrPreprocessingRecipe) -> tuple[float, float]:
    padding = window.width_mT * recipe.integration_detected_window_padding_width_multiplier
    return window.start_field_mT - padding, window.end_field_mT + padding


def _selected_single_integral(
    trace: ProcessedTrace,
    recipe: EsrPreprocessingRecipe,
    attempt: FitAttemptRecord,
) -> tuple[
    IntegralSummary,
    FitIntegratedCurves | None,
    IntegralSummary,
    PrimaryIntegratedCurves | None,
    IntegralSummary,
    PrimaryIntegratedCurves | None,
]:
    if not attempt.accepted:
        local_summary = _unavailable_selected_local_integral(attempt)
        fit_local_summary = _unavailable_fit_local_integral(local_summary)
        return _unavailable_primary_integral(local_summary), None, fit_local_summary, None, local_summary, None

    local_integral, local_curves = integrate_local_resonance_with_curves(
        trace,
        recipe,
        label="local_total",
        center_mT=attempt.fit.parameters["center_mT"],
        gamma_mT=attempt.fit.parameters["gamma_mT"],
        peak_window=attempt.source_window,
    )
    renamed_local_summary = _rename_integral_summary(local_integral, "local_total")
    renamed_local_curves = (
        None if local_curves is None else _rename_local_integrated_curves(local_curves, renamed_local_summary)
    )
    primary_integral = _build_fit_integral_summary(
        label="total",
        trace=trace,
        fit=attempt.fit,
        local_summary=renamed_local_summary,
    )
    primary_curves = _build_fit_integrated_curves(trace, attempt.fit)
    fit_local_integral, fit_local_curves = _build_window_matched_fit_curves_and_integral(
        trace=trace,
        fit=attempt.fit,
        local_summary=renamed_local_summary,
        label="fit_local_total",
    )
    disagreement_ratio, disagreement_flag, disagreement_reason = _compute_fit_local_disagreement(
        fit_local_area_integral=fit_local_integral.area_integral,
        local_area_integral=renamed_local_summary.area_integral,
        threshold=recipe.fit_local_disagreement_ratio_threshold,
    )
    _attach_primary_intensity(
        attempt.fit,
        peak_integral=primary_integral,
        fit_local_integral=fit_local_integral,
        local_integral=renamed_local_summary,
        local_diagnostic_reason=None,
        disagreement_ratio=disagreement_ratio,
        disagreement_flag=disagreement_flag,
        disagreement_reason=disagreement_reason,
    )
    return primary_integral, primary_curves, fit_local_integral, fit_local_curves, renamed_local_summary, renamed_local_curves


def _unavailable_selected_local_integral(attempt: FitAttemptRecord) -> IntegralSummary:
    start_field = float(attempt.fit.parameters.get("center_mT", 0.0))
    end_field = start_field
    if attempt.source_window is not None:
        start_field = attempt.source_window.start_field_mT
        end_field = attempt.source_window.end_field_mT
    return IntegralSummary(
        label="local_total",
        start_field_mT=start_field,
        end_field_mT=end_field,
        absorption_integral=None,
        area_integral=None,
        integration_kind="primary_local_window",
        window_source="fit_linewidth",
        baseline_polyorder=None,
        integration_window_clipped_by_detected_window=False,
    )


def _unavailable_primary_integral(local_summary: IntegralSummary) -> IntegralSummary:
    return IntegralSummary(
        label="total",
        start_field_mT=local_summary.start_field_mT,
        end_field_mT=local_summary.end_field_mT,
        absorption_integral=None,
        area_integral=None,
        integration_kind="primary_fit_model",
        window_source=local_summary.window_source,
        baseline_polyorder=local_summary.baseline_polyorder,
        integration_window_clipped_by_detected_window=local_summary.integration_window_clipped_by_detected_window,
    )


def _unavailable_fit_local_integral(
    local_summary: IntegralSummary,
    *,
    label: str = "fit_local_total",
) -> IntegralSummary:
    return IntegralSummary(
        label=label,
        start_field_mT=local_summary.start_field_mT,
        end_field_mT=local_summary.end_field_mT,
        absorption_integral=None,
        area_integral=None,
        integration_kind="fit_local_windowed_model",
        window_source=local_summary.window_source,
        baseline_polyorder=local_summary.baseline_polyorder,
        integration_window_clipped_by_detected_window=local_summary.integration_window_clipped_by_detected_window,
    )


def _full_axis_fit_signal(trace: ProcessedTrace, fit: FitResult):
    if {"amplitude", "center_mT", "gamma_mT", "offset"}.issubset(fit.parameters):
        return derivative_lorentzian(
            trace.field_mT,
            amplitude=fit.parameters["amplitude"],
            center_mT=fit.parameters["center_mT"],
            gamma_mT=fit.parameters["gamma_mT"],
            offset=fit.parameters["offset"],
        )
    return fit.fitted_signal


def _build_diagnostic_integral_summary(
    label: str,
    field,
    diagnostic_absorption_signal,
    diagnostic_area_signal,
) -> IntegralSummary:
    return IntegralSummary(
        label=label,
        start_field_mT=float(field[0]),
        end_field_mT=float(field[-1]),
        absorption_integral=scalar_integral(field, diagnostic_absorption_signal),
        area_integral=scalar_integral(field, diagnostic_area_signal),
        integration_kind="diagnostic_full_span",
        window_source=None,
        baseline_polyorder=None,
        integration_window_clipped_by_detected_window=False,
    )


def _compute_selected_resonance_metrics(
    *,
    trace: ProcessedTrace,
    selected_mode: Literal["single", "split"],
    single_fit: FitResult | None,
    peak_fits: list[PeakFitResult],
    local_total_integral: IntegralSummary,
    local_peak_integrals: list[IntegralSummary],
    config: ResonanceMetricsConfig,
) -> list[ResonanceModeMetrics]:
    if not config.compute_resonance_metrics:
        return []
    if selected_mode == "single":
        if single_fit is None:
            return []
        curves = _build_fit_integrated_curves(trace, single_fit)
        if curves is None:
            return []
        return [
            compute_absorption_mode_metrics(
                curves.field_mT,
                curves.absorption_signal,
                hres=float(single_fit.parameters["center_mT"]),
                config=config,
                support_start_field_mT=local_total_integral.start_field_mT,
                support_end_field_mT=local_total_integral.end_field_mT,
                owner_kind="selected",
                owner_id="selected",
                metadata={"mode": "single", "model_name": single_fit.model_name},
            )
        ]

    metrics: list[ResonanceModeMetrics] = []
    for peak_fit, local_summary in zip(peak_fits, local_peak_integrals, strict=True):
        curves = _build_fit_integrated_curves(trace, peak_fit.fit)
        if curves is None:
            continue
        metrics.append(
            compute_absorption_mode_metrics(
                curves.field_mT,
                curves.absorption_signal,
                hres=float(peak_fit.fit.parameters["center_mT"]),
                config=config,
                support_start_field_mT=local_summary.start_field_mT,
                support_end_field_mT=local_summary.end_field_mT,
                owner_kind="peak_fit",
                owner_id=peak_fit.label,
                metadata={"mode": "split", "model_name": peak_fit.fit.model_name},
            )
        )
    return metrics


def _rename_local_integrated_curves(
    curves: PrimaryIntegratedCurves,
    summary: IntegralSummary,
) -> PrimaryIntegratedCurves:
    return PrimaryIntegratedCurves(
        field_mT=curves.field_mT.copy(),
        absorption_signal=curves.absorption_signal.copy(),
        area_signal=curves.area_signal.copy(),
        start_field_mT=summary.start_field_mT,
        end_field_mT=summary.end_field_mT,
        integration_kind="primary_local_window",
        window_source="fit_linewidth",
        baseline_polyorder=summary.baseline_polyorder,
        integration_window_clipped_by_detected_window=summary.integration_window_clipped_by_detected_window,
    )


def _build_fit_integrated_curves(trace: ProcessedTrace, fit: FitResult) -> FitIntegratedCurves | None:
    required = {"amplitude", "center_mT", "gamma_mT"}
    if not required.issubset(fit.parameters):
        return None
    absorption_signal = np.asarray(
        absorption_lorentzian(
            trace.field_mT,
            amplitude=fit.parameters["amplitude"],
            center_mT=fit.parameters["center_mT"],
            gamma_mT=fit.parameters["gamma_mT"],
        ),
        dtype=float,
    )
    area_signal = np.asarray(cumulative_integral(trace.field_mT, absorption_signal), dtype=float)
    return FitIntegratedCurves(
        field_mT=trace.field_mT.copy(),
        absorption_signal=absorption_signal,
        area_signal=area_signal,
        integration_kind="primary_fit_model",
        model_name=fit.model_name,
    )


def _build_fit_integral_summary(
    *,
    label: str,
    trace: ProcessedTrace,
    fit: FitResult,
    local_summary: IntegralSummary,
) -> IntegralSummary:
    curves = _build_fit_integrated_curves(trace, fit)
    if curves is None:
        return IntegralSummary(
            label=label,
            start_field_mT=local_summary.start_field_mT,
            end_field_mT=local_summary.end_field_mT,
            absorption_integral=None,
            area_integral=None,
            integration_kind="primary_fit_model",
            window_source=local_summary.window_source,
            baseline_polyorder=local_summary.baseline_polyorder,
            integration_window_clipped_by_detected_window=local_summary.integration_window_clipped_by_detected_window,
        )
    return IntegralSummary(
        label=label,
        start_field_mT=local_summary.start_field_mT,
        end_field_mT=local_summary.end_field_mT,
        absorption_integral=None,
        area_integral=scalar_integral(curves.field_mT, curves.absorption_signal),
        integration_kind="primary_fit_model",
        window_source=local_summary.window_source,
        baseline_polyorder=local_summary.baseline_polyorder,
        integration_window_clipped_by_detected_window=local_summary.integration_window_clipped_by_detected_window,
    )


def _build_window_matched_fit_curves_and_integral(
    *,
    trace: ProcessedTrace,
    fit: FitResult,
    local_summary: IntegralSummary,
    label: str,
) -> tuple[IntegralSummary, PrimaryIntegratedCurves | None]:
    full_curves = _build_fit_integrated_curves(trace, fit)
    if full_curves is None:
        return (
            IntegralSummary(
                label=label,
                start_field_mT=local_summary.start_field_mT,
                end_field_mT=local_summary.end_field_mT,
                absorption_integral=None,
                area_integral=None,
                integration_kind="fit_local_windowed_model",
                window_source=local_summary.window_source,
                baseline_polyorder=local_summary.baseline_polyorder,
                integration_window_clipped_by_detected_window=local_summary.integration_window_clipped_by_detected_window,
            ),
            None,
        )
    field = trace.field_mT
    integration_mask = (field >= local_summary.start_field_mT) & (field <= local_summary.end_field_mT)
    if int(np.count_nonzero(integration_mask)) < 2:
        return (
            IntegralSummary(
                label=label,
                start_field_mT=local_summary.start_field_mT,
                end_field_mT=local_summary.end_field_mT,
                absorption_integral=None,
                area_integral=None,
                integration_kind="fit_local_windowed_model",
                window_source=local_summary.window_source,
                baseline_polyorder=local_summary.baseline_polyorder,
                integration_window_clipped_by_detected_window=local_summary.integration_window_clipped_by_detected_window,
            ),
            None,
        )
    field_segment = np.asarray(field[integration_mask], dtype=float)
    absorption_segment = np.asarray(full_curves.absorption_signal[integration_mask], dtype=float)
    area_segment = np.asarray(cumulative_integral(field_segment, absorption_segment), dtype=float)
    absorption_signal = np.full_like(field, np.nan, dtype=float)
    area_signal = np.full_like(field, np.nan, dtype=float)
    absorption_signal[integration_mask] = absorption_segment
    area_signal[integration_mask] = area_segment
    return (
        IntegralSummary(
            label=label,
            start_field_mT=float(field_segment[0]),
            end_field_mT=float(field_segment[-1]),
            absorption_integral=scalar_integral(field_segment, absorption_segment),
            area_integral=scalar_integral(field_segment, absorption_segment),
            integration_kind="fit_local_windowed_model",
            window_source=local_summary.window_source,
            baseline_polyorder=local_summary.baseline_polyorder,
            integration_window_clipped_by_detected_window=local_summary.integration_window_clipped_by_detected_window,
        ),
        PrimaryIntegratedCurves(
            field_mT=field.copy(),
            absorption_signal=absorption_signal,
            area_signal=area_signal,
            start_field_mT=float(field_segment[0]),
            end_field_mT=float(field_segment[-1]),
            integration_kind="fit_local_windowed_model",
            window_source="fit_linewidth",
            baseline_polyorder=local_summary.baseline_polyorder,
            integration_window_clipped_by_detected_window=local_summary.integration_window_clipped_by_detected_window,
            model_name=fit.model_name,
        ),
    )


def _compute_fit_local_disagreement(
    *,
    fit_local_area_integral: float | None,
    local_area_integral: float | None,
    threshold: float,
) -> tuple[float | None, bool, str | None]:
    if fit_local_area_integral is None or local_area_integral is None:
        return None, False, None
    denominator = max(abs(fit_local_area_integral), abs(local_area_integral), 1e-12)
    ratio = abs(fit_local_area_integral - local_area_integral) / denominator
    if ratio > threshold:
        return ratio, True, f"fit_local_window_vs_data_ratio_exceeds_threshold:{ratio:.3g}>{threshold:.3g}"
    return ratio, False, None


def _attach_primary_intensity(
    fit: FitResult,
    *,
    peak_integral: IntegralSummary,
    fit_local_integral: IntegralSummary,
    local_integral: IntegralSummary,
    local_diagnostic_reason: str | None,
    disagreement_ratio: float | None,
    disagreement_flag: bool,
    disagreement_reason: str | None,
) -> None:
    fit.derived["intensity_method"] = "fit_derived_lorentzian_area_integral"
    fit.derived["integration_kind"] = peak_integral.integration_kind
    fit.derived["integration_start_field_mT"] = local_integral.start_field_mT
    fit.derived["integration_end_field_mT"] = local_integral.end_field_mT
    fit.derived["integration_baseline_polyorder"] = local_integral.baseline_polyorder
    fit.derived["integration_window_clipped_by_detected_window"] = (
        local_integral.integration_window_clipped_by_detected_window
    )
    fit.derived["local_diagnostic_reason"] = local_diagnostic_reason
    fit.derived["local_diagnostic_available"] = local_integral.area_integral is not None
    fit.derived["fit_local_diagnostic_available"] = fit_local_integral.area_integral is not None
    fit.derived["fit_local_windowed_intensity_proxy"] = fit_local_integral.area_integral
    fit.derived["local_windowed_intensity_proxy"] = local_integral.area_integral
    fit.derived["fit_local_disagreement_ratio"] = disagreement_ratio
    fit.derived["fit_local_disagreement_flag"] = disagreement_flag
    fit.derived["fit_local_disagreement_reason"] = disagreement_reason
    if fit.feature_summary is not None:
        fit.feature_summary.integrated_intensity_proxy = peak_integral.area_integral


def _mark_selected_attempt(attempt: FitAttemptRecord) -> None:
    attempt.selected_for_primary = True
    attempt.fit.derived["selected_for_primary"] = True


def _rename_integral_summary(summary: IntegralSummary, label: str) -> IntegralSummary:
    return IntegralSummary(
        label=label,
        start_field_mT=summary.start_field_mT,
        end_field_mT=summary.end_field_mT,
        absorption_integral=summary.absorption_integral,
        area_integral=summary.area_integral,
        integration_kind=summary.integration_kind,
        window_source=summary.window_source,
        baseline_polyorder=summary.baseline_polyorder,
        integration_window_clipped_by_detected_window=summary.integration_window_clipped_by_detected_window,
    )


def _combine_integral_summaries(
    label: str,
    summaries: list[IntegralSummary],
    *,
    integration_kind: Literal["primary_fit_model", "fit_local_windowed_model", "primary_local_window"],
) -> IntegralSummary:
    if not summaries:
        return IntegralSummary(
            label=label,
            start_field_mT=0.0,
            end_field_mT=0.0,
            absorption_integral=None,
            area_integral=None,
            integration_kind=integration_kind,
            window_source="fit_linewidth",
            baseline_polyorder=None,
            integration_window_clipped_by_detected_window=False,
        )
    absorption_values = [summary.absorption_integral for summary in summaries]
    area_values = [summary.area_integral for summary in summaries]
    return IntegralSummary(
        label=label,
        start_field_mT=min(summary.start_field_mT for summary in summaries),
        end_field_mT=max(summary.end_field_mT for summary in summaries),
        absorption_integral=None if any(value is None for value in absorption_values) else sum(absorption_values),
        area_integral=None if any(value is None for value in area_values) else sum(area_values),
        integration_kind=integration_kind,
        window_source="fit_linewidth",
        baseline_polyorder=None if any(summary.baseline_polyorder is None for summary in summaries) else summaries[0].baseline_polyorder,
        integration_window_clipped_by_detected_window=any(
            summary.integration_window_clipped_by_detected_window for summary in summaries
        ),
    )


def _combine_fit_integrated_curves(
    curves: list[FitIntegratedCurves],
) -> FitIntegratedCurves | None:
    if not curves:
        return None

    field = curves[0].field_mT.copy()
    absorption_signal = np.sum(np.vstack([curve.absorption_signal for curve in curves]), axis=0)
    area_signal = np.sum(np.vstack([curve.area_signal for curve in curves]), axis=0)
    return FitIntegratedCurves(
        field_mT=field,
        absorption_signal=np.asarray(absorption_signal, dtype=float),
        area_signal=np.asarray(area_signal, dtype=float),
        integration_kind="primary_fit_model",
        model_name="split_derivative_lorentzian",
    )


def _combine_windowed_integrated_curves(
    curves: list[PrimaryIntegratedCurves],
) -> PrimaryIntegratedCurves | None:
    if not curves:
        return None

    field = curves[0].field_mT.copy()
    absorption_stack = [curve.absorption_signal for curve in curves]
    area_stack = [curve.area_signal for curve in curves]
    valid_absorption = np.logical_or.reduce([~np.isnan(signal) for signal in absorption_stack])
    valid_area = np.logical_or.reduce([~np.isnan(signal) for signal in area_stack])

    absorption_signal = np.nansum(np.vstack(absorption_stack), axis=0)
    area_signal = _stitch_windowed_area_signal(curves)
    absorption_signal = np.asarray(absorption_signal, dtype=float)
    area_signal = np.asarray(area_signal, dtype=float)
    absorption_signal[~valid_absorption] = np.nan
    area_signal[~valid_area] = np.nan

    return PrimaryIntegratedCurves(
        field_mT=field,
        absorption_signal=absorption_signal,
        area_signal=area_signal,
        start_field_mT=min(curve.start_field_mT for curve in curves),
        end_field_mT=max(curve.end_field_mT for curve in curves),
        integration_kind=curves[0].integration_kind,
        window_source="fit_linewidth",
        baseline_polyorder=None if any(curve.baseline_polyorder is None for curve in curves) else curves[0].baseline_polyorder,
        integration_window_clipped_by_detected_window=any(
            curve.integration_window_clipped_by_detected_window for curve in curves
        ),
        model_name=curves[0].model_name,
    )


def _stitch_windowed_area_signal(curves: list[PrimaryIntegratedCurves]) -> np.ndarray:
    field = curves[0].field_mT
    stitched = np.full_like(field, np.nan, dtype=float)
    cumulative_offset = 0.0
    for curve in sorted(curves, key=lambda item: (item.start_field_mT, item.end_field_mT)):
        valid_mask = ~np.isnan(curve.area_signal)
        if not np.any(valid_mask):
            continue
        segment = np.asarray(curve.area_signal[valid_mask], dtype=float) + cumulative_offset
        stitched[valid_mask] = segment
        cumulative_offset = float(segment[-1])
    return stitched


def _split_local_diagnostic_reason(peak_fits: list[PeakFitResult]) -> str | None:
    reasons = [
        f"{peak_fit.label}={peak_fit.fit.derived.get('local_diagnostic_reason')}"
        for peak_fit in peak_fits
        if peak_fit.fit.derived.get("local_diagnostic_reason")
    ]
    if not reasons:
        return None
    return f"split_local_diagnostic_unavailable:{'|'.join(reasons)}"


def _result_fit_local_disagreement(fit: FitResult | None) -> tuple[float | None, bool, str | None]:
    if fit is None:
        return None, False, None
    return (
        fit.derived.get("fit_local_disagreement_ratio"),
        bool(fit.derived.get("fit_local_disagreement_flag", False)),
        fit.derived.get("fit_local_disagreement_reason"),
    )
