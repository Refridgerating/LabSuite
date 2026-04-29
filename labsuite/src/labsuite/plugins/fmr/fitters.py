"""Trace fitting and QC helpers for field-swept FMR data."""

from __future__ import annotations

import numpy as np

from labsuite.core.recipes import FmrRecipe
from labsuite.plugins.fmr.candidate_generation import (
    detect_candidate_windows,
    detect_feature,
    median_step,
)
from labsuite.plugins.fmr.model_selection import fit_candidate_models
from labsuite.plugins.fmr.models import (
    FmrComponentFitResult,
    FmrTraceDataset,
    FmrTraceFitResult,
    FmrTraceModelResult,
)
from labsuite.plugins.fmr.preprocess import FmrProcessedTrace
from labsuite.plugins.fmr.spectral_models import (
    double_mixed_derivative_lorentzian,
    mixed_absorption_lorentzian,
    mixed_derivative_lorentzian,
)

__all__ = [
    "assess_trace_fit_quality",
    "detect_candidate_windows",
    "double_mixed_derivative_lorentzian",
    "fit_fmr_trace",
    "mixed_absorption_lorentzian",
    "mixed_derivative_lorentzian",
]


def fit_fmr_trace(
    raw_trace: FmrTraceDataset, processed_trace: FmrProcessedTrace, recipe: FmrRecipe
) -> FmrTraceFitResult:
    """Fit one processed FMR trace with one to three resonance components."""

    field = np.asarray(processed_trace.field_mT, dtype=float)
    signal = np.asarray(processed_trace.signal, dtype=float)
    trace_feature = detect_feature(field, signal, recipe.shape_pair_prominence_ratio)
    selected_fit, candidate_fits, selection_reason, improvement_ratio, windows, diagnostics = (
        fit_candidate_models(field, signal, recipe)
    )
    selected_mode = "single" if selected_fit.n_peaks == 1 else "double"
    if selected_fit.n_peaks == 3:
        selected_mode = "triple"
    selected_components = [
        _clone_component(component, raw_trace) for component in selected_fit.components
    ]
    return FmrTraceFitResult(
        trace_id=raw_trace.trace_id,
        source_file=raw_trace.source_file,
        sample_name=raw_trace.sample_name,
        frequency_GHz=raw_trace.frequency_GHz,
        angle_deg=raw_trace.angle_deg,
        temperature_K=raw_trace.temperature_K,
        model_name=selected_fit.model_name,
        signal_channel=str(
            raw_trace.metadata.get("selected_signal_channel", recipe.signal_channel)
        ),
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
        covariance=None
        if selected_fit.covariance is None
        else [list(row) for row in selected_fit.covariance],
        success=bool(selected_fit.success),
        accepted=False,
        rejection_reason=None,
        requested_mode=recipe.n_peaks if recipe.n_peaks != "auto" else recipe.fit_mode,
        selected_mode=selected_mode,
        selection_reason=selection_reason,
        n_peaks_selected=selected_fit.n_peaks,
        model_selection_method=recipe.multi_peak_selection,
        background_model=recipe.background_model,
        fit_aic=selected_fit.fit_aic,
        fit_bic=selected_fit.fit_bic,
        fit_red_chi2=selected_fit.fit_red_chi2,
        residual_rms=selected_fit.residual_rms,
        residual_structure_score=selected_fit.residual_structure_score,
        background_signal=None
        if selected_fit.background_signal is None
        else np.asarray(selected_fit.background_signal, dtype=float),
        candidate_window_count=len(windows),
        double_fit_improvement_ratio=improvement_ratio,
        double_fit_threshold=recipe.double_fit_min_improvement_ratio,
        candidate_windows=windows,
        single_fit=candidate_fits.get(1),
        double_fit=candidate_fits.get(2),
        selected_components=selected_components,
        partial_component_qc=False,
        r_squared=None
        if selected_fit.metrics.get("r_squared") is None
        else float(selected_fit.metrics["r_squared"]),
        signal_max_abs=None,
        residual_rmse_fraction=None,
        amplitude_snr=None,
        feature_center_mT=trace_feature["feature_center_mT"],
        feature_peak_to_peak_mT=trace_feature["feature_peak_to_peak_mT"],
        center_feature_disagreement_mT=None,
        critical_bound_hit_names=[],
        acceptance_checks={},
        warnings=list(selected_fit.warnings),
        preprocessing_steps=processed_trace.steps,
        baseline_summary=processed_trace.baseline_summary,
        metadata={
            **dict(raw_trace.metadata),
            "feature_positive_extremum_mT": trace_feature["positive_extremum_mT"],
            "feature_negative_extremum_mT": trace_feature["negative_extremum_mT"],
            "model_selection_diagnostics": diagnostics,
        },
    )


def assess_trace_fit_quality(
    fit_result: FmrTraceFitResult, *, recipe: FmrRecipe
) -> tuple[bool, str | None, list[str]]:
    """Apply component-level acceptance criteria without hiding rejected candidates."""

    warnings = list(fit_result.warnings)
    signal_max_abs = (
        float(np.max(np.abs(fit_result.processed_signal)))
        if fit_result.processed_signal.size
        else 0.0
    )
    rmse = fit_result.residual_summary.rmse
    residual_rmse_fraction = (
        0.0
        if signal_max_abs <= 1e-12 and rmse <= 1e-12
        else (float("inf") if signal_max_abs <= 1e-12 else float(rmse / signal_max_abs))
    )
    residual_std = fit_result.residual_summary.std
    strongest = max(
        (
            max(abs(component.amplitude_symmetric), abs(component.amplitude_antisymmetric))
            for component in fit_result.selected_components
        ),
        default=0.0,
    )
    fit_result.signal_max_abs = signal_max_abs
    fit_result.residual_rmse_fraction = residual_rmse_fraction
    fit_result.amplitude_snr = (
        float("inf") if residual_std <= 1e-12 else float(strongest / residual_std)
    )
    fit_result.acceptance_checks = {
        "fit_converged": bool(fit_result.convergence.success and fit_result.success),
        "has_selected_components": bool(fit_result.selected_components),
    }
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
    accepted_by_center: list[FmrComponentFitResult] = []
    for component in fit_result.selected_components:
        accepted, reason, component_warnings = _assess_component(
            component,
            fit_result.field_mT,
            signal_max_abs,
            residual_rmse_fraction,
            residual_std,
            recipe,
        )
        component.accepted = accepted
        component.rejection_reason = reason
        component.confidence = _component_confidence(component, recipe) if accepted else "low"
        component.warnings.extend(component_warnings)
        warnings.extend(f"{component.component_label}:{warning}" for warning in component_warnings)
        critical_hits.extend(
            f"{component.component_label}:{name}" for name in component.critical_bound_hit_names
        )
        if accepted:
            accepted_components += 1
            accepted_by_center.append(component)
        elif reason is not None:
            reasons.append(f"{component.component_label}={reason}")
    _reject_collapsed_components(accepted_by_center, recipe, reasons)
    fit_result.critical_bound_hit_names = critical_hits
    fit_result.partial_component_qc = (
        fit_result.n_peaks_selected > 1
        and 0 < accepted_components < len(fit_result.selected_components)
    )
    if fit_result.partial_component_qc:
        if fit_result.selected_mode == "double":
            warnings.append(
                "partial_double_fit:"
                + "|".join(
                    component.component_label
                    for component in fit_result.selected_components
                    if not component.accepted
                )
            )
        warnings.append(
            "partial_multi_peak_fit:"
            + "|".join(
                component.component_label
                for component in fit_result.selected_components
                if not component.accepted
            )
        )
    fit_result.accepted = any(component.accepted for component in fit_result.selected_components)
    fit_result.rejection_reason = (
        None
        if fit_result.accepted
        else (
            "all_components_rejected:" + "|".join(reasons) if reasons else "no_components_accepted"
        )
    )
    return fit_result.accepted, fit_result.rejection_reason, warnings


def _assess_component(
    component: FmrComponentFitResult,
    field: np.ndarray,
    signal_max_abs: float,
    residual_rmse_fraction: float,
    residual_std: float,
    recipe: FmrRecipe,
) -> tuple[bool, str | None, list[str]]:
    warnings: list[str] = []
    sweep = abs(float(field[-1] - field[0]))
    guard = max(median_step(field) * 2.0, recipe.field_guard_fraction * sweep)
    amplitude = max(abs(component.amplitude_symmetric), abs(component.amplitude_antisymmetric))
    amplitude_snr = float("inf") if residual_std <= 1e-12 else float(amplitude / residual_std)
    critical_hits = [name for name in ("H_res_mT", "DeltaH_mT") if component.bound_hits.get(name)]
    disagreement = (
        None
        if component.feature_center_mT is None
        else float(abs(component.H_res_mT - component.feature_center_mT))
    )
    component.signal_max_abs = signal_max_abs
    component.residual_rmse_fraction = residual_rmse_fraction
    component.amplitude_snr = amplitude_snr
    component.center_feature_disagreement_mT = disagreement
    component.critical_bound_hit_names = critical_hits
    max_linewidth = recipe.max_linewidth_mT or recipe.linewidth_max_sweep_fraction * sweep
    min_linewidth = recipe.min_linewidth_mT or 0.0
    component.acceptance_checks = {
        "center_inside_guard": bool(
            float(np.min(field)) + guard <= component.H_res_mT <= float(np.max(field)) - guard
        ),
        "linewidth_positive": bool(component.DeltaH_mT > 0.0),
        "linewidth_above_minimum": bool(component.DeltaH_mT >= min_linewidth),
        "linewidth_within_fraction_limit": bool(component.DeltaH_mT <= max_linewidth),
        "residual_rmse_within_limit": bool(
            residual_rmse_fraction <= recipe.residual_rmse_max_signal_fraction
        ),
        "amplitude_snr_within_limit": bool(amplitude_snr >= recipe.amplitude_snr_min),
        "critical_bound_hit_clear": not critical_hits,
    }
    if component.feature_center_mT is None:
        component.acceptance_checks["shape_center_consistent"] = True
    else:
        tolerance = max(
            recipe.shape_center_tolerance_min_mT,
            recipe.shape_center_tolerance_linewidth_fraction * max(component.DeltaH_mT, 0.0),
        )
        component.acceptance_checks["shape_center_consistent"] = bool(
            disagreement is not None and disagreement <= tolerance
        )
    if critical_hits and recipe.critical_bound_hit_policy == "warn":
        warnings.append(f"critical_parameter_hit_bound:{'|'.join(critical_hits)}")
    if (
        not component.acceptance_checks["shape_center_consistent"]
        and recipe.shape_consistency_policy == "warn"
    ):
        warnings.append("shape_center_inconsistent_with_feature_reference")
    if not component.acceptance_checks["center_inside_guard"]:
        return False, "resonance_field_outside_guard", warnings
    if not component.acceptance_checks["linewidth_positive"]:
        return False, "linewidth_not_positive", warnings
    if not component.acceptance_checks["linewidth_above_minimum"]:
        return False, "linewidth_below_minimum", warnings
    if not component.acceptance_checks["linewidth_within_fraction_limit"]:
        return False, "linewidth_exceeds_limit", warnings
    if not component.acceptance_checks["residual_rmse_within_limit"]:
        return False, "residual_rmse_exceeds_signal_fraction_limit", warnings
    if not component.acceptance_checks["amplitude_snr_within_limit"]:
        return False, "amplitude_snr_below_threshold", warnings
    if critical_hits and recipe.critical_bound_hit_policy == "reject":
        return False, f"critical_parameter_hit_bound:{'|'.join(critical_hits)}", warnings
    if (
        not component.acceptance_checks["shape_center_consistent"]
        and recipe.shape_consistency_policy == "reject"
    ):
        return False, "shape_center_inconsistent_with_feature_reference", warnings
    return True, None, warnings


def _reject_collapsed_components(
    components: list[FmrComponentFitResult],
    recipe: FmrRecipe,
    reasons: list[str],
) -> None:
    components.sort(key=lambda item: item.H_res_mT)
    for left, right in zip(components, components[1:], strict=False):
        if abs(right.H_res_mT - left.H_res_mT) >= recipe.min_peak_separation_mT:
            continue
        weaker = (
            right
            if max(abs(right.amplitude_symmetric), abs(right.amplitude_antisymmetric))
            < max(abs(left.amplitude_symmetric), abs(left.amplitude_antisymmetric))
            else left
        )
        weaker.accepted = False
        weaker.confidence = "low"
        weaker.rejection_reason = "peak_separation_below_minimum"
        reasons.append(f"{weaker.component_label}=peak_separation_below_minimum")


def _component_confidence(component: FmrComponentFitResult, recipe: FmrRecipe) -> str:
    if component.amplitude_snr is None:
        return "medium"
    if component.amplitude_snr >= recipe.amplitude_snr_min * 2.0:
        return "high"
    return "medium"


def _clone_component(
    component: FmrComponentFitResult, trace: FmrTraceDataset
) -> FmrComponentFitResult:
    metadata = dict(component.metadata)
    for name in (
        "field_polarity",
        "field_polarity_raw",
        "field_polarity_column",
        "sample_id",
        "measurement_id",
        "replicate_id",
        "geometry",
    ):
        if name in trace.metadata:
            metadata[name] = trace.metadata.get(name)
    return FmrComponentFitResult(
        component_id=f"{trace.trace_id}:{component.component_label}",
        component_label=component.component_label,
        H_res_mT=component.H_res_mT,
        DeltaH_mT=component.DeltaH_mT,
        amplitude_symmetric=component.amplitude_symmetric,
        amplitude_antisymmetric=component.amplitude_antisymmetric,
        field_mT=np.asarray(component.field_mT, dtype=float).copy(),
        component_signal=np.asarray(component.component_signal, dtype=float).copy(),
        absorption_signal=None
        if component.absorption_signal is None
        else np.asarray(component.absorption_signal, dtype=float).copy(),
        peak_index=component.peak_index,
        branch_id=component.branch_id,
        confidence=component.confidence,
        parameter_diagnostics=dict(component.parameter_diagnostics),
        bound_hits=dict(component.bound_hits),
        accepted=component.accepted,
        rejection_reason=component.rejection_reason,
        signal_max_abs=component.signal_max_abs,
        residual_rmse_fraction=component.residual_rmse_fraction,
        amplitude_snr=component.amplitude_snr,
        feature_center_mT=component.feature_center_mT,
        feature_peak_to_peak_mT=component.feature_peak_to_peak_mT,
        center_feature_disagreement_mT=component.center_feature_disagreement_mT,
        critical_bound_hit_names=list(component.critical_bound_hit_names),
        acceptance_checks=dict(component.acceptance_checks),
        warnings=list(component.warnings),
        metadata=metadata,
        resonance_metrics=component.resonance_metrics,
    )


def _trace_model_from_payload(payload: dict) -> FmrTraceModelResult:
    return FmrTraceModelResult(**payload)
