"""JSON export for reproducible workflow artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from labsuite.core.types import AnalysisResult


def export_analysis_json(result: AnalysisResult, destination: Path) -> Path:
    """Write the complete analysis payload to JSON."""

    destination.write_text(
        json.dumps(_analysis_result_to_dict(result), indent=2),
        encoding="utf-8",
    )
    return destination


def _analysis_result_to_dict(result: AnalysisResult) -> dict[str, Any]:
    return {
        "modality": result.dataset.modality,
        "source_file": str(result.dataset.source_path),
        "metadata": result.dataset.metadata,
        "recipe": {
            "name": result.recipe_name,
            "config": result.recipe_config,
        },
        "raw_trace": {
            "field_mT": result.dataset.field_mT.tolist(),
            "signal": result.dataset.signal.tolist(),
        },
        "processed_trace": {
            "field_mT": result.processed.field_mT.tolist(),
            "signal": result.processed.signal.tolist(),
            "steps": result.processed.steps,
        },
        "baseline_summaries": {
            "derivative": _baseline_summary_to_dict(result.derivative_baseline),
            "absorption": _baseline_summary_to_dict(result.absorption_baseline),
        },
        "primary_integrated_curves": (
            None
            if result.primary_integrated is None
            else _fit_integrated_curves_to_dict(result.primary_integrated)
        ),
        "fit_local_integrated_curves": (
            None
            if result.fit_local_integrated is None
            else _windowed_integrated_curves_to_dict(result.fit_local_integrated)
        ),
        "local_integrated_curves": (
            None
            if result.local_integrated is None
            else _windowed_integrated_curves_to_dict(result.local_integrated)
        ),
        "integrated_curves": {
            "integration_kind": "diagnostic_full_span",
            "field_mT": result.integrated.field_mT.tolist(),
            "absorption_signal": result.integrated.absorption_signal.tolist(),
            "area_signal": result.integrated.area_signal.tolist(),
            "steps": result.integrated.steps,
        },
        "qc": {
            "fit_local_disagreement_ratio": result.fit_local_disagreement_ratio,
            "fit_local_disagreement_flag": result.fit_local_disagreement_flag,
            "fit_local_disagreement_reason": result.fit_local_disagreement_reason,
        },
        "fit_selection": {
            "selected_mode": result.selected_mode,
            "decision": {
                "requested_mode": result.fit_decision.requested_mode,
                "selected_mode": result.fit_decision.selected_mode,
                "candidate_peak_count": result.fit_decision.candidate_peak_count,
                "split_improvement_ratio": result.fit_decision.split_improvement_ratio,
                "split_threshold": result.fit_decision.split_threshold,
                "reason": result.fit_decision.reason,
                "metrics": result.fit_decision.metrics,
            },
            "single_fit": None if result.single_fit is None else _fit_result_to_dict(result.single_fit),
            "single_fit_attempts": [_fit_attempt_to_dict(item) for item in result.single_fit_attempts],
            "peak_fits": [_peak_fit_result_to_dict(item) for item in result.peak_fits],
            "selected_fit_signal": result.selected_fit_signal.tolist(),
            "selected_residual": result.selected_residual.tolist(),
        },
        "integral_summaries": {
            "total": _integral_summary_to_dict(result.total_integral),
            "fit_local_total": _integral_summary_to_dict(result.fit_local_total_integral),
            "local_total": _integral_summary_to_dict(result.local_total_integral),
            "diagnostic_total": _integral_summary_to_dict(result.diagnostic_total_integral),
            "peaks": [_integral_summary_to_dict(item) for item in result.peak_integrals],
            "fit_local_peaks": [_integral_summary_to_dict(item) for item in result.fit_local_peak_integrals],
            "local_peaks": [_integral_summary_to_dict(item) for item in result.local_peak_integrals],
        },
    }


def _fit_result_to_dict(result) -> dict[str, Any]:
    return {
        "model_name": result.model_name,
        "parameters": result.parameters,
        "derived": result.derived,
        "metrics": result.metrics,
        "parameter_diagnostics": {
            name: _parameter_diagnostic_to_dict(diagnostic)
            for name, diagnostic in result.parameter_diagnostics.items()
        },
        "convergence": _convergence_to_dict(result.convergence),
        "residual_summary": _residual_summary_to_dict(result.residual_summary),
        "feature_summary": None if result.feature_summary is None else _feature_summary_to_dict(result.feature_summary),
        "bound_hits": result.bound_hits,
        "fitted_signal": result.fitted_signal.tolist(),
        "residual": result.residual.tolist(),
        "success": result.success,
    }


def _fit_attempt_to_dict(result) -> dict[str, Any]:
    return {
        "scope": result.scope,
        "fit": _fit_result_to_dict(result.fit),
        "source_window": None if result.source_window is None else {
            "start_index": result.source_window.start_index,
            "end_index": result.source_window.end_index,
            "start_field_mT": result.source_window.start_field_mT,
            "end_field_mT": result.source_window.end_field_mT,
            "peak_index": result.source_window.peak_index,
            "trough_index": result.source_window.trough_index,
            "peak_field_mT": result.source_window.peak_field_mT,
            "trough_field_mT": result.source_window.trough_field_mT,
            "width_mT": result.source_window.width_mT,
            "prominence": result.source_window.prominence,
        },
        "accepted": result.accepted,
        "rejection_reason": result.rejection_reason,
        "selected_for_primary": result.selected_for_primary,
    }


def _peak_fit_result_to_dict(result) -> dict[str, Any]:
    return {
        "label": result.label,
        "window": {
            "start_index": result.window.start_index,
            "end_index": result.window.end_index,
            "start_field_mT": result.window.start_field_mT,
            "end_field_mT": result.window.end_field_mT,
            "peak_index": result.window.peak_index,
            "trough_index": result.window.trough_index,
            "peak_field_mT": result.window.peak_field_mT,
            "trough_field_mT": result.window.trough_field_mT,
            "width_mT": result.window.width_mT,
            "prominence": result.window.prominence,
        },
        "fit": _fit_result_to_dict(result.fit),
        "component_signal": result.component_signal.tolist(),
        "attempts": [_fit_attempt_to_dict(item) for item in result.attempts],
    }


def _integral_summary_to_dict(result) -> dict[str, Any]:
    return {
        "label": result.label,
        "start_field_mT": result.start_field_mT,
        "end_field_mT": result.end_field_mT,
        "absorption_integral": result.absorption_integral,
        "area_integral": result.area_integral,
        "integration_kind": result.integration_kind,
        "window_source": result.window_source,
        "baseline_polyorder": result.baseline_polyorder,
        "integration_window_clipped_by_detected_window": result.integration_window_clipped_by_detected_window,
    }


def _fit_integrated_curves_to_dict(result) -> dict[str, Any]:
    return {
        "field_mT": result.field_mT.tolist(),
        "absorption_signal": result.absorption_signal.tolist(),
        "area_signal": result.area_signal.tolist(),
        "integration_kind": result.integration_kind,
        "model_name": result.model_name,
    }


def _windowed_integrated_curves_to_dict(result) -> dict[str, Any]:
    return {
        "field_mT": result.field_mT.tolist(),
        "absorption_signal": result.absorption_signal.tolist(),
        "area_signal": result.area_signal.tolist(),
        "start_field_mT": result.start_field_mT,
        "end_field_mT": result.end_field_mT,
        "integration_kind": result.integration_kind,
        "window_source": result.window_source,
        "baseline_polyorder": result.baseline_polyorder,
        "integration_window_clipped_by_detected_window": result.integration_window_clipped_by_detected_window,
        "model_name": result.model_name,
    }


def _baseline_summary_to_dict(result) -> dict[str, Any]:
    return {
        "target": result.target,
        "edge_points": result.edge_points,
        "slope": result.slope,
        "intercept": result.intercept,
    }


def _parameter_diagnostic_to_dict(result) -> dict[str, Any]:
    return {
        "value": result.value,
        "stderr": result.stderr,
        "relative_stderr": result.relative_stderr,
        "stderr_missing": result.stderr_missing,
        "min_bound": result.min_bound,
        "max_bound": result.max_bound,
        "hit_min_bound": result.hit_min_bound,
        "hit_max_bound": result.hit_max_bound,
    }


def _convergence_to_dict(result) -> dict[str, Any]:
    return {
        "success": result.success,
        "message": result.message,
        "nfev": result.nfev,
        "nvarys": result.nvarys,
        "errorbars": result.errorbars,
    }


def _residual_summary_to_dict(result) -> dict[str, Any]:
    return {
        "rss": result.rss,
        "rmse": result.rmse,
        "mae": result.mae,
        "max_abs": result.max_abs,
        "mean": result.mean,
        "std": result.std,
    }


def _feature_summary_to_dict(result) -> dict[str, Any]:
    return {
        "positive_extremum_field_mT": result.positive_extremum_field_mT,
        "negative_extremum_field_mT": result.negative_extremum_field_mT,
        "zero_crossing_field_mT": result.zero_crossing_field_mT,
        "peak_to_peak_separation_mT": result.peak_to_peak_separation_mT,
        "integrated_intensity_proxy": result.integrated_intensity_proxy,
    }
