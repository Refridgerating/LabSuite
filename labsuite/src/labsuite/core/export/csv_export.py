"""CSV export for tabular trace and fit data."""

from __future__ import annotations

import csv
from pathlib import Path

from labsuite.core.types import AnalysisResult, FitResult, PeakFitResult


def export_analysis_csv(
    result: AnalysisResult,
    destination: Path,
    summary_destination: Path | None = None,
) -> Path:
    """Export aligned raw, processed, and fitted traces to CSV."""

    component_signals = {peak_fit.label: peak_fit.component_signal for peak_fit in result.peak_fits}
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = [
            "field_mT",
            "raw_derivative_signal",
            "processed_derivative_signal",
            "primary_absorption_signal",
            "primary_area_signal",
            "fit_local_absorption_signal",
            "fit_local_area_signal",
            "local_absorption_signal",
            "local_area_signal",
            "diagnostic_absorption_signal",
            "diagnostic_area_signal",
            "selected_fit_signal",
            "selected_residual",
        ]
        header.extend(f"{label}_component_signal" for label in component_signals)
        writer.writerow(header)
        primary_absorption_signal = (
            result.primary_integrated.absorption_signal
            if result.primary_integrated is not None
            else result.integrated.field_mT * 0.0 + float("nan")
        )
        primary_area_signal = (
            result.primary_integrated.area_signal
            if result.primary_integrated is not None
            else result.integrated.field_mT * 0.0 + float("nan")
        )
        fit_local_absorption_signal = (
            result.fit_local_integrated.absorption_signal
            if result.fit_local_integrated is not None
            else result.integrated.field_mT * 0.0 + float("nan")
        )
        fit_local_area_signal = (
            result.fit_local_integrated.area_signal
            if result.fit_local_integrated is not None
            else result.integrated.field_mT * 0.0 + float("nan")
        )
        local_absorption_signal = (
            result.local_integrated.absorption_signal
            if result.local_integrated is not None
            else result.integrated.field_mT * 0.0 + float("nan")
        )
        local_area_signal = (
            result.local_integrated.area_signal
            if result.local_integrated is not None
            else result.integrated.field_mT * 0.0 + float("nan")
        )
        for row_index, row in enumerate(
            zip(
                result.dataset.field_mT,
                result.dataset.signal,
                result.processed.signal,
                primary_absorption_signal,
                primary_area_signal,
                fit_local_absorption_signal,
                fit_local_area_signal,
                local_absorption_signal,
                local_area_signal,
                result.integrated.absorption_signal,
                result.integrated.area_signal,
                result.selected_fit_signal,
                result.selected_residual,
                strict=True,
            )
        ):
            values = [f"{value:.10g}" for value in row]
            values.extend(f"{component_signals[label][row_index]:.10g}" for label in component_signals)
            writer.writerow(values)

    if summary_destination is not None:
        export_analysis_summary_csv(result, summary_destination)

    return destination


def export_analysis_summary_csv(result: AnalysisResult, destination: Path) -> Path:
    """Export scalar analysis diagnostics to a sidecar summary CSV."""

    fieldnames = [
        "target",
        "mode",
        "model_name",
        "success",
        "convergence_message",
        "nfev",
        "errorbars",
        "derivative_baseline_edge_points",
        "derivative_baseline_slope",
        "derivative_baseline_intercept",
        "absorption_baseline_edge_points",
        "absorption_baseline_slope",
        "absorption_baseline_intercept",
        "r_squared",
        "chi_square",
        "reduced_chi_square",
        "sum_squared_residuals",
        "residual_rss",
        "residual_rmse",
        "residual_mae",
        "residual_max_abs",
        "residual_mean",
        "residual_std",
        "positive_extremum_field_mT",
        "negative_extremum_field_mT",
        "zero_crossing_field_mT",
        "peak_to_peak_separation_mT",
        "integrated_intensity_proxy",
        "fit_local_windowed_intensity_proxy",
        "local_windowed_intensity_proxy",
        "fit_local_disagreement_ratio",
        "fit_local_disagreement_flag",
        "fit_local_disagreement_reason",
        "diagnostic_full_span_area_integral",
        "integral_kind",
        "integration_start_field_mT",
        "integration_end_field_mT",
        "baseline_polyorder",
        "fit_scope",
        "fit_valid",
        "fit_rejection_reason",
        "selected_for_primary",
        "integration_window_clipped_by_detected_window",
        "amplitude",
        "amplitude_stderr",
        "amplitude_relative_stderr",
        "amplitude_hit_bound",
        "center_mT",
        "center_mT_stderr",
        "center_mT_relative_stderr",
        "center_mT_hit_bound",
        "gamma_mT",
        "gamma_mT_stderr",
        "gamma_mT_relative_stderr",
        "gamma_mT_hit_bound",
        "offset",
        "offset_stderr",
        "offset_relative_stderr",
        "offset_hit_bound",
        "window_start_field_mT",
        "window_end_field_mT",
        "window_peak_field_mT",
        "window_trough_field_mT",
        "window_prominence",
        "window_width_mT",
    ]

    rows: list[dict[str, object]] = []
    if result.single_fit is not None:
        rows.append(
            _fit_summary_row(
                target="single_candidate" if result.selected_mode == "split" else "selected",
                mode="single",
                fit=result.single_fit,
                result=result,
            )
        )
    for peak_fit in result.peak_fits:
        rows.append(
            _fit_summary_row(
                target=peak_fit.label if result.selected_mode == "split" else f"{peak_fit.label}_candidate",
                mode="split",
                fit=peak_fit.fit,
                result=result,
                peak_fit=peak_fit,
            )
        )
    if result.selected_mode == "split" and result.peak_fits:
        rows.insert(
            0,
            _selected_split_row(result),
        )

    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return destination


def _fit_summary_row(
    *,
    target: str,
    mode: str,
    fit: FitResult,
    result: AnalysisResult,
    peak_fit: PeakFitResult | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "target": target,
        "mode": mode,
        "model_name": fit.model_name,
        "success": fit.success,
        "convergence_message": fit.convergence.message,
        "nfev": fit.convergence.nfev,
        "errorbars": fit.convergence.errorbars,
        "derivative_baseline_edge_points": result.derivative_baseline.edge_points,
        "derivative_baseline_slope": result.derivative_baseline.slope,
        "derivative_baseline_intercept": result.derivative_baseline.intercept,
        "absorption_baseline_edge_points": result.absorption_baseline.edge_points,
        "absorption_baseline_slope": result.absorption_baseline.slope,
        "absorption_baseline_intercept": result.absorption_baseline.intercept,
        "r_squared": fit.metrics.get("r_squared"),
        "chi_square": fit.metrics.get("chi_square"),
        "reduced_chi_square": fit.metrics.get("reduced_chi_square"),
        "sum_squared_residuals": fit.metrics.get("sum_squared_residuals"),
        "residual_rss": fit.residual_summary.rss,
        "residual_rmse": fit.residual_summary.rmse,
        "residual_mae": fit.residual_summary.mae,
        "residual_max_abs": fit.residual_summary.max_abs,
        "residual_mean": fit.residual_summary.mean,
        "residual_std": fit.residual_summary.std,
        "positive_extremum_field_mT": None if fit.feature_summary is None else fit.feature_summary.positive_extremum_field_mT,
        "negative_extremum_field_mT": None if fit.feature_summary is None else fit.feature_summary.negative_extremum_field_mT,
        "zero_crossing_field_mT": None if fit.feature_summary is None else fit.feature_summary.zero_crossing_field_mT,
        "peak_to_peak_separation_mT": None if fit.feature_summary is None else fit.feature_summary.peak_to_peak_separation_mT,
        "integrated_intensity_proxy": None if fit.feature_summary is None else fit.feature_summary.integrated_intensity_proxy,
        "fit_local_windowed_intensity_proxy": fit.derived.get("fit_local_windowed_intensity_proxy"),
        "local_windowed_intensity_proxy": fit.derived.get("local_windowed_intensity_proxy"),
        "fit_local_disagreement_ratio": fit.derived.get("fit_local_disagreement_ratio"),
        "fit_local_disagreement_flag": fit.derived.get("fit_local_disagreement_flag"),
        "fit_local_disagreement_reason": fit.derived.get("fit_local_disagreement_reason"),
        "diagnostic_full_span_area_integral": result.diagnostic_total_integral.area_integral,
        "integral_kind": fit.derived.get("integration_kind"),
        "integration_start_field_mT": fit.derived.get("integration_start_field_mT"),
        "integration_end_field_mT": fit.derived.get("integration_end_field_mT"),
        "baseline_polyorder": fit.derived.get("integration_baseline_polyorder"),
        "fit_scope": fit.derived.get("fit_scope"),
        "fit_valid": fit.derived.get("fit_valid"),
        "fit_rejection_reason": fit.derived.get("fit_rejection_reason"),
        "selected_for_primary": fit.derived.get("selected_for_primary"),
        "integration_window_clipped_by_detected_window": fit.derived.get(
            "integration_window_clipped_by_detected_window"
        ),
        "window_start_field_mT": None if peak_fit is None else peak_fit.window.start_field_mT,
        "window_end_field_mT": None if peak_fit is None else peak_fit.window.end_field_mT,
        "window_peak_field_mT": None if peak_fit is None else peak_fit.window.peak_field_mT,
        "window_trough_field_mT": None if peak_fit is None else peak_fit.window.trough_field_mT,
        "window_prominence": None if peak_fit is None else peak_fit.window.prominence,
        "window_width_mT": None if peak_fit is None else peak_fit.window.width_mT,
    }
    for parameter_name in ("amplitude", "center_mT", "gamma_mT", "offset"):
        diagnostic = fit.parameter_diagnostics.get(parameter_name)
        row[parameter_name] = None if diagnostic is None else diagnostic.value
        row[f"{parameter_name}_stderr"] = None if diagnostic is None else diagnostic.stderr
        row[f"{parameter_name}_relative_stderr"] = None if diagnostic is None else diagnostic.relative_stderr
        row[f"{parameter_name}_hit_bound"] = fit.bound_hits.get(parameter_name)
    return row


def _selected_split_row(result: AnalysisResult) -> dict[str, object]:
    return {
        "target": "selected",
        "mode": "split",
        "model_name": "split_derivative_lorentzian",
        "success": True,
        "convergence_message": "selected split fit",
        "nfev": sum(peak.fit.convergence.nfev or 0 for peak in result.peak_fits),
        "errorbars": all(peak.fit.convergence.errorbars for peak in result.peak_fits),
        "derivative_baseline_edge_points": result.derivative_baseline.edge_points,
        "derivative_baseline_slope": result.derivative_baseline.slope,
        "derivative_baseline_intercept": result.derivative_baseline.intercept,
        "absorption_baseline_edge_points": result.absorption_baseline.edge_points,
        "absorption_baseline_slope": result.absorption_baseline.slope,
        "absorption_baseline_intercept": result.absorption_baseline.intercept,
        "r_squared": result.fit_decision.metrics.get("split_r_squared"),
        "chi_square": float((result.selected_residual**2).sum()),
        "reduced_chi_square": float((result.selected_residual**2).sum()) / max(result.selected_residual.size - 1, 1),
        "sum_squared_residuals": float((result.selected_residual**2).sum()),
        "residual_rss": float((result.selected_residual**2).sum()),
        "residual_rmse": float((result.selected_residual**2).mean() ** 0.5),
        "residual_mae": float(abs(result.selected_residual).mean()),
        "residual_max_abs": float(abs(result.selected_residual).max()),
        "residual_mean": float(result.selected_residual.mean()),
        "residual_std": float(result.selected_residual.std()),
        "positive_extremum_field_mT": None,
        "negative_extremum_field_mT": None,
        "zero_crossing_field_mT": None,
        "peak_to_peak_separation_mT": None,
        "integrated_intensity_proxy": result.total_integral.area_integral,
        "fit_local_windowed_intensity_proxy": result.fit_local_total_integral.area_integral,
        "local_windowed_intensity_proxy": result.local_total_integral.area_integral,
        "fit_local_disagreement_ratio": result.fit_local_disagreement_ratio,
        "fit_local_disagreement_flag": result.fit_local_disagreement_flag,
        "fit_local_disagreement_reason": result.fit_local_disagreement_reason,
        "diagnostic_full_span_area_integral": result.diagnostic_total_integral.area_integral,
        "integral_kind": result.total_integral.integration_kind,
        "integration_start_field_mT": result.local_total_integral.start_field_mT,
        "integration_end_field_mT": result.local_total_integral.end_field_mT,
        "baseline_polyorder": result.local_total_integral.baseline_polyorder,
        "fit_scope": "split_selected",
        "fit_valid": True,
        "fit_rejection_reason": None,
        "selected_for_primary": True,
        "integration_window_clipped_by_detected_window": result.local_total_integral.integration_window_clipped_by_detected_window,
        "amplitude": None,
        "amplitude_stderr": None,
        "amplitude_relative_stderr": None,
        "amplitude_hit_bound": None,
        "center_mT": None,
        "center_mT_stderr": None,
        "center_mT_relative_stderr": None,
        "center_mT_hit_bound": None,
        "gamma_mT": None,
        "gamma_mT_stderr": None,
        "gamma_mT_relative_stderr": None,
        "gamma_mT_hit_bound": None,
        "offset": None,
        "offset_stderr": None,
        "offset_relative_stderr": None,
        "offset_hit_bound": None,
        "window_start_field_mT": None,
        "window_end_field_mT": None,
        "window_peak_field_mT": None,
        "window_trough_field_mT": None,
        "window_prominence": None,
        "window_width_mT": None,
    }
