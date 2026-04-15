"""Figure export for single-file analyses."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from labsuite.core.types import AnalysisResult, FitResult, PeakFitResult


def export_analysis_figure(
    result: AnalysisResult,
    destination: Path,
    *,
    show: bool = False,
) -> Path:
    """Save a diagnostic figure with raw, derivative, absorption, and area views."""

    figure, axes = plt.subplots(
        4,
        1,
        figsize=(11, 12),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(result.dataset.field_mT, result.dataset.signal, color="0.25", linewidth=1.25)
    axes[0].set_title("Raw ESR Trace")
    axes[0].set_ylabel("Signal (a.u.)")
    axes[0].grid(alpha=0.2)

    axes[1].plot(
        result.processed.field_mT,
        result.processed.signal,
        label="Processed derivative",
        color="#1f77b4",
        linewidth=1.5,
    )
    axes[1].plot(
        result.processed.field_mT,
        result.selected_fit_signal,
        label="Selected fit",
        color="#d62728",
        linewidth=1.5,
    )

    summary_fit = _selected_fit_for_annotation(result)
    if summary_fit is not None and summary_fit.feature_summary is not None:
        _draw_feature_markers(axes[1], summary_fit)
    _draw_selected_integration_windows(axes[1], result)

    for peak_fit in result.peak_fits:
        axes[1].plot(
            result.processed.field_mT,
            peak_fit.component_signal,
            linewidth=1.0,
            linestyle="--",
            label=f"{peak_fit.label} component",
        )
        axes[1].axvspan(
            peak_fit.window.start_field_mT,
            peak_fit.window.end_field_mT,
            color="#ffcc99",
            alpha=0.15,
        )
        if peak_fit.fit.feature_summary is not None:
            _draw_feature_markers(axes[1], peak_fit.fit, alpha=0.4)

    axes[1].set_title(f"Processed Derivative and Selected Fit ({result.selected_mode})")
    axes[1].set_ylabel("Derivative signal")
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="upper left")
    axes[1].text(
        0.99,
        0.02,
        _build_summary_text(result),
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.75"},
    )

    plotted_curves, absorption_title, area_title = _plotted_integrated_curves(result)
    for series in _integrated_curve_series(result):
        axes[2].plot(
            series["field_mT"],
            series["absorption_signal"],
            color="#2ca02c",
            linewidth=series["linewidth"],
            linestyle=series["linestyle"],
            alpha=series["alpha"],
            label=series["label"],
        )
    axes[2].set_title(absorption_title)
    axes[2].set_ylabel("Absorption")
    axes[2].grid(alpha=0.2)
    axes[2].legend(loc="upper right")

    for series in _integrated_curve_series(result):
        axes[3].plot(
            series["field_mT"],
            series["area_signal"],
            color="#9467bd",
            linewidth=series["linewidth"],
            linestyle=series["linestyle"],
            alpha=series["alpha"],
            label=series["label"],
        )
    axes[3].set_title(area_title)
    axes[3].set_xlabel("Field (mT)")
    axes[3].set_ylabel("Area")
    axes[3].grid(alpha=0.2)
    axes[3].legend(loc="upper right")

    figure.savefig(destination, dpi=200)
    if show:
        plt.show()
    plt.close(figure)
    return destination


def _selected_fit_for_annotation(result: AnalysisResult) -> FitResult | None:
    if result.selected_mode == "single":
        return result.single_fit
    return result.peak_fits[0].fit if result.peak_fits else None


def _draw_feature_markers(axis, fit: FitResult, *, alpha: float = 0.8) -> None:
    feature = fit.feature_summary
    if feature is None:
        return
    axis.axvline(feature.positive_extremum_field_mT, color="#2ca02c", linestyle=":", linewidth=1.0, alpha=alpha)
    axis.axvline(feature.negative_extremum_field_mT, color="#ff7f0e", linestyle=":", linewidth=1.0, alpha=alpha)
    axis.axvline(feature.zero_crossing_field_mT, color="#7f7f7f", linestyle="--", linewidth=1.0, alpha=alpha)


def _draw_selected_integration_windows(axis, result: AnalysisResult) -> None:
    windows = (
        result.local_peak_integrals
        if result.selected_mode == "split" and result.local_peak_integrals
        else [result.local_total_integral]
    )
    for window in windows:
        axis.axvspan(
            window.start_field_mT,
            window.end_field_mT,
            color="#c7e9c0",
            alpha=0.25,
        )


def _build_summary_text(result: AnalysisResult) -> str:
    lines = [
        f"mode: {result.selected_mode}",
        f"deriv baseline: m={result.derivative_baseline.slope:.3g}, b={result.derivative_baseline.intercept:.3g}",
        f"diag abs base:  m={result.absorption_baseline.slope:.3g}, b={result.absorption_baseline.intercept:.3g}",
    ]
    if result.primary_integrated is None:
        lines.append("plotting diagnostic fallback curves")
    else:
        lines.append("plotting fit-derived primary curves")
    if result.fit_local_integrated is not None:
        lines.append("dotted curves: window-matched fit-derived diagnostic")
    if result.local_integrated is not None:
        lines.append("dashed curves: local data-derived diagnostic")
    if result.fit_local_disagreement_flag:
        ratio_text = "NA" if result.fit_local_disagreement_ratio is None else f"{result.fit_local_disagreement_ratio:.3g}"
        lines.append(f"qc warning: fit-local vs data-local ratio={ratio_text}")
        lines.append(f"qc detail: {result.fit_local_disagreement_reason}")
    if result.selected_mode == "single" and result.single_fit is not None:
        if result.single_fit.derived.get("fit_scope") == "detected_window_fallback":
            lines.append("selected via fallback: detected_window_fallback")
        rejected_global = next(
            (
                attempt
                for attempt in result.single_fit_attempts
                if attempt.scope == "global_full_trace" and not attempt.accepted
            ),
            None,
        )
        if rejected_global is not None:
            lines.append(f"global rejected: {rejected_global.rejection_reason}")
        lines.extend(_fit_text_lines("selected", result.single_fit))
    else:
        lines.append(
            f"selected rss={float((result.selected_residual**2).sum()):.3g}, rmse={float(((result.selected_residual**2).mean())**0.5):.3g}"
        )
        for peak_fit in result.peak_fits:
            lines.extend(_fit_text_lines(peak_fit.label, peak_fit.fit, compact=True))
    return "\n".join(lines)


def _fit_text_lines(label: str, fit: FitResult, *, compact: bool = False) -> list[str]:
    feature = fit.feature_summary
    if feature is None:
        return [f"{label}: no feature summary"]
    uncertainty = fit.parameter_diagnostics.get("center_mT")
    center_err = "NA" if uncertainty is None or uncertainty.stderr is None else f"{uncertainty.stderr:.3g}"
    line = (
        f"{label}: B0={feature.zero_crossing_field_mT:.3f}±{center_err} mT, "
        f"pp={feature.peak_to_peak_separation_mT:.3f}, "
        f"R2={fit.metrics['r_squared']:.4f}, RMSE={fit.residual_summary.rmse:.3g}"
    )
    if compact:
        return [line, f"  conv={fit.convergence.success}, bounds={any(fit.bound_hits.values())}"]
    return [
        line,
        f"  conv={fit.convergence.success}, msg={fit.convergence.message}, bounds={any(fit.bound_hits.values())}",
        f"  Ifit={feature.integrated_intensity_proxy if feature.integrated_intensity_proxy is not None else float('nan'):.3g}",
    ]


def _plotted_integrated_curves(result: AnalysisResult):
    if result.primary_integrated is not None:
        return (
            result.primary_integrated,
            "Primary Fit-Derived Absorption Curve",
            "Primary Fit-Derived Area Curve",
        )
    return (
        result.integrated,
        "Diagnostic/Fallback Full-Span Absorption Curve",
        "Diagnostic/Fallback Full-Span Area Curve",
    )


def _integrated_curve_series(result: AnalysisResult):
    series = []
    if result.primary_integrated is not None:
        series.append(
            {
                "label": "Primary fit-derived",
                "field_mT": result.primary_integrated.field_mT,
                "absorption_signal": result.primary_integrated.absorption_signal,
                "area_signal": result.primary_integrated.area_signal,
                "linewidth": 1.4,
                "linestyle": "-",
                "alpha": 1.0,
            }
        )
    else:
        series.append(
            {
                "label": "Diagnostic full-span fallback",
                "field_mT": result.integrated.field_mT,
                "absorption_signal": result.integrated.absorption_signal,
                "area_signal": result.integrated.area_signal,
                "linewidth": 1.4,
                "linestyle": "-",
                "alpha": 1.0,
            }
        )
    if result.fit_local_integrated is not None:
        series.append(
            {
                "label": "Window-matched fit-derived diagnostic",
                "field_mT": result.fit_local_integrated.field_mT,
                "absorption_signal": result.fit_local_integrated.absorption_signal,
                "area_signal": result.fit_local_integrated.area_signal,
                "linewidth": 1.0,
                "linestyle": ":",
                "alpha": 0.95,
            }
        )
    if result.local_integrated is not None:
        series.append(
            {
                "label": "Local data-derived diagnostic",
                "field_mT": result.local_integrated.field_mT,
                "absorption_signal": result.local_integrated.absorption_signal,
                "area_signal": result.local_integrated.area_signal,
                "linewidth": 1.0,
                "linestyle": "--",
                "alpha": 0.85,
            }
        )
    return series
