"""Figure export for single-file analyses."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text

from labsuite.core.resonance_metrics import ResonanceAreaWindow, ResonanceModeMetrics
from labsuite.core.types import AnalysisResult, FitResult

_SUMMARY_FONT_SIZE = 8
_SUMMARY_FONT_FAMILY = "monospace"
_PLUS_MINUS = "\u00b1"
_DELTA = "\u0394"


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
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(hspace=0.22, h_pad=0.2)

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
    _place_axis_legend_below(axes[1], ncols=2, y_offset=-0.16)
    _place_axis_footer_text_below(axes[1], _build_summary_text(result), y_offset=-0.52)

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
    _draw_absorption_resonance_metrics(axes[2], result)
    _place_axis_legend_below(axes[2], ncols=1, y_offset=-0.18)

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
    _draw_area_window_annotations(axes[3], result)
    _place_axis_legend_below(axes[3], ncols=1, y_offset=-0.24)

    figure.savefig(destination, dpi=200)
    if show:
        plt.show()
    plt.close(figure)
    return destination


def _selected_fit_for_annotation(result: AnalysisResult) -> FitResult | None:
    if result.selected_mode == "single":
        return result.single_fit
    return result.peak_fits[0].fit if result.peak_fits else None


def _place_axis_legend_below(axis, *, ncols: int = 1, y_offset: float = -0.18):
    legend = axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, y_offset),
        ncols=ncols,
        frameon=True,
        borderaxespad=0.0,
    )
    legend.set_in_layout(True)
    return legend


def _place_axis_footer_text_below(axis, text: str, *, y_offset: float) -> None:
    axis.figure.canvas.draw()
    renderer = axis.figure.canvas.get_renderer()
    wrapped_text = _wrap_footer_text_to_axis_width(
        axis,
        text,
        renderer,
        fontsize=_SUMMARY_FONT_SIZE,
        family=_SUMMARY_FONT_FAMILY,
    )
    footer = axis.text(
        0.5,
        y_offset,
        wrapped_text,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=_SUMMARY_FONT_SIZE,
        family=_SUMMARY_FONT_FAMILY,
        clip_on=False,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "0.75",
        },
    )
    footer.set_in_layout(True)


def _wrap_footer_text_to_axis_width(
    axis, text: str, renderer, *, fontsize: float, family: str
) -> str:
    axis_width_px = axis.get_window_extent(renderer=renderer).width
    max_text_width_px = max(axis_width_px - 24.0, 160.0)
    font_properties = FontProperties(family=family, size=fontsize)
    char_width_px = (
        _measure_text_block_width(axis.figure, "M" * 40, renderer, font_properties) / 40.0
    )
    max_chars = max(28, int(max_text_width_px / max(char_width_px, 1.0)))
    wrapped_text = _wrap_footer_lines(text, width=max_chars)
    while max_chars > 28:
        wrapped_width = _measure_text_block_width(
            axis.figure, wrapped_text, renderer, font_properties
        )
        if wrapped_width <= max_text_width_px:
            break
        max_chars -= 2
        wrapped_text = _wrap_footer_lines(text, width=max_chars)
    return wrapped_text


def _measure_text_block_width(
    figure, text: str, renderer, font_properties: FontProperties
) -> float:
    probe = Text(0.0, 0.0, text, fontproperties=font_properties)
    probe.set_figure(figure)
    return probe.get_window_extent(renderer=renderer).width


def _wrap_footer_lines(text: str, *, width: int) -> str:
    wrapped_lines: list[str] = []
    for raw_line in text.splitlines():
        if not raw_line:
            wrapped_lines.append("")
            continue
        stripped = raw_line.lstrip()
        indent = raw_line[: len(raw_line) - len(stripped)]
        if ": " in stripped:
            prefix, remainder = stripped.split(": ", 1)
            wrapped_lines.extend(
                textwrap.wrap(
                    remainder,
                    width=max(width, 16),
                    initial_indent=f"{indent}{prefix}: ",
                    subsequent_indent=" " * (len(indent) + len(prefix) + 2),
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                or [f"{indent}{prefix}: "]
            )
            continue
        wrapped_lines.extend(
            textwrap.wrap(
                stripped,
                width=max(width - len(indent), 16),
                initial_indent=indent,
                subsequent_indent=indent,
                break_long_words=False,
                break_on_hyphens=False,
            )
            or [raw_line]
        )
    return "\n".join(wrapped_lines)


def _draw_feature_markers(axis, fit: FitResult, *, alpha: float = 0.8) -> None:
    feature = fit.feature_summary
    if feature is None:
        return
    axis.axvline(
        feature.positive_extremum_field_mT,
        color="#2ca02c",
        linestyle=":",
        linewidth=1.0,
        alpha=alpha,
    )
    axis.axvline(
        feature.negative_extremum_field_mT,
        color="#ff7f0e",
        linestyle=":",
        linewidth=1.0,
        alpha=alpha,
    )
    axis.axvline(
        feature.zero_crossing_field_mT, color="#7f7f7f", linestyle="--", linewidth=1.0, alpha=alpha
    )


def _draw_selected_integration_windows(axis, result: AnalysisResult) -> None:
    windows = (
        [window for window in result.local_peak_integrals if window.area_integral is not None]
        if result.selected_mode == "split" and result.local_peak_integrals
        else (
            [result.local_total_integral]
            if result.local_total_integral.area_integral is not None
            else []
        )
    )
    for window in windows:
        axis.axvspan(
            window.start_field_mT,
            window.end_field_mT,
            color="#c7e9c0",
            alpha=0.25,
        )


def _draw_absorption_resonance_metrics(axis, result: AnalysisResult) -> None:
    if not result.resonance_metrics:
        return
    if not result.resonance_metrics_config.get("plot_halfmax_markers", False):
        return
    for metrics in result.resonance_metrics:
        if not metrics.success:
            continue
        label_prefix = metrics.owner_id
        axis.axvline(
            metrics.hres,
            color="#111827",
            linestyle="--",
            linewidth=1.0,
            alpha=0.75,
            label=f"{label_prefix} hres",
        )
        if metrics.h_left_half is not None:
            axis.axvline(
                metrics.h_left_half,
                color="#0f766e",
                linestyle=":",
                linewidth=1.0,
                alpha=0.75,
                label=f"{label_prefix} half-max left",
            )
        if metrics.h_right_half is not None:
            axis.axvline(
                metrics.h_right_half,
                color="#b45309",
                linestyle=":",
                linewidth=1.0,
                alpha=0.75,
                label=f"{label_prefix} half-max right",
            )
    if result.resonance_metrics_config.get("plot_area_windows", False):
        for area_window in _visible_area_windows(result.resonance_metrics):
            if area_window.start_field_mT is None or area_window.end_field_mT is None:
                continue
            axis.axvspan(
                area_window.start_field_mT, area_window.end_field_mT, color="#fde68a", alpha=0.08
            )


def _draw_area_window_annotations(axis, result: AnalysisResult) -> None:
    if not result.resonance_metrics:
        return
    if not result.resonance_metrics_config.get("plot_area_windows", False):
        return
    for area_window in _visible_area_windows(result.resonance_metrics):
        if area_window.start_field_mT is None or area_window.end_field_mT is None:
            continue
        axis.axvspan(
            area_window.start_field_mT, area_window.end_field_mT, color="#ddd6fe", alpha=0.08
        )


def _visible_area_windows(metrics_records: list[ResonanceModeMetrics]) -> list[ResonanceAreaWindow]:
    visible: list[ResonanceAreaWindow] = []
    for metrics in metrics_records:
        if not metrics.success:
            continue
        for area_window in metrics.area_windows:
            if area_window.area is not None and round(area_window.multiplier, 9) == round(1.0, 9):
                visible.append(area_window)
    return visible


def _build_summary_text(result: AnalysisResult) -> str:
    derivative_baseline = result.derivative_baseline
    absorption_baseline = result.absorption_baseline
    lines = [
        f"mode: {result.selected_mode}",
        f"deriv baseline: m={derivative_baseline.slope:.3g}, b={derivative_baseline.intercept:.3g}",
        f"diag abs base:  m={absorption_baseline.slope:.3g}, b={absorption_baseline.intercept:.3g}",
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
        ratio_text = (
            "NA"
            if result.fit_local_disagreement_ratio is None
            else f"{result.fit_local_disagreement_ratio:.3g}"
        )
        lines.append(f"qc warning: fit-local vs data-local ratio={ratio_text}")
        lines.append(f"qc detail: {_format_summary_detail(result.fit_local_disagreement_reason)}")
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
        selected_rss = float((result.selected_residual**2).sum())
        selected_rmse = float(((result.selected_residual**2).mean()) ** 0.5)
        lines.append(f"selected rss={selected_rss:.3g}, rmse={selected_rmse:.3g}")
        for peak_fit in result.peak_fits:
            lines.extend(_fit_text_lines(peak_fit.label, peak_fit.fit, compact=True))
    return "\n".join(lines)


def _fit_text_lines(label: str, fit: FitResult, *, compact: bool = False) -> list[str]:
    feature = fit.feature_summary
    if feature is None:
        return [f"{label}: no feature summary"]
    uncertainty = fit.parameter_diagnostics.get("center_mT")
    center_err = (
        "NA" if uncertainty is None or uncertainty.stderr is None else f"{uncertainty.stderr:.3g}"
    )
    line = _format_fit_summary_line(
        label,
        zero_crossing_field_mT=feature.zero_crossing_field_mT,
        center_err=center_err,
        peak_to_peak_separation_mT=feature.peak_to_peak_separation_mT,
        r_squared=fit.metrics["r_squared"],
        rmse=fit.residual_summary.rmse,
    )
    if compact:
        return [line, f"  conv={fit.convergence.success}, bounds={any(fit.bound_hits.values())}"]
    bound_hit = any(fit.bound_hits.values())
    fit_integral = feature.integrated_intensity_proxy
    if fit_integral is None:
        fit_integral = float("nan")
    return [
        line,
        f"  conv={fit.convergence.success}, msg={fit.convergence.message}, bounds={bound_hit}",
        f"  Ifit={fit_integral:.3g}",
    ]


def _format_fit_summary_line(
    label: str,
    *,
    zero_crossing_field_mT: float,
    center_err: str,
    peak_to_peak_separation_mT: float,
    r_squared: float,
    rmse: float,
) -> str:
    return (
        f"{label}: Hres={zero_crossing_field_mT:.3f}{_PLUS_MINUS}{center_err} mT, "
        f"{_DELTA}Hpp={peak_to_peak_separation_mT:.3f}, "
        f"R2={r_squared:.4f}, RMSE={rmse:.3g}"
    )


def _format_summary_detail(detail: str | None) -> str:
    if not detail:
        return "NA"
    return re.sub(r"([:;|])(?=\S)", r"\1 ", detail)


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
