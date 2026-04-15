"""Explicit ESR preprocessing path for the first CLI slice."""

from __future__ import annotations

import numpy as np

from labsuite.core.preprocessing import (
    cumulative_integral,
    normalize_max_abs,
    savgol_smooth,
    scalar_integral,
    subtract_linear_edge_baseline,
)
from labsuite.core.recipes import EsrPreprocessingRecipe
from labsuite.core.types import (
    BaselineSummary,
    IntegratedCurves,
    IntegralSummary,
    PeakWindow,
    PrimaryIntegratedCurves,
    ProcessedTrace,
    TraceDataset,
)


def apply_esr_preprocessing(
    dataset: TraceDataset,
    recipe: EsrPreprocessingRecipe,
) -> tuple[ProcessedTrace, BaselineSummary]:
    """Apply the first ESR preprocessing chain to the raw trace."""

    baseline_corrected, _baseline_curve, baseline = subtract_linear_edge_baseline(
        dataset.field_mT,
        dataset.signal,
        edge_points=recipe.derivative_baseline_edge_points,
    )
    smoothed, window_length = savgol_smooth(
        baseline_corrected,
        window_length=recipe.savgol_window,
        polyorder=recipe.savgol_polyorder,
    )
    normalized_signal, scale_factor = normalize_max_abs(smoothed) if recipe.normalize else (smoothed, 1.0)

    return (
        ProcessedTrace(
            field_mT=dataset.field_mT.copy(),
            signal=normalized_signal,
            steps=[
                {
                    "name": "linear_edge_baseline_subtraction",
                    "parameters": {
                        "edge_points": recipe.derivative_baseline_edge_points,
                        "slope": baseline["slope"],
                        "intercept": baseline["intercept"],
                    },
                },
                {
                    "name": "savgol_smoothing",
                    "parameters": {
                        "window_length": window_length,
                        "polyorder": recipe.savgol_polyorder,
                    },
                },
                {
                    "name": "max_abs_normalization",
                    "parameters": {
                        "enabled": recipe.normalize,
                        "scale_factor": scale_factor,
                    },
                },
            ],
        ),
        BaselineSummary(
            target="derivative",
            edge_points=recipe.derivative_baseline_edge_points,
            slope=baseline["slope"],
            intercept=baseline["intercept"],
        ),
    )


def integrate_esr_trace(
    trace: ProcessedTrace,
    recipe: EsrPreprocessingRecipe,
) -> tuple[IntegratedCurves, BaselineSummary]:
    """Build diagnostic full-span absorption and area curves from a processed derivative trace."""

    absorption_raw = cumulative_integral(trace.field_mT, trace.signal)
    absorption_corrected, _absorption_baseline_curve, absorption_baseline = subtract_linear_edge_baseline(
        trace.field_mT,
        absorption_raw,
        edge_points=recipe.absorption_baseline_edge_points,
    )
    area_signal = cumulative_integral(trace.field_mT, absorption_corrected)

    return (
        IntegratedCurves(
            field_mT=trace.field_mT.copy(),
            absorption_signal=absorption_corrected,
            area_signal=area_signal,
            steps=[
                {
                    "name": "cumulative_absorption_integration",
                    "parameters": {
                        "method": "trapezoidal",
                        "kind": "diagnostic_full_span",
                    },
                },
                {
                    "name": "linear_edge_baseline_subtraction",
                    "parameters": {
                        "target": "absorption",
                        "edge_points": recipe.absorption_baseline_edge_points,
                        "slope": absorption_baseline["slope"],
                        "intercept": absorption_baseline["intercept"],
                    },
                },
                {
                    "name": "cumulative_area_integration",
                    "parameters": {
                        "method": "trapezoidal",
                        "kind": "diagnostic_full_span",
                    },
                },
            ],
        ),
        BaselineSummary(
            target="absorption",
            edge_points=recipe.absorption_baseline_edge_points,
            slope=absorption_baseline["slope"],
            intercept=absorption_baseline["intercept"],
        ),
    )


def integrate_local_resonance(
    trace: ProcessedTrace,
    recipe: EsrPreprocessingRecipe,
    *,
    label: str,
    center_mT: float,
    gamma_mT: float,
    peak_window: PeakWindow | None = None,
) -> IntegralSummary:
    """Compute a resonance-local integral from off-resonance baseline data only."""

    summary, _curves = integrate_local_resonance_with_curves(
        trace,
        recipe,
        label=label,
        center_mT=center_mT,
        gamma_mT=gamma_mT,
        peak_window=peak_window,
    )
    return summary


def integrate_local_resonance_with_curves(
    trace: ProcessedTrace,
    recipe: EsrPreprocessingRecipe,
    *,
    label: str,
    center_mT: float,
    gamma_mT: float,
    peak_window: PeakWindow | None = None,
) -> tuple[IntegralSummary, PrimaryIntegratedCurves | None]:
    """Compute resonance-local scalar integrals and full-axis local curves."""

    field = trace.field_mT
    signal = trace.signal
    if field.size < 2 or signal.size < 2:
        return (
            _unavailable_integral_summary(
                label,
                center_mT,
                center_mT,
                clipped_by_detected_window=False,
            ),
            None,
        )

    integration_half_width = max(
        recipe.integration_window_gamma_multiplier * abs(float(gamma_mT)),
        recipe.integration_window_min_half_width_mT,
    )
    baseline_half_width = max(
        recipe.integration_baseline_window_gamma_multiplier * abs(float(gamma_mT)),
        recipe.integration_baseline_window_min_half_width_mT,
    )

    start_field = max(float(field[0]), float(center_mT - integration_half_width))
    end_field = min(float(field[-1]), float(center_mT + integration_half_width))
    clipped_by_detected_window = False
    if peak_window is not None:
        guard_start, guard_end = _detected_window_guard_bounds(peak_window, recipe)
        clipped_start = max(start_field, guard_start)
        clipped_end = min(end_field, guard_end)
        clipped_by_detected_window = clipped_start > start_field or clipped_end < end_field
        start_field = clipped_start
        end_field = clipped_end
    integration_mask = (field >= start_field) & (field <= end_field)
    if int(np.count_nonzero(integration_mask)) < 2:
        return (
            _unavailable_integral_summary(
                label,
                start_field,
                end_field,
                clipped_by_detected_window=clipped_by_detected_window,
            ),
            None,
        )

    baseline_start = center_mT - baseline_half_width
    baseline_end = center_mT + baseline_half_width
    if peak_window is not None:
        guard_start, guard_end = _detected_window_guard_bounds(peak_window, recipe)
        baseline_start = max(baseline_start, guard_start)
        baseline_end = min(baseline_end, guard_end)
    baseline_mask = _local_baseline_mask(
        field=field,
        region_start=baseline_start,
        region_end=baseline_end,
        integration_mask=integration_mask,
    )
    required_points = max(20, 2 * (recipe.integration_baseline_polyorder + 1))
    actual_polyorder = recipe.integration_baseline_polyorder
    if int(np.count_nonzero(baseline_mask)) < required_points and peak_window is not None:
        expanded_start = max(float(field[0]), float(peak_window.start_field_mT - baseline_half_width))
        expanded_end = min(float(field[-1]), float(peak_window.end_field_mT + baseline_half_width))
        baseline_mask = _local_baseline_mask(
            field=field,
            region_start=expanded_start,
            region_end=expanded_end,
            integration_mask=integration_mask,
        )
    if int(np.count_nonzero(baseline_mask)) < required_points:
        actual_polyorder = 0
    if int(np.count_nonzero(baseline_mask)) == 0:
        return (
            _unavailable_integral_summary(
                label,
                start_field,
                end_field,
                clipped_by_detected_window=clipped_by_detected_window,
            ),
            None,
        )

    baseline_curve = _fit_local_baseline(
        field=field,
        signal=signal,
        baseline_mask=baseline_mask,
        polyorder=actual_polyorder,
    )
    field_segment = np.asarray(field[integration_mask], dtype=float)
    corrected_segment = np.asarray(signal[integration_mask] - baseline_curve[integration_mask], dtype=float)
    local_absorption = cumulative_integral(field_segment, corrected_segment)
    local_area = cumulative_integral(field_segment, local_absorption)

    summary = IntegralSummary(
        label=label,
        start_field_mT=float(field_segment[0]),
        end_field_mT=float(field_segment[-1]),
        absorption_integral=scalar_integral(field_segment, corrected_segment),
        area_integral=scalar_integral(field_segment, local_absorption),
        integration_kind="primary_local_window",
        window_source="fit_linewidth",
        baseline_polyorder=actual_polyorder,
        integration_window_clipped_by_detected_window=clipped_by_detected_window,
    )
    curves = _build_primary_local_curves(
        field=field,
        integration_mask=integration_mask,
        local_absorption=local_absorption,
        local_area=local_area,
        summary=summary,
    )
    return summary, curves


def _local_baseline_mask(
    *,
    field: np.ndarray,
    region_start: float,
    region_end: float,
    integration_mask: np.ndarray,
) -> np.ndarray:
    region_mask = (field >= region_start) & (field <= region_end)
    return region_mask & ~integration_mask


def _fit_local_baseline(
    *,
    field: np.ndarray,
    signal: np.ndarray,
    baseline_mask: np.ndarray,
    polyorder: int,
) -> np.ndarray:
    baseline_field = np.asarray(field[baseline_mask], dtype=float)
    baseline_signal = np.asarray(signal[baseline_mask], dtype=float)
    if polyorder == 0:
        return np.full_like(field, float(np.mean(baseline_signal)), dtype=float)
    coefficients = np.polyfit(baseline_field, baseline_signal, deg=polyorder)
    return np.asarray(np.polyval(coefficients, field), dtype=float)


def _build_primary_local_curves(
    *,
    field: np.ndarray,
    integration_mask: np.ndarray,
    local_absorption: np.ndarray,
    local_area: np.ndarray,
    summary: IntegralSummary,
) -> PrimaryIntegratedCurves:
    absorption_signal = np.full_like(field, np.nan, dtype=float)
    area_signal = np.full_like(field, np.nan, dtype=float)
    absorption_signal[integration_mask] = local_absorption
    area_signal[integration_mask] = local_area
    return PrimaryIntegratedCurves(
        field_mT=np.asarray(field, dtype=float).copy(),
        absorption_signal=absorption_signal,
        area_signal=area_signal,
        start_field_mT=summary.start_field_mT,
        end_field_mT=summary.end_field_mT,
        integration_kind="primary_local_window",
        window_source="fit_linewidth",
        baseline_polyorder=summary.baseline_polyorder,
        integration_window_clipped_by_detected_window=summary.integration_window_clipped_by_detected_window,
    )


def _detected_window_guard_bounds(
    peak_window: PeakWindow,
    recipe: EsrPreprocessingRecipe,
) -> tuple[float, float]:
    padding = peak_window.width_mT * recipe.integration_detected_window_padding_width_multiplier
    return (
        peak_window.start_field_mT - padding,
        peak_window.end_field_mT + padding,
    )


def _unavailable_integral_summary(
    label: str,
    start_field_mT: float,
    end_field_mT: float,
    *,
    clipped_by_detected_window: bool,
) -> IntegralSummary:
    return IntegralSummary(
        label=label,
        start_field_mT=start_field_mT,
        end_field_mT=end_field_mT,
        absorption_integral=None,
        area_integral=None,
        integration_kind="primary_local_window",
        window_source="fit_linewidth",
        baseline_polyorder=None,
        integration_window_clipped_by_detected_window=clipped_by_detected_window,
    )
