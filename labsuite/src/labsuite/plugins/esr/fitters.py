"""Derivative Lorentzian fitting and split-peak selection for ESR traces."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from lmfit import Model
from scipy.signal import find_peaks

from labsuite.core.exceptions import WorkflowError
from labsuite.core.recipes import EsrPreprocessingRecipe
from labsuite.core.types import (
    ConvergenceSummary,
    FeatureSummary,
    FitDecision,
    FitResult,
    ParameterDiagnostic,
    PeakFitResult,
    PeakWindow,
    ProcessedTrace,
    ResidualSummary,
)


def derivative_lorentzian(
    field_mT: np.ndarray,
    amplitude: float,
    center_mT: float,
    gamma_mT: float,
    offset: float,
) -> np.ndarray:
    """Derivative of a Lorentzian absorption profile."""

    scaled = (field_mT - center_mT) / gamma_mT
    return amplitude * ((-2.0 * scaled) / (1.0 + scaled**2) ** 2) + offset


def absorption_lorentzian(
    field_mT: np.ndarray,
    amplitude: float,
    center_mT: float,
    gamma_mT: float,
) -> np.ndarray:
    """Lorentzian absorption profile corresponding to the derivative model."""

    scaled = (field_mT - center_mT) / gamma_mT
    return amplitude * gamma_mT / (1.0 + scaled**2)


def fit_derivative_lorentzian(
    trace: ProcessedTrace,
    *,
    integrated_intensity_proxy: float | None = None,
) -> FitResult:
    """Fit the processed ESR trace to a derivative Lorentzian model."""

    field = trace.field_mT
    signal = trace.signal

    peak_index = int(np.argmax(signal))
    trough_index = int(np.argmin(signal))
    center_guess = float((field[peak_index] + field[trough_index]) / 2.0)
    extrema_spacing = abs(float(field[peak_index] - field[trough_index]))
    field_step = _median_step(field)
    gamma_guess = max(extrema_spacing * math.sqrt(3.0) / 2.0, field_step)
    offset_guess = float(np.mean(np.concatenate((signal[:5], signal[-5:]))))
    amplitude_scale = 9.0 / (8.0 * math.sqrt(3.0))
    amplitude_sign = 1.0 if field[peak_index] < field[trough_index] else -1.0
    amplitude_guess = amplitude_sign * float(
        max(abs(signal[peak_index]), abs(signal[trough_index])) / amplitude_scale
    )

    model = Model(derivative_lorentzian, independent_vars=["field_mT"])
    params = model.make_params(
        amplitude=amplitude_guess,
        center_mT=center_guess,
        gamma_mT=gamma_guess,
        offset=offset_guess,
    )
    params["center_mT"].set(min=float(np.min(field)), max=float(np.max(field)))
    params["gamma_mT"].set(min=max(field_step * 0.25, 1e-6))

    fit = model.fit(signal, params, field_mT=field)
    fitted_signal = np.asarray(fit.best_fit, dtype=float)
    residual = signal - fitted_signal

    metrics = _compute_fit_metrics(signal, residual)
    feature_summary = _compute_feature_summary(
        fit.params,
        integrated_intensity_proxy=integrated_intensity_proxy,
    )

    return FitResult(
        model_name="derivative_lorentzian",
        parameters={name: float(parameter.value) for name, parameter in fit.params.items()},
        derived={
            "peak_to_peak_separation_mT": feature_summary.peak_to_peak_separation_mT,
            "center_field_mT": feature_summary.zero_crossing_field_mT,
        },
        metrics=metrics,
        fitted_signal=fitted_signal,
        residual=residual,
        parameter_diagnostics=_build_parameter_diagnostics(fit.params),
        convergence=ConvergenceSummary(
            success=bool(fit.success),
            message=str(fit.message),
            nfev=None if fit.nfev is None else int(fit.nfev),
            nvarys=None if fit.nvarys is None else int(fit.nvarys),
            errorbars=bool(fit.errorbars),
        ),
        residual_summary=_build_residual_summary(residual),
        feature_summary=feature_summary,
        bound_hits=_build_bound_hits(fit.params),
        success=bool(fit.success),
    )


def detect_peak_windows(
    trace: ProcessedTrace,
    recipe: EsrPreprocessingRecipe,
) -> list[PeakWindow]:
    """Detect up to two resonance windows from a processed derivative trace."""

    field = trace.field_mT
    signal = trace.signal
    field_step = _median_step(field)
    distance_points = max(1, int(round(recipe.peak_min_distance_mT / field_step)))
    prominence = float(max(np.max(np.abs(signal)) * recipe.peak_min_prominence_ratio, 1e-9))

    positive_indices, positive_props = find_peaks(
        signal, prominence=prominence, distance=distance_points
    )
    negative_indices, negative_props = find_peaks(
        -signal, prominence=prominence, distance=distance_points
    )
    if positive_indices.size == 0 or negative_indices.size == 0:
        return []

    candidates: dict[tuple[int, int], PeakWindow] = {}
    for peak_index, peak_prominence in zip(
        positive_indices, positive_props["prominences"], strict=True
    ):
        nearest_trough_position = int(
            np.argmin(np.abs(field[negative_indices] - field[peak_index]))
        )
        trough_index = int(negative_indices[nearest_trough_position])
        trough_prominence = float(negative_props["prominences"][nearest_trough_position])
        width_mT = abs(float(field[trough_index] - field[peak_index]))
        if width_mT < recipe.peak_min_pair_width_mT:
            continue

        left_index = min(int(peak_index), trough_index)
        right_index = max(int(peak_index), trough_index)
        padding_points = max(5, int(round((width_mT * 1.5) / field_step)))
        start_index = max(0, left_index - padding_points)
        end_index = min(signal.size - 1, right_index + padding_points)
        candidate = PeakWindow(
            label="",
            start_index=start_index,
            end_index=end_index,
            start_field_mT=float(field[start_index]),
            end_field_mT=float(field[end_index]),
            peak_index=int(peak_index),
            trough_index=trough_index,
            peak_field_mT=float(field[peak_index]),
            trough_field_mT=float(field[trough_index]),
            width_mT=width_mT,
            prominence=float(min(peak_prominence, trough_prominence)),
        )
        key = (left_index, right_index)
        previous = candidates.get(key)
        if previous is None or candidate.prominence > previous.prominence:
            candidates[key] = candidate

    ranked_candidates = sorted(candidates.values(), key=lambda item: item.prominence, reverse=True)
    selected: list[PeakWindow] = []
    for candidate in ranked_candidates:
        overlaps = any(
            not (
                candidate.end_index < existing.start_index
                or candidate.start_index > existing.end_index
            )
            for existing in selected
        )
        if overlaps:
            continue
        selected.append(candidate)
        if len(selected) == 2:
            break

    selected.sort(key=lambda item: item.start_index)
    return [
        PeakWindow(
            label=f"peak_{index}",
            start_index=item.start_index,
            end_index=item.end_index,
            start_field_mT=item.start_field_mT,
            end_field_mT=item.end_field_mT,
            peak_index=item.peak_index,
            trough_index=item.trough_index,
            peak_field_mT=item.peak_field_mT,
            trough_field_mT=item.trough_field_mT,
            width_mT=item.width_mT,
            prominence=item.prominence,
        )
        for index, item in enumerate(selected, start=1)
    ]


def fit_peak_windows(
    trace: ProcessedTrace,
    windows: list[PeakWindow],
) -> list[PeakFitResult]:
    """Fit each detected resonance window independently."""

    peak_fits: list[PeakFitResult] = []
    field = trace.field_mT
    for window in windows:
        fit = fit_derivative_lorentzian_in_window(trace, window)
        parameters = fit.parameters
        component_signal = derivative_lorentzian(
            field,
            amplitude=parameters["amplitude"],
            center_mT=parameters["center_mT"],
            gamma_mT=parameters["gamma_mT"],
            offset=0.0,
        )
        peak_fits.append(
            PeakFitResult(
                label=window.label,
                window=window,
                fit=fit,
                component_signal=np.asarray(component_signal, dtype=float),
            )
        )
    return peak_fits


def build_split_fit(trace: ProcessedTrace, peak_fits: list[PeakFitResult]) -> FitResult:
    """Build a full-axis stitched split fit from independently fitted components."""

    if not peak_fits:
        raise WorkflowError("Cannot build a split fit without fitted peak components.")

    global_offset = float(
        np.mean([peak_fit.fit.parameters.get("offset", 0.0) for peak_fit in peak_fits])
    )
    fitted_signal = np.full_like(trace.signal, global_offset)
    for peak_fit in peak_fits:
        fitted_signal += peak_fit.component_signal

    residual = trace.signal - fitted_signal
    positive_index = int(np.argmax(fitted_signal))
    negative_index = int(np.argmin(fitted_signal))
    width = abs(float(trace.field_mT[negative_index] - trace.field_mT[positive_index]))
    intensity_values = [
        peak_fit.fit.feature_summary.integrated_intensity_proxy
        for peak_fit in peak_fits
        if peak_fit.fit.feature_summary is not None
    ]
    feature_summary = FeatureSummary(
        positive_extremum_field_mT=float(trace.field_mT[positive_index]),
        negative_extremum_field_mT=float(trace.field_mT[negative_index]),
        zero_crossing_field_mT=float(
            np.mean(
                [
                    peak_fit.fit.feature_summary.zero_crossing_field_mT
                    for peak_fit in peak_fits
                    if peak_fit.fit.feature_summary is not None
                ]
            )
        ),
        peak_to_peak_separation_mT=width,
        integrated_intensity_proxy=None
        if any(value is None for value in intensity_values)
        else float(sum(intensity_values)),
    )
    return FitResult(
        model_name="split_derivative_lorentzian",
        parameters={"global_offset": global_offset},
        derived={"component_count": float(len(peak_fits))},
        metrics=_compute_fit_metrics(trace.signal, residual),
        fitted_signal=fitted_signal,
        residual=residual,
        parameter_diagnostics={
            "global_offset": ParameterDiagnostic(
                value=global_offset,
                stderr=None,
                relative_stderr=None,
                stderr_missing=True,
                min_bound=None,
                max_bound=None,
                hit_min_bound=False,
                hit_max_bound=False,
            )
        },
        convergence=ConvergenceSummary(
            success=all(peak_fit.fit.success for peak_fit in peak_fits),
            message="split fit composed from independently converged component fits",
            nfev=sum(peak_fit.fit.convergence.nfev or 0 for peak_fit in peak_fits),
            nvarys=None,
            errorbars=all(peak_fit.fit.convergence.errorbars for peak_fit in peak_fits),
        ),
        residual_summary=_build_residual_summary(residual),
        feature_summary=feature_summary,
        bound_hits={"global_offset": False},
        success=all(peak_fit.fit.success for peak_fit in peak_fits),
    )


def fit_derivative_lorentzian_in_window(trace: ProcessedTrace, window: PeakWindow) -> FitResult:
    """Fit the derivative Lorentzian model using only a local trace window."""

    subtrace = ProcessedTrace(
        field_mT=trace.field_mT[window.start_index : window.end_index + 1],
        signal=trace.signal[window.start_index : window.end_index + 1],
        steps=trace.steps,
    )
    return fit_derivative_lorentzian(subtrace)


def select_fit_mode(
    requested_mode: Literal["auto", "single", "split"],
    single_fit: FitResult,
    split_fit: FitResult | None,
    peak_windows: list[PeakWindow],
    split_threshold: float,
) -> tuple[Literal["single", "split"], FitDecision]:
    """Select the analysis mode given the requested policy and candidate fits."""

    split_improvement_ratio: float | None = None
    single_ss_res = single_fit.metrics.get("sum_squared_residuals")
    split_ss_res = split_fit.metrics.get("sum_squared_residuals") if split_fit is not None else None
    if split_fit is not None and single_ss_res and single_ss_res > 0.0 and split_ss_res is not None:
        split_improvement_ratio = max(0.0, float((single_ss_res - split_ss_res) / single_ss_res))

    metrics = {
        "single_r_squared": single_fit.metrics.get("r_squared"),
        "split_r_squared": None if split_fit is None else split_fit.metrics.get("r_squared"),
    }
    if requested_mode == "single":
        return "single", FitDecision(
            requested_mode=requested_mode,
            selected_mode="single",
            candidate_peak_count=len(peak_windows),
            split_improvement_ratio=split_improvement_ratio,
            split_threshold=split_threshold,
            reason="single mode was explicitly requested",
            metrics=metrics,
        )

    if requested_mode == "split":
        if split_fit is None or len(peak_windows) < 2 or not split_fit.success:
            raise WorkflowError(
                "Split fit mode requested, but two valid peak windows were not available."
            )
        return "split", FitDecision(
            requested_mode=requested_mode,
            selected_mode="split",
            candidate_peak_count=len(peak_windows),
            split_improvement_ratio=split_improvement_ratio,
            split_threshold=split_threshold,
            reason="split mode was explicitly requested",
            metrics=metrics,
        )

    if split_fit is None or len(peak_windows) < 2 or not split_fit.success:
        return "single", FitDecision(
            requested_mode=requested_mode,
            selected_mode="single",
            candidate_peak_count=len(peak_windows),
            split_improvement_ratio=split_improvement_ratio,
            split_threshold=split_threshold,
            reason="auto mode found fewer than two valid split-fit peak windows",
            metrics=metrics,
        )

    if split_improvement_ratio is not None and split_improvement_ratio >= split_threshold:
        return "split", FitDecision(
            requested_mode=requested_mode,
            selected_mode="split",
            candidate_peak_count=len(peak_windows),
            split_improvement_ratio=split_improvement_ratio,
            split_threshold=split_threshold,
            reason="auto mode selected split because the stitched residual improved materially",
            metrics=metrics,
        )

    return "single", FitDecision(
        requested_mode=requested_mode,
        selected_mode="single",
        candidate_peak_count=len(peak_windows),
        split_improvement_ratio=split_improvement_ratio,
        split_threshold=split_threshold,
        reason="auto mode kept the single fit because split improvement was below threshold",
        metrics=metrics,
    )


def _median_step(field: np.ndarray) -> float:
    diffs = np.diff(field)
    non_zero_diffs = np.abs(diffs[diffs != 0.0])
    if non_zero_diffs.size == 0:
        return 1.0
    return float(np.median(non_zero_diffs))


def _compute_fit_metrics(signal: np.ndarray, residual: np.ndarray) -> dict[str, float]:
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((signal - np.mean(signal)) ** 2))
    r_squared = 1.0 if ss_tot == 0.0 else 1.0 - (ss_res / ss_tot)
    reduced_chi_square = ss_res / max(signal.size - 1, 1)
    return {
        "chi_square": ss_res,
        "reduced_chi_square": reduced_chi_square,
        "r_squared": r_squared,
        "sum_squared_residuals": ss_res,
    }


def _build_parameter_diagnostics(params) -> dict[str, ParameterDiagnostic]:
    diagnostics: dict[str, ParameterDiagnostic] = {}
    for name, parameter in params.items():
        stderr = None if parameter.stderr is None else float(parameter.stderr)
        relative_stderr = None
        if stderr is not None and parameter.value not in (None, 0.0):
            relative_stderr = float(abs(stderr / float(parameter.value)))
        min_bound = None if parameter.min in (-np.inf, None) else float(parameter.min)
        max_bound = None if parameter.max in (np.inf, None) else float(parameter.max)
        diagnostics[name] = ParameterDiagnostic(
            value=float(parameter.value),
            stderr=stderr,
            relative_stderr=relative_stderr,
            stderr_missing=stderr is None,
            min_bound=min_bound,
            max_bound=max_bound,
            hit_min_bound=_is_at_bound(float(parameter.value), min_bound),
            hit_max_bound=_is_at_bound(float(parameter.value), max_bound),
        )
    return diagnostics


def _build_bound_hits(params) -> dict[str, bool]:
    bound_hits: dict[str, bool] = {}
    for name, parameter in params.items():
        min_bound = None if parameter.min in (-np.inf, None) else float(parameter.min)
        max_bound = None if parameter.max in (np.inf, None) else float(parameter.max)
        value = float(parameter.value)
        bound_hits[name] = _is_at_bound(value, min_bound) or _is_at_bound(value, max_bound)
    return bound_hits


def _build_residual_summary(residual: np.ndarray) -> ResidualSummary:
    return ResidualSummary(
        rss=float(np.sum(residual**2)),
        rmse=float(np.sqrt(np.mean(residual**2))),
        mae=float(np.mean(np.abs(residual))),
        max_abs=float(np.max(np.abs(residual))),
        mean=float(np.mean(residual)),
        std=float(np.std(residual)),
    )


def _compute_feature_summary(
    params,
    *,
    integrated_intensity_proxy: float | None,
) -> FeatureSummary:
    center = float(params["center_mT"].value)
    gamma = float(params["gamma_mT"].value)
    delta = gamma / math.sqrt(3.0)
    return FeatureSummary(
        positive_extremum_field_mT=center - delta,
        negative_extremum_field_mT=center + delta,
        zero_crossing_field_mT=center,
        peak_to_peak_separation_mT=2.0 * delta,
        integrated_intensity_proxy=integrated_intensity_proxy,
    )


def _is_at_bound(value: float, bound: float | None) -> bool:
    if bound is None:
        return False
    tolerance = max(1e-9, abs(bound) * 1e-6)
    return abs(value - bound) <= tolerance
