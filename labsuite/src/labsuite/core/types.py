"""Shared data structures for raw traces, integrated curves, and fit outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class TraceDataset:
    """Raw parsed dataset preserved directly from an import source."""

    modality: str
    source_path: Path
    field_mT: FloatArray
    signal: FloatArray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProcessedTrace:
    """A trace after explicit preprocessing steps have been applied."""

    field_mT: FloatArray
    signal: FloatArray
    steps: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class IntegratedCurves:
    """Integrated curves derived from a processed derivative signal."""

    field_mT: FloatArray
    absorption_signal: FloatArray
    area_signal: FloatArray
    steps: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class BaselineSummary:
    """Structured baseline terms captured during preprocessing."""

    target: str
    edge_points: int
    slope: float
    intercept: float


@dataclass(slots=True)
class ParameterDiagnostic:
    """Detailed parameter fit diagnostics."""

    value: float
    stderr: float | None
    relative_stderr: float | None
    stderr_missing: bool
    min_bound: float | None
    max_bound: float | None
    hit_min_bound: bool
    hit_max_bound: bool


@dataclass(slots=True)
class ConvergenceSummary:
    """Optimizer convergence diagnostics."""

    success: bool
    message: str
    nfev: int | None
    nvarys: int | None
    errorbars: bool


@dataclass(slots=True)
class ResidualSummary:
    """Scalar residual diagnostics for a fit."""

    rss: float
    rmse: float
    mae: float
    max_abs: float
    mean: float
    std: float


@dataclass(slots=True)
class FeatureSummary:
    """Derived landmark fields for a fitted spectral feature."""

    positive_extremum_field_mT: float
    negative_extremum_field_mT: float
    zero_crossing_field_mT: float
    peak_to_peak_separation_mT: float
    integrated_intensity_proxy: float | None


@dataclass(slots=True)
class FitResult:
    """Result of fitting a processed trace to a model."""

    model_name: str
    parameters: dict[str, float]
    derived: dict[str, Any]
    metrics: dict[str, float]
    fitted_signal: FloatArray
    residual: FloatArray
    parameter_diagnostics: dict[str, ParameterDiagnostic]
    convergence: ConvergenceSummary
    residual_summary: ResidualSummary
    feature_summary: FeatureSummary | None
    bound_hits: dict[str, bool]
    success: bool = True


@dataclass(slots=True)
class FitAttemptRecord:
    """One fitting attempt with provenance and selection metadata."""

    scope: Literal["global_full_trace", "detected_window_fallback", "peak_window_local"]
    fit: FitResult
    source_window: "PeakWindow | None"
    accepted: bool
    rejection_reason: str | None
    selected_for_primary: bool


@dataclass(slots=True)
class PeakWindow:
    """Detected derivative peak window used for local fitting and integration."""

    label: str
    start_index: int
    end_index: int
    start_field_mT: float
    end_field_mT: float
    peak_index: int
    trough_index: int
    peak_field_mT: float
    trough_field_mT: float
    width_mT: float
    prominence: float


@dataclass(slots=True)
class PeakFitResult:
    """Fit result for a single detected resonance window."""

    label: str
    window: PeakWindow
    fit: FitResult
    component_signal: FloatArray
    attempts: list[FitAttemptRecord] = field(default_factory=list)


@dataclass(slots=True)
class FitDecision:
    """Decision metadata describing how the selected fit mode was chosen."""

    requested_mode: Literal["auto", "single", "split"]
    selected_mode: Literal["single", "split"]
    candidate_peak_count: int
    split_improvement_ratio: float | None
    split_threshold: float
    reason: str
    metrics: dict[str, float | None] = field(default_factory=dict)


@dataclass(slots=True)
class IntegralSummary:
    """Scalar integration summary for the full trace or a single detected peak."""

    label: str
    start_field_mT: float
    end_field_mT: float
    absorption_integral: float | None
    area_integral: float | None
    integration_kind: Literal["primary_fit_model", "fit_local_windowed_model", "primary_local_window", "diagnostic_full_span"]
    window_source: Literal["fit_linewidth"] | None
    baseline_polyorder: int | None
    integration_window_clipped_by_detected_window: bool


@dataclass(slots=True)
class FitIntegratedCurves:
    """Fit-derived primary absorption and area curves on the native field axis."""

    field_mT: FloatArray
    absorption_signal: FloatArray
    area_signal: FloatArray
    integration_kind: Literal["primary_fit_model"]
    model_name: str


@dataclass(slots=True)
class PrimaryIntegratedCurves:
    """Windowed diagnostic absorption and area curves on the native field axis."""

    field_mT: FloatArray
    absorption_signal: FloatArray
    area_signal: FloatArray
    start_field_mT: float
    end_field_mT: float
    integration_kind: Literal["primary_local_window", "fit_local_windowed_model"]
    window_source: Literal["fit_linewidth"]
    baseline_polyorder: int | None
    integration_window_clipped_by_detected_window: bool
    model_name: str | None = None


@dataclass(slots=True)
class AnalysisResult:
    """Complete reproducible payload for one single-file analysis."""

    dataset: TraceDataset
    processed: ProcessedTrace
    integrated: IntegratedCurves
    primary_integrated: FitIntegratedCurves | None
    fit_local_integrated: PrimaryIntegratedCurves | None
    local_integrated: PrimaryIntegratedCurves | None
    derivative_baseline: BaselineSummary
    absorption_baseline: BaselineSummary
    selected_mode: Literal["single", "split"]
    fit_decision: FitDecision
    single_fit: FitResult | None
    single_fit_attempts: list[FitAttemptRecord]
    peak_fits: list[PeakFitResult]
    selected_fit_signal: FloatArray
    selected_residual: FloatArray
    total_integral: IntegralSummary
    fit_local_total_integral: IntegralSummary
    local_total_integral: IntegralSummary
    diagnostic_total_integral: IntegralSummary
    peak_integrals: list[IntegralSummary]
    fit_local_peak_integrals: list[IntegralSummary]
    local_peak_integrals: list[IntegralSummary]
    fit_local_disagreement_ratio: float | None
    fit_local_disagreement_flag: bool
    fit_local_disagreement_reason: str | None
    recipe_name: str
    recipe_config: dict[str, Any]
