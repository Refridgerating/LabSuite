"""FMR-specific data models used by the PhaseFMR analysis workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from labsuite.core.resonance_metrics import ResonanceModeMetrics
from labsuite.core.types import ConvergenceSummary, ParameterDiagnostic, ResidualSummary

FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class FmrTraceDataset:
    """One standardized field-swept FMR trace extracted from a PhaseFMR log."""

    trace_id: str
    source_file: Path
    sample_name: str
    frequency_GHz: float
    angle_deg: float | None
    temperature_K: float | None
    field_mT: FloatArray
    signal: FloatArray
    field_units: str
    signal_units: str
    sweep_direction: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    i_signal: FloatArray | None = None
    q_signal: FloatArray | None = None
    fit_source_signal: FloatArray | None = None
    fit_signal: FloatArray | None = None
    aux_signal: FloatArray | None = None
    temp_K_signal: FloatArray | None = None
    time_s_signal: FloatArray | None = None


@dataclass(slots=True)
class FmrFileDataset:
    """Parsed PhaseFMR source file containing one or more traces."""

    source_path: Path
    sample_name: str
    replicate_id: str | None
    angle_deg: float | None
    nominal_temperature_K: float | None
    sweep_span_label: str | None
    measurement_mode: str | None
    traces: list[FmrTraceDataset] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FmrCandidateWindow:
    """Detected resonance candidate window on a processed derivative trace."""

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
    candidate_center_mT: float


@dataclass(slots=True)
class FmrComponentFitResult:
    """One accepted or rejected resonance component from a selected FMR trace fit."""

    component_id: str
    component_label: str
    H_res_mT: float
    DeltaH_mT: float
    amplitude_symmetric: float
    amplitude_antisymmetric: float
    field_mT: FloatArray
    component_signal: FloatArray
    absorption_signal: FloatArray | None = None
    parameter_diagnostics: dict[str, ParameterDiagnostic] = field(default_factory=dict)
    bound_hits: dict[str, bool] = field(default_factory=dict)
    accepted: bool = False
    rejection_reason: str | None = None
    signal_max_abs: float | None = None
    residual_rmse_fraction: float | None = None
    amplitude_snr: float | None = None
    feature_center_mT: float | None = None
    feature_peak_to_peak_mT: float | None = None
    center_feature_disagreement_mT: float | None = None
    critical_bound_hit_names: list[str] = field(default_factory=list)
    acceptance_checks: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    resonance_metrics: ResonanceModeMetrics | None = None


@dataclass(slots=True)
class FmrTraceModelResult:
    """One candidate full-trace FMR model fit, single or double resonance."""

    model_name: str
    success: bool
    parameters: dict[str, float]
    parameter_diagnostics: dict[str, ParameterDiagnostic]
    convergence: ConvergenceSummary
    residual_summary: ResidualSummary
    metrics: dict[str, float]
    bound_hits: dict[str, bool]
    covariance: list[list[float]] | None
    fitted_signal: FloatArray
    residual: FloatArray
    components: list[FmrComponentFitResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FmrTraceFitResult:
    """Per-trace fit-selection result and QC for one FMR trace."""

    trace_id: str
    source_file: Path
    sample_name: str
    frequency_GHz: float
    angle_deg: float | None
    temperature_K: float | None
    model_name: str
    signal_channel: str
    field_mT: FloatArray
    processed_signal: FloatArray
    fitted_signal: FloatArray
    residual: FloatArray
    parameters: dict[str, float]
    parameter_diagnostics: dict[str, ParameterDiagnostic]
    convergence: ConvergenceSummary
    residual_summary: ResidualSummary
    metrics: dict[str, float]
    bound_hits: dict[str, bool]
    covariance: list[list[float]] | None
    success: bool
    accepted: bool
    rejection_reason: str | None
    requested_mode: str = "auto"
    selected_mode: str = "single"
    selection_reason: str = ""
    candidate_window_count: int = 0
    double_fit_improvement_ratio: float | None = None
    double_fit_threshold: float | None = None
    candidate_windows: list[FmrCandidateWindow] = field(default_factory=list)
    single_fit: FmrTraceModelResult | None = None
    double_fit: FmrTraceModelResult | None = None
    selected_components: list[FmrComponentFitResult] = field(default_factory=list)
    partial_component_qc: bool = False
    r_squared: float | None = None
    signal_max_abs: float | None = None
    residual_rmse_fraction: float | None = None
    amplitude_snr: float | None = None
    feature_center_mT: float | None = None
    feature_peak_to_peak_mT: float | None = None
    center_feature_disagreement_mT: float | None = None
    critical_bound_hit_names: list[str] = field(default_factory=list)
    acceptance_checks: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    preprocessing_steps: list[dict[str, Any]] = field(default_factory=list)
    baseline_summary: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    resonance_metrics: list[ResonanceModeMetrics] = field(default_factory=list)


@dataclass(slots=True)
class FmrSeriesResult:
    """One mode-specific resonance series assembled from selected FMR components."""

    series_label: str
    sample_name: str
    angle_deg: float | None
    nominal_temperature_K: float | None
    measurement_mode: str | None
    frequency_GHz: FloatArray
    resonance_field_mT: FloatArray
    linewidth_mT: FloatArray
    amplitude_symmetric: FloatArray
    amplitude_antisymmetric: FloatArray
    resonance_field_stderr_mT: FloatArray
    linewidth_stderr_mT: FloatArray
    included_trace_ids: list[str] = field(default_factory=list)
    included_component_ids: list[str] = field(default_factory=list)
    excluded_trace_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FmrSeriesCollectionResult:
    """All mode-specific resonance series assembled from one FMR file or aggregate group."""

    sample_name: str
    angle_deg: float | None
    nominal_temperature_K: float | None
    measurement_mode: str | None
    series_by_label: dict[str, FmrSeriesResult] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FmrModelFitSummary:
    """Generic scalar summary for one higher-level FMR physics fit."""

    model_name: str
    success: bool
    message: str
    parameters: dict[str, float] = field(default_factory=dict)
    stderr: dict[str, float | None] = field(default_factory=dict)
    metrics: dict[str, float | None] = field(default_factory=dict)
    x: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    fitted_y: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FmrPhysicsResult:
    """Higher-level physics fits derived from one assembled resonance series."""

    sample_name: str
    angle_deg: float | None
    nominal_temperature_K: float | None
    measurement_mode: str | None
    kittel_fit: FmrModelFitSummary | None = None
    linewidth_fit: FmrModelFitSummary | None = None
    derived_parameters: dict[str, float | None] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FmrPhysicsCollectionResult:
    """Mode-specific physics fits derived from an FMR series collection."""

    sample_name: str
    angle_deg: float | None
    nominal_temperature_K: float | None
    measurement_mode: str | None
    physics_by_label: dict[str, FmrPhysicsResult] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
