"""Shared measurement-level models for modality-agnostic workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class SampleRecord:
    """Parsed sample identity and grouping metadata."""

    sample_id: str
    series_id: str
    filename_tokens: dict[str, Any] = field(default_factory=dict)
    grouping_keys: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class MeasurementRecord:
    """Shared metadata for one imported measurement."""

    modality: str
    source_path: Path
    sample: SampleRecord
    replicate_id: str | None
    condition_metadata: dict[str, Any] = field(default_factory=dict)
    raw_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FitResult:
    """Generic fit or diagnostic model summary."""

    model_name: str
    parameters: dict[str, float]
    metrics: dict[str, float | None]
    success: bool
    message: str
    selected_indices: list[int] = field(default_factory=list)
    fitted_x: list[float] = field(default_factory=list)
    fitted_y: list[float] = field(default_factory=list)
    residual_y: list[float] = field(default_factory=list)


@dataclass(slots=True)
class PlotManifest:
    """Saved figure metadata needed to regenerate a plot from JSON."""

    figure_type: str
    title: str
    series: list[dict[str, Any]] = field(default_factory=list)
    annotations: list[dict[str, Any]] = field(default_factory=list)
    theme: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MeasurementAnalysisResult:
    """Measurement-agnostic envelope around modality-specific analysis."""

    measurement: MeasurementRecord
    provenance: dict[str, Any]
    warnings: list[str]
    summary_metrics: dict[str, Any]
    artifacts: dict[str, Any]
    analysis_payload: dict[str, Any]
    plot_manifest: PlotManifest | None = None

    def to_dict(self) -> dict[str, Any]:
        return to_serializable(asdict(self))


def to_serializable(value: Any) -> Any:
    """Recursively convert dataclasses, arrays, and paths into JSON-safe values."""

    if is_dataclass(value):
        return to_serializable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value
