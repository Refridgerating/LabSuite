"""Shared resonance metrics computed from reconstructed absorption-like modes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from labsuite.core.preprocessing import scalar_integral

AreaWindowMode = Literal["side-aware", "symmetric"]
HalfmaxInterpolation = Literal["linear"]
MetricsSource = Literal["reconstructed_absorption"]


@dataclass(slots=True)
class ResonanceMetricsConfig:
    """Shared runtime configuration for resonance-metrics computation and export."""

    compute_resonance_metrics: bool = True
    area_window_mode: AreaWindowMode = "side-aware"
    area_window_multipliers: tuple[float, ...] = (1.0, 2.0, 3.0)
    compute_full_area: bool = False
    report_asymmetry: bool = True
    halfmax_interp: HalfmaxInterpolation = "linear"
    metrics_from: MetricsSource = "reconstructed_absorption"
    export_resonance_metrics: bool = False
    plot_halfmax_markers: bool = False
    plot_area_windows: bool = False

    def __post_init__(self) -> None:
        if self.area_window_mode not in {"side-aware", "symmetric"}:
            raise ValueError(f"Unsupported area_window_mode: {self.area_window_mode}")
        if self.halfmax_interp != "linear":
            raise ValueError(f"Unsupported halfmax_interp: {self.halfmax_interp}")
        if self.metrics_from != "reconstructed_absorption":
            raise ValueError(f"Unsupported metrics_from: {self.metrics_from}")
        multipliers = tuple(float(item) for item in self.area_window_multipliers)
        if not multipliers:
            raise ValueError("area_window_multipliers must not be empty")
        if any(item <= 0.0 for item in multipliers):
            raise ValueError("area_window_multipliers must all be positive")
        self.area_window_multipliers = multipliers

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["area_window_multipliers"] = list(self.area_window_multipliers)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> ResonanceMetricsConfig:
        if payload is None:
            return cls()
        return cls(
            compute_resonance_metrics=bool(payload.get("compute_resonance_metrics", True)),
            area_window_mode=str(payload.get("area_window_mode", "side-aware")),
            area_window_multipliers=tuple(
                float(item) for item in payload.get("area_window_multipliers", (1, 2, 3))
            ),
            compute_full_area=bool(payload.get("compute_full_area", False)),
            report_asymmetry=bool(payload.get("report_asymmetry", True)),
            halfmax_interp=str(payload.get("halfmax_interp", "linear")),
            metrics_from=str(payload.get("metrics_from", "reconstructed_absorption")),
            export_resonance_metrics=bool(payload.get("export_resonance_metrics", False)),
            plot_halfmax_markers=bool(payload.get("plot_halfmax_markers", False)),
            plot_area_windows=bool(payload.get("plot_area_windows", False)),
        )


@dataclass(slots=True)
class ResonanceAreaWindow:
    """One configured area window around a resonance mode."""

    multiplier: float
    label: str
    window_mode: AreaWindowMode
    start_field_mT: float | None
    end_field_mT: float | None
    area: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResonanceAreaWindow:
        return cls(
            multiplier=float(payload["multiplier"]),
            label=str(payload["label"]),
            window_mode=str(payload["window_mode"]),
            start_field_mT=None
            if payload.get("start_field_mT") is None
            else float(payload["start_field_mT"]),
            end_field_mT=None
            if payload.get("end_field_mT") is None
            else float(payload["end_field_mT"]),
            area=None if payload.get("area") is None else float(payload["area"]),
        )


@dataclass(slots=True)
class ResonanceModeMetrics:
    """Standardized metrics for one fitted resonance mode."""

    owner_kind: str
    owner_id: str
    success: bool
    failure_reason: str | None
    metrics_from: MetricsSource
    hres: float
    peak_field_abs: float | None
    peak_height_abs: float | None
    half_max_level: float | None
    h_left_half: float | None
    h_right_half: float | None
    fwhm: float | None
    hwhm_left: float | None
    hwhm_right: float | None
    asymmetry_ratio: float | None
    area_pm_1fwhm: float | None
    area_pm_2fwhm: float | None
    area_pm_3fwhm: float | None
    area_full: float | None
    support_start_field_mT: float | None
    support_end_field_mT: float | None
    local_baseline_edge_points: int | None
    local_baseline_slope: float | None
    local_baseline_intercept: float | None
    signal_polarity: float = 1.0
    area_windows: list[ResonanceAreaWindow] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner_kind": self.owner_kind,
            "owner_id": self.owner_id,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "metrics_from": self.metrics_from,
            "hres": self.hres,
            "peak_field_abs": self.peak_field_abs,
            "peak_height_abs": self.peak_height_abs,
            "half_max_level": self.half_max_level,
            "h_left_half": self.h_left_half,
            "h_right_half": self.h_right_half,
            "fwhm": self.fwhm,
            "hwhm_left": self.hwhm_left,
            "hwhm_right": self.hwhm_right,
            "asymmetry_ratio": self.asymmetry_ratio,
            "area_pm_1fwhm": self.area_pm_1fwhm,
            "area_pm_2fwhm": self.area_pm_2fwhm,
            "area_pm_3fwhm": self.area_pm_3fwhm,
            "area_full": self.area_full,
            "support_start_field_mT": self.support_start_field_mT,
            "support_end_field_mT": self.support_end_field_mT,
            "local_baseline_edge_points": self.local_baseline_edge_points,
            "local_baseline_slope": self.local_baseline_slope,
            "local_baseline_intercept": self.local_baseline_intercept,
            "signal_polarity": self.signal_polarity,
            "area_windows": [item.to_dict() for item in self.area_windows],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResonanceModeMetrics:
        return cls(
            owner_kind=str(payload["owner_kind"]),
            owner_id=str(payload["owner_id"]),
            success=bool(payload["success"]),
            failure_reason=payload.get("failure_reason"),
            metrics_from=str(payload.get("metrics_from", "reconstructed_absorption")),
            hres=float(payload["hres"]),
            peak_field_abs=None
            if payload.get("peak_field_abs") is None
            else float(payload["peak_field_abs"]),
            peak_height_abs=None
            if payload.get("peak_height_abs") is None
            else float(payload["peak_height_abs"]),
            half_max_level=None
            if payload.get("half_max_level") is None
            else float(payload["half_max_level"]),
            h_left_half=None
            if payload.get("h_left_half") is None
            else float(payload["h_left_half"]),
            h_right_half=None
            if payload.get("h_right_half") is None
            else float(payload["h_right_half"]),
            fwhm=None if payload.get("fwhm") is None else float(payload["fwhm"]),
            hwhm_left=None if payload.get("hwhm_left") is None else float(payload["hwhm_left"]),
            hwhm_right=None if payload.get("hwhm_right") is None else float(payload["hwhm_right"]),
            asymmetry_ratio=None
            if payload.get("asymmetry_ratio") is None
            else float(payload["asymmetry_ratio"]),
            area_pm_1fwhm=None
            if payload.get("area_pm_1fwhm") is None
            else float(payload["area_pm_1fwhm"]),
            area_pm_2fwhm=None
            if payload.get("area_pm_2fwhm") is None
            else float(payload["area_pm_2fwhm"]),
            area_pm_3fwhm=None
            if payload.get("area_pm_3fwhm") is None
            else float(payload["area_pm_3fwhm"]),
            area_full=None if payload.get("area_full") is None else float(payload["area_full"]),
            support_start_field_mT=None
            if payload.get("support_start_field_mT") is None
            else float(payload["support_start_field_mT"]),
            support_end_field_mT=None
            if payload.get("support_end_field_mT") is None
            else float(payload["support_end_field_mT"]),
            local_baseline_edge_points=None
            if payload.get("local_baseline_edge_points") is None
            else int(payload["local_baseline_edge_points"]),
            local_baseline_slope=None
            if payload.get("local_baseline_slope") is None
            else float(payload["local_baseline_slope"]),
            local_baseline_intercept=None
            if payload.get("local_baseline_intercept") is None
            else float(payload["local_baseline_intercept"]),
            signal_polarity=float(payload.get("signal_polarity", 1.0)),
            area_windows=[
                ResonanceAreaWindow.from_dict(item) for item in payload.get("area_windows", [])
            ],
            metadata=dict(payload.get("metadata", {})),
        )


def parse_area_window_multipliers(raw_value: str | Iterable[float]) -> tuple[float, ...]:
    """Parse CLI-style multiplier inputs into a validated tuple."""

    if isinstance(raw_value, str):
        parts = [part.strip() for part in raw_value.split(",") if part.strip()]
        if not parts:
            raise ValueError("area_window_multipliers must not be empty")
        return tuple(float(part) for part in parts)
    values = tuple(float(item) for item in raw_value)
    if not values:
        raise ValueError("area_window_multipliers must not be empty")
    return values


def compute_absorption_mode_metrics(
    field_mT: np.ndarray,
    absorption_signal: np.ndarray,
    *,
    hres: float,
    config: ResonanceMetricsConfig,
    support_start_field_mT: float | None = None,
    support_end_field_mT: float | None = None,
    owner_kind: str = "mode",
    owner_id: str = "mode",
    metadata: Mapping[str, Any] | None = None,
) -> ResonanceModeMetrics:
    """Compute standardized resonance metrics from one reconstructed absorption-like mode."""

    field = np.asarray(field_mT, dtype=float)
    signal = np.asarray(absorption_signal, dtype=float)
    finite_mask = np.isfinite(field) & np.isfinite(signal)
    if int(np.count_nonzero(finite_mask)) < 5:
        return _failed_metrics(
            owner_kind=owner_kind,
            owner_id=owner_id,
            hres=hres,
            config=config,
            failure_reason="insufficient_finite_points",
            support_start_field_mT=support_start_field_mT,
            support_end_field_mT=support_end_field_mT,
            metadata=metadata,
        )

    field = field[finite_mask]
    signal = signal[finite_mask]
    if field[0] > field[-1]:
        field = field[::-1]
        signal = signal[::-1]

    support_start = (
        float(field[0])
        if support_start_field_mT is None
        else float(max(support_start_field_mT, float(field[0])))
    )
    support_end = (
        float(field[-1])
        if support_end_field_mT is None
        else float(min(support_end_field_mT, float(field[-1])))
    )
    support_mask = (field >= support_start) & (field <= support_end)
    support_indices = np.flatnonzero(support_mask)
    if support_indices.size < 5:
        return _failed_metrics(
            owner_kind=owner_kind,
            owner_id=owner_id,
            hres=hres,
            config=config,
            failure_reason="insufficient_support_window_points",
            support_start_field_mT=support_start,
            support_end_field_mT=support_end,
            metadata=metadata,
        )

    edge_points = min(16, max(2, support_indices.size // 8))
    while edge_points >= 2 and support_indices.size < edge_points * 2:
        edge_points -= 1
    if edge_points < 2:
        return _failed_metrics(
            owner_kind=owner_kind,
            owner_id=owner_id,
            hres=hres,
            config=config,
            failure_reason="support_window_too_short_for_baseline",
            support_start_field_mT=support_start,
            support_end_field_mT=support_end,
            metadata=metadata,
        )

    baseline_field = np.concatenate(
        (field[support_indices[:edge_points]], field[support_indices[-edge_points:]])
    )
    baseline_signal = np.concatenate(
        (signal[support_indices[:edge_points]], signal[support_indices[-edge_points:]])
    )
    slope, intercept = np.polyfit(baseline_field, baseline_signal, deg=1)
    corrected_signal = signal - (slope * field + intercept)
    polarity = 1.0
    corrected_support = corrected_signal[support_mask]
    if abs(float(np.min(corrected_support))) > abs(float(np.max(corrected_support))):
        corrected_signal = -corrected_signal
        corrected_support = -corrected_support
        polarity = -1.0

    peak_height = float(np.max(corrected_support))
    if peak_height <= 0.0:
        return _failed_metrics(
            owner_kind=owner_kind,
            owner_id=owner_id,
            hres=hres,
            config=config,
            failure_reason="non_positive_absorption_peak",
            support_start_field_mT=support_start,
            support_end_field_mT=support_end,
            baseline_edge_points=edge_points,
            baseline_slope=float(slope),
            baseline_intercept=float(intercept),
            signal_polarity=polarity,
            metadata=metadata,
        )

    support_peak_indices = support_indices[
        np.isclose(
            corrected_signal[support_indices],
            peak_height,
            rtol=1e-9,
            atol=max(abs(peak_height) * 1e-9, 1e-12),
        )
    ]
    peak_index = int(
        support_peak_indices[np.argmin(np.abs(field[support_peak_indices] - float(hres)))]
    )
    peak_field_abs = float(field[peak_index])
    half_max_level = peak_height / 2.0
    left_crossing = _find_halfmax_crossing(
        field,
        corrected_signal,
        peak_index=peak_index,
        support_indices=support_indices,
        half_max_level=half_max_level,
        direction="left",
        interpolation=config.halfmax_interp,
    )
    right_crossing = _find_halfmax_crossing(
        field,
        corrected_signal,
        peak_index=peak_index,
        support_indices=support_indices,
        half_max_level=half_max_level,
        direction="right",
        interpolation=config.halfmax_interp,
    )
    if left_crossing is None or right_crossing is None:
        return _failed_metrics(
            owner_kind=owner_kind,
            owner_id=owner_id,
            hres=hres,
            config=config,
            failure_reason="halfmax_crossing_not_found",
            peak_field_abs=peak_field_abs,
            peak_height_abs=peak_height,
            half_max_level=half_max_level,
            support_start_field_mT=support_start,
            support_end_field_mT=support_end,
            baseline_edge_points=edge_points,
            baseline_slope=float(slope),
            baseline_intercept=float(intercept),
            signal_polarity=polarity,
            metadata=metadata,
        )

    h_left_half = float(left_crossing)
    h_right_half = float(right_crossing)
    hwhm_left = float(hres - h_left_half)
    hwhm_right = float(h_right_half - hres)
    fwhm = float(h_right_half - h_left_half)
    asymmetry_ratio = None
    if config.report_asymmetry and hwhm_left > 0.0:
        asymmetry_ratio = float(hwhm_right / hwhm_left)

    area_windows = compute_windowed_area_metrics(
        field,
        corrected_signal,
        hres=float(hres),
        fwhm=fwhm,
        hwhm_left=hwhm_left,
        hwhm_right=hwhm_right,
        config=config,
    )
    area_lookup = {round(item.multiplier, 9): item.area for item in area_windows}
    area_full = scalar_integral(field, corrected_signal) if config.compute_full_area else None
    return ResonanceModeMetrics(
        owner_kind=owner_kind,
        owner_id=owner_id,
        success=True,
        failure_reason=None,
        metrics_from=config.metrics_from,
        hres=float(hres),
        peak_field_abs=peak_field_abs,
        peak_height_abs=peak_height,
        half_max_level=half_max_level,
        h_left_half=h_left_half,
        h_right_half=h_right_half,
        fwhm=fwhm,
        hwhm_left=hwhm_left,
        hwhm_right=hwhm_right,
        asymmetry_ratio=asymmetry_ratio,
        area_pm_1fwhm=area_lookup.get(round(1.0, 9)),
        area_pm_2fwhm=area_lookup.get(round(2.0, 9)),
        area_pm_3fwhm=area_lookup.get(round(3.0, 9)),
        area_full=None if area_full is None else float(area_full),
        support_start_field_mT=support_start,
        support_end_field_mT=support_end,
        local_baseline_edge_points=edge_points,
        local_baseline_slope=float(slope),
        local_baseline_intercept=float(intercept),
        signal_polarity=polarity,
        area_windows=area_windows,
        metadata=dict(metadata or {}),
    )


def compute_windowed_area_metrics(
    field_mT: np.ndarray,
    absorption_signal: np.ndarray,
    *,
    hres: float,
    fwhm: float,
    hwhm_left: float,
    hwhm_right: float,
    config: ResonanceMetricsConfig,
) -> list[ResonanceAreaWindow]:
    """Compute configured area windows around one resonance mode."""

    field = np.asarray(field_mT, dtype=float)
    signal = np.asarray(absorption_signal, dtype=float)
    windows: list[ResonanceAreaWindow] = []
    for multiplier in config.area_window_multipliers:
        if config.area_window_mode == "side-aware":
            start_field = float(hres - multiplier * hwhm_left)
            end_field = float(hres + multiplier * hwhm_right)
        else:
            half_span = multiplier * fwhm / 2.0
            start_field = float(hres - half_span)
            end_field = float(hres + half_span)
        area = _integrate_between(field, signal, start_field, end_field)
        windows.append(
            ResonanceAreaWindow(
                multiplier=float(multiplier),
                label=_window_label(multiplier),
                window_mode=config.area_window_mode,
                start_field_mT=start_field,
                end_field_mT=end_field,
                area=area,
            )
        )
    return windows


def flatten_resonance_metrics(metrics: ResonanceModeMetrics) -> dict[str, Any]:
    """Flatten one mode-metrics record for CSV or summary-table export."""

    return {
        "owner_kind": metrics.owner_kind,
        "owner_id": metrics.owner_id,
        "resonance_metrics_success": metrics.success,
        "resonance_metrics_failure_reason": metrics.failure_reason,
        "hres": metrics.hres,
        "peak_field_abs": metrics.peak_field_abs,
        "peak_height_abs": metrics.peak_height_abs,
        "half_max_level": metrics.half_max_level,
        "h_left_half": metrics.h_left_half,
        "h_right_half": metrics.h_right_half,
        "fwhm": metrics.fwhm,
        "hwhm_left": metrics.hwhm_left,
        "hwhm_right": metrics.hwhm_right,
        "asymmetry_ratio": metrics.asymmetry_ratio,
        "area_pm_1fwhm": metrics.area_pm_1fwhm,
        "area_pm_2fwhm": metrics.area_pm_2fwhm,
        "area_pm_3fwhm": metrics.area_pm_3fwhm,
        "area_full": metrics.area_full,
        "support_start_field_mT": metrics.support_start_field_mT,
        "support_end_field_mT": metrics.support_end_field_mT,
        "local_baseline_edge_points": metrics.local_baseline_edge_points,
        "local_baseline_slope": metrics.local_baseline_slope,
        "local_baseline_intercept": metrics.local_baseline_intercept,
        "signal_polarity": metrics.signal_polarity,
    }


def _failed_metrics(
    *,
    owner_kind: str,
    owner_id: str,
    hres: float,
    config: ResonanceMetricsConfig,
    failure_reason: str,
    support_start_field_mT: float | None,
    support_end_field_mT: float | None,
    peak_field_abs: float | None = None,
    peak_height_abs: float | None = None,
    half_max_level: float | None = None,
    baseline_edge_points: int | None = None,
    baseline_slope: float | None = None,
    baseline_intercept: float | None = None,
    signal_polarity: float = 1.0,
    metadata: Mapping[str, Any] | None = None,
) -> ResonanceModeMetrics:
    return ResonanceModeMetrics(
        owner_kind=owner_kind,
        owner_id=owner_id,
        success=False,
        failure_reason=failure_reason,
        metrics_from=config.metrics_from,
        hres=float(hres),
        peak_field_abs=peak_field_abs,
        peak_height_abs=peak_height_abs,
        half_max_level=half_max_level,
        h_left_half=None,
        h_right_half=None,
        fwhm=None,
        hwhm_left=None,
        hwhm_right=None,
        asymmetry_ratio=None,
        area_pm_1fwhm=None,
        area_pm_2fwhm=None,
        area_pm_3fwhm=None,
        area_full=None,
        support_start_field_mT=support_start_field_mT,
        support_end_field_mT=support_end_field_mT,
        local_baseline_edge_points=baseline_edge_points,
        local_baseline_slope=baseline_slope,
        local_baseline_intercept=baseline_intercept,
        signal_polarity=signal_polarity,
        area_windows=[],
        metadata=dict(metadata or {}),
    )


def _find_halfmax_crossing(
    field: np.ndarray,
    signal: np.ndarray,
    *,
    peak_index: int,
    support_indices: np.ndarray,
    half_max_level: float,
    direction: Literal["left", "right"],
    interpolation: HalfmaxInterpolation,
) -> float | None:
    start = int(support_indices[0])
    end = int(support_indices[-1])
    if direction == "left":
        for index in range(peak_index, start, -1):
            y0 = float(signal[index - 1])
            y1 = float(signal[index])
            if (y0 <= half_max_level <= y1) or (y1 <= half_max_level <= y0):
                return _interpolate_crossing(
                    float(field[index - 1]),
                    y0,
                    float(field[index]),
                    y1,
                    half_max_level,
                    interpolation,
                )
        return None
    for index in range(peak_index, end):
        y0 = float(signal[index])
        y1 = float(signal[index + 1])
        if (y0 >= half_max_level >= y1) or (y1 >= half_max_level >= y0):
            return _interpolate_crossing(
                float(field[index]), y0, float(field[index + 1]), y1, half_max_level, interpolation
            )
    return None


def _interpolate_crossing(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    target: float,
    interpolation: HalfmaxInterpolation,
) -> float:
    if interpolation != "linear":
        raise ValueError(f"Unsupported halfmax interpolation: {interpolation}")
    if x0 == x1 or y0 == y1:
        return float((x0 + x1) / 2.0)
    fraction = (target - y0) / (y1 - y0)
    return float(x0 + fraction * (x1 - x0))


def _integrate_between(
    field: np.ndarray, signal: np.ndarray, start_field: float, end_field: float
) -> float | None:
    bounded_start = max(float(field[0]), float(start_field))
    bounded_end = min(float(field[-1]), float(end_field))
    if bounded_end <= bounded_start:
        return None
    inside_mask = (field > bounded_start) & (field < bounded_end)
    segment_field = np.concatenate(
        (
            np.asarray([bounded_start], dtype=float),
            np.asarray(field[inside_mask], dtype=float),
            np.asarray([bounded_end], dtype=float),
        )
    )
    segment_signal = np.concatenate(
        (
            np.asarray([np.interp(bounded_start, field, signal)], dtype=float),
            np.asarray(signal[inside_mask], dtype=float),
            np.asarray([np.interp(bounded_end, field, signal)], dtype=float),
        )
    )
    if segment_field.size < 2:
        return None
    return float(scalar_integral(segment_field, segment_signal))


def _window_label(multiplier: float) -> str:
    formatted = f"{multiplier:g}"
    return f"pm_{formatted}fwhm"
