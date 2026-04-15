"""Serialization helpers for ESR analysis payloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from labsuite.core.types import (
    AnalysisResult,
    BaselineSummary,
    ConvergenceSummary,
    FeatureSummary,
    FitAttemptRecord,
    FitDecision,
    FitIntegratedCurves,
    FitResult,
    IntegralSummary,
    IntegratedCurves,
    ParameterDiagnostic,
    PeakFitResult,
    PeakWindow,
    PrimaryIntegratedCurves,
    ProcessedTrace,
    ResidualSummary,
    TraceDataset,
)


def load_esr_analysis_result(path: Path) -> AnalysisResult:
    """Load a saved ESR JSON analysis payload back into dataclasses."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    dataset = TraceDataset(
        modality=payload["modality"],
        source_path=Path(payload["source_file"]),
        field_mT=np.asarray(payload["raw_trace"]["field_mT"], dtype=float),
        signal=np.asarray(payload["raw_trace"]["signal"], dtype=float),
        metadata=payload["metadata"],
    )
    processed = ProcessedTrace(
        field_mT=np.asarray(payload["processed_trace"]["field_mT"], dtype=float),
        signal=np.asarray(payload["processed_trace"]["signal"], dtype=float),
        steps=payload["processed_trace"]["steps"],
    )
    integrated = IntegratedCurves(
        field_mT=np.asarray(payload["integrated_curves"]["field_mT"], dtype=float),
        absorption_signal=np.asarray(payload["integrated_curves"]["absorption_signal"], dtype=float),
        area_signal=np.asarray(payload["integrated_curves"]["area_signal"], dtype=float),
        steps=payload["integrated_curves"]["steps"],
    )
    return AnalysisResult(
        dataset=dataset,
        processed=processed,
        integrated=integrated,
        primary_integrated=_load_fit_integrated(payload["primary_integrated_curves"]),
        fit_local_integrated=_load_windowed_integrated(payload["fit_local_integrated_curves"]),
        local_integrated=_load_windowed_integrated(payload["local_integrated_curves"]),
        derivative_baseline=_load_baseline(payload["baseline_summaries"]["derivative"]),
        absorption_baseline=_load_baseline(payload["baseline_summaries"]["absorption"]),
        selected_mode=payload["fit_selection"]["selected_mode"],
        fit_decision=_load_fit_decision(payload["fit_selection"]["decision"]),
        single_fit=_load_fit_result(payload["fit_selection"]["single_fit"]),
        single_fit_attempts=[_load_fit_attempt(item) for item in payload["fit_selection"]["single_fit_attempts"]],
        peak_fits=[_load_peak_fit(item) for item in payload["fit_selection"]["peak_fits"]],
        selected_fit_signal=np.asarray(payload["fit_selection"]["selected_fit_signal"], dtype=float),
        selected_residual=np.asarray(payload["fit_selection"]["selected_residual"], dtype=float),
        total_integral=_load_integral(payload["integral_summaries"]["total"]),
        fit_local_total_integral=_load_integral(payload["integral_summaries"]["fit_local_total"]),
        local_total_integral=_load_integral(payload["integral_summaries"]["local_total"]),
        diagnostic_total_integral=_load_integral(payload["integral_summaries"]["diagnostic_total"]),
        peak_integrals=[_load_integral(item) for item in payload["integral_summaries"]["peaks"]],
        fit_local_peak_integrals=[_load_integral(item) for item in payload["integral_summaries"]["fit_local_peaks"]],
        local_peak_integrals=[_load_integral(item) for item in payload["integral_summaries"]["local_peaks"]],
        fit_local_disagreement_ratio=payload["qc"]["fit_local_disagreement_ratio"],
        fit_local_disagreement_flag=payload["qc"]["fit_local_disagreement_flag"],
        fit_local_disagreement_reason=payload["qc"]["fit_local_disagreement_reason"],
        recipe_name=payload["recipe"]["name"],
        recipe_config=payload["recipe"]["config"],
    )


def build_esr_report(input_path: Path, output_path: Path | None = None, *, recursive: bool = True) -> Path:
    """Generate a Markdown report from saved ESR JSON analyses."""

    resolved_input = input_path.resolve()
    if resolved_input.is_file():
        payload = json.loads(resolved_input.read_text(encoding="utf-8"))
        destination = output_path.resolve() if output_path is not None else resolved_input.with_name(
            f"{resolved_input.stem.replace('_analysis', '')}_report.md"
        )
        destination.write_text(_build_single_report(payload), encoding="utf-8")
        return destination

    json_paths = sorted(
        resolved_input.rglob("*_analysis.json") if recursive else resolved_input.glob("*_analysis.json"),
        key=lambda path: str(path).lower(),
    )
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in json_paths]
    destination = output_path.resolve() if output_path is not None else resolved_input / "batch_report.md"
    destination.write_text(_build_batch_report(payloads), encoding="utf-8")
    return destination


def _load_fit_result(payload: dict[str, Any] | None) -> FitResult | None:
    if payload is None:
        return None
    return FitResult(
        model_name=payload["model_name"],
        parameters=payload["parameters"],
        derived=payload["derived"],
        metrics=payload["metrics"],
        fitted_signal=np.asarray(payload["fitted_signal"], dtype=float),
        residual=np.asarray(payload["residual"], dtype=float),
        parameter_diagnostics={
            name: ParameterDiagnostic(**diagnostic)
            for name, diagnostic in payload["parameter_diagnostics"].items()
        },
        convergence=ConvergenceSummary(**payload["convergence"]),
        residual_summary=ResidualSummary(**payload["residual_summary"]),
        feature_summary=None if payload["feature_summary"] is None else FeatureSummary(**payload["feature_summary"]),
        bound_hits=payload["bound_hits"],
        success=payload["success"],
    )


def _load_fit_attempt(payload: dict[str, Any]) -> FitAttemptRecord:
    source_window = payload["source_window"]
    return FitAttemptRecord(
        scope=payload["scope"],
        fit=_load_fit_result(payload["fit"]),
        source_window=None if source_window is None else PeakWindow(label="window", **source_window),
        accepted=payload["accepted"],
        rejection_reason=payload["rejection_reason"],
        selected_for_primary=payload["selected_for_primary"],
    )


def _load_peak_fit(payload: dict[str, Any]) -> PeakFitResult:
    window_payload = payload["window"]
    return PeakFitResult(
        label=payload["label"],
        window=PeakWindow(label=payload["label"], **window_payload),
        fit=_load_fit_result(payload["fit"]),
        component_signal=np.asarray(payload["component_signal"], dtype=float),
        attempts=[_load_fit_attempt(item) for item in payload["attempts"]],
    )


def _load_fit_decision(payload: dict[str, Any]) -> FitDecision:
    return FitDecision(
        requested_mode=payload["requested_mode"],
        selected_mode=payload["selected_mode"],
        candidate_peak_count=payload["candidate_peak_count"],
        split_improvement_ratio=payload["split_improvement_ratio"],
        split_threshold=payload["split_threshold"],
        reason=payload["reason"],
        metrics=payload["metrics"],
    )


def _load_integral(payload: dict[str, Any]) -> IntegralSummary:
    return IntegralSummary(
        label=payload["label"],
        start_field_mT=payload["start_field_mT"],
        end_field_mT=payload["end_field_mT"],
        absorption_integral=payload["absorption_integral"],
        area_integral=payload["area_integral"],
        integration_kind=payload["integration_kind"],
        window_source=payload["window_source"],
        baseline_polyorder=payload["baseline_polyorder"],
        integration_window_clipped_by_detected_window=payload["integration_window_clipped_by_detected_window"],
    )


def _load_fit_integrated(payload: dict[str, Any] | None) -> FitIntegratedCurves | None:
    if payload is None:
        return None
    return FitIntegratedCurves(
        field_mT=np.asarray(payload["field_mT"], dtype=float),
        absorption_signal=np.asarray(payload["absorption_signal"], dtype=float),
        area_signal=np.asarray(payload["area_signal"], dtype=float),
        integration_kind=payload["integration_kind"],
        model_name=payload["model_name"],
    )


def _load_windowed_integrated(payload: dict[str, Any] | None) -> PrimaryIntegratedCurves | None:
    if payload is None:
        return None
    return PrimaryIntegratedCurves(
        field_mT=np.asarray(payload["field_mT"], dtype=float),
        absorption_signal=np.asarray(payload["absorption_signal"], dtype=float),
        area_signal=np.asarray(payload["area_signal"], dtype=float),
        start_field_mT=payload["start_field_mT"],
        end_field_mT=payload["end_field_mT"],
        integration_kind=payload["integration_kind"],
        window_source=payload["window_source"],
        baseline_polyorder=payload["baseline_polyorder"],
        integration_window_clipped_by_detected_window=payload["integration_window_clipped_by_detected_window"],
        model_name=payload.get("model_name"),
    )


def _load_baseline(payload: dict[str, Any]) -> BaselineSummary:
    return BaselineSummary(
        target=payload["target"],
        edge_points=payload["edge_points"],
        slope=payload["slope"],
        intercept=payload["intercept"],
    )


def _build_single_report(payload: dict[str, Any]) -> str:
    selection = payload["fit_selection"]
    lines = [
        f"# ESR Report: {Path(payload['source_file']).name}",
        "",
        f"- Mode: `{selection['selected_mode']}`",
        f"- Recipe: `{payload['recipe']['name']}`",
        f"- Parser: `{payload['metadata']['parser']}`",
        "",
        "## Diagnostics",
        "",
        f"- Total area integral: `{payload['integral_summaries']['total']['area_integral']}`",
        f"- Diagnostic full-span area integral: `{payload['integral_summaries']['diagnostic_total']['area_integral']}`",
        f"- Fit/local disagreement flag: `{payload['qc']['fit_local_disagreement_flag']}`",
    ]
    return "\n".join(lines) + "\n"


def _build_batch_report(payloads: list[dict[str, Any]]) -> str:
    lines = [
        "# ESR Batch Report",
        "",
        f"- Measurements: `{len(payloads)}`",
        "",
        "| Source | Mode | Total area integral | Disagreement flag |",
        "| --- | --- | --- | --- |",
    ]
    for payload in payloads:
        lines.append(
            "| "
            + " | ".join(
                [
                    Path(payload["source_file"]).name,
                    payload["fit_selection"]["selected_mode"],
                    str(payload["integral_summaries"]["total"]["area_integral"]),
                    str(payload["qc"]["fit_local_disagreement_flag"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"
