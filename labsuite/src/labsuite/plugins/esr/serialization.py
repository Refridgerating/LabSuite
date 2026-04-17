"""Serialization helpers for ESR analysis payloads."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
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
from labsuite.plugins.esr.batch_qc import build_multi_bucket_slug, parse_esr_batch_identity


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


def export_esr_batch_overlay_figure(
    analyses: list[AnalysisResult],
    output_dir: Path,
) -> dict[str, Path]:
    """Export ESR batch figures grouped by sample/frequency bucket and replicate."""

    if not analyses:
        return {}

    identities = [
        parse_esr_batch_identity(analysis.dataset.source_path, analysis.dataset.metadata)
        for analysis in analyses
    ]
    slug_by_bucket = build_multi_bucket_slug(identities)

    grouped_entries: dict[tuple[str, float | None, str], list[dict[str, Any]]] = {}
    for analysis in analyses:
        identity = parse_esr_batch_identity(analysis.dataset.source_path, analysis.dataset.metadata)
        replicate_id = identity.replicate_id or "UNGROUPED"
        group_key = (identity.sample_id, identity.frequency_bucket_GHz, replicate_id)
        grouped_entries.setdefault(group_key, []).append(
            {
                "analysis": analysis,
                "source_stem": analysis.dataset.source_path.stem,
                "angle_deg": identity.nominal_angle_deg,
                "label": _esr_angle_display_label(
                    {
                        "source_stem": analysis.dataset.source_path.stem,
                        "angle_deg": identity.nominal_angle_deg,
                    }
                ),
                "sample_id": identity.sample_id,
                "frequency_GHz": identity.frequency_bucket_GHz,
            }
        )

    exported_paths: dict[str, Path] = {}
    for group_key, entries in sorted(grouped_entries.items(), key=lambda item: _esr_group_sort_key(item[0])):
        sample_id, frequency_GHz, replicate_id = group_key
        ordered_entries = sorted(
            entries,
            key=lambda entry: (
                float("inf") if entry["angle_deg"] is None else float(entry["angle_deg"]),
                str(entry["source_stem"]).lower(),
            ),
        )
        slug = slug_by_bucket.get((sample_id, frequency_GHz), "")
        suffix = f"{slug}_{replicate_id}" if slug else replicate_id
        offset_path = output_dir / f"batch_processed_offset_{suffix}.png"
        overlay_path = output_dir / f"batch_angle_overlay_{suffix}.png"
        _export_esr_group_figure(
            ordered_entries,
            replicate_id=replicate_id,
            sample_id=sample_id,
            frequency_GHz=frequency_GHz,
            destination=offset_path,
            plot_mode="offset",
        )
        _export_esr_group_figure(
            ordered_entries,
            replicate_id=replicate_id,
            sample_id=sample_id,
            frequency_GHz=frequency_GHz,
            destination=overlay_path,
            plot_mode="overlay",
        )
        exported_paths[f"batch_processed_offset_{suffix}"] = offset_path
        exported_paths[f"batch_angle_overlay_{suffix}"] = overlay_path
    return exported_paths


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


def _export_esr_group_figure(
    entries: list[dict[str, Any]],
    *,
    replicate_id: str,
    sample_id: str,
    frequency_GHz: float | None,
    destination: Path,
    plot_mode: str,
) -> None:
    figure, axis = plt.subplots(1, 1, figsize=(12.0, 7.5))
    colors = plt.cm.plasma(np.linspace(0.08, 0.92, max(len(entries), 1)))
    offset_step = _esr_overlay_offset_step([entry["analysis"] for entry in entries])

    for index, entry in enumerate(entries):
        analysis = entry["analysis"]
        processed_signal = np.asarray(analysis.processed.signal, dtype=float)
        field_mT = np.asarray(analysis.processed.field_mT, dtype=float)
        plotted_signal = processed_signal if plot_mode == "overlay" else processed_signal + index * offset_step
        axis.plot(
            field_mT,
            plotted_signal,
            color=colors[index],
            linewidth=1.3,
            label=entry["label"],
        )

    axis.set_title(_esr_group_title(sample_id, frequency_GHz, replicate_id, plot_mode))
    axis.set_xlabel("Field (mT)")
    axis.set_ylabel("Processed derivative" if plot_mode == "overlay" else "Processed derivative + offset")
    axis.grid(alpha=0.2)
    _place_esr_group_legend(figure, axis, len(entries))
    figure.tight_layout(rect=(0.03, 0.12, 0.98, 0.98))
    figure.savefig(destination, dpi=200)
    plt.close(figure)


def _place_esr_group_legend(figure, axis, entry_count: int) -> None:
    handles, labels = axis.get_legend_handles_labels()
    if not handles:
        return
    ncols = min(4, max(1, entry_count))
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncols=ncols,
        frameon=True,
    )


def _esr_overlay_offset_step(analyses: list[AnalysisResult]) -> float:
    amplitudes: list[float] = []
    for analysis in analyses:
        processed_signal = np.asarray(analysis.processed.signal, dtype=float)
        if processed_signal.size == 0:
            continue
        span = float(np.nanmax(processed_signal) - np.nanmin(processed_signal))
        if np.isfinite(span) and span > 0.0:
            amplitudes.append(span)
    if not amplitudes:
        return 1.0
    return max(amplitudes) * 1.2


def _parse_esr_filename_tokens(path: Path) -> dict[str, Any]:
    stem = path.stem.split(" - ", maxsplit=1)[0].strip()
    parts = [part for part in stem.split("-") if part]
    replicate_id: str | None = None
    angle_deg: float | None = None
    for part in parts:
        if replicate_id is None and re.fullmatch(r"R\d+", part, flags=re.IGNORECASE):
            replicate_id = part.upper()
            continue
        angle_match = re.fullmatch(r"(-?\d+(?:[.,]\d+)?)deg", part, flags=re.IGNORECASE)
        if angle_deg is None and angle_match is not None:
            angle_deg = float(angle_match.group(1).replace(",", "."))
    return {
        "source_stem": stem,
        "replicate_id": replicate_id,
        "angle_deg": angle_deg,
    }


def _esr_angle_display_label(filename_tokens: dict[str, Any]) -> str:
    angle_deg = filename_tokens["angle_deg"]
    if angle_deg is None:
        return str(filename_tokens["source_stem"])
    if float(angle_deg).is_integer():
        return f"{int(angle_deg)} deg"
    return f"{angle_deg:.3g} deg"


def _esr_group_title(sample_id: str, frequency_GHz: float | None, replicate_id: str, plot_mode: str) -> str:
    frequency_label = "" if frequency_GHz is None else f", {frequency_GHz:.3f} GHz"
    if plot_mode == "overlay":
        return f"ESR Angle Overlay ({sample_id}, {replicate_id}{frequency_label})"
    return f"ESR Processed Derivative Offset ({sample_id}, {replicate_id}{frequency_label})"


def _esr_group_sort_key(group_key: tuple[str, float | None, str]) -> tuple[str, float, tuple[int, str]]:
    sample_id, frequency_GHz, replicate_id = group_key
    return (
        sample_id.lower(),
        float("inf") if frequency_GHz is None else float(frequency_GHz),
        _esr_replicate_sort_key(replicate_id),
    )


def _esr_replicate_sort_key(replicate_id: str) -> tuple[int, str]:
    match = re.fullmatch(r"R(\d+)", replicate_id, flags=re.IGNORECASE)
    if match is None:
        return (1_000_000, replicate_id.lower())
    return (int(match.group(1)), replicate_id.lower())
