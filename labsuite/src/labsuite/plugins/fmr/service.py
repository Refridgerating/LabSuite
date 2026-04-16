"""Service layer, export, and report helpers for the FMR workflow."""

from __future__ import annotations

from collections import Counter
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from labsuite.core.exceptions import WorkflowError
from labsuite.core.measurement_models import (
    MeasurementAnalysisResult,
    MeasurementRecord,
    PlotManifest,
    SampleRecord,
    to_serializable,
)
from labsuite.core.recipes import FmrRecipe, load_fmr_recipe
from labsuite.plugins.fmr.derived import build_fmr_series, fit_fmr_physics
from labsuite.plugins.fmr.fitters import assess_trace_fit_quality, fit_fmr_trace
from labsuite.plugins.fmr.models import (
    FmrCandidateWindow,
    FmrComponentFitResult,
    FmrPhysicsCollectionResult,
    FmrPhysicsResult,
    FmrSeriesCollectionResult,
    FmrSeriesResult,
    FmrTraceFitResult,
    FmrTraceModelResult,
)
from labsuite.plugins.fmr.parser import parse_fmr_file
from labsuite.plugins.fmr.preprocess import apply_fmr_preprocessing


def analyze_fmr_file(source_path: Path, recipe_path: Path) -> MeasurementAnalysisResult:
    """Run the PhaseFMR-first single-file analysis pipeline."""

    file_dataset = parse_fmr_file(source_path.resolve())
    recipe = load_fmr_recipe(recipe_path.resolve())

    trace_fits: list[FmrTraceFitResult] = []
    processed_traces: list[dict[str, Any]] = []
    warnings = list(file_dataset.warnings)
    for trace in file_dataset.traces:
        selected_trace = _select_signal_channel(trace, recipe.signal_channel)
        processed_trace = apply_fmr_preprocessing(selected_trace, recipe)
        fit_result = fit_fmr_trace(selected_trace, processed_trace, recipe)
        accepted, rejection_reason, fit_warnings = assess_trace_fit_quality(fit_result, recipe=recipe)
        fit_result.accepted = accepted
        fit_result.rejection_reason = rejection_reason
        fit_result.warnings.extend(fit_warnings)
        fit_result.metadata["requested_signal_channel"] = recipe.signal_channel
        processed_traces.append(
            {
                "trace_id": processed_trace.trace_id,
                "frequency_GHz": selected_trace.frequency_GHz,
                "field_mT": processed_trace.field_mT,
                "signal": processed_trace.signal,
                "steps": processed_trace.steps,
                "baseline_summary": processed_trace.baseline_summary,
            }
        )
        trace_fits.append(fit_result)
        warnings.extend(f"{selected_trace.trace_id}:{warning}" for warning in fit_result.warnings)
        if rejection_reason is not None:
            warnings.append(f"{selected_trace.trace_id}:{rejection_reason}")

    series_collection = build_fmr_series(trace_fits, measurement_mode=file_dataset.measurement_mode)
    physics_collection = fit_fmr_physics(series_collection, recipe)
    warnings.extend(series_collection.warnings)
    warnings.extend(physics_collection.warnings)
    legacy_series, legacy_physics = _legacy_series_pair(series_collection, physics_collection)

    sample = SampleRecord(
        sample_id=file_dataset.sample_name,
        series_id=file_dataset.sample_name,
        filename_tokens={
            "replicate_id": file_dataset.replicate_id,
            "angle_deg": file_dataset.angle_deg,
            "sweep_span_label": file_dataset.sweep_span_label,
        },
        grouping_keys={
            "sample": file_dataset.sample_name,
            "sample_angle_temperature": _group_key(
                file_dataset.sample_name,
                file_dataset.angle_deg,
                file_dataset.nominal_temperature_K,
                file_dataset.measurement_mode,
            ),
        },
    )
    measurement = MeasurementRecord(
        modality="fmr",
        source_path=file_dataset.source_path,
        sample=sample,
        replicate_id=file_dataset.replicate_id,
        condition_metadata={
            "angle_deg": file_dataset.angle_deg,
            "temperature_K": file_dataset.nominal_temperature_K,
            "measurement_mode": file_dataset.measurement_mode,
        },
        raw_metadata=file_dataset.metadata,
    )

    summary_metrics = {
        "sample_id": file_dataset.sample_name,
        "series_id": file_dataset.sample_name,
        "replicate_id": file_dataset.replicate_id,
        "angle_deg": file_dataset.angle_deg,
        "temperature_K": file_dataset.nominal_temperature_K,
        "measurement_mode": file_dataset.measurement_mode,
        "trace_count": len(trace_fits),
        "has_multiple_frequencies": bool(file_dataset.metadata.get("has_multiple_frequencies")),
        "frequency_GHz_values": list(file_dataset.metadata.get("frequency_GHz_values", [])),
        "accepted_trace_count": len({fit.trace_id for fit in trace_fits if fit.accepted}),
        "accepted_component_count": sum(len(series.included_component_ids) for series in series_collection.series_by_label.values()),
        "excluded_trace_count": int(series_collection.metadata.get("excluded_trace_count", 0)),
        "mode_counts": dict(series_collection.metadata.get("mode_counts", {})),
        "series_labels": sorted(series_collection.series_by_label),
        "rejection_reason_histogram": _summarize_rejection_reasons(trace_fits),
        "kittel_success": any(item.kittel_fit is not None and item.kittel_fit.success for item in physics_collection.physics_by_label.values()),
        "linewidth_success": any(item.linewidth_fit is not None and item.linewidth_fit.success for item in physics_collection.physics_by_label.values()),
        "gamma_GHz_per_T": None if legacy_physics is None else legacy_physics.derived_parameters.get("gamma_GHz_per_T"),
        "g": None if legacy_physics is None else legacy_physics.derived_parameters.get("g"),
        "M_eff_mT": None if legacy_physics is None else legacy_physics.derived_parameters.get("M_eff_mT"),
        "alpha": None if legacy_physics is None else legacy_physics.derived_parameters.get("alpha"),
        "DeltaH0_mT": None if legacy_physics is None else legacy_physics.derived_parameters.get("DeltaH0_mT"),
        "mode_physics_success": {
            label: {
                "kittel": bool(item.kittel_fit is not None and item.kittel_fit.success),
                "linewidth": bool(item.linewidth_fit is not None and item.linewidth_fit.success),
            }
            for label, item in physics_collection.physics_by_label.items()
        },
        "warning_count": len(warnings),
        "warnings": warnings,
    }

    plot_manifest = PlotManifest(
        figure_type="fmr_trace_series_diagnostic",
        title=f"FMR Diagnostic: {file_dataset.sample_name}",
        series=[
            {"label": "Trace fits", "x": "field_mT", "y": "processed_signal"},
            {"label": "Hres vs frequency", "x": "frequency_GHz", "y": "resonance_field_mT"},
            {"label": "DeltaH vs frequency", "x": "frequency_GHz", "y": "linewidth_mT"},
        ],
        annotations=[
            {"label": "Accepted traces", "value": summary_metrics["accepted_trace_count"]},
            {"label": "Accepted components", "value": summary_metrics["accepted_component_count"]},
            {"label": "Series labels", "value": ",".join(summary_metrics["series_labels"])},
        ],
        theme={"trace_mode": "overlay"},
    )

    analysis_payload = {
        "file_dataset": to_serializable(file_dataset),
        "standardized_traces": to_serializable(file_dataset.traces),
        "processed_traces": to_serializable(processed_traces),
        "trace_fit_results": to_serializable(trace_fits),
        "series_collection_result": to_serializable(series_collection),
        "physics_collection_result": to_serializable(physics_collection),
        "series_result": None if legacy_series is None else to_serializable(legacy_series),
        "physics_result": None if legacy_physics is None else to_serializable(legacy_physics),
    }

    return MeasurementAnalysisResult(
        measurement=measurement,
        provenance={
            "parser": file_dataset.metadata.get("parser"),
            "source_format": file_dataset.metadata.get("source_format"),
            "recipe_name": recipe.name,
            "recipe_config": recipe.to_dict(),
            "canonical_signal_channel": recipe.signal_channel,
            "trace_fit_model": recipe.trace_fit_model,
            "physics_model": recipe.physics_model,
            "canonical_units": {
                "field": "mT",
                "raw_field": "Oe",
                "signal": "arb",
                "frequency": "GHz",
                "temperature": "K",
            },
        },
        warnings=warnings,
        summary_metrics=summary_metrics,
        artifacts={},
        analysis_payload=analysis_payload,
        plot_manifest=plot_manifest,
    )


def _select_signal_channel(trace, channel_name: str):
    channel_map = {
        "i": trace.i_signal,
        "q": trace.q_signal,
        "fit_source": trace.fit_source_signal,
        "fit": trace.fit_signal,
        "aux": trace.aux_signal,
    }
    selected_signal = channel_map.get(channel_name)
    if selected_signal is None:
        raise WorkflowError(f"Requested FMR signal channel is unavailable for {trace.trace_id}: {channel_name}")
    metadata = dict(trace.metadata)
    metadata["selected_signal_channel"] = channel_name
    return replace(trace, signal=np.asarray(selected_signal, dtype=float).copy(), metadata=metadata)


def _group_key(
    sample_id: str | None,
    angle_deg: float | None,
    temperature_K: float | None,
    measurement_mode: str | None,
) -> str:
    return "|".join(
        [
            str(sample_id or "unknown"),
            "na" if angle_deg is None else f"{float(angle_deg):.3f}",
            "na" if temperature_K is None else f"{float(temperature_K):.3f}",
            str(measurement_mode or "unknown"),
        ]
    )


def export_fmr_analysis_json(result: MeasurementAnalysisResult, destination: Path) -> Path:
    """Write the complete FMR analysis envelope to JSON."""

    destination.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return destination


def export_fmr_analysis_csv(result: MeasurementAnalysisResult | dict[str, Any], destination: Path) -> Path:
    """Export point-wise trace data and fit overlays to CSV."""

    payload = _normalize_result(result)
    processed_by_id = {
        item["trace_id"]: item for item in payload["analysis_payload"]["processed_traces"]
    }
    fits = payload["analysis_payload"]["trace_fit_results"]

    header = [
        "trace_id",
        "sample_id",
        "frequency_GHz",
        "angle_deg",
        "temperature_K",
        "selected_mode",
        "accepted",
        "field_mT",
        "raw_signal",
        "processed_signal",
        "selected_fit_signal",
        "residual",
        "single_unassigned_signal",
        "mode_1_signal",
        "mode_2_signal",
        "I",
        "Q",
        "fit_source",
        "fit",
        "aux",
        "temp_K",
        "time_s",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for fit in fits:
            trace_id = fit["trace_id"]
            processed = processed_by_id[trace_id]
            raw_trace = _trace_lookup(payload, trace_id)
            field = processed["field_mT"]
            raw_signal = raw_trace["signal"]
            component_signals = {
                component["component_label"]: component["component_signal"]
                for component in fit.get("selected_components", [])
            }
            for index in range(len(field)):
                writer.writerow(
                    [
                        trace_id,
                        fit["sample_name"],
                        _format_value(fit["frequency_GHz"]),
                        _format_value(fit["angle_deg"]),
                        _format_value(fit["temperature_K"]),
                        fit.get("selected_mode", ""),
                        str(bool(fit["accepted"])).lower(),
                        _format_value(field[index]),
                        _format_value(raw_signal[index]),
                        _format_value(processed["signal"][index]),
                        _format_value(fit["fitted_signal"][index]),
                        _format_value(fit["residual"][index]),
                        _format_optional_signal(component_signals.get("single_unassigned"), index),
                        _format_optional_signal(component_signals.get("mode_1"), index),
                        _format_optional_signal(component_signals.get("mode_2"), index),
                        _format_optional_signal(raw_trace.get("i_signal"), index),
                        _format_optional_signal(raw_trace.get("q_signal"), index),
                        _format_optional_signal(raw_trace.get("fit_source_signal"), index),
                        _format_optional_signal(raw_trace.get("fit_signal"), index),
                        _format_optional_signal(raw_trace.get("aux_signal"), index),
                        _format_optional_signal(raw_trace.get("temp_K_signal"), index),
                        _format_optional_signal(raw_trace.get("time_s_signal"), index),
                    ]
                )
    return destination


def export_fmr_summary_csv(result: MeasurementAnalysisResult | dict[str, Any], destination: Path) -> Path:
    """Export one summary row per FMR trace fit."""

    payload = _normalize_result(result)
    fieldnames = [
        "trace_id",
        "sample_id",
        "frequency_GHz",
        "angle_deg",
        "temperature_K",
        "trace_accepted",
        "trace_rejection_reason",
        "requested_mode",
        "selected_mode",
        "selection_reason",
        "candidate_window_count",
        "double_fit_improvement_ratio",
        "partial_component_qc",
        "component_label",
        "component_id",
        "component_accepted",
        "component_rejection_reason",
        "model_name",
        "H_res_mT",
        "H_res_mT_stderr",
        "DeltaH_mT",
        "DeltaH_mT_stderr",
        "amplitude_symmetric",
        "amplitude_symmetric_stderr",
        "amplitude_antisymmetric",
        "amplitude_antisymmetric_stderr",
        "baseline_offset",
        "baseline_offset_stderr",
        "baseline_slope",
        "baseline_slope_stderr",
        "r_squared",
        "rmse",
        "rss",
        "residual_rmse_fraction",
        "amplitude_snr",
        "signal_max_abs",
        "feature_center_mT",
        "feature_peak_to_peak_mT",
        "center_feature_disagreement_mT",
        "critical_bound_hit_names",
        "acceptance_checks",
        "convergence_message",
        "warning_count",
        "warnings",
        "included_in_series",
    ]

    collection = payload["analysis_payload"]["series_collection_result"]
    included_component_ids = {
        component_id
        for series in collection.get("series_by_label", {}).values()
        for component_id in series.get("included_component_ids", [])
    }
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for fit in payload["analysis_payload"]["trace_fit_results"]:
            diagnostics = fit["parameter_diagnostics"]
            trace_warnings = fit.get("warnings", [])
            for component in fit.get("selected_components", []):
                component_diagnostics = component.get("parameter_diagnostics", {})
                writer.writerow(
                    {
                        "trace_id": fit["trace_id"],
                        "sample_id": fit["sample_name"],
                        "frequency_GHz": fit["frequency_GHz"],
                        "angle_deg": fit["angle_deg"],
                        "temperature_K": fit["temperature_K"],
                        "trace_accepted": fit["accepted"],
                        "trace_rejection_reason": fit["rejection_reason"],
                        "requested_mode": fit.get("requested_mode"),
                        "selected_mode": fit.get("selected_mode"),
                        "selection_reason": fit.get("selection_reason"),
                        "candidate_window_count": fit.get("candidate_window_count"),
                        "double_fit_improvement_ratio": fit.get("double_fit_improvement_ratio"),
                        "partial_component_qc": fit.get("partial_component_qc"),
                        "component_label": component.get("component_label"),
                        "component_id": component.get("component_id"),
                        "component_accepted": component.get("accepted"),
                        "component_rejection_reason": component.get("rejection_reason"),
                        "model_name": fit["model_name"],
                        "H_res_mT": component.get("H_res_mT"),
                        "H_res_mT_stderr": _diagnostic_stderr(component_diagnostics, "H_res_mT"),
                        "DeltaH_mT": component.get("DeltaH_mT"),
                        "DeltaH_mT_stderr": _diagnostic_stderr(component_diagnostics, "DeltaH_mT"),
                        "amplitude_symmetric": component.get("amplitude_symmetric"),
                        "amplitude_symmetric_stderr": _diagnostic_stderr(component_diagnostics, "amplitude_symmetric"),
                        "amplitude_antisymmetric": component.get("amplitude_antisymmetric"),
                        "amplitude_antisymmetric_stderr": _diagnostic_stderr(component_diagnostics, "amplitude_antisymmetric"),
                        "baseline_offset": fit["parameters"].get("baseline_offset"),
                        "baseline_offset_stderr": _diagnostic_stderr(diagnostics, "baseline_offset"),
                        "baseline_slope": fit["parameters"].get("baseline_slope"),
                        "baseline_slope_stderr": _diagnostic_stderr(diagnostics, "baseline_slope"),
                        "r_squared": fit["metrics"]["r_squared"],
                        "rmse": fit["residual_summary"]["rmse"],
                        "rss": fit["residual_summary"]["rss"],
                        "residual_rmse_fraction": component.get("residual_rmse_fraction"),
                        "amplitude_snr": component.get("amplitude_snr"),
                        "signal_max_abs": component.get("signal_max_abs"),
                        "feature_center_mT": component.get("feature_center_mT"),
                        "feature_peak_to_peak_mT": component.get("feature_peak_to_peak_mT"),
                        "center_feature_disagreement_mT": component.get("center_feature_disagreement_mT"),
                        "critical_bound_hit_names": "|".join(component.get("critical_bound_hit_names", [])),
                        "acceptance_checks": _format_acceptance_checks(component.get("acceptance_checks", {})),
                        "convergence_message": fit["convergence"]["message"],
                        "warning_count": len(trace_warnings) + len(component.get("warnings", [])),
                        "warnings": "|".join([*trace_warnings, *component.get("warnings", [])]),
                        "included_in_series": component.get("component_id") in included_component_ids,
                    }
                )
    return destination


def export_fmr_series_csv(result: MeasurementAnalysisResult | dict[str, Any], destination: Path) -> Path:
    """Export the per-file resonance series and higher-level physics fits."""

    payload = _normalize_result(result)
    collection = payload["analysis_payload"]["series_collection_result"]
    physics_collection = payload["analysis_payload"]["physics_collection_result"]

    fieldnames = [
        "series_label",
        "mode_label",
        "row_type",
        "frequency_GHz",
        "resonance_field_mT",
        "linewidth_mT",
        "trace_id",
        "component_id",
        "label",
        "value",
        "extra",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for series_label, series in collection.get("series_by_label", {}).items():
            for frequency, resonance, linewidth, trace_id, component_id in zip(
                series["frequency_GHz"],
                series["resonance_field_mT"],
                series["linewidth_mT"],
                series.get("included_trace_ids", []),
                series.get("included_component_ids", []),
                strict=True,
            ):
                writer.writerow(
                    {
                        "series_label": series_label,
                        "mode_label": series_label,
                        "row_type": "series_point",
                        "frequency_GHz": frequency,
                        "resonance_field_mT": resonance,
                        "linewidth_mT": linewidth,
                        "trace_id": trace_id,
                        "component_id": component_id,
                        "label": "H_res_mT",
                        "value": linewidth,
                        "extra": "DeltaH_mT",
                    }
                )
            physics = physics_collection.get("physics_by_label", {}).get(series_label)
            if physics is None:
                continue
            for fit_key in ("kittel_fit", "linewidth_fit"):
                fit = physics.get(fit_key)
                if fit is None:
                    continue
                writer.writerow(
                    {
                        "series_label": series_label,
                        "mode_label": series_label,
                        "row_type": fit_key,
                        "frequency_GHz": "",
                        "resonance_field_mT": "",
                        "linewidth_mT": "",
                        "trace_id": "",
                        "component_id": "",
                        "label": "success",
                        "value": fit["success"],
                        "extra": fit["message"],
                    }
                )
                for name, value in fit.get("parameters", {}).items():
                    writer.writerow(
                        {
                            "series_label": series_label,
                            "mode_label": series_label,
                            "row_type": fit_key,
                            "frequency_GHz": "",
                            "resonance_field_mT": "",
                            "linewidth_mT": "",
                            "trace_id": "",
                            "component_id": "",
                            "label": name,
                            "value": value,
                            "extra": fit.get("stderr", {}).get(name),
                        }
                    )
            for name, value in physics.get("derived_parameters", {}).items():
                writer.writerow(
                    {
                        "series_label": series_label,
                        "mode_label": series_label,
                        "row_type": "derived_parameter",
                        "frequency_GHz": "",
                        "resonance_field_mT": "",
                        "linewidth_mT": "",
                        "trace_id": "",
                        "component_id": "",
                        "label": name,
                        "value": value,
                        "extra": "",
                    }
                )
    return destination


def export_fmr_analysis_figure(result: MeasurementAnalysisResult | dict[str, Any], destination: Path) -> Path:
    """Save the FMR diagnostic figure for one analyzed file."""

    payload = _normalize_result(result)
    summary = payload["summary_metrics"]
    fits = payload["analysis_payload"]["trace_fit_results"]
    collection = payload["analysis_payload"]["series_collection_result"]
    physics_collection = payload["analysis_payload"]["physics_collection_result"]

    figure, axes = plt.subplots(3, 1, figsize=(10.5, 11.0), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(fits), 1)))
    for index, fit in enumerate(fits):
        color = colors[index]
        processed = _processed_trace_lookup(payload, fit["trace_id"])
        amplitude = np.asarray(processed["signal"], dtype=float)
        offset = index * 0.35 * max(1.0, float(np.nanmax(np.abs(amplitude)) or 1.0))
        label = f"{fit['frequency_GHz']:.3g} GHz"
        axes[0].plot(processed["field_mT"], amplitude + offset, color=color, linewidth=1.1, label=label)
        axes[0].plot(fit["field_mT"], np.asarray(fit["fitted_signal"], dtype=float) + offset, color=color, linewidth=1.1, linestyle="--")
    axes[0].set_title(f"FMR Trace Fits: {summary['sample_id']}")
    axes[0].set_xlabel("Field (mT)")
    axes[0].set_ylabel("Signal + offset")
    axes[0].grid(alpha=0.2)
    if fits:
        axes[0].legend(loc="best", ncols=2)

    series_colors = {"single_unassigned": "#1d4ed8", "mode_1": "#047857", "mode_2": "#b91c1c"}
    for label, series in collection.get("series_by_label", {}).items():
        if series["frequency_GHz"]:
            axes[1].scatter(series["frequency_GHz"], series["resonance_field_mT"], color=series_colors.get(label, "#1d4ed8"), s=28, label=f"{label} data")
        physics = physics_collection.get("physics_by_label", {}).get(label)
        if physics and physics.get("kittel_fit") and physics["kittel_fit"]["success"]:
            fitted_frequency = np.asarray(physics["kittel_fit"]["fitted_y"], dtype=float)
            fitted_resonance = np.asarray(physics["kittel_fit"]["x"], dtype=float)
            order = np.argsort(fitted_frequency)
            axes[1].plot(fitted_frequency[order], fitted_resonance[order], color=series_colors.get(label, "#1d4ed8"), linewidth=1.3, linestyle="--", label=f"{label} Kittel")
    axes[1].set_title("Resonance Field vs Frequency")
    axes[1].set_xlabel("Frequency (GHz)")
    axes[1].set_ylabel("H_res (mT)")
    axes[1].grid(alpha=0.2)
    if collection.get("series_by_label"):
        axes[1].legend(loc="best")

    for label, series in collection.get("series_by_label", {}).items():
        if series["frequency_GHz"]:
            axes[2].scatter(series["frequency_GHz"], series["linewidth_mT"], color=series_colors.get(label, "#047857"), s=28, label=f"{label} data")
        physics = physics_collection.get("physics_by_label", {}).get(label)
        if physics and physics.get("linewidth_fit") and physics["linewidth_fit"]["success"]:
            axes[2].plot(physics["linewidth_fit"]["x"], physics["linewidth_fit"]["fitted_y"], color=series_colors.get(label, "#047857"), linewidth=1.3, linestyle="--", label=f"{label} linewidth")
    axes[2].set_title("Linewidth vs Frequency")
    axes[2].set_xlabel("Frequency (GHz)")
    axes[2].set_ylabel("DeltaH (mT)")
    axes[2].grid(alpha=0.2)
    if collection.get("series_by_label"):
        axes[2].legend(loc="best")
    axes[2].text(
        0.02,
        0.98,
        _build_metric_summary(payload),
        transform=axes[2].transAxes,
        va="top",
        ha="left",
        fontsize=8,
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "0.75", "boxstyle": "round,pad=0.35"},
    )

    figure.savefig(destination, dpi=200)
    plt.close(figure)
    return destination


def export_fmr_trace_diagnostic_figures(
    result: MeasurementAnalysisResult | dict[str, Any],
    destination_dir: Path,
) -> dict[str, Path]:
    """Save one per-trace diagnostic PNG for every fitted FMR trace."""

    payload = _normalize_result(result)
    destination_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}
    for fit in payload["analysis_payload"]["trace_fit_results"]:
        trace_id = fit["trace_id"]
        safe_trace_id = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in trace_id)
        output_path = destination_dir / f"{safe_trace_id}.png"
        _export_single_trace_diagnostic(payload, fit, output_path)
        exported[trace_id] = output_path
    return exported


def export_fmr_bundle_from_json(analysis_json_path: Path, output_dir: Path | None = None) -> dict[str, Path]:
    """Regenerate FMR CSV and figure exports from saved JSON."""

    payload = load_fmr_analysis_json(analysis_json_path)
    destination_dir = output_dir.resolve() if output_dir is not None else analysis_json_path.resolve().parent
    destination_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(payload["measurement"]["source_path"]).stem
    csv_path = destination_dir / f"{stem}_trace.csv"
    summary_path = destination_dir / f"{stem}_summary.csv"
    figure_path = destination_dir / f"{stem}_figure.png"
    series_path = destination_dir / f"{stem}_series.csv"
    diagnostics_dir = destination_dir / "trace_diagnostics"
    export_fmr_analysis_csv(payload, csv_path)
    export_fmr_summary_csv(payload, summary_path)
    export_fmr_series_csv(payload, series_path)
    export_fmr_analysis_figure(payload, figure_path)
    export_fmr_trace_diagnostic_figures(payload, diagnostics_dir)
    return {
        "json_path": analysis_json_path.resolve(),
        "csv_path": csv_path,
        "summary_csv_path": summary_path,
        "figure_path": figure_path,
        "series_csv_path": series_path,
        "trace_diagnostics_dir": diagnostics_dir,
    }


def load_fmr_analysis_json(path: Path) -> dict[str, Any]:
    """Load a saved FMR analysis JSON payload."""

    return json.loads(path.read_text(encoding="utf-8"))


def build_fmr_report(input_path: Path, output_path: Path | None = None, *, recursive: bool = True) -> Path:
    """Generate a Markdown report from one or many saved FMR JSON analyses."""

    resolved_input = input_path.resolve()
    if resolved_input.is_file():
        payload = load_fmr_analysis_json(resolved_input)
        destination = output_path.resolve() if output_path is not None else resolved_input.with_name(
            f"{resolved_input.stem.replace('_analysis', '')}_report.md"
        )
        destination.write_text(_build_single_report_text(payload), encoding="utf-8")
        return destination

    if not resolved_input.is_dir():
        raise WorkflowError(f"Report input is neither a file nor directory: {resolved_input}")

    json_paths = sorted(
        resolved_input.rglob("*_analysis.json") if recursive else resolved_input.glob("*_analysis.json"),
        key=lambda path: str(path).lower(),
    )
    if not json_paths:
        raise WorkflowError(f"No analysis JSON files found under {resolved_input}")

    payloads = [load_fmr_analysis_json(path) for path in json_paths]
    destination = output_path.resolve() if output_path is not None else resolved_input / "batch_report.md"
    destination.write_text(_build_batch_report_text(payloads), encoding="utf-8")
    return destination


def aggregate_fmr_payloads(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group many saved FMR analyses into aggregated series/physics summaries."""

    grouped: dict[str, list[FmrTraceFitResult]] = {}
    group_metadata: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        summary = payload["summary_metrics"]
        group_key = _group_key(
            summary.get("sample_id"),
            summary.get("angle_deg"),
            summary.get("temperature_K"),
            summary.get("measurement_mode"),
        )
        grouped.setdefault(group_key, [])
        group_metadata[group_key] = {
            "sample_id": summary.get("sample_id"),
            "angle_deg": summary.get("angle_deg"),
            "temperature_K": summary.get("temperature_K"),
            "measurement_mode": summary.get("measurement_mode"),
        }
        for trace_fit_payload in payload["analysis_payload"]["trace_fit_results"]:
            grouped[group_key].append(_trace_fit_from_payload(trace_fit_payload))

    aggregate_results: list[dict[str, Any]] = []
    for group_key, trace_fits in sorted(grouped.items()):
        recipe = _recipe_from_payload(payloads[0])
        series_collection = build_fmr_series(
            trace_fits,
            measurement_mode=group_metadata[group_key]["measurement_mode"],
        )
        physics_collection = fit_fmr_physics(series_collection, recipe)
        aggregate_results.append(
            {
                "group_key": group_key,
                "group_metadata": group_metadata[group_key],
                "series_collection_result": to_serializable(series_collection),
                "physics_collection_result": to_serializable(physics_collection),
            }
        )
    return aggregate_results


def _normalize_result(result: MeasurementAnalysisResult | dict[str, Any]) -> dict[str, Any]:
    if isinstance(result, MeasurementAnalysisResult):
        return result.to_dict()
    return result


def _trace_lookup(payload: dict[str, Any], trace_id: str) -> dict[str, Any]:
    for trace in payload["analysis_payload"]["standardized_traces"]:
        if trace["trace_id"] == trace_id:
            return trace
    raise KeyError(trace_id)


def _processed_trace_lookup(payload: dict[str, Any], trace_id: str) -> dict[str, Any]:
    for trace in payload["analysis_payload"]["processed_traces"]:
        if trace["trace_id"] == trace_id:
            return trace
    raise KeyError(trace_id)


def _format_optional_signal(signal: list[float] | None, index: int) -> str:
    if signal is None:
        return ""
    return _format_value(signal[index])


def _diagnostic_stderr(diagnostics: dict[str, Any], parameter_name: str) -> Any:
    item = diagnostics.get(parameter_name)
    if item is None:
        return None
    return item.get("stderr")


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(numeric):
        return ""
    return f"{numeric:.10g}"


def _recipe_from_payload(payload: dict[str, Any]) -> FmrRecipe:
    config = payload["provenance"]["recipe_config"]
    return FmrRecipe(**config)


def _legacy_series_pair(series_collection: FmrSeriesCollectionResult, physics_collection: FmrPhysicsCollectionResult) -> tuple[FmrSeriesResult | None, FmrPhysicsResult | None]:
    if len(series_collection.series_by_label) != 1:
        return None, None
    label = next(iter(series_collection.series_by_label))
    return series_collection.series_by_label[label], physics_collection.physics_by_label.get(label)


def _trace_fit_from_payload(payload: dict[str, Any]) -> FmrTraceFitResult:
    from labsuite.core.types import ConvergenceSummary, ParameterDiagnostic, ResidualSummary

    return FmrTraceFitResult(
        trace_id=payload["trace_id"],
        source_file=Path(payload["source_file"]),
        sample_name=payload["sample_name"],
        frequency_GHz=float(payload["frequency_GHz"]),
        angle_deg=payload["angle_deg"],
        temperature_K=payload["temperature_K"],
        model_name=payload["model_name"],
        signal_channel=payload["signal_channel"],
        field_mT=np.asarray(payload["field_mT"], dtype=float),
        processed_signal=np.asarray(payload["processed_signal"], dtype=float),
        fitted_signal=np.asarray(payload["fitted_signal"], dtype=float),
        residual=np.asarray(payload["residual"], dtype=float),
        parameters={key: float(value) for key, value in payload["parameters"].items()},
        parameter_diagnostics={
            key: ParameterDiagnostic(**value)
            for key, value in payload["parameter_diagnostics"].items()
        },
        convergence=ConvergenceSummary(**payload["convergence"]),
        residual_summary=ResidualSummary(**payload["residual_summary"]),
        metrics=payload["metrics"],
        bound_hits=payload["bound_hits"],
        covariance=payload.get("covariance"),
        success=payload["success"],
        accepted=payload["accepted"],
        rejection_reason=payload["rejection_reason"],
        requested_mode=payload.get("requested_mode", "auto"),
        selected_mode=payload.get("selected_mode", "single"),
        selection_reason=payload.get("selection_reason", ""),
        candidate_window_count=payload.get("candidate_window_count", 0),
        double_fit_improvement_ratio=payload.get("double_fit_improvement_ratio"),
        double_fit_threshold=payload.get("double_fit_threshold"),
        candidate_windows=[FmrCandidateWindow(**item) for item in payload.get("candidate_windows", [])],
        single_fit=_trace_model_from_payload(payload.get("single_fit")),
        double_fit=_trace_model_from_payload(payload.get("double_fit")),
        selected_components=[_component_from_payload(item) for item in payload.get("selected_components", [])],
        partial_component_qc=payload.get("partial_component_qc", False),
        signal_max_abs=payload.get("signal_max_abs"),
        residual_rmse_fraction=payload.get("residual_rmse_fraction"),
        amplitude_snr=payload.get("amplitude_snr"),
        feature_center_mT=payload.get("feature_center_mT"),
        feature_peak_to_peak_mT=payload.get("feature_peak_to_peak_mT"),
        center_feature_disagreement_mT=payload.get("center_feature_disagreement_mT"),
        critical_bound_hit_names=payload.get("critical_bound_hit_names", []),
        acceptance_checks=payload.get("acceptance_checks", {}),
        warnings=payload.get("warnings", []),
        preprocessing_steps=payload.get("preprocessing_steps", []),
        baseline_summary=payload.get("baseline_summary"),
        metadata=payload.get("metadata", {}),
    )


def _trace_model_from_payload(payload: dict[str, Any] | None) -> FmrTraceModelResult | None:
    if payload is None:
        return None
    from labsuite.core.types import ConvergenceSummary, ParameterDiagnostic, ResidualSummary

    return FmrTraceModelResult(
        model_name=payload["model_name"],
        success=payload["success"],
        parameters={key: float(value) for key, value in payload.get("parameters", {}).items()},
        parameter_diagnostics={key: ParameterDiagnostic(**value) for key, value in payload.get("parameter_diagnostics", {}).items()},
        convergence=ConvergenceSummary(**payload["convergence"]),
        residual_summary=ResidualSummary(**payload["residual_summary"]),
        metrics=payload["metrics"],
        bound_hits=payload["bound_hits"],
        covariance=payload.get("covariance"),
        fitted_signal=np.asarray(payload["fitted_signal"], dtype=float),
        residual=np.asarray(payload["residual"], dtype=float),
        components=[_component_from_payload(item) for item in payload.get("components", [])],
        warnings=payload.get("warnings", []),
    )


def _component_from_payload(payload: dict[str, Any]) -> FmrComponentFitResult:
    from labsuite.core.types import ParameterDiagnostic

    return FmrComponentFitResult(
        component_id=payload.get("component_id", ""),
        component_label=payload["component_label"],
        H_res_mT=float(payload["H_res_mT"]),
        DeltaH_mT=float(payload["DeltaH_mT"]),
        amplitude_symmetric=float(payload["amplitude_symmetric"]),
        amplitude_antisymmetric=float(payload["amplitude_antisymmetric"]),
        field_mT=np.asarray(payload["field_mT"], dtype=float),
        component_signal=np.asarray(payload["component_signal"], dtype=float),
        parameter_diagnostics={key: ParameterDiagnostic(**value) for key, value in payload.get("parameter_diagnostics", {}).items()},
        bound_hits=payload.get("bound_hits", {}),
        accepted=payload.get("accepted", False),
        rejection_reason=payload.get("rejection_reason"),
        signal_max_abs=payload.get("signal_max_abs"),
        residual_rmse_fraction=payload.get("residual_rmse_fraction"),
        amplitude_snr=payload.get("amplitude_snr"),
        feature_center_mT=payload.get("feature_center_mT"),
        feature_peak_to_peak_mT=payload.get("feature_peak_to_peak_mT"),
        center_feature_disagreement_mT=payload.get("center_feature_disagreement_mT"),
        critical_bound_hit_names=payload.get("critical_bound_hit_names", []),
        acceptance_checks=payload.get("acceptance_checks", {}),
        warnings=payload.get("warnings", []),
        metadata=payload.get("metadata", {}),
    )


def _build_metric_summary(payload: dict[str, Any]) -> str:
    summary = payload["summary_metrics"]
    lines = [
        f"accepted = {summary.get('accepted_trace_count')}/{summary.get('trace_count')}",
        f"components = {summary.get('accepted_component_count')}",
        f"multi_freq = {summary.get('has_multiple_frequencies')}",
        f"series = {','.join(summary.get('series_labels', []))}",
        f"kittel = {summary.get('kittel_success')}",
        f"linewidth = {summary.get('linewidth_success')}",
        f"gamma = {_format_value(summary.get('gamma_GHz_per_T'))} GHz/T",
        f"g = {_format_value(summary.get('g'))}",
        f"M_eff = {_format_value(summary.get('M_eff_mT'))} mT",
        f"alpha = {_format_value(summary.get('alpha'))}",
        f"DeltaH0 = {_format_value(summary.get('DeltaH0_mT'))} mT",
    ]
    return "\n".join(lines)


def _build_single_report_text(payload: dict[str, Any]) -> str:
    summary = payload["summary_metrics"]
    physics_collection = payload["analysis_payload"]["physics_collection_result"]
    series_collection = payload["analysis_payload"]["series_collection_result"]
    rejected_count = int(summary.get("excluded_trace_count") or 0)
    diagnostics_dir = payload.get("artifacts", {}).get("trace_diagnostics_dir")
    rejection_lines = _format_rejection_histogram(_count_rejection_reasons(payload))
    lines = [
        f"# FMR Report: {summary['sample_id']}",
        "",
        f"- Source: `{payload['measurement']['source_path']}`",
        f"- Replicate: `{summary.get('replicate_id')}`",
        f"- Angle: `{summary.get('angle_deg')}` deg",
        f"- Temperature: `{summary.get('temperature_K')}` K",
        f"- Measurement mode: `{summary.get('measurement_mode')}`",
        f"- Accepted traces: `{summary.get('accepted_trace_count')}` / `{summary.get('trace_count')}`",
        f"- Accepted components: `{summary.get('accepted_component_count')}`",
        f"- Rejected traces: `{rejected_count}`",
        f"- Multi-frequency file: `{summary.get('has_multiple_frequencies')}`",
        f"- Diagnostics folder: `{diagnostics_dir}`",
        "",
        "## Fit Modes",
        "",
        f"- single: `{summary.get('mode_counts', {}).get('single', 0)}`",
        f"- double: `{summary.get('mode_counts', {}).get('double', 0)}`",
        f"- partial_double: `{summary.get('mode_counts', {}).get('partial_double', 0)}`",
    ]
    lines.extend(["", "## Series Buckets", ""])
    for label, series in series_collection.get("series_by_label", {}).items():
        physics = physics_collection.get("physics_by_label", {}).get(label, {})
        lines.extend(
            [
                f"- `{label}` traces: `{len(series.get('included_trace_ids', []))}`",
                f"- `{label}` components: `{len(series.get('included_component_ids', []))}`",
                f"- `{label}` Kittel: `{physics.get('kittel_fit', {}).get('success') if physics else False}`",
                f"- `{label}` linewidth: `{physics.get('linewidth_fit', {}).get('success') if physics else False}`",
            ]
        )
    lines.extend(["", "## Rejection Reasons", ""])
    lines.extend(rejection_lines)
    if physics_collection.get("warnings"):
        lines.extend(["", "## Physics Warnings", ""])
        lines.extend(f"- `{warning}`" for warning in physics_collection["warnings"])
    if summary.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- `{warning}`" for warning in summary["warnings"])
    return "\n".join(lines) + "\n"


def _build_batch_report_text(payloads: list[dict[str, Any]]) -> str:
    aggregates = aggregate_fmr_payloads(payloads)
    total_rejection_counts = Counter()
    for payload in payloads:
        total_rejection_counts.update(_count_rejection_reasons(payload))
    lines = [
        "# FMR Batch Report",
        "",
        f"- Measurements: `{len(payloads)}`",
        f"- Aggregate groups: `{len(aggregates)}`",
        "",
        "## Rejection Reasons",
        "",
        *_format_rejection_histogram(total_rejection_counts),
        "",
    ]
    for item in aggregates:
        metadata = item["group_metadata"]
        series_collection = item["series_collection_result"]
        physics_collection = item["physics_collection_result"]
        lines.extend(
            [
                f"## {metadata['sample_id']}",
                "",
                f"- Group key: `{item['group_key']}`",
                f"- Angle: `{metadata.get('angle_deg')}` deg",
                f"- Temperature: `{metadata.get('temperature_K')}` K",
                f"- Measurement mode: `{metadata.get('measurement_mode')}`",
                f"- Series labels: `{','.join(sorted(series_collection.get('series_by_label', {})))}`",
                "",
            ]
        )
        for label, series in series_collection.get("series_by_label", {}).items():
            physics = physics_collection.get("physics_by_label", {}).get(label, {})
            lines.extend(
                [
                    f"- `{label}` accepted components: `{len(series.get('included_component_ids', []))}`",
                    f"- `{label}` Kittel: `{physics.get('kittel_fit', {}).get('success') if physics else False}`",
                    f"- `{label}` linewidth: `{physics.get('linewidth_fit', {}).get('success') if physics else False}`",
                ]
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _export_single_trace_diagnostic(payload: dict[str, Any], fit: dict[str, Any], destination: Path) -> None:
    raw_trace = _trace_lookup(payload, fit["trace_id"])
    processed_trace = _processed_trace_lookup(payload, fit["trace_id"])
    field = np.asarray(processed_trace["field_mT"], dtype=float)
    raw_signal = np.asarray(raw_trace["signal"], dtype=float)
    processed_signal = np.asarray(processed_trace["signal"], dtype=float)
    fitted_signal = np.asarray(fit["fitted_signal"], dtype=float)
    residual = np.asarray(fit["residual"], dtype=float)

    figure, axes = plt.subplots(2, 1, figsize=(10.0, 7.0), constrained_layout=True, sharex=True)
    axes[0].plot(field, raw_signal, color="#94a3b8", linewidth=1.0, label="Raw")
    axes[0].plot(field, processed_signal, color="#2563eb", linewidth=1.2, label="Processed")
    axes[0].plot(field, fitted_signal, color="#dc2626", linewidth=1.2, linestyle="--", label="Fit")
    for window in fit.get("candidate_windows", []):
        axes[0].axvspan(float(window["start_field_mT"]), float(window["end_field_mT"]), color="#f59e0b", alpha=0.08)
        axes[0].axvline(float(window["candidate_center_mT"]), color="#f59e0b", linestyle=":", linewidth=0.9)
    component_colors = {"single_unassigned": "#7c3aed", "mode_1": "#047857", "mode_2": "#b91c1c"}
    for component in fit.get("selected_components", []):
        axes[0].plot(field, np.asarray(component["component_signal"], dtype=float), color=component_colors.get(component["component_label"], "#6b7280"), linewidth=1.0, alpha=0.9, label=component["component_label"])
        axes[0].axvline(float(component["H_res_mT"]), color=component_colors.get(component["component_label"], "#6b7280"), linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Signal")
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="best")
    status = "accepted" if fit.get("accepted") else "rejected"
    axes[0].set_title(f"{fit['sample_name']} | {fit['frequency_GHz']:.3g} GHz | {fit.get('selected_mode')} | {status}")

    axes[1].plot(field, residual, color="#111827", linewidth=1.0, label="Residual")
    axes[1].axhline(0.0, color="#9ca3af", linewidth=0.9, linestyle="--")
    for component in fit.get("selected_components", []):
        axes[1].axvline(float(component["H_res_mT"]), color=component_colors.get(component["component_label"], "#6b7280"), linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Field (mT)")
    axes[1].set_ylabel("Residual")
    axes[1].grid(alpha=0.2)

    warnings = fit.get("warnings", [])
    text_lines = [
        f"selected_mode = {fit.get('selected_mode')}",
        f"reason = {fit.get('selection_reason')}",
        f"accepted = {fit.get('accepted')}",
        f"trace_reject = {fit.get('rejection_reason') or 'none'}",
        f"RMSE frac = {_format_value(fit.get('residual_rmse_fraction'))}",
        f"SNR = {_format_value(fit.get('amplitude_snr'))}",
    ]
    for component in fit.get("selected_components", []):
        text_lines.append(
            f"{component['component_label']}: H={_format_value(component.get('H_res_mT'))} dH={_format_value(component.get('DeltaH_mT'))} accepted={component.get('accepted')}"
        )
    if warnings:
        text_lines.append(f"warnings = {' | '.join(str(item) for item in warnings)}")
    axes[1].text(
        0.02,
        0.98,
        "\n".join(text_lines),
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=8,
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "0.75", "boxstyle": "round,pad=0.35"},
    )

    figure.savefig(destination, dpi=200)
    plt.close(figure)


def _summarize_rejection_reasons(trace_fits: list[FmrTraceFitResult]) -> list[str]:
    counts = Counter(fit.rejection_reason for fit in trace_fits if fit.rejection_reason)
    return [f"{reason}:{counts[reason]}" for reason in sorted(counts)]


def _count_rejection_reasons(payload: dict[str, Any]) -> Counter[str]:
    return Counter(
        fit.get("rejection_reason")
        for fit in payload["analysis_payload"]["trace_fit_results"]
        if fit.get("rejection_reason")
    )


def _format_rejection_histogram(counts: Counter[str]) -> list[str]:
    if not counts:
        return ["- `none`"]
    return [f"- `{reason}`: `{counts[reason]}`" for reason in sorted(counts)]


def _format_acceptance_checks(checks: dict[str, Any]) -> str:
    if not checks:
        return ""
    return "|".join(f"{name}={'pass' if bool(value) else 'fail'}" for name, value in sorted(checks.items()))
