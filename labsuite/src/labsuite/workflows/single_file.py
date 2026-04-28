"""Single-file workflows shared by the CLI and future GUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from labsuite.core.export import (
    export_analysis_csv,
    export_analysis_figure,
    export_analysis_json,
)
from labsuite.core.resonance_metrics import ResonanceMetricsConfig
from labsuite.core.sample_registry import (
    AnalysisSampleContext,
    RegistryWorkflowOptions,
    record_processed_result,
    resolve_analysis_context,
    to_serializable,
)
from labsuite.core.types import AnalysisResult
from labsuite.plugins.esr.service import analyze_esr_file
from labsuite.plugins.fmr.service import (
    analyze_fmr_file,
    export_fmr_analysis_csv,
    export_fmr_analysis_figure,
    export_fmr_analysis_json,
    export_fmr_polarity_diagnostics_figure,
    export_fmr_series_csv,
    export_fmr_summary_csv,
    export_fmr_trace_diagnostic_figures,
)
from labsuite.plugins.vsm.service import (
    analyze_vsm_file,
    export_vsm_analysis_csv,
    export_vsm_analysis_figure,
    export_vsm_analysis_json,
    export_vsm_summary_csv,
)


@dataclass(slots=True)
class WorkflowArtifacts:
    """Paths created by the single-file workflow."""

    json_path: Path
    csv_path: Path
    summary_csv_path: Path
    figure_path: Path


def run_esr_single_file_workflow(
    source_path: Path,
    recipe_path: Path,
    output_dir: Path,
    *,
    fit_mode: Literal["auto", "single", "split"] | None = None,
    show_raw: bool = False,
    resonance_metrics_config: ResonanceMetricsConfig | None = None,
    registry_options: RegistryWorkflowOptions | None = None,
) -> tuple[AnalysisResult, WorkflowArtifacts]:
    """Execute the first ESR-only end-to-end workflow."""

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_context = resolve_analysis_context(
        source_path=source_path,
        measurement_type="esr",
        options=registry_options,
    )
    analysis = analyze_esr_file(
        source_path=source_path,
        recipe_path=recipe_path,
        fit_mode=fit_mode,
        resonance_metrics_config=resonance_metrics_config,
        sample_context=sample_context,
    )

    stem = source_path.stem
    artifacts = WorkflowArtifacts(
        json_path=output_dir / f"{stem}_analysis.json",
        csv_path=output_dir / f"{stem}_trace.csv",
        summary_csv_path=output_dir / f"{stem}_summary.csv",
        figure_path=output_dir / f"{stem}_figure.png",
    )

    export_analysis_json(analysis, artifacts.json_path)
    export_analysis_csv(analysis, artifacts.csv_path, artifacts.summary_csv_path)
    export_analysis_figure(analysis, artifacts.figure_path, show=show_raw)
    record_processed_result(
        sample_context=sample_context,
        measurement_type="esr",
        processed_path=artifacts.json_path,
        recipe_path=recipe_path,
        analysis=analysis,
        options=registry_options,
    )
    _write_analysis_provenance(
        output_dir=output_dir,
        modality="esr",
        source_path=source_path,
        recipe_path=recipe_path,
        sample_context=sample_context,
        registry_options=registry_options,
        extra_config={
            "fit_mode": fit_mode,
            "resonance_metrics_config": _maybe_to_dict(resonance_metrics_config),
        },
    )

    return analysis, artifacts


def run_vsm_single_file_workflow(
    source_path: Path,
    recipe_path: Path,
    output_dir: Path,
    *,
    registry_options: RegistryWorkflowOptions | None = None,
) -> tuple[object, WorkflowArtifacts]:
    """Execute the first VSM single-file workflow."""

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_context = resolve_analysis_context(
        source_path=source_path,
        measurement_type="vsm",
        options=registry_options,
    )
    analysis = analyze_vsm_file(
        source_path=source_path, recipe_path=recipe_path, sample_context=sample_context
    )

    stem = source_path.stem
    artifacts = WorkflowArtifacts(
        json_path=output_dir / f"{stem}_analysis.json",
        csv_path=output_dir / f"{stem}_trace.csv",
        summary_csv_path=output_dir / f"{stem}_summary.csv",
        figure_path=output_dir / f"{stem}_figure.png",
    )
    analysis.artifacts.update(
        {
            "json_path": str(artifacts.json_path),
            "csv_path": str(artifacts.csv_path),
            "summary_csv_path": str(artifacts.summary_csv_path),
            "figure_path": str(artifacts.figure_path),
        }
    )

    export_vsm_analysis_json(analysis, artifacts.json_path)
    export_vsm_analysis_csv(analysis, artifacts.csv_path)
    export_vsm_summary_csv(analysis, artifacts.summary_csv_path)
    export_vsm_analysis_figure(analysis, artifacts.figure_path)
    record_processed_result(
        sample_context=sample_context,
        measurement_type="vsm",
        processed_path=artifacts.json_path,
        recipe_path=recipe_path,
        analysis=analysis,
        options=registry_options,
    )
    _write_analysis_provenance(
        output_dir=output_dir,
        modality="vsm",
        source_path=source_path,
        recipe_path=recipe_path,
        sample_context=sample_context,
        registry_options=registry_options,
    )

    return analysis, artifacts


def run_fmr_single_file_workflow(
    source_path: Path,
    recipe_path: Path,
    output_dir: Path,
    *,
    resonance_metrics_config: ResonanceMetricsConfig | None = None,
    registry_options: RegistryWorkflowOptions | None = None,
    fmr_recipe_overrides: dict[str, object] | None = None,
) -> tuple[object, WorkflowArtifacts]:
    """Execute the first FMR single-file workflow."""

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_context = resolve_analysis_context(
        source_path=source_path,
        measurement_type="fmr",
        options=registry_options,
    )
    analysis = analyze_fmr_file(
        source_path=source_path,
        recipe_path=recipe_path,
        resonance_metrics_config=resonance_metrics_config,
        sample_context=sample_context,
        recipe_overrides=fmr_recipe_overrides,
    )

    stem = source_path.stem
    artifacts = WorkflowArtifacts(
        json_path=output_dir / f"{stem}_analysis.json",
        csv_path=output_dir / f"{stem}_trace.csv",
        summary_csv_path=output_dir / f"{stem}_summary.csv",
        figure_path=output_dir / f"{stem}_figure.png",
    )
    series_csv_path = output_dir / f"{stem}_series.csv"
    polarity_diagnostics_path = output_dir / f"{stem}_polarity_diagnostics.png"
    diagnostics_dir = output_dir / "trace_diagnostics"
    diagnostic_paths = export_fmr_trace_diagnostic_figures(analysis, diagnostics_dir)
    polarity_plot_path = None
    if (
        analysis.provenance.get("recipe_config", {})
        .get("field_polarity_correction", {})
        .get("plot_diagnostics")
    ):
        polarity_plot_path = export_fmr_polarity_diagnostics_figure(
            analysis,
            polarity_diagnostics_path,
        )
    analysis.artifacts.update(
        {
            "json_path": str(artifacts.json_path),
            "csv_path": str(artifacts.csv_path),
            "summary_csv_path": str(artifacts.summary_csv_path),
            "figure_path": str(artifacts.figure_path),
            "series_csv_path": str(series_csv_path),
            "polarity_diagnostics_path": None
            if polarity_plot_path is None
            else str(polarity_plot_path),
            "trace_diagnostics_dir": str(diagnostics_dir),
            "trace_diagnostic_paths": {
                trace_id: str(path) for trace_id, path in diagnostic_paths.items()
            },
        }
    )

    export_fmr_analysis_csv(analysis, artifacts.csv_path)
    export_fmr_summary_csv(analysis, artifacts.summary_csv_path)
    export_fmr_series_csv(analysis, series_csv_path)
    export_fmr_analysis_figure(analysis, artifacts.figure_path)
    export_fmr_analysis_json(analysis, artifacts.json_path)
    record_processed_result(
        sample_context=sample_context,
        measurement_type="fmr",
        processed_path=artifacts.json_path,
        recipe_path=recipe_path,
        analysis=analysis,
        options=registry_options,
    )
    _write_analysis_provenance(
        output_dir=output_dir,
        modality="fmr",
        source_path=source_path,
        recipe_path=recipe_path,
        sample_context=sample_context,
        registry_options=registry_options,
        extra_config={
            "resonance_metrics_config": _maybe_to_dict(resonance_metrics_config),
            "fmr_recipe_overrides": fmr_recipe_overrides,
        },
    )

    return analysis, artifacts


def _write_analysis_provenance(
    *,
    output_dir: Path,
    modality: str,
    source_path: Path,
    recipe_path: Path,
    sample_context: AnalysisSampleContext | None,
    registry_options: RegistryWorkflowOptions | None,
    extra_config: dict[str, object] | None = None,
) -> None:
    config = {
        "modality": modality,
        "source_path": source_path.resolve(),
        "original_source_path": None
        if registry_options is None
        else registry_options.original_source_path,
        "recipe_path": recipe_path.resolve(),
        "registry_options": registry_options,
        "resolved_sample": None if sample_context is None else sample_context.to_dict(),
    }
    if extra_config:
        config.update(extra_config)
    (output_dir / "analysis_config.yaml").write_text(
        yaml.safe_dump(to_serializable(config), sort_keys=False),
        encoding="utf-8",
    )
    snapshot = (
        {}
        if sample_context is None or sample_context.registry_snapshot is None
        else sample_context.registry_snapshot
    )
    (output_dir / "sample_registry_snapshot.yaml").write_text(
        yaml.safe_dump(to_serializable(snapshot), sort_keys=False),
        encoding="utf-8",
    )
    measurement_snapshot = (
        {}
        if sample_context is None or sample_context.measurement_ledger_snapshot is None
        else sample_context.measurement_ledger_snapshot
    )
    (output_dir / "measurement_ledger_snapshot.yaml").write_text(
        yaml.safe_dump(to_serializable(measurement_snapshot), sort_keys=False),
        encoding="utf-8",
    )
    processed_snapshot = (
        {}
        if sample_context is None or sample_context.processed_ledger_snapshot is None
        else sample_context.processed_ledger_snapshot
    )
    (output_dir / "processed_ledger_snapshot.yaml").write_text(
        yaml.safe_dump(to_serializable(processed_snapshot), sort_keys=False),
        encoding="utf-8",
    )


def _maybe_to_dict(value: object | None) -> object | None:
    if value is None:
        return None
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return value
