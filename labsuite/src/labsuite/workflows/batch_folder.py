"""Legacy ESR batch workflow wrapper kept for CLI compatibility."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal, Sequence

from labsuite.core.recipes import load_esr_recipe
from labsuite.core.resonance_metrics import ResonanceMetricsConfig
from labsuite.core.sample_registry import RegistryWorkflowOptions
from labsuite.plugins.esr.batch_qc import (
    EsrBatchQcRecord,
    compute_esr_qc_metrics,
    export_esr_batch_qc_csv,
    select_best_runs,
)
from labsuite.plugins.esr.serialization import (
    export_esr_batch_overlay_figure,
    load_esr_analysis_result,
)
from labsuite.workflows.measurement_batch import (
    BatchItemResult,
    BatchRunResult,
    build_default_batch_output_dir as build_default_measurement_batch_output_dir,
    discover_source_files,
    run_batch_workflow,
    write_batch_outputs,
)
from labsuite.workflows.single_file import run_esr_single_file_workflow


def discover_esr_source_files(
    inputs: Sequence[Path],
    pattern: str = "*.dsc",
    recursive: bool = False,
) -> list[Path]:
    """Resolve file and folder inputs into a deterministic list of ESR descriptor paths."""

    return discover_source_files(
        inputs,
        allowed_suffixes={".dsc"},
        pattern=pattern,
        recursive=recursive,
        source_label="ESR descriptor",
    )


def build_default_batch_output_dir(
    output_root: Path,
    primary_input: Path,
    run_timestamp: datetime | None = None,
) -> Path:
    """Construct the default ESR batch output directory name."""

    return build_default_measurement_batch_output_dir(
        output_root=output_root,
        primary_input=primary_input,
        run_timestamp=run_timestamp,
    )


def run_esr_batch_workflow(
    inputs: Sequence[Path],
    recipe_path: Path,
    output_dir: Path,
    *,
    pattern: str = "*.dsc",
    recursive: bool = False,
    fit_mode: Literal["auto", "single", "split"] | None = None,
    show_raw: bool = False,
    resonance_metrics_config: ResonanceMetricsConfig | None = None,
    registry_options: RegistryWorkflowOptions | None = None,
) -> BatchRunResult:
    """Run the ESR workflow across every discovered source file."""

    batch_result = run_batch_workflow(
        inputs=inputs,
        recipe_path=recipe_path,
        output_dir=output_dir,
        allowed_suffixes={".dsc"},
        pattern=pattern,
        recursive=recursive,
        source_label="ESR descriptor",
        run_single_workflow=run_esr_single_file_workflow,
        summarize_analysis=_summarize_esr_analysis,
        export_batch_figure=None,
        workflow_options={
            "fit_mode": fit_mode,
            "show_raw": show_raw,
            "resonance_metrics_config": resonance_metrics_config,
            "registry_options": registry_options,
        },
    )

    recipe = load_esr_recipe(recipe_path.resolve())
    qc_records, successful_analyses = _build_qc_records(batch_result, recipe)
    select_best_runs(qc_records)
    qc_by_source = {record.source_path.resolve(): record for record in qc_records}
    _merge_qc_summary_metrics(batch_result.succeeded_items, qc_by_source)
    _merge_qc_summary_metrics(batch_result.failed_items, qc_by_source)

    export_esr_batch_qc_csv(qc_records, output_dir / "batch_qc.csv")
    selected_analyses = _selected_analyses(successful_analyses, qc_records)
    batch_result.batch_figure_paths = (
        export_esr_batch_overlay_figure(selected_analyses, output_dir) if selected_analyses else {}
    )

    all_items = sorted(
        [*batch_result.succeeded_items, *batch_result.failed_items, *batch_result.unresolved_items],
        key=lambda item: str(item.source_path).lower(),
    )
    batch_result.summary_csv_path, batch_result.manifest_json_path = write_batch_outputs(
        inputs=[path.resolve() for path in inputs],
        pattern=pattern,
        recursive=recursive,
        output_dir=output_dir,
        items=all_items,
        batch_figure_paths=batch_result.batch_figure_paths,
        resonance_metrics_csv_path=batch_result.resonance_metrics_csv_path,
        unresolved_csv_path=batch_result.unresolved_csv_path,
    )
    return batch_result


def _summarize_esr_analysis(analysis) -> dict[str, object]:
    """Extract ESR batch-summary fields from one analysis result."""

    return {
        "selected_mode": analysis.selected_mode,
        "candidate_peak_count": analysis.fit_decision.candidate_peak_count,
        "split_improvement_ratio": analysis.fit_decision.split_improvement_ratio,
        "total_area_integral": analysis.total_integral.area_integral,
        "local_area_integral": analysis.local_total_integral.area_integral,
        "fit_local_disagreement_ratio": analysis.fit_local_disagreement_ratio,
        "fit_local_disagreement_flag": analysis.fit_local_disagreement_flag,
        "fit_local_disagreement_reason": analysis.fit_local_disagreement_reason,
        "resonance_metrics_mode_count": len(getattr(analysis, "resonance_metrics", [])),
        "resonance_metrics_failure_count": sum(1 for item in getattr(analysis, "resonance_metrics", []) if not item.success),
    }


def _build_qc_records(
    batch_result: BatchRunResult,
    recipe,
) -> tuple[list[EsrBatchQcRecord], dict[Path, object]]:
    successful_analyses: dict[Path, object] = {}
    records: list[EsrBatchQcRecord] = []

    for item in batch_result.succeeded_items:
        if item.json_path is None:
            records.append(
                compute_esr_qc_metrics(
                    None,
                    source_path=item.source_path,
                    recipe=recipe,
                    error_message="missing_analysis_json",
                )
            )
            continue
        analysis = load_esr_analysis_result(item.json_path)
        successful_analyses[item.source_path.resolve()] = analysis
        records.append(
            compute_esr_qc_metrics(
                analysis,
                source_path=item.source_path,
                recipe=recipe,
            )
        )

    for item in batch_result.failed_items:
        records.append(
            compute_esr_qc_metrics(
                None,
                source_path=item.source_path,
                recipe=recipe,
                error_message=item.error_message,
            )
        )

    return records, successful_analyses


def _selected_analyses(
    successful_analyses: dict[Path, object],
    qc_records: Sequence[EsrBatchQcRecord],
) -> list[object]:
    return [
        successful_analyses[record.source_path.resolve()]
        for record in qc_records
        if record.accepted_for_plot and record.selected_as_best and record.source_path.resolve() in successful_analyses
    ]


def _merge_qc_summary_metrics(
    items: Sequence[BatchItemResult],
    qc_by_source: dict[Path, EsrBatchQcRecord],
) -> None:
    for item in items:
        record = qc_by_source.get(item.source_path.resolve())
        if record is None:
            continue
        item.summary_metrics.update(record.summary_metrics())
