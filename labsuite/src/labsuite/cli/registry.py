"""Static modality registry for the measurement CLI."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from labsuite.core.export import export_analysis_csv, export_analysis_figure
from labsuite.plugins.esr.serialization import (
    build_esr_report,
    export_esr_batch_overlay_figure,
    load_esr_analysis_result,
)
from labsuite.plugins.fmr.service import build_fmr_report, export_fmr_bundle_from_json
from labsuite.plugins.vsm.batch import run_vsm_batch_workflow
from labsuite.plugins.vsm.service import (
    build_vsm_report,
    export_vsm_batch_overlay_figure,
    export_vsm_bundle_from_json,
)
from labsuite.workflows.batch_folder import run_esr_batch_workflow
from labsuite.workflows.single_file import (
    run_esr_single_file_workflow,
    run_fmr_single_file_workflow,
    run_vsm_single_file_workflow,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed"


@dataclass(frozen=True, slots=True)
class ModalityCliSpec:
    """Static CLI behavior for one measurement modality."""

    name: str
    source_label: str
    allowed_suffixes: set[str]
    default_pattern: str
    default_recipe: Path
    run_single_workflow: Callable[..., tuple[Any, Any]]
    summarize_analysis: Callable[[Any], dict[str, Any]]
    export_from_json: Callable[[Path, Path | None], dict[str, Path]]
    build_report: Callable[[Path, Path | None, bool], Path]
    export_batch_figure: Callable[[list[Any], Path], dict[str, Path]] | None = None
    run_batch_workflow: Callable[..., Any] | None = None


def summarize_esr_analysis(analysis) -> dict[str, Any]:
    """Extract batch-summary fields from an ESR analysis result."""

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
        "resonance_metrics_failure_count": sum(
            1 for item in getattr(analysis, "resonance_metrics", []) if not item.success
        ),
    }


def summarize_vsm_analysis(analysis) -> dict[str, Any]:
    """Extract batch-summary fields from a VSM analysis result."""

    return dict(analysis.summary_metrics)


def summarize_fmr_analysis(analysis) -> dict[str, Any]:
    """Extract batch-summary fields from an FMR analysis result."""

    return dict(analysis.summary_metrics)


def export_esr_bundle_from_json(
    analysis_json_path: Path, output_dir: Path | None = None
) -> dict[str, Path]:
    """Regenerate ESR CSV and figure exports from a saved JSON result."""

    analysis = load_esr_analysis_result(analysis_json_path)
    destination_dir = (
        output_dir.resolve() if output_dir is not None else analysis_json_path.resolve().parent
    )
    destination_dir.mkdir(parents=True, exist_ok=True)
    stem = analysis.dataset.source_path.stem
    csv_path = destination_dir / f"{stem}_trace.csv"
    summary_path = destination_dir / f"{stem}_summary.csv"
    figure_path = destination_dir / f"{stem}_figure.png"
    export_analysis_csv(analysis, csv_path, summary_path)
    export_analysis_figure(analysis, figure_path)
    return {
        "json_path": analysis_json_path.resolve(),
        "csv_path": csv_path,
        "summary_csv_path": summary_path,
        "figure_path": figure_path,
    }


def build_esr_report_entry(
    input_path: Path, output_path: Path | None = None, recursive: bool = True
) -> Path:
    """Wrap ESR report generation for the registry."""

    return build_esr_report(input_path, output_path=output_path, recursive=recursive)


def build_vsm_report_entry(
    input_path: Path, output_path: Path | None = None, recursive: bool = True
) -> Path:
    """Wrap VSM report generation for the registry."""

    return build_vsm_report(input_path, output_path=output_path, recursive=recursive)


def build_fmr_report_entry(
    input_path: Path, output_path: Path | None = None, recursive: bool = True
) -> Path:
    """Wrap FMR report generation for the registry."""

    return build_fmr_report(input_path, output_path=output_path, recursive=recursive)


MODALITY_SPECS: dict[str, ModalityCliSpec] = {
    "esr": ModalityCliSpec(
        name="esr",
        source_label="ESR descriptor",
        allowed_suffixes={".dsc"},
        default_pattern="*.dsc",
        default_recipe=PROJECT_ROOT / "recipes" / "esr" / "default.yaml",
        run_single_workflow=run_esr_single_file_workflow,
        summarize_analysis=summarize_esr_analysis,
        export_from_json=export_esr_bundle_from_json,
        build_report=build_esr_report_entry,
        export_batch_figure=export_esr_batch_overlay_figure,
        run_batch_workflow=run_esr_batch_workflow,
    ),
    "vsm": ModalityCliSpec(
        name="vsm",
        source_label="VSM loop",
        allowed_suffixes={".dat"},
        default_pattern="*.dat",
        default_recipe=PROJECT_ROOT / "recipes" / "vsm" / "default.yaml",
        run_single_workflow=run_vsm_single_file_workflow,
        summarize_analysis=summarize_vsm_analysis,
        export_from_json=export_vsm_bundle_from_json,
        build_report=build_vsm_report_entry,
        export_batch_figure=export_vsm_batch_overlay_figure,
        run_batch_workflow=run_vsm_batch_workflow,
    ),
    "fmr": ModalityCliSpec(
        name="fmr",
        source_label="FMR log",
        allowed_suffixes={".log"},
        default_pattern="*.log",
        default_recipe=PROJECT_ROOT / "recipes" / "fmr" / "default.yaml",
        run_single_workflow=run_fmr_single_file_workflow,
        summarize_analysis=summarize_fmr_analysis,
        export_from_json=export_fmr_bundle_from_json,
        build_report=build_fmr_report_entry,
        export_batch_figure=None,
    ),
}
