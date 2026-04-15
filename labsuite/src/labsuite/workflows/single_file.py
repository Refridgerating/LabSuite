"""Single-file workflows shared by the CLI and future GUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from labsuite.core.export import (
    export_analysis_csv,
    export_analysis_figure,
    export_analysis_json,
)
from labsuite.core.types import AnalysisResult
from labsuite.plugins.esr.service import analyze_esr_file


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
) -> tuple[AnalysisResult, WorkflowArtifacts]:
    """Execute the first ESR-only end-to-end workflow."""

    output_dir.mkdir(parents=True, exist_ok=True)
    analysis = analyze_esr_file(source_path=source_path, recipe_path=recipe_path, fit_mode=fit_mode)

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

    return analysis, artifacts
