"""Legacy ESR batch workflow wrapper kept for CLI compatibility."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal, Sequence

from labsuite.cli.registry import summarize_esr_analysis
from labsuite.plugins.esr.serialization import export_esr_batch_overlay_figure
from labsuite.workflows.measurement_batch import (
    BatchRunResult,
    build_default_batch_output_dir as build_default_measurement_batch_output_dir,
    discover_source_files,
    run_batch_workflow,
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
) -> BatchRunResult:
    """Run the ESR workflow across every discovered source file."""

    return run_batch_workflow(
        inputs=inputs,
        recipe_path=recipe_path,
        output_dir=output_dir,
        allowed_suffixes={".dsc"},
        pattern=pattern,
        recursive=recursive,
        source_label="ESR descriptor",
        run_single_workflow=run_esr_single_file_workflow,
        summarize_analysis=summarize_esr_analysis,
        export_batch_figure=export_esr_batch_overlay_figure,
        workflow_options={
            "fit_mode": fit_mode,
            "show_raw": show_raw,
        },
    )
