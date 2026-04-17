"""Generic measurement workflow helpers shared by modality CLIs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from labsuite.core.exceptions import LabSuiteError, WorkflowError
from labsuite.workflows.single_file import WorkflowArtifacts


@dataclass(slots=True)
class BatchItemResult:
    """Outcome for one source file in a batch run."""

    source_path: Path
    status: str
    error_message: str | None
    output_dir: Path
    json_path: Path | None
    csv_path: Path | None
    summary_csv_path: Path | None
    figure_path: Path | None
    report_path: Path | None = None
    summary_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BatchRunResult:
    """Generic batch workflow result."""

    discovered_sources: list[Path]
    succeeded_items: list[BatchItemResult]
    failed_items: list[BatchItemResult]
    output_dir: Path
    summary_csv_path: Path
    manifest_json_path: Path
    batch_figure_paths: dict[str, Path] = field(default_factory=dict)


def discover_source_files(
    inputs: Sequence[Path],
    *,
    allowed_suffixes: set[str],
    pattern: str,
    recursive: bool,
    source_label: str,
) -> list[Path]:
    """Resolve file and folder inputs into a deterministic list of source files."""

    if not inputs:
        raise WorkflowError("At least one input path is required.")

    discovered: set[Path] = set()
    for raw_input in inputs:
        resolved_input = raw_input.resolve()
        if not resolved_input.exists():
            raise WorkflowError(f"Input path does not exist: {resolved_input}")
        if resolved_input.is_file():
            if resolved_input.suffix.lower() not in allowed_suffixes:
                suffix_list = ", ".join(sorted(allowed_suffixes))
                raise WorkflowError(
                    f"Direct file input must be a {source_label} source with one of suffixes {suffix_list}: {resolved_input.name}"
                )
            discovered.add(resolved_input)
            continue
        if not resolved_input.is_dir():
            raise WorkflowError(f"Input path is neither a file nor a directory: {resolved_input}")

        iterator = resolved_input.rglob(pattern) if recursive else resolved_input.glob(pattern)
        for candidate in iterator:
            if candidate.is_file() and candidate.suffix.lower() in allowed_suffixes:
                discovered.add(candidate.resolve())

    sources = sorted(discovered, key=lambda path: str(path).lower())
    if not sources:
        raise WorkflowError(f"No {source_label} source files were discovered for the provided inputs.")
    return sources


def resolve_single_source(
    input_path: Path,
    *,
    allowed_suffixes: set[str],
    pattern: str,
    recursive: bool,
    source_label: str,
) -> Path:
    """Resolve a single source file from either a file or a folder input."""

    discovered_sources = discover_source_files(
        [input_path.resolve()],
        allowed_suffixes=allowed_suffixes,
        pattern=pattern,
        recursive=recursive,
        source_label=source_label,
    )
    if len(discovered_sources) != 1:
        raise WorkflowError(
            f"single requires exactly one discovered source file, found {len(discovered_sources)}"
        )
    return discovered_sources[0]


def build_default_batch_output_dir(
    output_root: Path,
    primary_input: Path,
    run_timestamp: datetime | None = None,
) -> Path:
    """Construct the default batch output directory name."""

    timestamp = (run_timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    input_name = primary_input.stem if primary_input.is_file() else primary_input.name
    return output_root / f"{input_name}_{timestamp}"


def run_batch_workflow(
    *,
    inputs: Sequence[Path],
    recipe_path: Path,
    output_dir: Path,
    allowed_suffixes: set[str],
    pattern: str,
    recursive: bool,
    source_label: str,
    run_single_workflow: Callable[..., tuple[Any, WorkflowArtifacts]],
    summarize_analysis: Callable[[Any], dict[str, Any]],
    export_batch_figure: Callable[[Sequence[Any], Path], dict[str, Path]] | None = None,
    workflow_options: dict[str, Any] | None = None,
) -> BatchRunResult:
    """Run a modality-specific single-file workflow across all discovered sources."""

    resolved_inputs = [path.resolve() for path in inputs]
    discovered_sources = discover_source_files(
        resolved_inputs,
        allowed_suffixes=allowed_suffixes,
        pattern=pattern,
        recursive=recursive,
        source_label=source_label,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    succeeded_items: list[BatchItemResult] = []
    failed_items: list[BatchItemResult] = []
    successful_analyses: list[Any] = []
    item_output_dirs = _build_item_output_dirs(output_dir, discovered_sources)
    resolved_recipe = recipe_path.resolve()
    options = workflow_options or {}

    for source_path in discovered_sources:
        item_output_dir = item_output_dirs[source_path]
        try:
            analysis, artifacts = run_single_workflow(
                source_path=source_path,
                recipe_path=resolved_recipe,
                output_dir=item_output_dir,
                **options,
            )
        except LabSuiteError as exc:
            failed_items.append(
                BatchItemResult(
                    source_path=source_path,
                    status="failed",
                    error_message=str(exc),
                    output_dir=item_output_dir,
                    json_path=None,
                    csv_path=None,
                    summary_csv_path=None,
                    figure_path=None,
                )
            )
            continue

        succeeded_items.append(
            BatchItemResult(
                source_path=source_path,
                status="success",
                error_message=None,
                output_dir=item_output_dir,
                json_path=artifacts.json_path,
                csv_path=artifacts.csv_path,
                summary_csv_path=artifacts.summary_csv_path,
                figure_path=artifacts.figure_path,
                summary_metrics=summarize_analysis(analysis),
            )
        )
        successful_analyses.append(analysis)

    batch_figure_paths = {} if export_batch_figure is None else export_batch_figure(successful_analyses, output_dir)
    all_items = sorted([*succeeded_items, *failed_items], key=lambda item: str(item.source_path).lower())
    summary_csv_path, manifest_json_path = write_batch_outputs(
        inputs=resolved_inputs,
        pattern=pattern,
        recursive=recursive,
        output_dir=output_dir,
        items=all_items,
        batch_figure_paths=batch_figure_paths,
    )
    return BatchRunResult(
        discovered_sources=discovered_sources,
        succeeded_items=succeeded_items,
        failed_items=failed_items,
        output_dir=output_dir,
        summary_csv_path=summary_csv_path,
        manifest_json_path=manifest_json_path,
        batch_figure_paths=batch_figure_paths,
    )


def write_batch_outputs(
    *,
    inputs: Sequence[Path],
    pattern: str,
    recursive: bool,
    output_dir: Path,
    items: Sequence[BatchItemResult],
    batch_figure_paths: dict[str, Path],
) -> tuple[Path, Path]:
    """Write batch summary and manifest artifacts for a completed run."""

    summary_csv_path = output_dir / "batch_summary.csv"
    manifest_json_path = output_dir / "batch_manifest.json"
    _write_batch_summary_csv(items, summary_csv_path)
    _write_batch_manifest_json(
        inputs=inputs,
        pattern=pattern,
        recursive=recursive,
        output_dir=output_dir,
        items=items,
        batch_figure_paths=batch_figure_paths,
        destination=manifest_json_path,
    )
    return summary_csv_path, manifest_json_path


def _build_item_output_dirs(output_dir: Path, sources: Sequence[Path]) -> dict[Path, Path]:
    counts: dict[str, int] = {}
    directories: dict[Path, Path] = {}
    for source_path in sources:
        stem = source_path.stem
        counts[stem] = counts.get(stem, 0) + 1
        suffix = "" if counts[stem] == 1 else f"__{counts[stem]}"
        directories[source_path] = output_dir / f"{stem}{suffix}"
    return directories


def _write_batch_summary_csv(items: Sequence[BatchItemResult], destination: Path) -> None:
    metric_fields = sorted({key for item in items for key in item.summary_metrics})
    fieldnames = [
        "source_file",
        "source_stem",
        "status",
        "error_message",
        "output_dir",
        "analysis_json",
        "trace_csv",
        "summary_csv",
        "figure_png",
        *metric_fields,
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            row = {
                "source_file": str(item.source_path),
                "source_stem": item.source_path.stem,
                "status": item.status,
                "error_message": item.error_message,
                "output_dir": str(item.output_dir),
                "analysis_json": "" if item.json_path is None else str(item.json_path),
                "trace_csv": "" if item.csv_path is None else str(item.csv_path),
                "summary_csv": "" if item.summary_csv_path is None else str(item.summary_csv_path),
                "figure_png": "" if item.figure_path is None else str(item.figure_path),
            }
            for key in metric_fields:
                value = item.summary_metrics.get(key)
                if isinstance(value, list):
                    row[key] = "|".join(str(entry) for entry in value)
                else:
                    row[key] = value
            writer.writerow(row)


def _write_batch_manifest_json(
    *,
    inputs: Sequence[Path],
    pattern: str,
    recursive: bool,
    output_dir: Path,
    items: Sequence[BatchItemResult],
    batch_figure_paths: dict[str, Path],
    destination: Path,
) -> None:
    serialized_batch_figures = {
        name: str(path)
        for name, path in sorted(batch_figure_paths.items())
    }
    batch_figure_png = None
    if len(serialized_batch_figures) == 1:
        batch_figure_png = next(iter(serialized_batch_figures.values()))
    payload = {
        "inputs": [str(path) for path in inputs],
        "scan": {
            "pattern": pattern,
            "recursive": recursive,
        },
        "output_dir": str(output_dir),
        "batch_figures": serialized_batch_figures,
        "batch_figure_png": batch_figure_png,
        "items": [
            {
                "source_file": str(item.source_path),
                "status": item.status,
                "error_message": item.error_message,
                "output_dir": str(item.output_dir),
                "analysis_json": None if item.json_path is None else str(item.json_path),
                "trace_csv": None if item.csv_path is None else str(item.csv_path),
                "summary_csv": None if item.summary_csv_path is None else str(item.summary_csv_path),
                "figure_png": None if item.figure_path is None else str(item.figure_path),
                "summary_metrics": item.summary_metrics,
            }
            for item in items
        ],
        "succeeded_count": sum(1 for item in items if item.status == "success"),
        "failed_count": sum(1 for item in items if item.status == "failed"),
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
