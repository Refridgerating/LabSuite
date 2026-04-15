"""Folder-based ESR workflows shared by the CLI and future GUI."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Sequence

from labsuite.core.exceptions import LabSuiteError, WorkflowError
from labsuite.workflows.single_file import run_esr_single_file_workflow


@dataclass(slots=True)
class BatchItemResult:
    """Outcome for one source file in a batch run."""

    source_path: Path
    status: Literal["success", "failed"]
    error_message: str | None
    output_dir: Path
    json_path: Path | None
    csv_path: Path | None
    summary_csv_path: Path | None
    figure_path: Path | None
    selected_mode: str | None = None
    candidate_peak_count: int | None = None
    split_improvement_ratio: float | None = None
    total_area_integral: float | None = None
    local_area_integral: float | None = None
    fit_local_disagreement_ratio: float | None = None
    fit_local_disagreement_flag: bool | None = None
    fit_local_disagreement_reason: str | None = None


@dataclass(slots=True)
class BatchRunResult:
    """Batch workflow result containing per-file outcomes and aggregate artifacts."""

    discovered_sources: list[Path]
    succeeded_items: list[BatchItemResult]
    failed_items: list[BatchItemResult]
    output_dir: Path
    summary_csv_path: Path
    manifest_json_path: Path


def discover_esr_source_files(
    inputs: Sequence[Path],
    pattern: str = "*.dsc",
    recursive: bool = False,
) -> list[Path]:
    """Resolve file and folder inputs into a deterministic list of ESR descriptor paths."""

    if not inputs:
        raise WorkflowError("At least one input path is required.")

    discovered: set[Path] = set()
    for raw_input in inputs:
        resolved_input = raw_input.resolve()
        if not resolved_input.exists():
            raise WorkflowError(f"Input path does not exist: {resolved_input}")
        if resolved_input.is_file():
            if resolved_input.suffix.lower() != ".dsc":
                raise WorkflowError(
                    f"Direct file input must be a Bruker descriptor with .dsc suffix: {resolved_input.name}"
                )
            discovered.add(resolved_input)
            continue
        if not resolved_input.is_dir():
            raise WorkflowError(f"Input path is neither a file nor a directory: {resolved_input}")

        iterator = resolved_input.rglob(pattern) if recursive else resolved_input.glob(pattern)
        for candidate in iterator:
            if candidate.is_file() and candidate.suffix.lower() == ".dsc":
                discovered.add(candidate.resolve())

    sources = sorted(discovered, key=lambda path: str(path).lower())
    if not sources:
        raise WorkflowError("No ESR descriptor files were discovered for the provided inputs.")
    return sources


def build_default_batch_output_dir(
    output_root: Path,
    primary_input: Path,
    run_timestamp: datetime | None = None,
) -> Path:
    """Construct the default batch output directory name."""

    timestamp = (run_timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    input_name = primary_input.stem if primary_input.is_file() else primary_input.name
    return output_root / f"{input_name}_{timestamp}"


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

    resolved_inputs = [path.resolve() for path in inputs]
    discovered_sources = discover_esr_source_files(resolved_inputs, pattern=pattern, recursive=recursive)
    output_dir.mkdir(parents=True, exist_ok=True)

    succeeded_items: list[BatchItemResult] = []
    failed_items: list[BatchItemResult] = []
    item_output_dirs = _build_item_output_dirs(output_dir, discovered_sources)
    resolved_recipe = recipe_path.resolve()

    for source_path in discovered_sources:
        item_output_dir = item_output_dirs[source_path]
        try:
            analysis, artifacts = run_esr_single_file_workflow(
                source_path=source_path,
                recipe_path=resolved_recipe,
                output_dir=item_output_dir,
                fit_mode=fit_mode,
                show_raw=show_raw,
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
                selected_mode=analysis.selected_mode,
                candidate_peak_count=analysis.fit_decision.candidate_peak_count,
                split_improvement_ratio=analysis.fit_decision.split_improvement_ratio,
                total_area_integral=analysis.total_integral.area_integral,
                local_area_integral=analysis.local_total_integral.area_integral,
                fit_local_disagreement_ratio=analysis.fit_local_disagreement_ratio,
                fit_local_disagreement_flag=analysis.fit_local_disagreement_flag,
                fit_local_disagreement_reason=analysis.fit_local_disagreement_reason,
            )
        )

    summary_csv_path = output_dir / "batch_summary.csv"
    manifest_json_path = output_dir / "batch_manifest.json"
    all_items = [*succeeded_items, *failed_items]
    all_items.sort(key=lambda item: str(item.source_path).lower())
    _write_batch_summary_csv(all_items, summary_csv_path)
    _write_batch_manifest_json(
        inputs=resolved_inputs,
        pattern=pattern,
        recursive=recursive,
        output_dir=output_dir,
        items=all_items,
        destination=manifest_json_path,
    )

    return BatchRunResult(
        discovered_sources=discovered_sources,
        succeeded_items=succeeded_items,
        failed_items=failed_items,
        output_dir=output_dir,
        summary_csv_path=summary_csv_path,
        manifest_json_path=manifest_json_path,
    )


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
    fieldnames = [
        "source_file",
        "source_stem",
        "status",
        "error_message",
        "selected_mode",
        "candidate_peak_count",
        "split_improvement_ratio",
        "total_area_integral",
        "local_area_integral",
        "fit_local_disagreement_ratio",
        "fit_local_disagreement_flag",
        "fit_local_disagreement_reason",
        "output_dir",
        "analysis_json",
        "summary_csv",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "source_file": str(item.source_path),
                    "source_stem": item.source_path.stem,
                    "status": item.status,
                    "error_message": item.error_message,
                    "selected_mode": item.selected_mode,
                    "candidate_peak_count": item.candidate_peak_count,
                    "split_improvement_ratio": item.split_improvement_ratio,
                    "total_area_integral": item.total_area_integral,
                    "local_area_integral": item.local_area_integral,
                    "fit_local_disagreement_ratio": item.fit_local_disagreement_ratio,
                    "fit_local_disagreement_flag": item.fit_local_disagreement_flag,
                    "fit_local_disagreement_reason": item.fit_local_disagreement_reason,
                    "output_dir": str(item.output_dir),
                    "analysis_json": "" if item.json_path is None else str(item.json_path),
                    "summary_csv": "" if item.summary_csv_path is None else str(item.summary_csv_path),
                }
            )


def _write_batch_manifest_json(
    *,
    inputs: Sequence[Path],
    pattern: str,
    recursive: bool,
    output_dir: Path,
    items: Sequence[BatchItemResult],
    destination: Path,
) -> None:
    payload = {
        "inputs": [str(path) for path in inputs],
        "scan": {
            "pattern": pattern,
            "recursive": recursive,
        },
        "output_dir": str(output_dir),
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
                "selected_mode": item.selected_mode,
                "candidate_peak_count": item.candidate_peak_count,
                "split_improvement_ratio": item.split_improvement_ratio,
                "total_area_integral": item.total_area_integral,
                "local_area_integral": item.local_area_integral,
                "fit_local_disagreement_ratio": item.fit_local_disagreement_ratio,
                "fit_local_disagreement_flag": item.fit_local_disagreement_flag,
                "fit_local_disagreement_reason": item.fit_local_disagreement_reason,
            }
            for item in items
        ],
        "succeeded_count": sum(1 for item in items if item.status == "success"),
        "failed_count": sum(1 for item in items if item.status == "failed"),
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
