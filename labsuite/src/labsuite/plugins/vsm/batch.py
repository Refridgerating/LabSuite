"""VSM-specific batch diagnostics layered over the generic batch runner."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from labsuite.core.recipes import load_vsm_recipe
from labsuite.plugins.vsm.quality import weighted_ms_summary
from labsuite.plugins.vsm.service import export_vsm_batch_overlay_figure
from labsuite.workflows.measurement_batch import (
    BatchRunResult,
    run_batch_workflow,
    write_batch_outputs,
)
from labsuite.workflows.single_file import run_vsm_single_file_workflow


def run_vsm_batch_workflow(
    *,
    inputs: Sequence[Path],
    recipe_path: Path,
    output_dir: Path,
    pattern: str,
    recursive: bool,
    **workflow_options: Any,
) -> BatchRunResult:
    """Run VSM batch analysis and write subtraction-quality aggregate artifacts."""

    runner_options = dict(workflow_options)
    diagnostics_override = runner_options.pop("vsm_diagnostics_out", None)
    batch_result = run_batch_workflow(
        inputs=inputs,
        recipe_path=recipe_path,
        output_dir=output_dir,
        allowed_suffixes={".dat"},
        pattern=pattern,
        recursive=recursive,
        source_label="VSM loop",
        run_single_workflow=run_vsm_single_file_workflow,
        summarize_analysis=lambda analysis: dict(analysis.summary_metrics),
        export_batch_figure=export_vsm_batch_overlay_figure,
        workflow_options=runner_options,
    )
    recipe = load_vsm_recipe(recipe_path)
    _apply_recipe_overrides(recipe, workflow_options.get("vsm_recipe_overrides"))
    diagnostic_rows = _build_quality_rows(batch_result)
    diagnostics_path = Path(
        diagnostics_override or output_dir / "vsm_subtraction_quality.csv"
    )
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    _write_quality_csv(diagnostic_rows, diagnostics_path)
    summary_rows = _build_weighted_summary_rows(
        diagnostic_rows,
        accept_downweighted=recipe.vsm_accept_downweighted,
        min_weight=recipe.vsm_min_weight,
    )
    summary_csv_path = output_dir / "vsm_ms_weighted_summary.csv"
    summary_json_path = output_dir / "vsm_ms_weighted_summary.json"
    _write_weighted_summary_csv(summary_rows, summary_csv_path)
    summary_json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    batch_result.extra_artifact_paths.update(
        {
            "vsm_subtraction_quality": diagnostics_path,
            "vsm_ms_weighted_summary_csv": summary_csv_path,
            "vsm_ms_weighted_summary_json": summary_json_path,
        }
    )
    all_items = sorted(
        [
            *batch_result.succeeded_items,
            *batch_result.failed_items,
            *batch_result.unresolved_items,
        ],
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
        raw_import_map_path=batch_result.raw_import_map_path,
        extra_artifact_paths=batch_result.extra_artifact_paths,
    )
    return batch_result


QUALITY_FIELDNAMES = [
    "sample_id",
    "source_file",
    "chosen_method",
    "hcut_fraction",
    "ms_emu",
    "weight",
    "status",
    "reasons",
    "residual_slope_pos",
    "residual_slope_neg",
    "symmetry_error",
    "cutoff_cv",
    "tail_rmse",
    "legacy_background_mode",
    "legacy_background_correction_accepted",
    "legacy_background_decision_reason",
    "legacy_background_qc_passed",
]


SUMMARY_FIELDNAMES = [
    "sample_id",
    "item_count",
    "included_count",
    "accepted_count",
    "downweighted_count",
    "rejected_count",
    "unweighted_mean_ms_emu",
    "weighted_mean_ms_emu",
    "weighted_std_ms_emu",
]


def _build_quality_rows(batch_result: BatchRunResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in batch_result.succeeded_items:
        summary = item.summary_metrics
        rows.append(
            {
                "sample_id": summary.get("sample_id") or "batch",
                "source_file": str(item.source_path),
                "chosen_method": summary.get("background_subtraction_mode"),
                "hcut_fraction": summary.get("background_tail_fraction")
                or summary.get("vsm_quality_hcut_fraction"),
                "ms_emu": summary.get("Ms_emu"),
                "weight": summary.get("vsm_quality_weight"),
                "status": summary.get("vsm_quality_status"),
                "reasons": "|".join(
                    str(reason) for reason in summary.get("vsm_quality_reasons", [])
                ),
                "residual_slope_pos": summary.get("vsm_quality_residual_slope_pos"),
                "residual_slope_neg": summary.get("vsm_quality_residual_slope_neg"),
                "symmetry_error": summary.get("vsm_quality_symmetry_error"),
                "cutoff_cv": summary.get("vsm_quality_cutoff_cv"),
                "tail_rmse": summary.get("vsm_quality_tail_rmse"),
                "legacy_background_mode": summary.get("legacy_background_mode"),
                "legacy_background_correction_accepted": summary.get(
                    "legacy_background_correction_accepted"
                ),
                "legacy_background_decision_reason": summary.get(
                    "legacy_background_decision_reason"
                ),
                "legacy_background_qc_passed": summary.get("legacy_background_qc_passed"),
            }
        )
    return rows


def _build_weighted_summary_rows(
    diagnostic_rows: Sequence[dict[str, Any]],
    *,
    accept_downweighted: bool,
    min_weight: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in diagnostic_rows:
        grouped.setdefault(str(row.get("sample_id") or "batch"), []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for sample_id in sorted(grouped):
        summary = weighted_ms_summary(
            grouped[sample_id],
            accept_downweighted=accept_downweighted,
            min_weight=min_weight,
        )
        summary_rows.append({"sample_id": sample_id, **summary})
    return summary_rows


def _write_quality_csv(rows: Sequence[dict[str, Any]], destination: Path) -> None:
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUALITY_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in QUALITY_FIELDNAMES})


def _write_weighted_summary_csv(rows: Sequence[dict[str, Any]], destination: Path) -> None:
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in SUMMARY_FIELDNAMES})


def _apply_recipe_overrides(recipe, overrides: Any) -> None:
    if not isinstance(overrides, dict):
        return
    for key, value in overrides.items():
        if value is not None and hasattr(recipe, key):
            setattr(recipe, key, value)
