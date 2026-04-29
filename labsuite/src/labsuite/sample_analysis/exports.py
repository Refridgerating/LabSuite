"""Exports for sample-level derived analysis results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml

from labsuite.core.sample_registry import to_serializable
from labsuite.sample_analysis.manifest import SampleAnalysisManifest
from labsuite.sample_analysis.recipe import SampleAnalysisRecipe


def write_sample_analysis_outputs(
    *,
    result: dict[str, Any],
    output_dir: Path,
    manifest: SampleAnalysisManifest,
    recipe: SampleAnalysisRecipe,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    (output_dir / "warnings").mkdir(exist_ok=True)
    (output_dir / "provenance").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    paths = {
        "summary_json": output_dir / "sample_analysis_summary.json",
        "summary_csv": output_dir / "sample_analysis_summary.csv",
        "report_md": output_dir / "sample_analysis_report.md",
        "registry_snapshot": output_dir / "provenance" / "sample_registry_snapshot.yaml",
        "measurement_ledger_snapshot": output_dir
        / "provenance"
        / "measurement_ledger_snapshot.yaml",
        "processed_ledger_snapshot": output_dir / "provenance" / "processed_ledger_snapshot.yaml",
        "recipe_snapshot": output_dir / "provenance" / "analysis_recipe_snapshot.yaml",
        "processed_manifest": output_dir / "provenance" / "processed_inputs_manifest.json",
        "fmr_table": output_dir / "tables" / "fmr_branch_parameters.csv",
        "fmr_branch_summary_table": output_dir / "tables" / "fmr_branch_summary.csv",
        "vsm_table": output_dir / "tables" / "vsm_parameters.csv",
        "esr_table": output_dir / "tables" / "esr_parameters.csv",
        "anisotropy_table": output_dir / "tables" / "anisotropy_parameters.csv",
        "damping_table": output_dir / "tables" / "damping_parameters.csv",
        "readiness_table": output_dir / "tables" / "readiness_matrix.csv",
        "warnings_table": output_dir / "warnings" / "analysis_warnings.csv",
    }
    paths["summary_json"].write_text(
        json.dumps(to_serializable(result), indent=2), encoding="utf-8"
    )
    _write_summary_csv(result, paths["summary_csv"])
    paths["report_md"].write_text(_build_report(result), encoding="utf-8")
    paths["registry_snapshot"].write_text(
        yaml.safe_dump(manifest.registry_snapshot, sort_keys=False), encoding="utf-8"
    )
    paths["measurement_ledger_snapshot"].write_text(
        yaml.safe_dump(manifest.measurement_ledger_snapshot, sort_keys=False), encoding="utf-8"
    )
    paths["processed_ledger_snapshot"].write_text(
        yaml.safe_dump(manifest.processed_ledger_snapshot, sort_keys=False), encoding="utf-8"
    )
    paths["recipe_snapshot"].write_text(
        yaml.safe_dump(recipe.to_dict(), sort_keys=False), encoding="utf-8"
    )
    paths["processed_manifest"].write_text(
        json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
    )
    _write_rows(result["tables"]["fmr_branch_parameters"], paths["fmr_table"])
    _write_rows(result["summary"].get("fmr_branches", []), paths["fmr_branch_summary_table"])
    _write_rows(result["tables"]["vsm_parameters"], paths["vsm_table"])
    _write_rows(result["tables"]["esr_parameters"], paths["esr_table"])
    _write_rows(result["tables"]["anisotropy_parameters"], paths["anisotropy_table"])
    _write_rows(result["tables"]["damping_parameters"], paths["damping_table"])
    _write_rows(_readiness_rows(result), paths["readiness_table"])
    _write_rows(result["warnings"], paths["warnings_table"])
    figure_path = _write_summary_figure(result, output_dir / "figures")
    if figure_path is not None:
        paths["summary_figure"] = figure_path
    return paths


def _write_summary_csv(result: dict[str, Any], path: Path) -> None:
    summary = result["summary"]
    fields = [
        "sample_id",
        "readiness",
        "usable_processed_inputs",
        "warning_count",
        "Ms_A_per_m",
        "primary_Meff_mT",
        "primary_g",
        "primary_alpha_eff",
        "fmr_branch_count",
    ]
    row = {
        "sample_id": summary["sample_id"],
        "readiness": summary["readiness"],
        "usable_processed_inputs": summary["usable_processed_inputs"],
        "warning_count": len(result["warnings"]),
        "Ms_A_per_m": summary.get("Ms_A_per_m"),
        "primary_Meff_mT": summary.get("primary_Meff_mT"),
        "primary_g": summary.get("primary_g"),
        "primary_alpha_eff": summary.get("primary_alpha_eff"),
        "fmr_branch_count": len(summary.get("fmr_branches", [])),
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    if not fields:
        fields = ["status"]
        rows = [{"status": "no_rows"}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _readiness_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": result["summary"]["sample_id"],
            "readiness": label,
            "ready": value,
            "primary_readiness": result["summary"]["readiness"],
        }
        for label, value in result["readiness_matrix"].items()
    ]


def _build_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        f"# Sample Analysis Report: {summary['sample_id']}",
        "",
        f"- Readiness: `{summary['readiness']}`",
        f"- Usable processed inputs: `{summary['usable_processed_inputs']}`",
        f"- Warnings: `{len(result['warnings'])}`",
        "",
        "## Key Parameters",
        "",
        f"- Ms: `{_format_value(summary.get('Ms_A_per_m'))}` A/m",
        f"- Primary Meff: `{_format_value(summary.get('primary_Meff_mT'))}` mT",
        f"- Primary g: `{_format_value(summary.get('primary_g'))}`",
        f"- Primary alpha_eff: `{_format_value(summary.get('primary_alpha_eff'))}`",
    ]
    if summary.get("fmr_branches"):
        lines.extend(["", "## FMR Branches", ""])
        for branch in summary["fmr_branches"]:
            lines.append(
                "- "
                f"`{branch.get('geometry')}:{branch.get('branch_label')}` "
                f"Meff=`{_format_value(branch.get('Meff_mT'))}` mT, "
                f"g=`{_format_value(branch.get('g'))}`, "
                f"Ms=`{_format_value(branch.get('Ms_A_per_m'))}` A/m, "
                f"alpha_eff=`{_format_value(branch.get('alpha_eff'))}`"
            )
    if result["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(
            f"- `{item.get('code')}`: {item.get('message')}" for item in result["warnings"]
        )
    return "\n".join(lines) + "\n"


def _write_summary_figure(result: dict[str, Any], figures_dir: Path) -> Path | None:
    fmr_rows = [
        row
        for row in result["tables"]["fmr_branch_parameters"]
        if row.get("row_type") == "series_point" and row.get("field_variant") == "canonical"
    ]
    damping_rows = result["tables"]["damping_parameters"]
    if not fmr_rows and not damping_rows:
        return None
    destination = figures_dir / "sample_analysis_overview.png"
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in fmr_rows:
        grouped.setdefault((str(row.get("geometry")), str(row.get("branch_label"))), []).append(row)
    for (geometry, branch), rows in grouped.items():
        ordered = sorted(rows, key=lambda item: float(item["frequency_GHz"]))
        axes[0].scatter(
            [row["frequency_GHz"] for row in ordered],
            [row["Hres_mT"] for row in ordered],
            label=f"{geometry}:{branch}",
            s=28,
        )
    axes[0].set_xlabel("Frequency (GHz)")
    axes[0].set_ylabel("Hres (mT)")
    axes[0].grid(alpha=0.25)
    if grouped:
        axes[0].legend(loc="best", fontsize=8)
    for row in damping_rows:
        axes[1].scatter(
            row.get("frequency_min_GHz"),
            row.get("alpha_eff"),
            label=f"{row.get('geometry')}:{row.get('branch_label')}",
            s=32,
        )
    axes[1].set_xlabel("Frequency min (GHz)")
    axes[1].set_ylabel("alpha_eff")
    axes[1].grid(alpha=0.25)
    if damping_rows:
        axes[1].legend(loc="best", fontsize=8)
    figure.suptitle(f"Sample Analysis: {result['summary']['sample_id']}")
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return "|".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return value


def _format_value(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)
