"""Service entrypoints for registry-aware sample derived analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from labsuite.core.sample_registry import find_sample, load_registry
from labsuite.sample_analysis.calculations import (
    classify_readiness,
    derive_anisotropy_parameters,
    derive_damping_parameters,
    derive_esr_parameters,
    derive_fmr_parameters,
    derive_vsm_parameters,
)
from labsuite.sample_analysis.exports import write_sample_analysis_outputs
from labsuite.sample_analysis.manifest import SampleAnalysisManifest, build_sample_manifest
from labsuite.sample_analysis.recipe import SampleAnalysisRecipe, load_sample_analysis_recipe


@dataclass(slots=True)
class SampleAnalysisRunResult:
    sample_id: str
    output_dir: Path | None
    result: dict[str, Any]
    artifacts: dict[str, Path]


def build_sample_readiness(
    *,
    sample_id: str,
    registry_path: Path,
    measurement_ledger_path: Path,
    processed_ledger_path: Path,
    recipe_path: Path | None = None,
) -> dict[str, Any]:
    recipe = _load_recipe(recipe_path)
    manifest = build_sample_manifest(
        sample_id=sample_id,
        registry_path=registry_path,
        measurement_ledger_path=measurement_ledger_path,
        processed_ledger_path=processed_ledger_path,
    )
    return _build_result(manifest, recipe)


def analyze_sample(
    *,
    sample_id: str,
    registry_path: Path,
    measurement_ledger_path: Path,
    processed_ledger_path: Path,
    recipe_path: Path,
    output_dir: Path,
) -> SampleAnalysisRunResult:
    recipe = load_sample_analysis_recipe(recipe_path)
    manifest = build_sample_manifest(
        sample_id=sample_id,
        registry_path=registry_path,
        measurement_ledger_path=measurement_ledger_path,
        processed_ledger_path=processed_ledger_path,
    )
    result = _build_result(manifest, recipe)
    artifacts = write_sample_analysis_outputs(
        result=result,
        output_dir=output_dir.resolve(),
        manifest=manifest,
        recipe=recipe,
    )
    return SampleAnalysisRunResult(
        sample_id=manifest.sample_id,
        output_dir=output_dir.resolve(),
        result=result,
        artifacts=artifacts,
    )


def analyze_sample_batch(
    *,
    registry_path: Path,
    measurement_ledger_path: Path,
    processed_ledger_path: Path,
    recipe_path: Path,
    output_dir: Path,
) -> list[SampleAnalysisRunResult]:
    registry = load_registry(registry_path.resolve())
    results: list[SampleAnalysisRunResult] = []
    for sample_id in sorted(registry.samples):
        sample = find_sample(registry, sample_id)
        if sample is None:
            continue
        results.append(
            analyze_sample(
                sample_id=sample.sample_id,
                registry_path=registry_path,
                measurement_ledger_path=measurement_ledger_path,
                processed_ledger_path=processed_ledger_path,
                recipe_path=recipe_path,
                output_dir=output_dir.resolve() / sample.sample_id,
            )
        )
    return results


def _build_result(
    manifest: SampleAnalysisManifest,
    recipe: SampleAnalysisRecipe,
) -> dict[str, Any]:
    if not manifest.processed_inputs:
        readiness, matrix = classify_readiness(0, [], [], [], [])
        return {
            "summary": {
                "sample_id": manifest.sample_id,
                "readiness": readiness,
                "usable_processed_inputs": 0,
                "registered_measurements": 0,
                "Ms_A_per_m": None,
                "primary_Meff_mT": None,
                "primary_g": None,
                "primary_alpha_eff": None,
            },
            "readiness_matrix": matrix,
            "warnings": _dedupe_warnings(manifest.warnings),
            "manifest": manifest.to_dict(),
            "tables": {
                "vsm_parameters": [],
                "fmr_branch_parameters": [],
                "esr_parameters": [],
                "anisotropy_parameters": [],
                "damping_parameters": [],
            },
            "recipe": recipe.to_dict(),
        }
    usable = [item for item in manifest.processed_inputs if item.status == "usable"]
    vsm_rows, vsm_warnings, vsm_context = derive_vsm_parameters(
        manifest.sample, manifest.processed_inputs
    )
    fmr_rows, fmr_warnings, fmr_context = derive_fmr_parameters(
        manifest.sample, manifest.processed_inputs, recipe
    )
    kittel_rows = fmr_context["kittel_rows"]
    anisotropy_rows, anisotropy_warnings = derive_anisotropy_parameters(
        manifest.sample_id,
        vsm_context["selected_ms"],
        kittel_rows,
        recipe,
    )
    damping_rows, damping_warnings = derive_damping_parameters(
        manifest.sample_id,
        fmr_rows,
        kittel_rows,
        recipe,
    )
    esr_rows, esr_warnings = derive_esr_parameters(
        manifest.sample_id, manifest.processed_inputs, recipe
    )
    readiness, matrix = classify_readiness(
        len(usable),
        vsm_rows,
        kittel_rows,
        anisotropy_rows,
        damping_rows,
    )
    warnings = [
        *manifest.warnings,
        *vsm_warnings,
        *fmr_warnings,
        *anisotropy_warnings,
        *damping_warnings,
        *esr_warnings,
    ]
    selected_ms = vsm_context["selected_ms"]
    primary_kittel = _primary_kittel(kittel_rows)
    primary_damping = _primary_damping(damping_rows)
    return {
        "summary": {
            "sample_id": manifest.sample_id,
            "readiness": readiness,
            "usable_processed_inputs": len(usable),
            "registered_measurements": len(manifest.processed_inputs),
            "Ms_A_per_m": None if selected_ms is None else selected_ms.get("Ms_A_per_m"),
            "primary_Meff_mT": None if primary_kittel is None else primary_kittel.get("Meff_mT"),
            "primary_g": None if primary_kittel is None else primary_kittel.get("g"),
            "primary_alpha_eff": None
            if primary_damping is None
            else primary_damping.get("alpha_eff"),
        },
        "readiness_matrix": matrix,
        "warnings": _dedupe_warnings(warnings),
        "manifest": manifest.to_dict(),
        "tables": {
            "vsm_parameters": vsm_rows,
            "fmr_branch_parameters": fmr_rows,
            "esr_parameters": esr_rows,
            "anisotropy_parameters": anisotropy_rows,
            "damping_parameters": damping_rows,
        },
        "recipe": recipe.to_dict(),
    }


def _load_recipe(recipe_path: Path | None) -> SampleAnalysisRecipe:
    if recipe_path is None:
        return SampleAnalysisRecipe()
    return load_sample_analysis_recipe(recipe_path)


def _primary_kittel(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    successful = [
        row for row in rows if row.get("success") and row.get("result_variant") == "canonical"
    ]
    if not successful:
        return None
    return sorted(
        successful, key=lambda row: (row.get("geometry") != "ip", str(row.get("branch_label")))
    )[0]


def _primary_damping(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    successful = [row for row in rows if row.get("success") and row.get("alpha_eff") is not None]
    if not successful:
        return None
    return sorted(
        successful,
        key=lambda row: (not bool(row.get("preferred_for_alpha")), str(row.get("branch_label"))),
    )[0]


def _dedupe_warnings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str]] = set()
    output: list[dict[str, Any]] = []
    for row in rows:
        key = (
            str(row.get("code")),
            str(row.get("message")),
            str(row.get("measurement_id")),
            str(row.get("branch_label")),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output
