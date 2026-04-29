"""Service layer and artifact export for the VSM workflow."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from labsuite.core.exceptions import WorkflowError
from labsuite.core.measurement_models import (
    MeasurementAnalysisResult,
    MeasurementRecord,
    PlotManifest,
    SampleRecord,
)
from labsuite.core.recipes import load_vsm_recipe
from labsuite.core.sample_registry import AnalysisSampleContext
from labsuite.plugins.vsm.derived import build_vsm_output_layers, summarize_loop_quality
from labsuite.plugins.vsm.models import CenteringResult
from labsuite.plugins.vsm.parser import parse_vsm_file
from labsuite.plugins.vsm.preprocess import (
    apply_vsm_preprocessing,
    fit_background_slope,
    split_vsm_branches,
)

_SUMMARY_FIELDS = [
    "sample_id",
    "series_id",
    "replicate_id",
    "temperature_k",
    "point_count",
    "branch_count",
    "Ms_emu",
    "ms_error",
    "ms_std_pos",
    "ms_std_neg",
    "ms_n_points_pos",
    "ms_n_points_neg",
    "Mr_emu",
    "mr_error",
    "mr_zero_window_width_mT",
    "mr_interp_method",
    "Hc_mT",
    "hc_error",
    "hc_error_pos",
    "hc_error_neg",
    "switching_slope_pos",
    "switching_slope_neg",
    "squareness",
    "squareness_error",
    "exchange_bias_mT",
    "hex_error",
    "loop_area_emu_mT",
    "loop_area_error",
    "saturation_confidence",
    "branch_asymmetry",
    "switching_complexity",
    "switching_complexity_label",
    "ambiguity_flags",
    "coercive_field_negative_mT",
    "coercive_field_positive_mT",
    "coercive_field_mean_abs_mT",
    "loop_shift_mT",
    "remanence_positive_emu",
    "remanence_negative_emu",
    "remanence_mean_abs_emu",
    "vertical_shift_emu",
    "saturation_moment_positive_emu",
    "saturation_moment_negative_emu",
    "saturation_moment_mean_abs_emu",
    "max_corrected_moment_emu",
    "min_corrected_moment_emu",
    "background_slope_emu_per_mT",
    "background_intercept_emu",
    "background_slope_positive_emu_per_mT",
    "background_slope_negative_emu_per_mT",
    "background_intercept_positive_emu",
    "background_intercept_negative_emu",
    "background_mode",
    "background_subtraction_mode",
    "background_correction_accepted",
    "background_decision_reason",
    "background_qc_passed",
    "vsm_quality_model",
    "vsm_quality_status",
    "vsm_quality_weight",
    "vsm_quality_reasons",
    "vsm_quality_hcut_fraction",
    "vsm_quality_slope_score",
    "vsm_quality_symmetry_score",
    "vsm_quality_stability_score",
    "vsm_quality_rmse_score",
    "vsm_quality_residual_slope_pos",
    "vsm_quality_residual_slope_neg",
    "vsm_quality_tail_rmse",
    "vsm_quality_symmetry_error",
    "vsm_quality_cutoff_cv",
    "legacy_background_mode",
    "legacy_background_correction_accepted",
    "legacy_background_decision_reason",
    "legacy_background_qc_passed",
    "background_score_raw",
    "background_score_corrected",
    "background_score_delta",
    "raw_plateau_slope_positive_normalized",
    "raw_plateau_slope_negative_normalized",
    "corrected_plateau_slope_positive_normalized",
    "corrected_plateau_slope_negative_normalized",
    "background_flatness_gain_positive",
    "background_flatness_gain_negative",
    "background_flatness_gain_score",
    "background_flatness_gain_balance_score",
    "background_flatness_gain_balance_ok",
    "background_soft_override_passed",
    "background_tail_slope_symmetry_score",
    "background_saturation_magnitude_symmetry_score",
    "raw_switching_width_mT",
    "corrected_switching_width_mT",
    "background_switching_width_relative_change",
    "raw_zero_crossing_candidate_count",
    "corrected_zero_crossing_candidate_count",
    "raw_plateau_flatness_ratio",
    "corrected_plateau_flatness_ratio",
    "raw_saturation_consistency_ratio",
    "corrected_saturation_consistency_ratio",
    "raw_branch_asymmetry",
    "corrected_branch_asymmetry",
    "raw_loop_closure_error",
    "corrected_loop_closure_error",
    "raw_coercive_ambiguity_count",
    "corrected_coercive_ambiguity_count",
    "positive_tail_fit_r_squared",
    "negative_tail_fit_r_squared",
    "positive_tail_fit_r_squared_soft_warning",
    "negative_tail_fit_r_squared_soft_warning",
    "positive_tail_fit_r_squared_catastrophic",
    "negative_tail_fit_r_squared_catastrophic",
    "corrected_positive_tail_slope_emu_per_mT",
    "corrected_negative_tail_slope_emu_per_mT",
    "corrected_tail_slope_abs_mismatch_emu_per_mT",
    "positive_tail_flatness_ratio",
    "negative_tail_flatness_ratio",
    "raw_tail_slope_disagreement_ratio",
    "tail_window_selection_mode",
    "positive_tail_window_initial_point_count",
    "negative_tail_window_initial_point_count",
    "positive_tail_window_selected_point_count",
    "negative_tail_window_selected_point_count",
    "positive_tail_window_soft_r_squared_rescue_attempted",
    "negative_tail_window_soft_r_squared_rescue_attempted",
    "positive_tail_window_rescue_changed_selection",
    "negative_tail_window_rescue_changed_selection",
    "centering_field_offset_mT",
    "centering_moment_offset_emu",
    "centering_applied",
    "warning_count",
    "warnings",
]


def analyze_vsm_file(
    source_path: Path,
    recipe_path: Path,
    sample_context: AnalysisSampleContext | None = None,
    recipe_overrides: dict[str, Any] | None = None,
) -> MeasurementAnalysisResult:
    """Run the first VSM loop-analysis pipeline."""

    dataset = parse_vsm_file(source_path.resolve())
    recipe = load_vsm_recipe(recipe_path)
    _apply_vsm_recipe_overrides(recipe, recipe_overrides)
    sample_record, filename_warnings = _parse_vsm_filename_metadata(dataset.source_path)
    filename_sample_id = sample_record.sample_id
    if sample_context is not None and sample_context.sample_id:
        resolved_replicate = (
            sample_context.sample.replicate
            if sample_context.sample is not None and sample_context.sample.replicate is not None
            else sample_record.filename_tokens.get("replicate_id")
        )
        sample_record = SampleRecord(
            sample_id=sample_context.sample_id,
            series_id=sample_context.sample_id,
            filename_tokens={
                **sample_record.filename_tokens,
                "filename_sample_id": filename_sample_id,
                "replicate_id": resolved_replicate,
            },
            grouping_keys={
                "series": sample_context.sample_id,
                "series_replicate": f"{sample_context.sample_id}:{sample_context.sample.replicate}"
                if sample_context.sample is not None and sample_context.sample.replicate
                else sample_context.sample_id,
            },
        )

    processed_moment, preprocessing_steps, preprocessing_warnings = apply_vsm_preprocessing(
        dataset, recipe
    )
    branch_ids, branches = split_vsm_branches(dataset.field_mT)
    background_fit, loop_variants, tail_masks, background_warnings = fit_background_slope(
        dataset.field_mT,
        processed_moment,
        dataset.moment_std_err_emu,
        branches,
        recipe,
        temperature_k=dataset.temperature_k,
    )
    combined_background = background_fit["combined_background"]
    background_qc = combined_background["qc"]
    uncorrected_metrics = background_qc["raw_metrics"]
    corrected_candidate_metrics = background_qc["corrected_candidate_metrics"]
    selected_uncentered_metrics = (
        corrected_candidate_metrics
        if combined_background["background_mode"] == "slope_only"
        else uncorrected_metrics
    )

    centering = CenteringResult(
        field_offset_mT=float(selected_uncentered_metrics.get("loop_shift_mT") or 0.0),
        moment_offset_emu=float(selected_uncentered_metrics.get("vertical_shift_emu") or 0.0),
        applied=bool(recipe.center_loop),
    )

    centered_field = np.asarray(dataset.field_mT - centering.field_offset_mT, dtype=float)
    centered_moment = np.asarray(
        loop_variants["final_moment_emu"] - centering.moment_offset_emu, dtype=float
    )
    final_field = centered_field if centering.applied else np.asarray(dataset.field_mT, dtype=float)
    final_moment = (
        centered_moment
        if centering.applied
        else np.asarray(loop_variants["final_moment_emu"], dtype=float)
    )

    final_metrics = summarize_loop_quality(
        field_mT=final_field,
        moment_emu=final_moment,
        branches=branches,
        positive_tail_indices=np.asarray(
            combined_background["selected_positive_indices"], dtype=int
        ),
        negative_tail_indices=np.asarray(
            combined_background["selected_negative_indices"], dtype=int
        ),
        temperature_k=dataset.temperature_k,
    )

    warnings = list(
        dict.fromkeys(
            [
                *filename_warnings,
                *preprocessing_warnings,
                *background_warnings,
                *uncorrected_metrics.get("warnings", []),
                *corrected_candidate_metrics.get("warnings", []),
                *final_metrics.get("warnings", []),
            ]
        )
    )
    direct_observables, trust_diagnostics, uncertainty_estimates = build_vsm_output_layers(
        field_mT=final_field,
        moment_emu=final_moment,
        branches=branches,
        detailed_metrics=final_metrics,
        background_qc=background_qc,
        background_details=combined_background,
        warnings=warnings,
        moment_std_err_emu=dataset.moment_std_err_emu,
        recipe=recipe,
    )

    measurement = MeasurementRecord(
        modality="vsm",
        source_path=dataset.source_path,
        sample=sample_record,
        replicate_id=sample_record.filename_tokens.get("replicate_id"),
        condition_metadata={
            "temperature_k": final_metrics.get("temperature_k"),
            "registry_geometry": None if sample_context is None else sample_context.geometry,
            "registry_measurement_id": None
            if sample_context is None
            else sample_context.measurement_id,
        },
        raw_metadata=dataset.metadata,
    )
    summary_metrics = {
        **final_metrics,
        **direct_observables,
        "ms_error": uncertainty_estimates["ms_error"],
        "ms_std_pos": uncertainty_estimates["ms_std_pos"],
        "ms_std_neg": uncertainty_estimates["ms_std_neg"],
        "ms_n_points_pos": uncertainty_estimates["ms_n_points_pos"],
        "ms_n_points_neg": uncertainty_estimates["ms_n_points_neg"],
        "mr_error": uncertainty_estimates["mr_error"],
        "mr_zero_window_width_mT": uncertainty_estimates["mr_zero_window_width_mT"],
        "mr_interp_method": uncertainty_estimates["mr_interp_method"],
        "hc_error": uncertainty_estimates["hc_error"],
        "hc_error_pos": uncertainty_estimates["hc_error_pos"],
        "hc_error_neg": uncertainty_estimates["hc_error_neg"],
        "switching_slope_pos": uncertainty_estimates["switching_slope_pos"],
        "switching_slope_neg": uncertainty_estimates["switching_slope_neg"],
        "hex_error": uncertainty_estimates["hex_error"],
        "squareness_error": uncertainty_estimates["squareness_error"],
        "loop_area_error": uncertainty_estimates["loop_area_error"],
        "sample_id": sample_record.sample_id,
        "series_id": sample_record.series_id,
        "replicate_id": sample_record.filename_tokens.get("replicate_id"),
        "filename_sample_id": filename_sample_id,
        "registry_measurement_id": None
        if sample_context is None
        else sample_context.measurement_id,
        "registry_geometry": None if sample_context is None else sample_context.geometry,
        "acquisition_index": sample_record.filename_tokens.get("acquisition_index"),
        "saturation_confidence": trust_diagnostics["saturation_confidence"],
        "ambiguity_flags": trust_diagnostics["ambiguity_flags"],
        "branch_asymmetry": trust_diagnostics["branch_asymmetry"],
        "switching_complexity": trust_diagnostics["switching_complexity"],
        "switching_complexity_label": trust_diagnostics["switching_complexity_label"],
        "background_mode": combined_background["background_mode"],
        "background_slope_emu_per_mT": combined_background["slope_emu_per_mT"],
        "background_intercept_emu": combined_background["intercept_emu"],
        "background_slope_positive_emu_per_mT": combined_background["positive_slope_emu_per_mT"],
        "background_slope_negative_emu_per_mT": combined_background["negative_slope_emu_per_mT"],
        "background_intercept_positive_emu": combined_background["positive_intercept_emu"],
        "background_intercept_negative_emu": combined_background["negative_intercept_emu"],
        "background_subtraction_mode": combined_background["subtraction_mode"],
        "background_correction_accepted": combined_background["correction_accepted"],
        "background_decision_reason": combined_background["decision_reason"],
        "background_qc_passed": combined_background["qc_passed"],
        "vsm_quality_model": combined_background["quality_model"],
        "vsm_quality_status": combined_background["quality_status"],
        "vsm_quality_weight": combined_background["quality_weight"],
        "vsm_quality_reasons": combined_background["quality_reasons"],
        "vsm_quality_hcut_fraction": combined_background["quality"]["hcut_fraction"],
        "vsm_quality_slope_score": combined_background["quality"]["slope_score"],
        "vsm_quality_symmetry_score": combined_background["quality"]["symmetry_score"],
        "vsm_quality_stability_score": combined_background["quality"]["stability_score"],
        "vsm_quality_rmse_score": combined_background["quality"]["rmse_score"],
        "vsm_quality_residual_slope_pos": combined_background["quality"][
            "residual_slope_pos"
        ],
        "vsm_quality_residual_slope_neg": combined_background["quality"][
            "residual_slope_neg"
        ],
        "vsm_quality_tail_rmse": combined_background["quality"]["tail_rmse"],
        "vsm_quality_symmetry_error": combined_background["quality"]["symmetry_error"],
        "vsm_quality_cutoff_cv": combined_background["quality"]["cutoff_cv"],
        "legacy_background_mode": combined_background["legacy_background_mode"],
        "legacy_background_correction_accepted": combined_background[
            "legacy_correction_accepted"
        ],
        "legacy_background_decision_reason": combined_background["legacy_decision_reason"],
        "legacy_background_qc_passed": combined_background["legacy_qc_passed"],
        "background_score_raw": background_qc["score_raw"],
        "background_score_corrected": background_qc["score_corrected"],
        "background_score_delta": background_qc["score_delta"],
        "raw_plateau_slope_positive_normalized": background_qc["comparison"][
            "raw_plateau_slope_positive_normalized"
        ],
        "raw_plateau_slope_negative_normalized": background_qc["comparison"][
            "raw_plateau_slope_negative_normalized"
        ],
        "corrected_plateau_slope_positive_normalized": background_qc["comparison"][
            "corrected_plateau_slope_positive_normalized"
        ],
        "corrected_plateau_slope_negative_normalized": background_qc["comparison"][
            "corrected_plateau_slope_negative_normalized"
        ],
        "background_flatness_gain_positive": background_qc["comparison"][
            "background_flatness_gain_positive"
        ],
        "background_flatness_gain_negative": background_qc["comparison"][
            "background_flatness_gain_negative"
        ],
        "background_flatness_gain_score": background_qc["comparison"][
            "background_flatness_gain_score"
        ],
        "background_flatness_gain_balance_score": background_qc["comparison"][
            "background_flatness_gain_balance_score"
        ],
        "background_flatness_gain_balance_ok": background_qc["comparison"][
            "background_flatness_gain_balance_ok"
        ],
        "background_soft_override_passed": background_qc["comparison"][
            "background_soft_override_passed"
        ],
        "background_tail_slope_symmetry_score": background_qc["comparison"][
            "background_tail_slope_symmetry_score"
        ],
        "background_saturation_magnitude_symmetry_score": background_qc["comparison"][
            "background_saturation_magnitude_symmetry_score"
        ],
        "raw_switching_width_mT": background_qc["comparison"]["raw_switching_width_mT"],
        "corrected_switching_width_mT": background_qc["comparison"]["corrected_switching_width_mT"],
        "background_switching_width_relative_change": background_qc["comparison"][
            "background_switching_width_relative_change"
        ],
        "raw_zero_crossing_candidate_count": background_qc["comparison"][
            "raw_zero_crossing_candidate_count"
        ],
        "corrected_zero_crossing_candidate_count": background_qc["comparison"][
            "corrected_zero_crossing_candidate_count"
        ],
        "raw_plateau_flatness_ratio": uncorrected_metrics["plateau_flatness_ratio"],
        "corrected_plateau_flatness_ratio": corrected_candidate_metrics["plateau_flatness_ratio"],
        "raw_saturation_consistency_ratio": uncorrected_metrics["saturation_consistency_ratio"],
        "corrected_saturation_consistency_ratio": corrected_candidate_metrics[
            "saturation_consistency_ratio"
        ],
        "raw_branch_asymmetry": uncorrected_metrics["branch_asymmetry"],
        "corrected_branch_asymmetry": corrected_candidate_metrics["branch_asymmetry"],
        "raw_loop_closure_error": uncorrected_metrics["loop_closure_error"],
        "corrected_loop_closure_error": corrected_candidate_metrics["loop_closure_error"],
        "raw_coercive_ambiguity_count": uncorrected_metrics["coercive_ambiguity_count"],
        "corrected_coercive_ambiguity_count": corrected_candidate_metrics[
            "coercive_ambiguity_count"
        ],
        "positive_tail_fit_r_squared": background_qc["positive_tail_fit_r_squared"],
        "negative_tail_fit_r_squared": background_qc["negative_tail_fit_r_squared"],
        "positive_tail_fit_r_squared_soft_warning": background_qc[
            "positive_tail_fit_r_squared_soft_warning"
        ],
        "negative_tail_fit_r_squared_soft_warning": background_qc[
            "negative_tail_fit_r_squared_soft_warning"
        ],
        "positive_tail_fit_r_squared_catastrophic": background_qc[
            "positive_tail_fit_r_squared_catastrophic"
        ],
        "negative_tail_fit_r_squared_catastrophic": background_qc[
            "negative_tail_fit_r_squared_catastrophic"
        ],
        "corrected_positive_tail_slope_emu_per_mT": background_qc[
            "corrected_positive_tail_slope_emu_per_mT"
        ],
        "corrected_negative_tail_slope_emu_per_mT": background_qc[
            "corrected_negative_tail_slope_emu_per_mT"
        ],
        "corrected_tail_slope_abs_mismatch_emu_per_mT": background_qc[
            "corrected_tail_slope_abs_mismatch_emu_per_mT"
        ],
        "positive_tail_flatness_ratio": background_qc["positive_tail_flatness_ratio"],
        "negative_tail_flatness_ratio": background_qc["negative_tail_flatness_ratio"],
        "raw_tail_slope_disagreement_ratio": background_qc["raw_tail_slope_disagreement_ratio"],
        "tail_window_selection_mode": background_qc["tail_window_selection_mode"],
        "positive_tail_window_initial_point_count": background_qc[
            "positive_tail_window_initial_point_count"
        ],
        "negative_tail_window_initial_point_count": background_qc[
            "negative_tail_window_initial_point_count"
        ],
        "positive_tail_window_selected_point_count": background_qc[
            "positive_tail_window_selected_point_count"
        ],
        "negative_tail_window_selected_point_count": background_qc[
            "negative_tail_window_selected_point_count"
        ],
        "positive_tail_window_selected_field_min_mT": background_qc[
            "positive_tail_window_selected_field_min_mT"
        ],
        "negative_tail_window_selected_field_max_mT": background_qc[
            "negative_tail_window_selected_field_max_mT"
        ],
        "positive_tail_window_soft_r_squared_rescue_attempted": background_qc[
            "positive_tail_window_soft_r_squared_rescue_attempted"
        ],
        "negative_tail_window_soft_r_squared_rescue_attempted": background_qc[
            "negative_tail_window_soft_r_squared_rescue_attempted"
        ],
        "positive_tail_window_rescue_changed_selection": background_qc[
            "positive_tail_window_rescue_changed_selection"
        ],
        "negative_tail_window_rescue_changed_selection": background_qc[
            "negative_tail_window_rescue_changed_selection"
        ],
        "centering_field_offset_mT": centering.field_offset_mT,
        "centering_moment_offset_emu": centering.moment_offset_emu,
        "centering_applied": centering.applied,
        "warning_count": len(warnings),
        "warnings": warnings,
        "mean_moment_std_err_emu": _nanmean_or_none(dataset.moment_std_err_emu),
    }

    analysis_payload = {
        "raw_data": {
            "acquisition_index": dataset.acquisition_index,
            "field_oe": dataset.field_oe,
            "field_mT": dataset.field_mT,
            "moment_emu": dataset.moment_emu,
            "moment_std_err_emu": dataset.moment_std_err_emu,
            "temperature_k": dataset.temperature_k,
        },
        "processed_data": {
            "processed_moment_emu": processed_moment,
            "uncorrected_moment_emu": loop_variants["uncorrected_moment_emu"],
            "slope_corrected_moment_emu": loop_variants["slope_corrected_moment_emu"],
            "corrected_moment_emu": loop_variants["slope_corrected_moment_emu"],
            "selected_moment_emu": loop_variants["final_moment_emu"],
            "centered_field_mT": centered_field,
            "centered_moment_emu": centered_moment,
            "final_field_mT": final_field,
            "final_moment_emu": final_moment,
            "branch_id": branch_ids,
            "branch_direction": [branches[branch_id].direction for branch_id in branch_ids],
        },
        "branches": [
            {
                "branch_id": branch.branch_id,
                "direction": branch.direction,
                "start_index": branch.start_index,
                "end_index": branch.end_index,
                "point_count": branch.point_count,
                "field_start_mT": branch.field_start_mT,
                "field_end_mT": branch.field_end_mT,
            }
            for branch in branches
        ],
        "background_fit": {
            "positive_tail_fit": background_fit["positive_tail_fit"],
            "negative_tail_fit": background_fit["negative_tail_fit"],
            "evaluation": background_qc,
            "combined_background": {
                **combined_background,
                "positive_tail_mask": tail_masks["positive_tail_mask"],
                "negative_tail_mask": tail_masks["negative_tail_mask"],
                "combined_tail_mask": tail_masks["combined_tail_mask"],
            },
        },
        "centering": {
            "field_offset_mT": centering.field_offset_mT,
            "moment_offset_emu": centering.moment_offset_emu,
            "applied": centering.applied,
        },
        "metrics": {
            "uncorrected": uncorrected_metrics,
            "corrected_candidate": corrected_candidate_metrics,
            "final": final_metrics,
        },
        "direct_observables": direct_observables,
        "trust_diagnostics": trust_diagnostics,
        "uncertainty_estimates": uncertainty_estimates,
        "uncertainty": {
            "source_column_present": bool(np.any(np.isfinite(dataset.moment_std_err_emu))),
            "mean_moment_std_err_emu": _nanmean_or_none(dataset.moment_std_err_emu),
            "uncertainty_scale": recipe.uncertainty_scale,
        },
    }
    plot_manifest = _build_plot_manifest(summary_metrics, centering)

    return MeasurementAnalysisResult(
        measurement=measurement,
        provenance={
            "parser": dataset.metadata.get("parser"),
            "source_format": dataset.metadata.get("source_format"),
            "recipe_name": recipe.name,
            "recipe_config": recipe.to_dict(),
            "preprocessing_steps": preprocessing_steps,
            "sample_registry": None if sample_context is None else sample_context.to_dict(),
            "canonical_units": {
                "field": "mT",
                "raw_field": "Oe",
                "moment": "emu",
                "temperature": "K",
            },
        },
        warnings=warnings,
        summary_metrics=summary_metrics,
        artifacts={},
        analysis_payload=analysis_payload,
        plot_manifest=plot_manifest,
    )


def export_vsm_analysis_json(result: MeasurementAnalysisResult, destination: Path) -> Path:
    """Write the complete VSM analysis envelope to JSON."""

    destination.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return destination


def _apply_vsm_recipe_overrides(
    recipe,
    overrides: dict[str, Any] | None,
) -> None:
    if not overrides:
        return
    for key, value in overrides.items():
        if value is None:
            continue
        if not hasattr(recipe, key):
            continue
        setattr(recipe, key, value)


def export_vsm_analysis_csv(
    result: MeasurementAnalysisResult | dict[str, Any], destination: Path
) -> Path:
    """Export evaluated VSM loop data to CSV."""

    payload = _normalize_result(result)
    raw_data = payload["analysis_payload"]["raw_data"]
    processed_data = payload["analysis_payload"]["processed_data"]
    background_fit = payload["analysis_payload"]["background_fit"]
    centering = payload["analysis_payload"]["centering"]

    header = [
        "acquisition_index",
        "raw_field_oe",
        "field_mT",
        "raw_moment_emu",
        "processed_moment_emu",
        "uncorrected_moment_emu",
        "slope_corrected_moment_emu",
        "background_fit_emu",
        "positive_tail_fit_emu",
        "negative_tail_fit_emu",
        "branch_id",
        "branch_direction",
        "temperature_k",
        "moment_std_err_emu",
        "selected_moment_emu",
        "final_field_mT",
        "final_moment_emu",
    ]
    if centering["applied"]:
        header.extend(["centered_field_mT", "centered_moment_emu"])

    combined_background = background_fit["combined_background"]
    positive_fit = background_fit["positive_tail_fit"]
    negative_fit = background_fit["negative_tail_fit"]
    slope = float(combined_background["slope_emu_per_mT"])
    positive_tail_index_map = {
        int(index): float(fit_value)
        for index, fit_value in zip(
            positive_fit["selected_indices"], positive_fit["fitted_y"], strict=True
        )
    }
    negative_tail_index_map = {
        int(index): float(fit_value)
        for index, fit_value in zip(
            negative_fit["selected_indices"], negative_fit["fitted_y"], strict=True
        )
    }

    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index in range(len(raw_data["field_mT"])):
            field_value = float(raw_data["field_mT"][index])
            row = [
                int(raw_data["acquisition_index"][index]),
                _format_float(raw_data["field_oe"][index]),
                _format_float(field_value),
                _format_float(raw_data["moment_emu"][index]),
                _format_float(processed_data["processed_moment_emu"][index]),
                _format_float(processed_data["uncorrected_moment_emu"][index]),
                _format_float(processed_data["slope_corrected_moment_emu"][index]),
                _format_float(slope * field_value),
                _format_float(positive_tail_index_map.get(index)),
                _format_float(negative_tail_index_map.get(index)),
                int(processed_data["branch_id"][index]),
                processed_data["branch_direction"][index],
                _format_float(raw_data["temperature_k"][index]),
                _format_float(raw_data["moment_std_err_emu"][index]),
                _format_float(processed_data["selected_moment_emu"][index]),
                _format_float(processed_data["final_field_mT"][index]),
                _format_float(processed_data["final_moment_emu"][index]),
            ]
            if centering["applied"]:
                row.extend(
                    [
                        _format_float(processed_data["centered_field_mT"][index]),
                        _format_float(processed_data["centered_moment_emu"][index]),
                    ]
                )
            writer.writerow(row)
    return destination


def export_vsm_summary_csv(
    result: MeasurementAnalysisResult | dict[str, Any], destination: Path
) -> Path:
    """Export the scalar VSM summary row to CSV."""

    payload = _normalize_result(result)
    summary_metrics = payload["summary_metrics"]
    row: dict[str, Any] = {}
    for field in _SUMMARY_FIELDS:
        value = summary_metrics.get(field)
        row[field] = "|".join(str(item) for item in value) if isinstance(value, list) else value
    row["warnings"] = "|".join(summary_metrics.get("warnings", []))
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    return destination


def export_vsm_analysis_figure(
    result: MeasurementAnalysisResult | dict[str, Any], destination: Path
) -> Path:
    """Save the VSM diagnostic figure."""

    payload = _normalize_result(result)
    raw_data = payload["analysis_payload"]["raw_data"]
    processed_data = payload["analysis_payload"]["processed_data"]
    background_fit = payload["analysis_payload"]["background_fit"]
    summary_metrics = payload["summary_metrics"]
    branches = payload["analysis_payload"]["branches"]
    centering = payload["analysis_payload"]["centering"]

    figure, axes = plt.subplots(2, 1, figsize=(10.5, 9.0), sharex=False, constrained_layout=True)

    field_mT = np.asarray(raw_data["field_mT"], dtype=float)
    raw_moment = np.asarray(raw_data["moment_emu"], dtype=float)
    uncorrected_moment = np.asarray(processed_data["uncorrected_moment_emu"], dtype=float)
    slope_corrected_moment = np.asarray(processed_data["slope_corrected_moment_emu"], dtype=float)
    selected_moment = np.asarray(processed_data["selected_moment_emu"], dtype=float)
    final_field = np.asarray(processed_data["final_field_mT"], dtype=float)
    final_moment = np.asarray(processed_data["final_moment_emu"], dtype=float)
    combined_background = background_fit["combined_background"]
    positive_fit = background_fit["positive_tail_fit"]
    negative_fit = background_fit["negative_tail_fit"]
    slope = float(combined_background["slope_emu_per_mT"])
    applied_background = slope * field_mT

    axes[0].plot(field_mT, raw_moment, color="#5f6b7a", linewidth=1.1, label="Raw moment")
    axes[0].plot(
        field_mT,
        uncorrected_moment,
        color="#0f766e",
        linewidth=1.1,
        label="Uncorrected analysis moment",
    )
    axes[0].plot(
        field_mT,
        slope_corrected_moment,
        color="#2563eb",
        linewidth=1.1,
        label="Slope-corrected candidate",
    )
    axes[0].plot(
        field_mT,
        applied_background,
        color="#c2410c",
        linewidth=1.1,
        linestyle="--",
        label="Evaluated slope-only background",
    )
    positive_fit_x = np.asarray(positive_fit["fitted_x"], dtype=float)
    positive_fit_y = np.asarray(positive_fit["fitted_y"], dtype=float)
    negative_fit_x = np.asarray(negative_fit["fitted_x"], dtype=float)
    negative_fit_y = np.asarray(negative_fit["fitted_y"], dtype=float)
    if positive_fit_x.size:
        axes[0].plot(
            positive_fit_x,
            positive_fit_y,
            color="#b91c1c",
            linewidth=1.4,
            label="Positive tail fit",
        )
        axes[0].scatter(
            positive_fit_x,
            uncorrected_moment[np.asarray(positive_fit["selected_indices"], dtype=int)],
            s=14,
            color="#b91c1c",
            alpha=0.8,
        )
    if negative_fit_x.size:
        axes[0].plot(
            negative_fit_x,
            negative_fit_y,
            color="#7c2d12",
            linewidth=1.4,
            label="Negative tail fit",
        )
        axes[0].scatter(
            negative_fit_x,
            uncorrected_moment[np.asarray(negative_fit["selected_indices"], dtype=int)],
            s=14,
            color="#7c2d12",
            alpha=0.8,
        )
    axes[0].set_title(f"Raw and Evaluated Background Fits: {summary_metrics['sample_id']}")
    axes[0].set_ylabel("Moment (emu)")
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="best")

    color_cycle = ["#2563eb", "#dc2626", "#16a34a", "#7c3aed", "#ea580c"]
    for branch in branches:
        start = int(branch["start_index"])
        end = int(branch["end_index"]) + 1
        color = color_cycle[int(branch["branch_id"]) % len(color_cycle)]
        label = f"Branch {branch['branch_id']} ({branch['direction']})"
        axes[1].plot(
            final_field[start:end], final_moment[start:end], color=color, linewidth=1.3, label=label
        )

    if summary_metrics.get("background_mode") != "slope_only":
        axes[1].plot(
            field_mT,
            slope_corrected_moment,
            color="0.55",
            linewidth=0.9,
            linestyle=":",
            label="Slope-corrected candidate",
        )
    if centering["applied"]:
        axes[1].plot(
            field_mT,
            selected_moment,
            color="0.45",
            linewidth=0.9,
            linestyle="-.",
            label="Selected uncentered",
        )

    axes[1].axhline(0.0, color="0.3", linewidth=0.8)
    axes[1].axvline(0.0, color="0.3", linewidth=0.8)
    axes[1].set_title(f"Final Loop ({summary_metrics.get('background_mode')})")
    axes[1].set_xlabel("Field (mT)")
    axes[1].set_ylabel("Moment (emu)")
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="best", ncols=2)
    axes[1].text(
        0.02,
        0.98,
        _build_metric_summary(payload),
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=8,
        family="monospace",
        bbox={
            "facecolor": "white",
            "alpha": 0.86,
            "edgecolor": "0.75",
            "boxstyle": "round,pad=0.35",
        },
    )

    figure.savefig(destination, dpi=200)
    plt.close(figure)
    return destination


def export_vsm_batch_overlay_figure(
    analyses: list[MeasurementAnalysisResult],
    output_dir: Path,
) -> dict[str, Path]:
    """Export one batch-level VSM hysteresis overlay figure."""

    if not analyses:
        return {}

    destination = output_dir / "batch_hysteresis_overlay.png"
    figure, axis = plt.subplots(1, 1, figsize=(10.5, 8.0), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, max(len(analyses), 1)))

    for index, analysis in enumerate(analyses):
        payload = analysis.analysis_payload["processed_data"]
        final_field = np.asarray(payload["final_field_mT"], dtype=float)
        final_moment = np.asarray(payload["final_moment_emu"], dtype=float)
        axis.plot(
            final_field,
            final_moment,
            color=colors[index],
            linewidth=1.25,
            label=_vsm_batch_overlay_label(analysis),
        )

    axis.axhline(0.0, color="0.35", linewidth=0.8)
    axis.axvline(0.0, color="0.35", linewidth=0.8)
    axis.set_title("VSM Batch Hysteresis Overlay")
    axis.set_xlabel("Field (mT)")
    axis.set_ylabel("Moment (emu)")
    axis.grid(alpha=0.2)
    axis.legend(loc="best", ncols=2)

    figure.savefig(destination, dpi=200)
    plt.close(figure)
    return {"batch_hysteresis_overlay": destination}


def export_vsm_bundle_from_json(
    analysis_json_path: Path, output_dir: Path | None = None
) -> dict[str, Path]:
    """Regenerate CSV and figure artifacts from a saved VSM JSON result."""

    payload = load_vsm_analysis_json(analysis_json_path)
    destination_dir = (
        output_dir.resolve() if output_dir is not None else analysis_json_path.resolve().parent
    )
    destination_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(payload["measurement"]["source_path"]).stem
    csv_path = destination_dir / f"{stem}_trace.csv"
    summary_path = destination_dir / f"{stem}_summary.csv"
    figure_path = destination_dir / f"{stem}_figure.png"
    export_vsm_analysis_csv(payload, csv_path)
    export_vsm_summary_csv(payload, summary_path)
    export_vsm_analysis_figure(payload, figure_path)
    return {
        "json_path": analysis_json_path.resolve(),
        "csv_path": csv_path,
        "summary_csv_path": summary_path,
        "figure_path": figure_path,
    }


def load_vsm_analysis_json(path: Path) -> dict[str, Any]:
    """Load a saved VSM analysis JSON payload."""

    return json.loads(path.read_text(encoding="utf-8"))


def build_vsm_report(
    input_path: Path, output_path: Path | None = None, *, recursive: bool = True
) -> Path:
    """Generate a Markdown report from one or many saved VSM JSON analyses."""

    resolved_input = input_path.resolve()
    if resolved_input.is_file():
        payload = load_vsm_analysis_json(resolved_input)
        destination = (
            output_path.resolve()
            if output_path is not None
            else resolved_input.with_name(
                f"{resolved_input.stem.replace('_analysis', '')}_report.md"
            )
        )
        destination.write_text(_build_single_report_text(payload), encoding="utf-8")
        return destination

    if not resolved_input.is_dir():
        raise WorkflowError(f"Report input is neither a file nor directory: {resolved_input}")

    json_paths = sorted(
        resolved_input.rglob("*_analysis.json")
        if recursive
        else resolved_input.glob("*_analysis.json"),
        key=lambda path: str(path).lower(),
    )
    if not json_paths:
        raise WorkflowError(f"No analysis JSON files found under {resolved_input}")

    payloads = [load_vsm_analysis_json(path) for path in json_paths]
    destination = (
        output_path.resolve() if output_path is not None else resolved_input / "batch_report.md"
    )
    destination.write_text(_build_batch_report_text(payloads), encoding="utf-8")
    return destination


def _normalize_result(result: MeasurementAnalysisResult | dict[str, Any]) -> dict[str, Any]:
    if isinstance(result, MeasurementAnalysisResult):
        return result.to_dict()
    return result


def _parse_vsm_filename_metadata(path: Path) -> tuple[SampleRecord, list[str]]:
    stem = path.stem
    match = re.match(
        r"^(?P<sample>.+)-(?P<temperature>\d+(?:\.\d+)?)K-(?P<replicate>R\d+)(?:_(?P<acquisition>\d+))?$",
        stem,
    )
    if match is None:
        return (
            SampleRecord(
                sample_id=stem,
                series_id=stem,
                filename_tokens={"source_stem": stem},
                grouping_keys={"series": stem, "series_replicate": stem},
            ),
            ["filename_metadata_parse_failed"],
        )

    sample_id = match.group("sample")
    replicate_id = match.group("replicate")
    acquisition_index = match.group("acquisition")
    return (
        SampleRecord(
            sample_id=sample_id,
            series_id=sample_id,
            filename_tokens={
                "source_stem": stem,
                "temperature_k": float(match.group("temperature")),
                "replicate_id": replicate_id,
                "acquisition_index": acquisition_index,
            },
            grouping_keys={
                "series": sample_id,
                "series_replicate": f"{sample_id}:{replicate_id}",
            },
        ),
        [],
    )


def _build_plot_manifest(
    summary_metrics: dict[str, Any], centering: CenteringResult
) -> PlotManifest:
    return PlotManifest(
        figure_type="vsm_loop_diagnostic",
        title=f"VSM Loop Diagnostic: {summary_metrics['sample_id']}",
        series=[
            {"label": "Raw moment", "x": "field_mT", "y": "moment_emu"},
            {
                "label": "Slope-corrected candidate",
                "x": "field_mT",
                "y": "slope_corrected_moment_emu",
            },
            {"label": "Final loop", "x": "final_field_mT", "y": "final_moment_emu"},
            {
                "label": "Evaluated slope-only background",
                "x": "field_mT",
                "y": "background_fit_emu",
            },
        ],
        annotations=[
            {"label": "Hc-", "value": summary_metrics.get("coercive_field_negative_mT")},
            {"label": "Hc+", "value": summary_metrics.get("coercive_field_positive_mT")},
            {"label": "Mr+", "value": summary_metrics.get("remanence_positive_emu")},
            {"label": "Mr-", "value": summary_metrics.get("remanence_negative_emu")},
        ],
        theme={
            "raw_color": "#5f6b7a",
            "corrected_color": "#2563eb",
            "centering_applied": centering.applied,
        },
    )


def _nanmean_or_none(values: np.ndarray) -> float | None:
    if not np.any(np.isfinite(values)):
        return None
    return float(np.nanmean(values))


def _format_float(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(numeric_value):
        return ""
    return f"{numeric_value:.10g}"


def _summary_float(summary: dict[str, Any], key: str) -> str:
    return _format_float(summary.get(key))


def _summary_pair(summary: dict[str, Any], first_key: str, second_key: str) -> str:
    return f"{_summary_float(summary, first_key)} / {_summary_float(summary, second_key)}"


def _summary_raw_pair(summary: dict[str, Any], first_key: str, second_key: str) -> str:
    return f"{summary.get(first_key)} / {summary.get(second_key)}"


def _summary_pm(summary: dict[str, Any], value_key: str, error_key: str) -> str:
    return f"{_summary_float(summary, value_key)} +/- {_summary_float(summary, error_key)}"


def _build_metric_summary(payload: dict[str, Any]) -> str:
    summary_metrics = payload["summary_metrics"]
    raw_tail_slopes = _summary_pair(
        summary_metrics,
        "raw_plateau_slope_positive_normalized",
        "raw_plateau_slope_negative_normalized",
    )
    corrected_tail_slopes = _summary_pair(
        summary_metrics,
        "corrected_plateau_slope_positive_normalized",
        "corrected_plateau_slope_negative_normalized",
    )
    tail_fit_r2 = _summary_pair(
        summary_metrics,
        "positive_tail_fit_r_squared",
        "negative_tail_fit_r_squared",
    )
    tail_fit_soft_warnings = _summary_raw_pair(
        summary_metrics,
        "positive_tail_fit_r_squared_soft_warning",
        "negative_tail_fit_r_squared_soft_warning",
    )
    tail_fit_catastrophic = _summary_raw_pair(
        summary_metrics,
        "positive_tail_fit_r_squared_catastrophic",
        "negative_tail_fit_r_squared_catastrophic",
    )
    flatness_gain_pair = _summary_pair(
        summary_metrics,
        "background_flatness_gain_positive",
        "background_flatness_gain_negative",
    )
    flatness_gain_score = _summary_float(summary_metrics, "background_flatness_gain_score")
    tail_rescue_attempted = _summary_raw_pair(
        summary_metrics,
        "positive_tail_window_soft_r_squared_rescue_attempted",
        "negative_tail_window_soft_r_squared_rescue_attempted",
    )
    tail_rescue_changed = _summary_raw_pair(
        summary_metrics,
        "positive_tail_window_rescue_changed_selection",
        "negative_tail_window_rescue_changed_selection",
    )
    switching_width = _summary_pair(
        summary_metrics,
        "raw_switching_width_mT",
        "corrected_switching_width_mT",
    )
    zero_crossings = _summary_pair(
        summary_metrics,
        "raw_zero_crossing_candidate_count",
        "corrected_zero_crossing_candidate_count",
    )
    lines = [
        f"Ms = {_summary_pm(summary_metrics, 'Ms_emu', 'ms_error')} emu",
        f"Mr = {_summary_pm(summary_metrics, 'Mr_emu', 'mr_error')} emu",
        f"Hc = {_summary_pm(summary_metrics, 'Hc_mT', 'hc_error')} mT",
        f"Hex = {_summary_pm(summary_metrics, 'exchange_bias_mT', 'hex_error')} mT",
        f"squareness = {_summary_pm(summary_metrics, 'squareness', 'squareness_error')}",
        f"vertical shift = {_format_float(summary_metrics.get('vertical_shift_emu'))} emu",
        f"loop area = {_summary_pm(summary_metrics, 'loop_area_emu_mT', 'loop_area_error')} emu*mT",
        f"saturation confidence = {_format_float(summary_metrics.get('saturation_confidence'))}",
        f"background mode = {summary_metrics.get('background_mode')}",
        f"background accepted = {summary_metrics.get('background_correction_accepted')}",
        f"background reason = {summary_metrics.get('background_decision_reason')}",
        f"background qc passed = {summary_metrics.get('background_qc_passed')}",
        f"tail selection mode = {summary_metrics.get('tail_window_selection_mode')}",
        "tail window pts init sel +/- = "
        f"{_summary_float(summary_metrics, 'positive_tail_window_initial_point_count')}:"
        f"{_summary_float(summary_metrics, 'positive_tail_window_selected_point_count')} / "
        f"{_summary_float(summary_metrics, 'negative_tail_window_initial_point_count')}:"
        f"{_summary_float(summary_metrics, 'negative_tail_window_selected_point_count')}",
        f"tail slopes raw +/- = {raw_tail_slopes}",
        f"tail slopes corr +/- = {corrected_tail_slopes}",
        f"tail fit r^2 +/- = {tail_fit_r2}",
        f"tail fit soft warn +/- = {tail_fit_soft_warnings}",
        f"tail fit catastrophic +/- = {tail_fit_catastrophic}",
        f"flatness gain +/- = {flatness_gain_pair}",
        f"flatness gain score = {flatness_gain_score}",
        "flatness gain balance = "
        f"{_format_float(summary_metrics.get('background_flatness_gain_balance_score'))}",
        f"flatness gain balance ok = {summary_metrics.get('background_flatness_gain_balance_ok')}",
        f"soft override passed = {summary_metrics.get('background_soft_override_passed')}",
        f"tail rescue attempted +/- = {tail_rescue_attempted}",
        f"tail rescue changed +/- = {tail_rescue_changed}",
        "tail slope symmetry = "
        f"{_summary_float(summary_metrics, 'background_tail_slope_symmetry_score')}",
        "sat magnitude symmetry = "
        f"{_summary_float(summary_metrics, 'background_saturation_magnitude_symmetry_score')}",
        f"switching width raw/corr = {switching_width}",
        f"zero crossings raw/corr = {zero_crossings}",
        "switching width rel change = "
        f"{_summary_float(summary_metrics, 'background_switching_width_relative_change')}",
        f"center applied = {summary_metrics.get('centering_applied')}",
        f"secondary branch asymmetry = {_format_float(summary_metrics.get('branch_asymmetry'))}",
        "secondary switching complexity = "
        f"{_summary_float(summary_metrics, 'switching_complexity')} "
        f"({summary_metrics.get('switching_complexity_label')})",
    ]
    ambiguity_flags = summary_metrics.get("ambiguity_flags", [])
    if ambiguity_flags:
        lines.append(f"ambiguity flags = {len(ambiguity_flags)}")
    warnings = summary_metrics.get("warnings", [])
    if warnings:
        lines.append(f"warnings = {len(warnings)}")
    return "\n".join(lines)


def _vsm_batch_overlay_label(result: MeasurementAnalysisResult) -> str:
    summary = result.summary_metrics
    stem = result.measurement.source_path.stem
    temperature_k = summary.get("temperature_k")
    replicate_id = summary.get("replicate_id")
    suffix_parts = [part for part in [_format_float(temperature_k), replicate_id] if part]
    if not suffix_parts:
        return stem
    return f"{stem} ({', '.join(suffix_parts)})"


def _build_single_report_text(payload: dict[str, Any]) -> str:
    measurement = payload["measurement"]
    summary = payload["summary_metrics"]
    artifacts = payload.get("artifacts", {})
    trust = payload["analysis_payload"]["trust_diagnostics"]
    raw_tail_slopes = _summary_pair(
        summary,
        "raw_plateau_slope_positive_normalized",
        "raw_plateau_slope_negative_normalized",
    )
    corrected_tail_slopes = _summary_pair(
        summary,
        "corrected_plateau_slope_positive_normalized",
        "corrected_plateau_slope_negative_normalized",
    )
    tail_fit_soft_warnings = _summary_raw_pair(
        summary,
        "positive_tail_fit_r_squared_soft_warning",
        "negative_tail_fit_r_squared_soft_warning",
    )
    tail_fit_catastrophic = _summary_raw_pair(
        summary,
        "positive_tail_fit_r_squared_catastrophic",
        "negative_tail_fit_r_squared_catastrophic",
    )
    tail_rescue_attempted = _summary_raw_pair(
        summary,
        "positive_tail_window_soft_r_squared_rescue_attempted",
        "negative_tail_window_soft_r_squared_rescue_attempted",
    )
    tail_rescue_changed = _summary_raw_pair(
        summary,
        "positive_tail_window_rescue_changed_selection",
        "negative_tail_window_rescue_changed_selection",
    )
    zero_crossings = _summary_pair(
        summary,
        "raw_zero_crossing_candidate_count",
        "corrected_zero_crossing_candidate_count",
    )
    coercive_fields = _summary_pair(
        summary,
        "coercive_field_negative_mT",
        "coercive_field_positive_mT",
    )
    saturation_moments = _summary_pair(
        summary,
        "saturation_moment_positive_emu",
        "saturation_moment_negative_emu",
    )
    lines = [
        f"# VSM Report: {summary['sample_id']}",
        "",
        f"- Source: `{measurement['source_path']}`",
        f"- Replicate: `{summary.get('replicate_id')}`",
        f"- Temperature: `{_format_float(summary.get('temperature_k'))}` K",
        f"- Point count: `{summary.get('point_count')}`",
        f"- Branch count: `{summary.get('branch_count')}`",
        "",
        "## Direct Observables",
        "",
        f"- Ms: `{_summary_pm(summary, 'Ms_emu', 'ms_error')}` emu",
        f"- Mr: `{_summary_pm(summary, 'Mr_emu', 'mr_error')}` emu",
        f"- Hc: `{_summary_pm(summary, 'Hc_mT', 'hc_error')}` mT",
        f"- Squareness: `{_summary_pm(summary, 'squareness', 'squareness_error')}`",
        f"- Exchange bias: `{_summary_pm(summary, 'exchange_bias_mT', 'hex_error')}` mT",
        f"- Vertical shift: `{_format_float(summary.get('vertical_shift_emu'))}` emu",
        f"- Loop area: `{_summary_pm(summary, 'loop_area_emu_mT', 'loop_area_error')}` emu*mT",
        "",
        "## Trust Diagnostics",
        "",
        f"- Saturation confidence: `{_format_float(summary.get('saturation_confidence'))}`",
        f"- Background mode: `{summary.get('background_mode')}`",
        f"- Background accepted: `{summary.get('background_correction_accepted')}`",
        f"- Background decision reason: `{summary.get('background_decision_reason')}`",
        f"- Background QC passed: `{summary.get('background_qc_passed')}`",
        f"- Tail selection mode: `{summary.get('tail_window_selection_mode')}`",
        "- Tail window counts (+ init/selected, - init/selected): "
        f"`{_summary_float(summary, 'positive_tail_window_initial_point_count')}` / "
        f"`{_summary_float(summary, 'positive_tail_window_selected_point_count')}` / "
        f"`{_summary_float(summary, 'negative_tail_window_initial_point_count')}` / "
        f"`{_summary_float(summary, 'negative_tail_window_selected_point_count')}`",
        f"- Raw residual tail slopes (+ / -): `{raw_tail_slopes}`",
        f"- Corrected residual tail slopes (+ / -): `{corrected_tail_slopes}`",
        "- Tail fit R^2 (+ / -): "
        f"`{_summary_pair(summary, 'positive_tail_fit_r_squared', 'negative_tail_fit_r_squared')}`",
        f"- Tail fit soft warning (+ / -): `{tail_fit_soft_warnings}`",
        f"- Tail fit catastrophic (+ / -): `{tail_fit_catastrophic}`",
        "- Flatness gain (+ / - / score): "
        f"`{_summary_float(summary, 'background_flatness_gain_positive')}` / "
        f"`{_summary_float(summary, 'background_flatness_gain_negative')}` / "
        f"`{_summary_float(summary, 'background_flatness_gain_score')}`",
        "- Flatness gain balance / ok / override: "
        f"`{_summary_float(summary, 'background_flatness_gain_balance_score')}` / "
        f"`{summary.get('background_flatness_gain_balance_ok')}` / "
        f"`{summary.get('background_soft_override_passed')}`",
        f"- Tail rescue attempted (+ / -): `{tail_rescue_attempted}`",
        f"- Tail rescue changed (+ / -): `{tail_rescue_changed}`",
        "- Corrected tail slope symmetry: "
        f"`{_summary_float(summary, 'background_tail_slope_symmetry_score')}`",
        "- Corrected saturation magnitude symmetry: "
        f"`{_summary_float(summary, 'background_saturation_magnitude_symmetry_score')}`",
        "- Switching width raw / corrected / rel change: "
        f"`{_summary_float(summary, 'raw_switching_width_mT')}` / "
        f"`{_summary_float(summary, 'corrected_switching_width_mT')}` / "
        f"`{_summary_float(summary, 'background_switching_width_relative_change')}`",
        f"- Zero crossing candidates raw / corrected: `{zero_crossings}`",
        f"- Background mode detail: `{trust['background_fit_details'].get('background_mode')}`",
        "",
        "## Secondary Diagnostics",
        "",
        "- Background score raw / corrected / delta: "
        f"`{_summary_float(summary, 'background_score_raw')}` / "
        f"`{_summary_float(summary, 'background_score_corrected')}` / "
        f"`{_summary_float(summary, 'background_score_delta')}`",
        f"- Branch asymmetry: `{_format_float(summary.get('branch_asymmetry'))}`",
        "- Switching complexity: "
        f"`{_summary_float(summary, 'switching_complexity')}` "
        f"(`{summary.get('switching_complexity_label')}`)",
        f"- Ambiguity flags: `{'|'.join(summary.get('ambiguity_flags', []))}`",
        "- Raw / corrected loop closure: "
        f"`{_summary_pair(summary, 'raw_loop_closure_error', 'corrected_loop_closure_error')}`",
        "",
        "## Detailed Fields",
        "",
        f"- Hc- / Hc+: `{coercive_fields}` mT",
        "- Mr+ / Mr-: "
        f"`{_summary_pair(summary, 'remanence_positive_emu', 'remanence_negative_emu')}` emu",
        f"- Ms+ / Ms-: `{saturation_moments}` emu",
        f"- Ms tail scatter: `{_summary_pair(summary, 'ms_std_pos', 'ms_std_neg')}` emu",
        "- Switching slopes: "
        f"`{_summary_pair(summary, 'switching_slope_neg', 'switching_slope_pos')}` emu/mT",
        "",
        "## Artifacts",
        "",
        f"- JSON: `{artifacts.get('json_path', '')}`",
        f"- Trace CSV: `{artifacts.get('csv_path', '')}`",
        f"- Summary CSV: `{artifacts.get('summary_csv_path', '')}`",
        f"- Figure: `{artifacts.get('figure_path', '')}`",
    ]
    warnings = summary.get("warnings", [])
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- `{warning}`" for warning in warnings)
    return "\n".join(lines) + "\n"


def _build_batch_report_text(payloads: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for payload in payloads:
        summary = payload["summary_metrics"]
        key = summary.get("series_id") or summary.get("sample_id") or "ungrouped"
        grouped.setdefault(key, []).append(payload)

    lines = [
        "# VSM Batch Report",
        "",
        f"- Measurements: `{len(payloads)}`",
        f"- Series groups: `{len(grouped)}`",
        "",
        "## Series Summary",
        "",
    ]
    for series_id in sorted(grouped):
        series_payloads = sorted(
            grouped[series_id],
            key=lambda payload: (
                float(payload["summary_metrics"].get("temperature_k") or float("inf")),
                str(payload["summary_metrics"].get("replicate_id") or ""),
            ),
        )
        lines.extend(
            [
                f"### {series_id}",
                "",
                "| Temperature (K) | Replicate | Hc +/- err (mT) | Mr +/- err (emu) | "
                "Ms +/- err (emu) | Sat. Conf. | Ambiguity |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for payload in series_payloads:
            summary = payload["summary_metrics"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        _format_float(summary.get("temperature_k")),
                        str(summary.get("replicate_id") or ""),
                        _summary_pm(summary, "Hc_mT", "hc_error"),
                        _summary_pm(summary, "Mr_emu", "mr_error"),
                        _summary_pm(summary, "Ms_emu", "ms_error"),
                        _format_float(summary.get("saturation_confidence")),
                        str(len(summary.get("ambiguity_flags", []))),
                    ]
                )
                + " |"
            )
        lines.extend(["", "Temperature trend:", ""])
        for payload in series_payloads:
            summary = payload["summary_metrics"]
            h_c = _summary_pm(summary, "Hc_mT", "hc_error")
            m_r = _summary_pm(summary, "Mr_emu", "mr_error")
            m_s = _summary_pm(summary, "Ms_emu", "ms_error")
            area = _summary_pm(summary, "loop_area_emu_mT", "loop_area_error")
            lines.append(
                f"- `{_format_float(summary.get('temperature_k'))}` K: "
                f"Hc=`{h_c}` mT, "
                f"Mr=`{m_r}` emu, "
                f"Ms=`{m_s}` emu, "
                f"area=`{area}` emu*mT, "
                f"trust=`{_format_float(summary.get('saturation_confidence'))}`"
            )
        lines.append("")
    return "\n".join(lines) + "\n"
