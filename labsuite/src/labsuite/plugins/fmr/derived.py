"""Series assembly and higher-level physics fits for FMR trace results."""

from __future__ import annotations

import math

import numpy as np
from scipy.constants import physical_constants
from scipy.optimize import curve_fit

from labsuite.core.exceptions import WorkflowError
from labsuite.core.recipes import FmrRecipe
from labsuite.plugins.fmr.branch_tracking import infer_branch_confidence
from labsuite.plugins.fmr.kittel import fit_kittel_branch, gamma_over_2pi_from_g
from labsuite.plugins.fmr.linewidth import fit_linewidth_branch
from labsuite.plugins.fmr.models import (
    FmrModelFitSummary,
    FmrPhysicsCollectionResult,
    FmrPhysicsResult,
    FmrSeriesCollectionResult,
    FmrSeriesResult,
    FmrTraceFitResult,
)
from labsuite.plugins.fmr.polarity_matching import (
    PolarityMatchConfig,
    match_positive_negative_points,
)

_HBAR = physical_constants["Planck constant over 2 pi"][0]
_MU_B = physical_constants["Bohr magneton"][0]
GONZALEZ_FUENTES_SINGLE_POLARITY_WARNING = (
    "Gonzalez-Fuentes polarity averaging requires both +H and -H sweeps for the same frequency "
    "and mode. Only one polarity was found, so raw Hres will be used."
)


def build_fmr_series(
    trace_fit_results: list[FmrTraceFitResult],
    *,
    measurement_mode: str | None = None,
    recipe: FmrRecipe | None = None,
    replicate_id: str | None = None,
    geometry: str | None = None,
) -> FmrSeriesCollectionResult:
    ordered = sorted(
        trace_fit_results,
        key=lambda item: (
            float(item.frequency_GHz),
            float("inf") if item.angle_deg is None else float(item.angle_deg),
            item.trace_id,
        ),
    )
    first = ordered[0] if ordered else None
    sample_name = first.sample_name if first is not None else "unknown"
    angle_deg = first.angle_deg if first is not None else None
    nominal_temperature_K = first.temperature_K if first is not None else None
    grouped: dict[str, list[tuple[FmrTraceFitResult, object]]] = {}
    mode_counts = {"single": 0, "double": 0, "triple": 0, "partial_double": 0}
    warnings: list[str] = []
    excluded_trace_ids = [item.trace_id for item in ordered if not item.accepted]
    for item in ordered:
        if item.selected_mode == "single":
            mode_counts["single"] += 1
        elif item.selected_mode == "double":
            mode_counts["double"] += 1
            if item.partial_component_qc:
                mode_counts["partial_double"] += 1
        elif item.selected_mode == "triple":
            mode_counts["triple"] += 1
        for component in item.selected_components:
            if not component.accepted:
                continue
            if recipe is not None and recipe.enable_branch_tracking:
                if not component.branch_id or component.confidence in {"low", "unassigned"}:
                    continue
                label = component.branch_id
            else:
                label = component.branch_id or component.component_label
            grouped.setdefault(label, []).append((item, component))
    series_by_label: dict[str, FmrSeriesResult] = {}
    for label, entries in grouped.items():
        if not entries:
            continue
        if (
            recipe is not None
            and recipe.enable_branch_tracking
            and len(entries) < recipe.kittel_min_points
        ):
            warnings.append(f"{label}:branch_skipped_insufficient_points:{len(entries)}")
            continue
        series_by_label[label] = _build_single_series(
            label,
            entries,
            measurement_mode=measurement_mode,
            excluded_trace_ids=excluded_trace_ids,
            recipe=recipe,
            replicate_id=replicate_id,
            geometry=geometry,
        )
    if not series_by_label:
        warnings.append("no_accepted_trace_components")
    for label, series in series_by_label.items():
        for warning in series.warnings:
            if warning == GONZALEZ_FUENTES_SINGLE_POLARITY_WARNING:
                warnings.append(warning)
            else:
                warnings.append(f"{label}:{warning}")
    return FmrSeriesCollectionResult(
        sample_name=sample_name,
        angle_deg=angle_deg,
        nominal_temperature_K=nominal_temperature_K,
        measurement_mode=measurement_mode,
        series_by_label=series_by_label,
        warnings=warnings,
        metadata={
            "trace_count": len(ordered),
            "excluded_trace_count": len(excluded_trace_ids),
            "mode_counts": mode_counts,
            "field_polarity_correction": _collection_correction_summary(series_by_label),
        },
    )


def fit_fmr_physics(
    series_collection: FmrSeriesCollectionResult,
    recipe: FmrRecipe,
    *,
    g_mode: str = "float",
    g_value: float | None = None,
) -> FmrPhysicsCollectionResult:
    physics_by_label: dict[str, FmrPhysicsResult] = {}
    warnings = list(series_collection.warnings)
    branch_locked_labels = set(recipe.branch_locked_g) | set(
        recipe.branch_locked_gamma_over_2pi_GHz_per_T
    )
    branch_locked_labels |= set(recipe.branch_locked_Hk_mT)
    produced_labels = set(series_collection.series_by_label)
    for label in sorted(branch_locked_labels - produced_labels):
        warnings.append(f"{label}:branch_lock_label_not_found")
    for label, series in series_collection.series_by_label.items():
        physics_by_label[label] = _fit_single_series_physics(
            series,
            recipe,
            g_mode=g_mode,
            g_value=g_value,
        )
        warnings.extend(f"{label}:{warning}" for warning in physics_by_label[label].warnings)
    return FmrPhysicsCollectionResult(
        sample_name=series_collection.sample_name,
        angle_deg=series_collection.angle_deg,
        nominal_temperature_K=series_collection.nominal_temperature_K,
        measurement_mode=series_collection.measurement_mode,
        physics_by_label=physics_by_label,
        warnings=warnings,
        metadata={"series_labels": sorted(physics_by_label), "series_count": len(physics_by_label)},
    )


def _build_single_series(
    series_label: str,
    entries: list[tuple[FmrTraceFitResult, object]],
    *,
    measurement_mode: str | None,
    excluded_trace_ids: list[str],
    recipe: FmrRecipe | None,
    replicate_id: str | None,
    geometry: str | None,
) -> FmrSeriesResult:
    ordered = sorted(entries, key=lambda item: (float(item[0].frequency_GHz), item[0].trace_id))
    first_trace = ordered[0][0]
    series = FmrSeriesResult(
        series_label=series_label,
        sample_name=first_trace.sample_name,
        angle_deg=first_trace.angle_deg,
        nominal_temperature_K=first_trace.temperature_K,
        measurement_mode=measurement_mode,
        frequency_GHz=np.asarray(
            [trace.frequency_GHz for trace, _component in ordered], dtype=float
        ),
        resonance_field_mT=np.asarray(
            [component.H_res_mT for _trace, component in ordered], dtype=float
        ),
        linewidth_mT=np.asarray(
            [component.DeltaH_mT for _trace, component in ordered], dtype=float
        ),
        amplitude_symmetric=np.asarray(
            [component.amplitude_symmetric for _trace, component in ordered], dtype=float
        ),
        amplitude_antisymmetric=np.asarray(
            [component.amplitude_antisymmetric for _trace, component in ordered], dtype=float
        ),
        resonance_field_stderr_mT=np.asarray(
            [_stderr_or_nan(component, "H_res_mT") for _trace, component in ordered], dtype=float
        ),
        linewidth_stderr_mT=np.asarray(
            [_stderr_or_nan(component, "DeltaH_mT") for _trace, component in ordered], dtype=float
        ),
        included_trace_ids=[trace.trace_id for trace, _component in ordered],
        included_component_ids=[component.component_id for _trace, component in ordered],
        excluded_trace_ids=list(excluded_trace_ids),
        warnings=[],
        metadata={
            "accepted_component_count": len(ordered),
            "accepted_trace_count": len({trace.trace_id for trace, _component in ordered}),
            "branch_id": series_label,
            "branch_confidence": infer_branch_confidence(
                [component for _trace, component in ordered]
            ),
            "polarity_points": _raw_polarity_points(
                series_label,
                ordered,
                replicate_id=replicate_id,
                geometry=geometry,
            ),
            "field_polarity_correction": {
                "enabled": False,
                "status": "disabled",
                "fit_field": "Hres",
            },
        },
    )
    if recipe is not None and recipe.field_polarity_correction.enabled:
        _apply_field_polarity_correction(series, recipe)
    return series


def _fit_single_series_physics(
    series: FmrSeriesResult,
    recipe: FmrRecipe,
    *,
    g_mode: str,
    g_value: float | None,
) -> FmrPhysicsResult:
    warnings = list(series.warnings)
    kittel_fit = None
    linewidth_fit = None
    derived_parameters: dict[str, float | None] = {
        "gamma_GHz_per_T": None,
        "gamma_over_2pi_GHz_per_T": None,
        "gamma_rad_per_s_T": None,
        "g": None,
        "M_eff_mT": None,
        "M_eff_T": None,
        "mu0_Meff_T": None,
        "mu0_Ms_apparent_T": None,
        "alpha": None,
        "alpha_eff": None,
        "DeltaH0_mT": None,
        "mu0_deltaH0_T": None,
    }
    if series.frequency_GHz.size >= recipe.kittel_min_points:
        branch_g = recipe.branch_locked_g.get(series.series_label)
        branch_gamma = recipe.branch_locked_gamma_over_2pi_GHz_per_T.get(series.series_label)
        branch_hk = recipe.branch_locked_Hk_mT.get(series.series_label)
        effective_g_value = branch_g if branch_g is not None else (g_value or 2.0)
        effective_gamma = branch_gamma
        if effective_gamma is None and effective_g_value is not None:
            effective_gamma = gamma_over_2pi_from_g(effective_g_value)
        kittel_fit = fit_kittel_branch(
            series.frequency_GHz,
            series.resonance_field_mT,
            model=recipe.physics_model,
            g_locked=effective_g_value if g_mode == "fixed" or branch_g is not None else None,
            gamma_locked_GHz_per_T=effective_gamma,
            fit_g=recipe.fit_g or recipe.fit_g_diagnostic,
            Hk_locked_mT=branch_hk,
            fit_Hk=recipe.fit_Hk,
        )
        warnings.extend(kittel_fit.warnings)
        if kittel_fit.success:
            gamma = kittel_fit.parameters["gamma_over_2pi_GHz_per_T"]
            gamma_rad = 2.0 * math.pi * gamma * 1e9
            derived_parameters["gamma_GHz_per_T"] = gamma
            derived_parameters["gamma_over_2pi_GHz_per_T"] = gamma
            derived_parameters["gamma_rad_per_s_T"] = gamma_rad
            derived_parameters["g"] = gamma_rad * _HBAR / _MU_B
            derived_parameters["M_eff_T"] = kittel_fit.parameters["M_eff_T"]
            derived_parameters["M_eff_mT"] = kittel_fit.parameters["M_eff_T"] * 1_000.0
            derived_parameters["mu0_Meff_T"] = kittel_fit.parameters["mu0_Meff_T"]
            derived_parameters["mu0_Ms_apparent_T"] = kittel_fit.parameters[
                "mu0_Ms_apparent_T"
            ]
        else:
            warnings.append("kittel_fit_failed")
    else:
        warnings.append(
            f"kittel_fit_insufficient_points:{series.frequency_GHz.size}<{recipe.kittel_min_points}"
        )
    if series.frequency_GHz.size >= recipe.linewidth_min_points:
        linewidth_fit = fit_linewidth_branch(
            series.frequency_GHz,
            series.linewidth_mT,
            gamma_over_2pi_GHz_per_T=derived_parameters["gamma_over_2pi_GHz_per_T"],
            min_points=recipe.linewidth_min_high_confidence_points,
        )
        if linewidth_fit.success:
            derived_parameters["DeltaH0_mT"] = linewidth_fit.parameters["DeltaH0_mT"]
            derived_parameters["mu0_deltaH0_T"] = linewidth_fit.parameters["mu0_deltaH0_T"]
            derived_parameters["alpha"] = linewidth_fit.parameters.get("alpha_eff")
            derived_parameters["alpha_eff"] = linewidth_fit.parameters.get("alpha_eff")
        else:
            warnings.append("linewidth_fit_failed")
    else:
        warnings.append(
            f"linewidth_fit_insufficient_points:{series.frequency_GHz.size}<{recipe.linewidth_min_points}"
        )
    warnings.append("anisotropy_K_deferred_requires_explicit_model")
    metadata = {
        "series_label": series.series_label,
        "accepted_component_count": int(series.frequency_GHz.size),
        "g_mode": g_mode,
        "g_value": g_value,
        "branch_confidence": series.metadata.get("branch_confidence"),
        "anisotropy_K": "deferred",
        "field_polarity_correction": series.metadata.get("field_polarity_correction", {}),
    }
    if (
        recipe.field_polarity_correction.enabled
        and recipe.field_polarity_correction.run_comparison_fits
    ):
        metadata["polarity_comparison_fits"] = _build_polarity_comparison_fits(
            series,
            g_mode=g_mode,
            g_value=g_value,
            min_points=recipe.kittel_min_points,
        )
    return FmrPhysicsResult(
        sample_name=series.sample_name,
        angle_deg=series.angle_deg,
        nominal_temperature_K=series.nominal_temperature_K,
        measurement_mode=series.measurement_mode,
        kittel_fit=kittel_fit,
        linewidth_fit=linewidth_fit,
        derived_parameters=derived_parameters,
        warnings=warnings,
        metadata=metadata,
    )


def _raw_polarity_points(
    series_label: str,
    entries: list[tuple[FmrTraceFitResult, object]],
    *,
    replicate_id: str | None,
    geometry: str | None,
) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for trace, component in entries:
        field_polarity = component.metadata.get("field_polarity") or trace.metadata.get(
            "field_polarity"
        )
        raw_label = component.metadata.get("field_polarity_raw") or trace.metadata.get(
            "field_polarity_raw"
        )
        points.append(
            {
                "series_label": series_label,
                "mode_id": series_label,
                "branch_id": component.branch_id or series_label,
                "sample_id": trace.sample_name,
                "replicate_id": replicate_id,
                "geometry": geometry,
                "measurement_id": trace.metadata.get("measurement_id"),
                "source_file": str(trace.source_file),
                "confidence": component.confidence,
                "frequency_GHz": float(trace.frequency_GHz),
                "field_polarity": field_polarity,
                "field_polarity_raw": raw_label,
                "polarity_pair_id": None,
                "polarity_pair_status": "raw_no_correction",
                "trace_id": trace.trace_id,
                "component_id": component.component_id,
                "Hres_raw_mT": float(component.H_res_mT),
                "Hres_raw_abs_mT": abs(float(component.H_res_mT)),
                "Hres_pos_mT": None,
                "Hres_neg_mT": None,
                "Hres_avg_mT": None,
                "Hres_offset_mT": None,
                "Hres_split_mT": None,
                "Hres_asymmetry_mT": None,
                "matched_pair_id": None,
                "matching_confidence": None,
                "DeltaH_raw_mT": float(component.DeltaH_mT),
                "DeltaH_fit_mT": float(component.DeltaH_mT),
                "fit_field": "Hres",
            }
        )
    return points


def _apply_field_polarity_correction(series: FmrSeriesResult, recipe: FmrRecipe) -> None:
    config = recipe.field_polarity_correction
    raw_points = [dict(point) for point in series.metadata.get("polarity_points", [])]
    series.metadata["raw_polarity_points"] = [dict(point) for point in raw_points]
    polarities = {
        point.get("field_polarity")
        for point in raw_points
        if point.get("field_polarity") in {"positive", "negative"}
    }
    summary = {
        "enabled": True,
        "method": config.method,
        "status": "skipped",
        "fit_field": "Hres",
        "paired_point_count": 0,
        "raw_point_count": len(raw_points),
        "warning": None,
    }
    if len(polarities) <= 1:
        series.warnings.append(GONZALEZ_FUENTES_SINGLE_POLARITY_WARNING)
        summary["status"] = "skipped_single_polarity"
        summary["warning"] = GONZALEZ_FUENTES_SINGLE_POLARITY_WARNING
        series.metadata["field_polarity_correction"] = summary
        series.metadata["polarity_points"] = raw_points
        return

    paired_points, warnings = _pair_polarity_points(raw_points, recipe)
    series.warnings.extend(warnings)
    fit_points = [
        point
        for point in paired_points
        if point.get("polarity_pair_status") == "paired"
        or config.on_unpaired == "warn_and_keep_raw"
    ]
    if config.on_unpaired == "drop":
        fit_points = [
            point for point in paired_points if point.get("polarity_pair_status") == "paired"
        ]
    if not fit_points:
        summary["status"] = "skipped_no_pairable_points"
        summary["warning"] = "field_polarity_correction_no_pairable_points"
        series.warnings.append("field_polarity_correction_no_pairable_points")
        series.metadata["field_polarity_correction"] = summary
        series.metadata["polarity_points"] = paired_points
        return

    series.frequency_GHz = np.asarray([point["frequency_GHz"] for point in fit_points], dtype=float)
    series.resonance_field_mT = np.asarray(
        [_fit_hres_value(point, config.fit_field) for point in fit_points], dtype=float
    )
    series.linewidth_mT = np.asarray(
        [point.get("DeltaH_fit_mT") for point in fit_points], dtype=float
    )
    series.included_trace_ids = [str(point.get("trace_id") or "") for point in fit_points]
    series.included_component_ids = [str(point.get("component_id") or "") for point in fit_points]
    summary["paired_point_count"] = sum(
        1 for point in paired_points if point.get("polarity_pair_status") == "paired"
    )
    summary["status"] = "applied" if summary["paired_point_count"] else "skipped_no_pairs"
    summary["fit_field"] = config.fit_field if summary["paired_point_count"] else "Hres"
    series.metadata["field_polarity_correction"] = summary
    series.metadata["polarity_points"] = paired_points


def _pair_polarity_points(
    raw_points: list[dict[str, object]],
    recipe: FmrRecipe,
) -> tuple[list[dict[str, object]], list[str]]:
    config = recipe.field_polarity_correction
    output, warnings = match_positive_negative_points(
        [dict(point) for point in raw_points],
        config=PolarityMatchConfig(
            frequency_tolerance_GHz=config.max_pair_frequency_tolerance_ghz,
            allow_low_confidence=recipe.allow_low_confidence_pos_neg_matching,
        ),
    )
    warnings = [
        warning.replace("polarity_matching_", "field_polarity_correction_")
        for warning in warnings
    ]
    if config.max_pair_hres_split_mT is not None:
        for point in output:
            if (
                point.get("polarity_pair_status") == "paired"
                and point.get("Hres_asymmetry_mT") is not None
                and float(point["Hres_asymmetry_mT"]) > config.max_pair_hres_split_mT
            ):
                point["polarity_pair_status"] = "pair_rejected_hres_split"
                point["Hres_avg_mT"] = None
                warnings.append(
                    f"field_polarity_correction_pair_rejected_hres_split:{point.get('matched_pair_id')}"
                )
    if config.on_unpaired == "fail" and any(
        point.get("polarity_pair_status") != "paired" for point in output
    ):
        raise WorkflowError("field_polarity_correction_unpaired_points")
    return sorted(
        output,
        key=lambda item: (
            float(item.get("frequency_GHz") or 0.0),
            str(item.get("component_id") or ""),
        ),
    ), warnings


def _paired_point(
    pair_id: str,
    positive: dict[str, object],
    negative: dict[str, object],
    recipe: FmrRecipe,
) -> tuple[dict[str, object], list[str]]:
    warnings: list[str] = []
    pos_abs = abs(float(positive["Hres_raw_mT"]))
    neg_abs = abs(float(negative["Hres_raw_mT"]))
    hres_avg = (pos_abs + neg_abs) / 2.0
    hres_offset = (pos_abs - neg_abs) / 2.0
    hres_split = abs(pos_abs - neg_abs)
    status = "paired"
    if (
        recipe.field_polarity_correction.max_pair_hres_split_mT is not None
        and hres_split > recipe.field_polarity_correction.max_pair_hres_split_mT
    ):
        status = "pair_rejected_hres_split"
        warnings.append(f"field_polarity_correction_pair_rejected_hres_split:{pair_id}")
    point = dict(positive)
    point.update(
        {
            "frequency_GHz": float(
                np.mean([float(positive["frequency_GHz"]), float(negative["frequency_GHz"])])
            ),
            "field_polarity": "paired",
            "field_polarity_raw": (
                f"{positive.get('field_polarity_raw')}|{negative.get('field_polarity_raw')}"
            ),
            "polarity_pair_id": pair_id,
            "polarity_pair_status": status,
            "trace_id": f"{positive.get('trace_id')}|{negative.get('trace_id')}",
            "component_id": f"{positive.get('component_id')}|{negative.get('component_id')}",
            "Hres_pos_mT": float(positive["Hres_raw_mT"]),
            "Hres_neg_mT": float(negative["Hres_raw_mT"]),
            "Hres_avg_mT": hres_avg if status == "paired" else None,
            "Hres_offset_mT": hres_offset if status == "paired" else None,
            "Hres_split_mT": hres_split,
            "DeltaH_fit_mT": float(
                np.mean([float(positive["DeltaH_raw_mT"]), float(negative["DeltaH_raw_mT"])])
            ),
            "fit_field": recipe.field_polarity_correction.fit_field
            if status == "paired"
            else "Hres",
        }
    )
    if status != "paired":
        point = _unpaired_point(point, status, recipe)
    return point, warnings


def _unpaired_point(
    point: dict[str, object],
    status: str,
    recipe: FmrRecipe,
) -> dict[str, object]:
    item = dict(point)
    item["polarity_pair_status"] = status
    item["fit_field"] = "Hres"
    if recipe.field_polarity_correction.on_unpaired == "drop":
        item["fit_field"] = "dropped"
    return item


def _fit_hres_value(point: dict[str, object], fit_field: str) -> float:
    if point.get("polarity_pair_status") == "paired":
        if fit_field == "Hres_avg" and point.get("Hres_avg_mT") is not None:
            return float(point["Hres_avg_mT"])
        if fit_field == "Hres_pos" and point.get("Hres_pos_mT") is not None:
            return abs(float(point["Hres_pos_mT"]))
        if fit_field == "Hres_neg" and point.get("Hres_neg_mT") is not None:
            return abs(float(point["Hres_neg_mT"]))
    return float(point["Hres_raw_mT"])


def _build_polarity_comparison_fits(
    series: FmrSeriesResult,
    *,
    g_mode: str,
    g_value: float | None,
    min_points: int,
) -> dict[str, object]:
    corrected_points = list(series.metadata.get("polarity_points", []))
    raw_points = list(series.metadata.get("raw_polarity_points", corrected_points))
    comparisons: dict[str, object] = {}
    specs = (
        ("raw", raw_points, "Hres_raw_mT", None),
        ("positive", raw_points, "Hres_raw_mT", "positive"),
        ("negative", raw_points, "Hres_raw_mT", "negative"),
        ("corrected", corrected_points, "Hres_avg_mT", None),
    )
    for label, source_points, field_name, polarity in specs:
        rows = [
            point
            for point in source_points
            if point.get(field_name) is not None
            and (polarity is None or point.get("field_polarity") == polarity)
        ]
        if len(rows) < min_points:
            comparisons[label] = {
                "success": False,
                "message": f"insufficient_points:{len(rows)}<{min_points}",
            }
            continue
        fit = _fit_kittel(
            np.asarray([float(point["frequency_GHz"]) for point in rows], dtype=float),
            np.asarray([abs(float(point[field_name])) for point in rows], dtype=float),
            g_mode=g_mode,
            g_value=g_value,
        )
        comparisons[label] = {
            "success": fit.success,
            "message": fit.message,
            "parameters": fit.parameters,
            "metrics": fit.metrics,
            "warnings": fit.warnings,
        }
    return comparisons


def _collection_correction_summary(
    series_by_label: dict[str, FmrSeriesResult],
) -> dict[str, object]:
    summaries = [
        series.metadata.get("field_polarity_correction", {})
        for series in series_by_label.values()
        if series.metadata.get("field_polarity_correction")
    ]
    enabled = any(bool(item.get("enabled")) for item in summaries)
    paired = sum(int(item.get("paired_point_count") or 0) for item in summaries)
    statuses = sorted({str(item.get("status")) for item in summaries if item.get("status")})
    return {"enabled": enabled, "paired_point_count": paired, "statuses": statuses}


def _fit_kittel(
    frequency_GHz: np.ndarray,
    resonance_field_mT: np.ndarray,
    *,
    g_mode: str,
    g_value: float | None,
) -> FmrModelFitSummary:
    resonance_field_T = np.asarray(resonance_field_mT, dtype=float) / 1_000.0
    warnings: list[str] = []
    normalized_mode = g_mode if g_mode in {"float", "fixed", "bounded"} else "float"
    if normalized_mode != g_mode:
        warnings.append(f"unknown_g_mode_using_float:{g_mode}")
    gamma_from_g = None if g_value is None else _gamma_ghz_per_t_from_g(float(g_value))
    try:
        if normalized_mode == "fixed" and gamma_from_g is not None:
            params, covariance = curve_fit(
                lambda field_t, m_eff_t: _ip_field_swept_kittel(field_t, gamma_from_g, m_eff_t),
                resonance_field_T,
                np.asarray(frequency_GHz, dtype=float),
                p0=(max(float(np.nanmax(resonance_field_T)), 0.1),),
                bounds=((0.0,), (10.0,)),
                maxfev=20000,
            )
            m_eff_t = float(params[0])
            fitted_y = _ip_field_swept_kittel(resonance_field_T, gamma_from_g, m_eff_t)
            return FmrModelFitSummary(
                model_name="ip_field_swept_kittel_gamma_fixed",
                success=True,
                message="fit converged",
                parameters={"gamma_GHz_per_T": float(gamma_from_g), "M_eff_T": m_eff_t},
                stderr={"gamma_GHz_per_T": None, "M_eff_T": _single_stderr(covariance)},
                metrics=_fit_metrics(
                    np.asarray(frequency_GHz, dtype=float), np.asarray(fitted_y, dtype=float)
                ),
                x=resonance_field_mT.tolist(),
                y=frequency_GHz.tolist(),
                fitted_y=np.asarray(fitted_y, dtype=float).tolist(),
                warnings=warnings,
            )
        if normalized_mode == "fixed":
            warnings.append("fixed_g_mode_requires_g_value_using_float")
        if normalized_mode == "bounded" and gamma_from_g is None:
            warnings.append("bounded_g_mode_requires_g_value_using_float")
            normalized_mode = "float"
        if normalized_mode == "bounded" and gamma_from_g is not None:
            gamma_lower = gamma_from_g * 0.90
            gamma_upper = gamma_from_g * 1.10
            p0 = (gamma_from_g, max(float(np.nanmax(resonance_field_T)), 0.1))
            bounds = ((gamma_lower, 0.0), (gamma_upper, 10.0))
        else:
            p0 = (28.0, max(float(np.nanmax(resonance_field_T)), 0.1))
            bounds = ((1.0, 0.0), (80.0, 10.0))
        params, covariance = curve_fit(
            _ip_field_swept_kittel,
            resonance_field_T,
            np.asarray(frequency_GHz, dtype=float),
            p0=p0,
            bounds=bounds,
            maxfev=20000,
        )
    except RuntimeError as exc:
        return FmrModelFitSummary(
            model_name="ip_field_swept_kittel",
            success=False,
            message=str(exc),
            x=resonance_field_mT.tolist(),
            y=frequency_GHz.tolist(),
            warnings=warnings,
        )
    fitted_y = _ip_field_swept_kittel(resonance_field_T, *params)
    model_name = (
        "ip_field_swept_kittel_gamma_bounded"
        if normalized_mode == "bounded"
        else "ip_field_swept_kittel"
    )
    return FmrModelFitSummary(
        model_name=model_name,
        success=True,
        message="fit converged",
        parameters={"gamma_GHz_per_T": float(params[0]), "M_eff_T": float(params[1])},
        stderr=_stderr_dict(["gamma_GHz_per_T", "M_eff_T"], covariance),
        metrics=_fit_metrics(
            np.asarray(frequency_GHz, dtype=float), np.asarray(fitted_y, dtype=float)
        ),
        x=resonance_field_mT.tolist(),
        y=frequency_GHz.tolist(),
        fitted_y=np.asarray(fitted_y, dtype=float).tolist(),
        warnings=warnings,
    )


def _fit_linewidth(frequency_GHz: np.ndarray, linewidth_mT: np.ndarray) -> FmrModelFitSummary:
    try:
        params, covariance = curve_fit(
            _line_width_model,
            np.asarray(frequency_GHz, dtype=float),
            np.asarray(linewidth_mT, dtype=float),
            p0=(float(np.nanmin(linewidth_mT)), 0.1),
            maxfev=20000,
        )
    except RuntimeError as exc:
        return FmrModelFitSummary(
            model_name="linewidth_vs_frequency_linear",
            success=False,
            message=str(exc),
            x=frequency_GHz.tolist(),
            y=linewidth_mT.tolist(),
        )
    fitted_y = _line_width_model(np.asarray(frequency_GHz, dtype=float), *params)
    return FmrModelFitSummary(
        model_name="linewidth_vs_frequency_linear",
        success=True,
        message="fit converged",
        parameters={"DeltaH0_mT": float(params[0]), "slope_mT_per_GHz": float(params[1])},
        stderr=_stderr_dict(["DeltaH0_mT", "slope_mT_per_GHz"], covariance),
        metrics=_fit_metrics(
            np.asarray(linewidth_mT, dtype=float), np.asarray(fitted_y, dtype=float)
        ),
        x=frequency_GHz.tolist(),
        y=linewidth_mT.tolist(),
        fitted_y=np.asarray(fitted_y, dtype=float).tolist(),
    )


def _ip_field_swept_kittel(
    resonance_field_T: np.ndarray, gamma_GHz_per_T: float, M_eff_T: float
) -> np.ndarray:
    return gamma_GHz_per_T * np.sqrt(resonance_field_T * (resonance_field_T + M_eff_T))


def _gamma_ghz_per_t_from_g(g_value: float) -> float:
    return float(g_value * _MU_B / (2.0 * math.pi * _HBAR) / 1e9)


def _single_stderr(covariance: np.ndarray | None) -> float | None:
    if covariance is None:
        return None
    diagonal = np.sqrt(np.diag(np.asarray(covariance, dtype=float)))
    if diagonal.size == 0:
        return None
    return float(diagonal[0])


def _line_width_model(
    frequency_GHz: np.ndarray, DeltaH0_mT: float, slope_mT_per_GHz: float
) -> np.ndarray:
    return DeltaH0_mT + slope_mT_per_GHz * frequency_GHz


def _stderr_or_nan(component, parameter_name: str) -> float:
    diagnostic = component.parameter_diagnostics.get(parameter_name)
    return (
        float("nan")
        if diagnostic is None or diagnostic.stderr is None
        else float(diagnostic.stderr)
    )


def _stderr_dict(names: list[str], covariance: np.ndarray | None) -> dict[str, float | None]:
    if covariance is None:
        return {name: None for name in names}
    diagonal = np.sqrt(np.diag(np.asarray(covariance, dtype=float)))
    return {name: float(diagonal[index]) for index, name in enumerate(names)}


def _fit_metrics(y_true: np.ndarray, y_fit: np.ndarray) -> dict[str, float]:
    residual = y_true - y_fit
    rss = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "rss": rss,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r_squared": 1.0 if ss_tot == 0.0 else 1.0 - (rss / ss_tot),
    }
