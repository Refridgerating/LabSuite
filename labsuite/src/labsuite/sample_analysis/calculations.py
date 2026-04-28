"""Scientific calculations for sample-level derived analysis."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.constants import physical_constants
from scipy.optimize import curve_fit

from labsuite.core.sample_registry import SampleRecord, resolve_sample_magnetic_volume
from labsuite.sample_analysis.manifest import ProcessedInput
from labsuite.sample_analysis.recipe import SampleAnalysisRecipe

MU0 = 4.0 * math.pi * 1e-7
HBAR = physical_constants["Planck constant over 2 pi"][0]
H_PLANCK = physical_constants["Planck constant"][0]
MU_B = physical_constants["Bohr magneton"][0]


def warning(code: str, message: str, **context: Any) -> dict[str, Any]:
    return {"code": code, "message": message, **context}


def derive_vsm_parameters(
    sample: SampleRecord,
    inputs: list[ProcessedInput],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    volume = _resolve_magnetic_volume(sample)
    if volume is None:
        warnings.append(
            warning("MISSING_VMAG", "Magnetic volume is required to convert VSM moment to Ms.")
        )
    elif volume["estimated"]:
        warnings.append(
            warning(
                "VMAG_ESTIMATED", "Magnetic volume was estimated from area and magnetic thickness."
            )
        )
    if volume is not None:
        for message in volume["warnings"]:
            warnings.append(warning("VMAG_WARNING", str(message)))
    for item in inputs:
        if item.modality != "vsm" or item.status != "usable" or item.payload is None:
            continue
        summary = item.payload.get("summary_metrics", {})
        ms_emu = _float_or_none(summary.get("Ms_emu"))
        ms_error_emu = _float_or_none(summary.get("ms_error"))
        if ms_emu is None:
            warnings.append(
                warning(
                    "MISSING_MS",
                    "Processed VSM result does not contain Ms_emu.",
                    measurement_id=item.measurement_id,
                )
            )
            continue
        row = {
            "sample_id": sample.sample_id,
            "measurement_id": item.measurement_id,
            "processed_json_path": str(item.processed_json_path),
            "Ms_emu": ms_emu,
            "Ms_emu_uncertainty": ms_error_emu,
            "magnetic_volume_m3": None if volume is None else volume["value_m3"],
            "magnetic_volume_uncertainty_m3": None if volume is None else volume["uncertainty_m3"],
            "magnetic_volume_source": None if volume is None else volume["source"],
            "Ms_A_per_m": None,
            "Ms_uncertainty_A_per_m": None,
        }
        if volume is not None:
            moment_a_m2 = ms_emu * 1e-3
            ms_a_per_m = moment_a_m2 / volume["value_m3"]
            rel_terms = []
            if ms_error_emu is not None and ms_emu != 0.0:
                rel_terms.append(abs(ms_error_emu / ms_emu))
            if volume["uncertainty_m3"] is not None and volume["value_m3"] != 0.0:
                rel_terms.append(abs(volume["uncertainty_m3"] / volume["value_m3"]))
            row["Ms_A_per_m"] = ms_a_per_m
            row["Ms_uncertainty_A_per_m"] = (
                None
                if not rel_terms
                else abs(ms_a_per_m) * math.sqrt(sum(term * term for term in rel_terms))
            )
        rows.append(row)
    if not rows:
        warnings.append(warning("MISSING_MS", "No usable processed VSM Ms result was found."))
    selected = next((row for row in rows if row.get("Ms_A_per_m") is not None), None)
    return rows, warnings, {"selected_ms": selected, "volume": volume}


def derive_fmr_parameters(
    sample: SampleRecord,
    inputs: list[ProcessedInput],
    recipe: SampleAnalysisRecipe,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for item in inputs:
        if item.modality != "fmr" or item.status != "usable" or item.payload is None:
            continue
        rows.extend(_extract_fmr_points(item, recipe, warnings))
        rows.extend(_extract_processed_fmr_physics(sample, item, recipe, warnings))
    meff_rows = [row for row in rows if row.get("fit_kind") == "kittel" and row.get("success")]
    return rows, warnings, {"kittel_rows": meff_rows, "damping_points": rows}


def derive_anisotropy_parameters(
    sample_id: str,
    selected_ms: dict[str, Any] | None,
    kittel_rows: list[dict[str, Any]],
    recipe: SampleAnalysisRecipe,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not recipe.anisotropy_enabled:
        return [], []
    if selected_ms is None or selected_ms.get("Ms_A_per_m") is None:
        return [], [warning("MISSING_MS", "Ms is required for anisotropy estimates.")]
    ms = float(selected_ms["Ms_A_per_m"])
    rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    canonical = [row for row in kittel_rows if row.get("result_variant") == "canonical"]
    for row in canonical:
        meff_t = _float_or_none(row.get("Meff_T"))
        if meff_t is None:
            continue
        meff_a_per_m = meff_t / MU0
        rows.append(
            {
                "sample_id": sample_id,
                "parameter": "K_perp",
                "value_J_per_m3": 0.5 * MU0 * ms * (ms - meff_a_per_m),
                "geometry": row.get("geometry"),
                "branch_label": row.get("branch_label"),
                "sign_convention": "positive when Ms exceeds Meff in K_perp = 0.5*mu0*Ms*(Ms-Meff)",
            }
        )
    ip = next((row for row in canonical if row.get("geometry") == "ip"), None)
    oop = next((row for row in canonical if row.get("geometry") == "oop"), None)
    if ip is not None and oop is not None:
        meff_ip = float(ip["Meff_T"]) / MU0
        meff_oop = float(oop["Meff_T"]) / MU0
        rows.extend(
            [
                {
                    "sample_id": sample_id,
                    "parameter": "K2_eff",
                    "value_J_per_m3": 0.5 * MU0 * ms * meff_ip,
                    "geometry": "ip_oop",
                    "branch_label": ip.get("branch_label"),
                    "sign_convention": "K2_eff = 0.5*mu0*Ms*Meff_ip",
                },
                {
                    "sample_id": sample_id,
                    "parameter": "K4_eff",
                    "value_J_per_m3": 0.5 * MU0 * ms * (meff_oop - meff_ip),
                    "geometry": "ip_oop",
                    "branch_label": ip.get("branch_label"),
                    "sign_convention": "K4_eff = 0.5*mu0*Ms*(Meff_oop-Meff_ip)",
                },
            ]
        )
        denom = max(abs(meff_ip), abs(meff_oop), 1e-12)
        if abs(meff_oop - meff_ip) / denom >= recipe.thresholds.meff_mismatch_relative:
            warnings.append(
                warning("IP_OOP_MEFF_MISMATCH", "IP and OOP Meff differ beyond threshold.")
            )
            warnings.append(
                warning(
                    "K4_POSSIBLE",
                    "IP/OOP Meff mismatch suggests fourth-order anisotropy may be needed.",
                )
            )
    return rows, warnings


def derive_damping_parameters(
    sample_id: str,
    fmr_rows: list[dict[str, Any]],
    kittel_rows: list[dict[str, Any]],
    recipe: SampleAnalysisRecipe,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not recipe.damping_enabled:
        return [], []
    rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    linewidth_rows = [
        row for row in fmr_rows if row.get("fit_kind") == "linewidth" and row.get("success")
    ]
    has_oop = any(row.get("geometry") == "oop" for row in linewidth_rows)
    for row in linewidth_rows:
        geometry = str(row.get("geometry") or "unknown")
        branch = row.get("branch_label")
        if geometry == "ip":
            warnings.append(
                warning(
                    "IP_DAMPING_MAY_INCLUDE_TWO_MAGNON",
                    "IP linewidth may include two-magnon scattering.",
                    branch_label=branch,
                )
            )
        if (
            row.get("r_squared") is not None
            and float(row["r_squared"]) < recipe.thresholds.linewidth_nonlinear_r2_min
        ):
            warnings.append(
                warning(
                    "LINEWIDTH_NONLINEAR",
                    "Linewidth vs frequency is below linearity threshold.",
                    branch_label=branch,
                )
            )
        rows.append(
            {
                "sample_id": sample_id,
                "geometry": geometry,
                "branch_label": branch,
                "point_count": row.get("point_count"),
                "frequency_min_GHz": row.get("frequency_min_GHz"),
                "frequency_max_GHz": row.get("frequency_max_GHz"),
                "DeltaH0_mT": row.get("DeltaH0_mT"),
                "slope_mT_per_GHz": row.get("slope_mT_per_GHz"),
                "alpha_eff": row.get("alpha_eff"),
                "r_squared": row.get("r_squared"),
                "rmse_mT": row.get("rmse_mT"),
                "success": row.get("success"),
                "preferred_for_alpha": geometry == "oop" or (geometry == "ip" and not has_oop),
            }
        )
    return rows, warnings


def derive_esr_parameters(
    sample_id: str,
    inputs: list[ProcessedInput],
    recipe: SampleAnalysisRecipe,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not recipe.esr_enabled:
        return [], []
    rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for item in inputs:
        if item.modality != "esr" or item.status != "usable" or item.payload is None:
            continue
        payload = item.payload
        freq = _float_or_none((payload.get("metadata") or {}).get("frequency_GHz"))
        fits: list[tuple[str, dict[str, Any]]] = []
        single = (payload.get("fit_selection") or {}).get("single_fit")
        if isinstance(single, dict):
            fits.append(("selected", single))
        for peak in (payload.get("fit_selection") or {}).get("peak_fits") or []:
            if isinstance(peak, dict) and isinstance(peak.get("fit"), dict):
                fits.append((str(peak.get("label") or "peak"), peak["fit"]))
        for label, fit in fits:
            params = fit.get("parameters") or {}
            hres_m_t = _float_or_none(params.get("center_mT"))
            linewidth_m_t = _float_or_none(params.get("gamma_mT"))
            g_eff = None
            if freq is not None and hres_m_t not in {None, 0.0}:
                g_eff = H_PLANCK * freq * 1e9 / (MU_B * abs(float(hres_m_t)) * 1e-3)
            rows.append(
                {
                    "sample_id": sample_id,
                    "measurement_id": item.measurement_id,
                    "geometry": item.geometry,
                    "branch_label": label,
                    "frequency_GHz": freq,
                    "Hres_mT": hres_m_t,
                    "linewidth_mT": linewidth_m_t,
                    "g_eff": g_eff,
                    "fit_success": fit.get("success"),
                    "processed_json_path": str(item.processed_json_path),
                }
            )
        warnings.append(
            warning(
                "ESR_G_EFFECTIVE_ONLY",
                "ESR g is reported as effective g only.",
                measurement_id=item.measurement_id,
            )
        )
    return rows, warnings


def classify_readiness(
    usable_input_count: int,
    vsm_rows: list[dict[str, Any]],
    kittel_rows: list[dict[str, Any]],
    anisotropy_rows: list[dict[str, Any]],
    damping_rows: list[dict[str, Any]],
) -> tuple[str, dict[str, bool]]:
    has_ms = any(row.get("Ms_A_per_m") is not None for row in vsm_rows)
    has_meff = any(row.get("success") and row.get("Meff_T") is not None for row in kittel_rows)
    has_anis = bool(anisotropy_rows)
    has_damping = any(
        row.get("success") and row.get("alpha_eff") is not None for row in damping_rows
    )
    matrix = {
        "READY_BASIC": usable_input_count > 0,
        "READY_MEFF_ONLY": has_meff,
        "READY_MEFF_PLUS_MS": has_meff and has_ms,
        "READY_ANISOTROPY": has_anis,
        "READY_DAMPING": has_damping,
        "READY_SMIT_BELJERS": False,
        "INSUFFICIENT_DATA": not (
            usable_input_count > 0 and (has_ms or has_meff or has_anis or has_damping)
        ),
    }
    for label in (
        "READY_DAMPING",
        "READY_ANISOTROPY",
        "READY_MEFF_PLUS_MS",
        "READY_MEFF_ONLY",
        "READY_BASIC",
    ):
        if matrix[label]:
            return label, matrix
    return "INSUFFICIENT_DATA", matrix


def _extract_fmr_points(
    item: ProcessedInput,
    recipe: SampleAnalysisRecipe,
    warnings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    payload = item.payload or {}
    collection = (payload.get("analysis_payload") or {}).get("series_collection_result") or {}
    rows: list[dict[str, Any]] = []
    for branch_label, series in (collection.get("series_by_label") or {}).items():
        polarity_by_component = {
            point.get("component_id"): point
            for point in (series.get("metadata") or {}).get("polarity_points", [])
        }
        freqs = series.get("frequency_GHz") or []
        hres = series.get("resonance_field_mT") or []
        linewidth = series.get("linewidth_mT") or []
        trace_ids = series.get("included_trace_ids") or []
        component_ids = series.get("included_component_ids") or []
        for index, frequency in enumerate(freqs):
            component_id = component_ids[index] if index < len(component_ids) else ""
            point = polarity_by_component.get(component_id, {})
            raw_hres = _float_or_none(point.get("Hres_raw_mT"))
            if raw_hres is None:
                raw_hres = _float_or_none(hres[index] if index < len(hres) else None)
            avg_hres = _float_or_none(point.get("Hres_avg_mT"))
            offset = _float_or_none(point.get("Hres_offset_mT"))
            paired = point.get("polarity_pair_status") == "paired" and avg_hres is not None
            canonical_hres = (
                avg_hres if recipe.positive_negative_pairing_enabled and paired else raw_hres
            )
            if recipe.positive_negative_pairing_enabled and point and not paired:
                warnings.append(
                    warning(
                        "NO_POS_NEG_PAIRING",
                        "No usable positive/negative Hres pair.",
                        measurement_id=item.measurement_id,
                        branch_label=branch_label,
                    )
                )
            if offset is not None and abs(offset) >= recipe.thresholds.field_offset_detected_mT:
                warnings.append(
                    warning(
                        "FIELD_OFFSET_DETECTED",
                        "Positive/negative Hres pair indicates field offset.",
                        field_offset_mT=offset,
                        measurement_id=item.measurement_id,
                        branch_label=branch_label,
                    )
                )
            common = {
                "row_type": "series_point",
                "sample_id": item.sample_id,
                "measurement_id": item.measurement_id,
                "geometry": item.geometry,
                "branch_label": branch_label,
                "frequency_GHz": float(frequency),
                "linewidth_mT": _float_or_none(
                    linewidth[index] if index < len(linewidth) else None
                ),
                "trace_id": trace_ids[index] if index < len(trace_ids) else "",
                "component_id": component_id,
                "processed_json_path": str(item.processed_json_path),
                "polarity_pair_status": point.get("polarity_pair_status"),
                "field_offset_mT": offset,
            }
            if canonical_hres is not None:
                rows.append(
                    {**common, "field_variant": "canonical", "Hres_mT": abs(canonical_hres)}
                )
            if raw_hres is not None:
                rows.append({**common, "field_variant": "raw", "Hres_mT": abs(raw_hres)})
            if avg_hres is not None:
                rows.append({**common, "field_variant": "averaged", "Hres_mT": abs(avg_hres)})
    return rows


def _extract_processed_fmr_physics(
    sample: SampleRecord,
    item: ProcessedInput,
    recipe: SampleAnalysisRecipe,
    warnings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    payload = item.payload or {}
    summary = (
        payload.get("summary_metrics") if isinstance(payload.get("summary_metrics"), dict) else {}
    )
    collection = (payload.get("analysis_payload") or {}).get("physics_collection_result") or {}
    physics_by_label = collection.get("physics_by_label") or {}
    series_by_label = (
        (payload.get("analysis_payload") or {}).get("series_collection_result") or {}
    ).get("series_by_label") or {}
    rows: list[dict[str, Any]] = []
    field_pair_count = _float_or_none(summary.get("field_polarity_pair_count"))
    if field_pair_count in {None, 0.0}:
        warnings.append(
            warning(
                "NO_POS_NEG_PAIRING",
                "No processed positive/negative field pairing was recorded.",
                measurement_id=item.measurement_id,
            )
        )
    correction = summary.get("field_polarity_correction")
    if isinstance(correction, dict):
        offsets = correction.get("offsets_mT") or correction.get("field_offsets_mT") or []
        if any(
            abs(float(value)) >= recipe.thresholds.field_offset_detected_mT
            for value in offsets
            if _float_or_none(value) is not None
        ):
            warnings.append(
                warning(
                    "FIELD_OFFSET_DETECTED",
                    "Processed FMR result reports a field offset.",
                    measurement_id=item.measurement_id,
                )
            )
    for label, physics in physics_by_label.items():
        if not isinstance(physics, dict):
            continue
        series = series_by_label.get(label) if isinstance(series_by_label.get(label), dict) else {}
        geometry = str(
            item.geometry
            or (
                item.processed_record.summary.get("geometry")
                if item.processed_record
                else "unknown"
            )
        )
        kittel = physics.get("kittel_fit") if isinstance(physics.get("kittel_fit"), dict) else {}
        physics_derived = (
            physics.get("derived_parameters")
            if isinstance(physics.get("derived_parameters"), dict)
            else {}
        )
        if kittel:
            metrics = kittel.get("metrics") if isinstance(kittel.get("metrics"), dict) else {}
            meff_t = _meff_t_from_current_schema(physics_derived)
            rows.append(
                {
                    "row_type": "fit",
                    "fit_kind": "kittel",
                    "sample_id": sample.sample_id,
                    "measurement_id": item.measurement_id,
                    "geometry": geometry,
                    "branch_label": str(label),
                    "result_variant": "canonical",
                    "point_count": len(series.get("frequency_GHz") or []),
                    "frequency_min_GHz": _min_or_none(series.get("frequency_GHz") or []),
                    "frequency_max_GHz": _max_or_none(series.get("frequency_GHz") or []),
                    "success": bool(kittel.get("success")),
                    "gamma_prime_GHz_per_T": _first_number(
                        physics_derived, names=("gamma_GHz_per_T", "gamma_prime_GHz_per_T")
                    ),
                    "g": _first_number(physics_derived, names=("g",)),
                    "Meff_mT": None if meff_t is None else meff_t * 1000.0,
                    "Meff_T": meff_t,
                    "r_squared": metrics.get("r_squared"),
                    "rmse_GHz": metrics.get("rmse") or metrics.get("rmse_GHz"),
                    "processed_json_path": str(item.processed_json_path),
                }
            )
        linewidth = (
            physics.get("linewidth_fit") if isinstance(physics.get("linewidth_fit"), dict) else {}
        )
        if linewidth:
            params = (
                linewidth.get("parameters") if isinstance(linewidth.get("parameters"), dict) else {}
            )
            metrics = linewidth.get("metrics") if isinstance(linewidth.get("metrics"), dict) else {}
            rows.append(
                {
                    "row_type": "fit",
                    "fit_kind": "linewidth",
                    "sample_id": sample.sample_id,
                    "measurement_id": item.measurement_id,
                    "geometry": geometry,
                    "branch_label": str(label),
                    "point_count": len(series.get("frequency_GHz") or []),
                    "frequency_min_GHz": _min_or_none(series.get("frequency_GHz") or []),
                    "frequency_max_GHz": _max_or_none(series.get("frequency_GHz") or []),
                    "success": bool(linewidth.get("success")),
                    "DeltaH0_mT": _first_number(
                        physics_derived, params, names=("DeltaH0_mT", "intercept_mT")
                    ),
                    "slope_mT_per_GHz": _first_number(params, names=("slope_mT_per_GHz",)),
                    "alpha_eff": _first_number(physics_derived, names=("alpha",)),
                    "r_squared": metrics.get("r_squared"),
                    "rmse_mT": metrics.get("rmse") or metrics.get("rmse_mT"),
                    "processed_json_path": str(item.processed_json_path),
                }
            )
    g_mode = summary.get("g_mode") or sample.defaults.g_mode
    if g_mode == "fixed":
        warnings.append(
            warning(
                "G_LOCKED_BY_REGISTRY",
                "FMR g was fixed by processed/registry metadata.",
                measurement_id=item.measurement_id,
            )
        )
    if g_mode == "bounded":
        warnings.append(
            warning(
                "G_BOUNDED_BY_REGISTRY",
                "FMR g was bounded by processed/registry metadata.",
                measurement_id=item.measurement_id,
            )
        )
    return rows


def _first_number(*mappings: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for mapping in mappings:
        for name in names:
            value = _float_or_none(mapping.get(name))
            if value is not None:
                return value
    return None


def _meff_t_from_current_schema(derived: dict[str, Any]) -> float | None:
    meff_t = _first_number(derived, names=("M_eff_T", "Meff_T"))
    if meff_t is not None:
        return meff_t
    meff_m_t = _first_number(derived, names=("M_eff_mT", "Meff_mT"))
    return None if meff_m_t is None else meff_m_t / 1000.0


def _min_or_none(values: list[Any]) -> float | None:
    numeric = [_float_or_none(value) for value in values]
    numeric = [value for value in numeric if value is not None]
    return None if not numeric else min(numeric)


def _max_or_none(values: list[Any]) -> float | None:
    numeric = [_float_or_none(value) for value in values]
    numeric = [value for value in numeric if value is not None]
    return None if not numeric else max(numeric)


def _fit_fmr_group(
    sample: SampleRecord,
    geometry: str,
    branch_label: str,
    points: list[dict[str, Any]],
    recipe: SampleAnalysisRecipe,
    warnings: list[dict[str, Any]],
    *,
    result_variant: str = "canonical",
) -> list[dict[str, Any]]:
    if len(points) < recipe.thresholds.min_kittel_points:
        return []
    frequency = np.asarray([row["frequency_GHz"] for row in points], dtype=float)
    hres = np.asarray([row["Hres_mT"] for row in points], dtype=float)
    if (
        float(np.nanmax(frequency) - np.nanmin(frequency))
        < recipe.thresholds.min_frequency_span_GHz
    ):
        warnings.append(
            warning(
                "LOW_FREQUENCY_SPAN",
                "FMR frequency span is below threshold.",
                geometry=geometry,
                branch_label=branch_label,
            )
        )
    g_mode = (
        sample.defaults.g_mode if recipe.fmr_g_mode == "inherit_registry" else recipe.fmr_g_mode
    )
    fit = _fit_kittel(
        frequency, hres, geometry=geometry, g_mode=g_mode, g_value=sample.defaults.g_value
    )
    if g_mode == "fixed":
        warnings.append(
            warning(
                "G_LOCKED_BY_REGISTRY",
                "FMR g was fixed from registry defaults.",
                geometry=geometry,
                branch_label=branch_label,
            )
        )
    if g_mode == "bounded":
        warnings.append(
            warning(
                "G_BOUNDED_BY_REGISTRY",
                "FMR g was bounded from registry defaults.",
                geometry=geometry,
                branch_label=branch_label,
            )
        )
    row = {
        "row_type": "fit",
        "fit_kind": "kittel",
        "sample_id": sample.sample_id,
        "geometry": geometry,
        "branch_label": branch_label,
        "result_variant": result_variant,
        "point_count": len(points),
        "frequency_min_GHz": float(np.nanmin(frequency)),
        "frequency_max_GHz": float(np.nanmax(frequency)),
        **fit,
    }
    if (
        g_mode == "float"
        and fit.get("g_stderr") is not None
        and fit.get("g") not in {None, 0.0}
        and abs(float(fit["g_stderr"]) / float(fit["g"]))
        > recipe.thresholds.g_float_relative_stderr_max
    ):
        warnings.append(
            warning(
                "G_FLOAT_UNSTABLE",
                "Floating g uncertainty exceeds threshold.",
                geometry=geometry,
                branch_label=branch_label,
            )
        )
    return [row]


def _fit_kittel(
    frequency_GHz: np.ndarray,
    hres_mT: np.ndarray,
    *,
    geometry: str,
    g_mode: str,
    g_value: float | None,
) -> dict[str, Any]:
    hres_t = np.asarray(hres_mT, dtype=float) / 1000.0
    freq = np.asarray(frequency_GHz, dtype=float)
    gamma_from_g = None if g_value is None else _gamma_from_g(g_value)
    normalized = g_mode if g_mode in {"float", "fixed", "bounded"} else "float"
    try:
        if normalized == "fixed" and gamma_from_g is not None:
            params, covariance = curve_fit(
                lambda field_t, meff_t: _kittel_model(field_t, gamma_from_g, meff_t, geometry),
                hres_t,
                freq,
                p0=(max(float(np.nanmedian(hres_t)), 0.01),),
                bounds=((-10.0,), (10.0,)),
                maxfev=20000,
            )
            gamma = gamma_from_g
            meff = float(params[0])
            stderr = {"gamma_prime_GHz_per_T": None, "Meff_T": _stderr(covariance, 0)}
        else:
            if normalized == "bounded" and gamma_from_g is not None:
                bounds = ((gamma_from_g * 0.90, -10.0), (gamma_from_g * 1.10, 10.0))
                p0 = (gamma_from_g, max(float(np.nanmedian(hres_t)), 0.01))
            else:
                bounds = ((1.0, -10.0), (80.0, 10.0))
                p0 = (28.0, max(float(np.nanmedian(hres_t)), 0.01))
            params, covariance = curve_fit(
                lambda field_t, gamma_value, meff_t: _kittel_model(
                    field_t, gamma_value, meff_t, geometry
                ),
                hres_t,
                freq,
                p0=p0,
                bounds=bounds,
                maxfev=20000,
            )
            gamma = float(params[0])
            meff = float(params[1])
            stderr = {
                "gamma_prime_GHz_per_T": _stderr(covariance, 0),
                "Meff_T": _stderr(covariance, 1),
            }
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        return {"success": False, "message": str(exc), "model_name": f"{geometry}_kittel"}
    fitted = _kittel_model(hres_t, gamma, meff, geometry)
    metrics = _metrics(freq, fitted)
    gamma_rad = 2.0 * math.pi * gamma * 1e9
    g = gamma_rad * HBAR / MU_B
    g_stderr = (
        None
        if stderr["gamma_prime_GHz_per_T"] is None
        else stderr["gamma_prime_GHz_per_T"] * 2.0 * math.pi * 1e9 * HBAR / MU_B
    )
    return {
        "success": True,
        "message": "fit converged",
        "model_name": f"{geometry}_kittel",
        "gamma_prime_GHz_per_T": gamma,
        "gamma_prime_stderr_GHz_per_T": stderr["gamma_prime_GHz_per_T"],
        "gamma_rad_per_s_T": gamma_rad,
        "g": g,
        "g_stderr": g_stderr,
        "Meff_T": meff,
        "Meff_mT": meff * 1000.0,
        "Meff_stderr_T": stderr["Meff_T"],
        "covariance": None if covariance is None else np.asarray(covariance, dtype=float).tolist(),
        "r_squared": metrics["r_squared"],
        "rmse_GHz": metrics["rmse"],
    }


def _kittel_model(field_t: np.ndarray, gamma: float, meff_t: float, geometry: str) -> np.ndarray:
    if geometry == "oop":
        return gamma * (field_t - meff_t)
    argument = field_t * (field_t + meff_t)
    return gamma * np.sqrt(np.maximum(argument, 0.0))


def _linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    if x.size < 2:
        return {"success": False}
    params = np.polyfit(x, y, 1)
    fitted = params[0] * x + params[1]
    metrics = _metrics(y, fitted)
    return {
        "success": True,
        "slope_mT_per_GHz": float(params[0]),
        "intercept_mT": float(params[1]),
        "r_squared": metrics["r_squared"],
        "rmse_mT": metrics["rmse"],
    }


def _metrics(y_true: np.ndarray, y_fit: np.ndarray) -> dict[str, float]:
    residual = np.asarray(y_true, dtype=float) - np.asarray(y_fit, dtype=float)
    rss = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "r_squared": 1.0 if ss_tot == 0.0 else 1.0 - rss / ss_tot,
        "rmse": float(np.sqrt(np.mean(residual**2))),
    }


def _gamma_from_g(g_value: float) -> float:
    return float(g_value * MU_B / (2.0 * math.pi * HBAR) / 1e9)


def _stderr(covariance: np.ndarray | None, index: int) -> float | None:
    if covariance is None:
        return None
    diagonal = np.sqrt(np.diag(np.asarray(covariance, dtype=float)))
    if diagonal.size <= index:
        return None
    return float(diagonal[index])


def _group_points(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row.get("geometry") or "unknown"), str(row.get("branch_label") or "default")), []
        ).append(row)
    return grouped


def _resolve_magnetic_volume(sample: SampleRecord) -> dict[str, Any] | None:
    resolution = resolve_sample_magnetic_volume(sample)
    if not resolution.is_available or resolution.magnetic_volume_m3 is None:
        return None
    return {
        "value_m3": resolution.magnetic_volume_m3,
        "uncertainty_m3": None,
        "source": resolution.source,
        "estimated": resolution.is_estimated,
        "warnings": list(resolution.warnings),
    }


def _convert_unit(value: float, unit: str, table: dict[str, float]) -> float:
    key = unit.strip().lower().replace("µ", "u").replace("²", "^2").replace("³", "^3")
    if key not in table:
        raise ValueError(f"Unsupported unit: {unit}")
    return value * table[key]


def _float_or_none(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric
