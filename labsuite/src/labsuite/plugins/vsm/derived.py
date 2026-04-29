"""Derived loop metrics for VSM hysteresis analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from labsuite.core.recipes import VsmPreprocessingRecipe
from labsuite.plugins.vsm.models import BranchSegment


def extract_loop_metrics(
    field_mT: np.ndarray,
    moment_emu: np.ndarray,
    temperature_k: np.ndarray,
    branches: list[BranchSegment],
    tail_mask: np.ndarray,
) -> tuple[dict[str, Any], list[str]]:
    """Extract hysteresis metrics from corrected loop data."""

    warnings: list[str] = []
    decreasing_branch = _select_primary_branch(branches, "decreasing")
    increasing_branch = _select_primary_branch(branches, "increasing")

    positive_remanence = None
    negative_remanence = None
    negative_coercive_field = None
    positive_coercive_field = None

    if decreasing_branch is None:
        warnings.append("missing_primary_decreasing_branch")
    else:
        branch_field, branch_moment = _branch_arrays(field_mT, moment_emu, decreasing_branch)
        positive_remanence = interpolate_y_at_x(branch_field, branch_moment, 0.0)
        negative_coercive_field = interpolate_zero_crossing(branch_field, branch_moment)
        if positive_remanence is None:
            warnings.append("decreasing_branch_remanence_unavailable")
        if negative_coercive_field is None:
            warnings.append("decreasing_branch_coercive_field_unavailable")

    if increasing_branch is None:
        warnings.append("missing_primary_increasing_branch")
    else:
        branch_field, branch_moment = _branch_arrays(field_mT, moment_emu, increasing_branch)
        negative_remanence = interpolate_y_at_x(branch_field, branch_moment, 0.0)
        positive_coercive_field = interpolate_zero_crossing(branch_field, branch_moment)
        if negative_remanence is None:
            warnings.append("increasing_branch_remanence_unavailable")
        if positive_coercive_field is None:
            warnings.append("increasing_branch_coercive_field_unavailable")

    positive_tail = tail_mask & (field_mT > 0.0)
    negative_tail = tail_mask & (field_mT < 0.0)
    positive_saturation = (
        float(np.mean(moment_emu[positive_tail])) if np.any(positive_tail) else None
    )
    negative_saturation = (
        float(np.mean(moment_emu[negative_tail])) if np.any(negative_tail) else None
    )
    if positive_saturation is None:
        warnings.append("positive_saturation_unavailable")
    if negative_saturation is None:
        warnings.append("negative_saturation_unavailable")

    loop_shift = _pair_average(positive_coercive_field, negative_coercive_field)
    vertical_shift = _pair_average(positive_remanence, negative_remanence)

    temperature_value = (
        float(np.nanmean(temperature_k)) if np.any(np.isfinite(temperature_k)) else None
    )
    metrics: dict[str, Any] = {
        "coercive_field_negative_mT": negative_coercive_field,
        "coercive_field_positive_mT": positive_coercive_field,
        "coercive_field_mean_abs_mT": _pair_mean_abs(
            positive_coercive_field, negative_coercive_field
        ),
        "loop_shift_mT": loop_shift,
        "remanence_positive_emu": positive_remanence,
        "remanence_negative_emu": negative_remanence,
        "remanence_mean_abs_emu": _pair_mean_abs(positive_remanence, negative_remanence),
        "vertical_shift_emu": vertical_shift,
        "saturation_moment_positive_emu": positive_saturation,
        "saturation_moment_negative_emu": negative_saturation,
        "saturation_moment_mean_abs_emu": _pair_mean_abs(positive_saturation, negative_saturation),
        "max_corrected_moment_emu": float(np.max(moment_emu)),
        "min_corrected_moment_emu": float(np.min(moment_emu)),
        "point_count": int(moment_emu.size),
        "branch_count": len(branches),
        "increasing_branch_count": sum(
            1 for branch in branches if branch.direction == "increasing"
        ),
        "decreasing_branch_count": sum(
            1 for branch in branches if branch.direction == "decreasing"
        ),
        "temperature_k": temperature_value,
        "selected_decreasing_branch_id": None
        if decreasing_branch is None
        else decreasing_branch.branch_id,
        "selected_increasing_branch_id": None
        if increasing_branch is None
        else increasing_branch.branch_id,
    }
    return metrics, warnings


def summarize_loop_quality(
    *,
    field_mT: np.ndarray,
    moment_emu: np.ndarray,
    branches: list[BranchSegment],
    positive_tail_indices: np.ndarray,
    negative_tail_indices: np.ndarray,
    temperature_k: np.ndarray | None = None,
) -> dict[str, Any]:
    """Summarize loop metrics plus background-comparison diagnostics."""

    if temperature_k is None:
        temperature_k = np.full(field_mT.shape, np.nan, dtype=float)

    tail_mask = np.zeros(field_mT.size, dtype=bool)
    tail_mask[np.asarray(positive_tail_indices, dtype=int)] = True
    tail_mask[np.asarray(negative_tail_indices, dtype=int)] = True
    loop_metrics, warnings = extract_loop_metrics(
        field_mT=field_mT,
        moment_emu=moment_emu,
        temperature_k=temperature_k,
        branches=branches,
        tail_mask=tail_mask,
    )
    direct_observables = _build_direct_observables(
        field_mT=field_mT,
        moment_emu=moment_emu,
        branches=branches,
        detailed_metrics=loop_metrics,
    )
    branch_asymmetry, branch_asymmetry_components = _compute_branch_asymmetry(
        field_mT=field_mT,
        moment_emu=moment_emu,
        branches=branches,
        detailed_metrics=loop_metrics,
    )
    loop_closure_error, loop_closure_components = _compute_loop_closure_error(
        field_mT=field_mT,
        moment_emu=moment_emu,
        branches=branches,
        positive_tail_indices=positive_tail_indices,
        negative_tail_indices=negative_tail_indices,
    )
    coercive_ambiguity_count, coercive_ambiguity_components = _compute_coercive_crossing_ambiguity(
        field_mT=field_mT,
        moment_emu=moment_emu,
        branches=branches,
        detailed_metrics=loop_metrics,
    )

    full_moment_span = max(float(np.max(moment_emu) - np.min(moment_emu)), 1e-18)
    positive_plateau_slope = _slope_from_indices(
        field_mT, moment_emu, np.asarray(positive_tail_indices, dtype=int)
    )
    negative_plateau_slope = _slope_from_indices(
        field_mT, moment_emu, np.asarray(negative_tail_indices, dtype=int)
    )
    H_max_mT = max(float(np.max(np.abs(field_mT))), 1e-18)
    Ms_reference_emu = loop_metrics.get("saturation_moment_mean_abs_emu")
    Ms_reference_emu = (
        float(Ms_reference_emu) if Ms_reference_emu is not None else 0.5 * full_moment_span
    )
    plateau_slope_scale = max(abs(Ms_reference_emu) / H_max_mT, 1e-18)
    positive_plateau_slope_normalized = abs(positive_plateau_slope) / plateau_slope_scale
    negative_plateau_slope_normalized = abs(negative_plateau_slope) / plateau_slope_scale
    positive_flatness_ratio = (
        abs(positive_plateau_slope)
        * max(
            float(np.ptp(field_mT[np.asarray(positive_tail_indices, dtype=int)]))
            if np.size(positive_tail_indices)
            else 0.0,
            1.0,
        )
        / full_moment_span
    )
    negative_flatness_ratio = (
        abs(negative_plateau_slope)
        * max(
            float(np.ptp(field_mT[np.asarray(negative_tail_indices, dtype=int)]))
            if np.size(negative_tail_indices)
            else 0.0,
            1.0,
        )
        / full_moment_span
    )
    plateau_flatness_ratio = max(positive_flatness_ratio, negative_flatness_ratio)
    saturation_consistency_ratio = _relative_difference(
        abs(loop_metrics.get("saturation_moment_positive_emu"))
        if loop_metrics.get("saturation_moment_positive_emu") is not None
        else None,
        abs(loop_metrics.get("saturation_moment_negative_emu"))
        if loop_metrics.get("saturation_moment_negative_emu") is not None
        else None,
    )
    if max(abs(positive_plateau_slope_normalized), abs(negative_plateau_slope_normalized)) <= 1e-6:
        tail_slope_symmetry_score = 1.0
    else:
        tail_slope_symmetry_score = float(
            np.clip(
                1.0
                - _relative_difference(
                    abs(positive_plateau_slope_normalized),
                    abs(negative_plateau_slope_normalized),
                ),
                0.0,
                1.0,
            )
        )
    saturation_magnitude_symmetry_score = float(
        np.clip(1.0 - saturation_consistency_ratio, 0.0, 1.0)
    )
    switching_width_mT = _compute_switching_width(
        positive_coercive_field_mT=loop_metrics.get("coercive_field_positive_mT"),
        negative_coercive_field_mT=loop_metrics.get("coercive_field_negative_mT"),
    )
    switching_asymmetry_ratio = _relative_difference(
        abs(loop_metrics.get("coercive_field_positive_mT"))
        if loop_metrics.get("coercive_field_positive_mT") is not None
        else None,
        abs(loop_metrics.get("coercive_field_negative_mT"))
        if loop_metrics.get("coercive_field_negative_mT") is not None
        else None,
    )
    zero_crossing_candidate_count = int(
        coercive_ambiguity_components.get("positive_branch_zero_crossing_candidates", 0)
        + coercive_ambiguity_components.get("negative_branch_zero_crossing_candidates", 0)
    )

    return {
        **loop_metrics,
        **direct_observables,
        "warnings": warnings,
        "plateau_slope_positive_emu_per_mT": positive_plateau_slope,
        "plateau_slope_negative_emu_per_mT": negative_plateau_slope,
        "plateau_slope_positive_normalized": positive_plateau_slope_normalized,
        "plateau_slope_negative_normalized": negative_plateau_slope_normalized,
        "plateau_flatness_ratio_positive": positive_flatness_ratio,
        "plateau_flatness_ratio_negative": negative_flatness_ratio,
        "plateau_flatness_ratio": plateau_flatness_ratio,
        "saturation_consistency_ratio": saturation_consistency_ratio,
        "tail_slope_symmetry_score": tail_slope_symmetry_score,
        "saturation_magnitude_symmetry_score": saturation_magnitude_symmetry_score,
        "branch_asymmetry": branch_asymmetry,
        "branch_asymmetry_components": branch_asymmetry_components,
        "loop_closure_error": loop_closure_error,
        "loop_closure_components": loop_closure_components,
        "switching_width_mT": switching_width_mT,
        "switching_asymmetry_ratio": switching_asymmetry_ratio,
        "zero_crossing_candidate_count": zero_crossing_candidate_count,
        "coercive_ambiguity_count": coercive_ambiguity_count,
        "coercive_crossing_ambiguous": coercive_ambiguity_count > 0,
        "coercive_ambiguity_components": coercive_ambiguity_components,
    }


def build_vsm_output_layers(
    *,
    field_mT: np.ndarray,
    moment_emu: np.ndarray,
    branches: list[BranchSegment],
    detailed_metrics: dict[str, Any],
    background_qc: dict[str, Any],
    background_details: dict[str, Any],
    warnings: list[str],
    moment_std_err_emu: np.ndarray,
    recipe: VsmPreprocessingRecipe,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build canonical direct observables, uncertainties, and trust diagnostics."""

    direct_observables = _build_direct_observables(
        field_mT=field_mT,
        moment_emu=moment_emu,
        branches=branches,
        detailed_metrics=detailed_metrics,
    )
    uncertainty_estimates = _build_uncertainty_estimates(
        field_mT=field_mT,
        moment_emu=moment_emu,
        moment_std_err_emu=moment_std_err_emu,
        branches=branches,
        detailed_metrics=detailed_metrics,
        direct_observables=direct_observables,
        background_details=background_details,
        recipe=recipe,
    )
    trust_diagnostics = _build_trust_diagnostics(
        field_mT=field_mT,
        moment_emu=moment_emu,
        branches=branches,
        detailed_metrics=detailed_metrics,
        background_qc=background_qc,
        background_details=background_details,
        direct_observables=direct_observables,
        warnings=warnings,
        uncertainty_flags=uncertainty_estimates["uncertainty_flags"],
    )
    return direct_observables, trust_diagnostics, uncertainty_estimates


def interpolate_y_at_x(x: np.ndarray, y: np.ndarray, target_x: float) -> float | None:
    """Interpolate a y value at a target x from one branch."""

    x_sorted, y_sorted = _sorted_unique_xy(x, y)
    if x_sorted.size < 2:
        return None
    if target_x < float(x_sorted[0]) or target_x > float(x_sorted[-1]):
        return None
    return float(np.interp(target_x, x_sorted, y_sorted))


def interpolate_zero_crossing(x: np.ndarray, y: np.ndarray) -> float | None:
    """Find the field where the moment crosses zero."""

    x_sorted, y_sorted = _sorted_unique_xy(x, y)
    if x_sorted.size < 2:
        return None

    direct_zero = np.flatnonzero(np.isclose(y_sorted, 0.0))
    if direct_zero.size:
        return float(x_sorted[direct_zero[np.argmin(np.abs(x_sorted[direct_zero]))]])

    candidates: list[float] = []
    for index in range(x_sorted.size - 1):
        y0 = float(y_sorted[index])
        y1 = float(y_sorted[index + 1])
        if y0 == y1:
            continue
        if y0 * y1 > 0.0:
            continue
        x0 = float(x_sorted[index])
        x1 = float(x_sorted[index + 1])
        crossing = x0 - y0 * (x1 - x0) / (y1 - y0)
        candidates.append(crossing)
    if not candidates:
        return None
    return min(candidates, key=abs)


def _select_primary_branch(branches: list[BranchSegment], direction: str) -> BranchSegment | None:
    candidates = [branch for branch in branches if branch.direction == direction]
    if not candidates:
        return None
    return max(candidates, key=lambda branch: branch.point_count)


def _branch_arrays(
    field_mT: np.ndarray,
    moment_emu: np.ndarray,
    branch: BranchSegment,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(field_mT[branch.start_index : branch.end_index + 1], dtype=float),
        np.asarray(moment_emu[branch.start_index : branch.end_index + 1], dtype=float),
    )


def _branch_arrays_with_std(
    field_mT: np.ndarray,
    moment_emu: np.ndarray,
    moment_std_err_emu: np.ndarray,
    branch: BranchSegment,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(field_mT[branch.start_index : branch.end_index + 1], dtype=float),
        np.asarray(moment_emu[branch.start_index : branch.end_index + 1], dtype=float),
        np.asarray(moment_std_err_emu[branch.start_index : branch.end_index + 1], dtype=float),
    )


def _sorted_unique_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    sorted_x = np.asarray(x[order], dtype=float)
    sorted_y = np.asarray(y[order], dtype=float)
    unique_x, inverse = np.unique(sorted_x, return_inverse=True)
    if unique_x.size == sorted_x.size:
        return sorted_x, sorted_y
    unique_y = np.zeros(unique_x.size, dtype=float)
    counts = np.zeros(unique_x.size, dtype=float)
    for index, value in enumerate(sorted_y):
        unique_y[inverse[index]] += value
        counts[inverse[index]] += 1.0
    unique_y /= np.maximum(counts, 1.0)
    return unique_x, unique_y


def _sorted_unique_xy_std(
    x: np.ndarray,
    y: np.ndarray,
    std_err: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(x)
    sorted_x = np.asarray(x[order], dtype=float)
    sorted_y = np.asarray(y[order], dtype=float)
    sorted_std = np.asarray(std_err[order], dtype=float)
    unique_x, inverse = np.unique(sorted_x, return_inverse=True)
    if unique_x.size == sorted_x.size:
        return sorted_x, sorted_y, sorted_std

    unique_y = np.zeros(unique_x.size, dtype=float)
    unique_std = np.full(unique_x.size, np.nan, dtype=float)
    counts = np.zeros(unique_x.size, dtype=float)
    std_counts = np.zeros(unique_x.size, dtype=float)
    for index, value in enumerate(sorted_y):
        unique_y[inverse[index]] += value
        counts[inverse[index]] += 1.0
        std_value = float(sorted_std[index])
        if np.isfinite(std_value):
            if np.isnan(unique_std[inverse[index]]):
                unique_std[inverse[index]] = 0.0
            unique_std[inverse[index]] += std_value
            std_counts[inverse[index]] += 1.0
    unique_y /= np.maximum(counts, 1.0)
    valid_std = std_counts > 0.0
    unique_std[valid_std] /= std_counts[valid_std]
    return unique_x, unique_y, unique_std


def _pair_average(first: float | None, second: float | None) -> float | None:
    values = [value for value in (first, second) if value is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def _pair_mean_abs(first: float | None, second: float | None) -> float | None:
    values = [abs(value) for value in (first, second) if value is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def _pair_mean_error(first: float | None, second: float | None) -> float | None:
    values = [float(value) for value in (first, second) if value is not None]
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return float(0.5 * np.sqrt(values[0] ** 2 + values[1] ** 2))


def _build_direct_observables(
    *,
    field_mT: np.ndarray,
    moment_emu: np.ndarray,
    branches: list[BranchSegment],
    detailed_metrics: dict[str, Any],
) -> dict[str, Any]:
    Ms_emu = detailed_metrics.get("saturation_moment_mean_abs_emu")
    Mr_emu = detailed_metrics.get("remanence_mean_abs_emu")
    Hc_mT = detailed_metrics.get("coercive_field_mean_abs_mT")
    loop_area_emu_mT, loop_area_signed_emu_mT, overlap_start_mT, overlap_end_mT = (
        _compute_loop_area(
            field_mT,
            moment_emu,
            branches,
        )
    )
    squareness = None
    if Ms_emu is not None and abs(Ms_emu) > 1e-18 and Mr_emu is not None:
        squareness = float(Mr_emu / Ms_emu)

    return {
        "Ms_emu": Ms_emu,
        "Mr_emu": Mr_emu,
        "Hc_mT": Hc_mT,
        "squareness": squareness,
        "exchange_bias_mT": detailed_metrics.get("loop_shift_mT"),
        "vertical_shift_emu": detailed_metrics.get("vertical_shift_emu"),
        "loop_area_emu_mT": loop_area_emu_mT,
        "loop_area_signed_emu_mT": loop_area_signed_emu_mT,
        "loop_area_overlap_start_mT": overlap_start_mT,
        "loop_area_overlap_end_mT": overlap_end_mT,
    }


def _build_uncertainty_estimates(
    *,
    field_mT: np.ndarray,
    moment_emu: np.ndarray,
    moment_std_err_emu: np.ndarray,
    branches: list[BranchSegment],
    detailed_metrics: dict[str, Any],
    direct_observables: dict[str, Any],
    background_details: dict[str, Any],
    recipe: VsmPreprocessingRecipe,
) -> dict[str, Any]:
    flags: list[str] = []

    positive_tail_indices = np.asarray(
        background_details.get("selected_positive_indices", []), dtype=int
    )
    negative_tail_indices = np.asarray(
        background_details.get("selected_negative_indices", []), dtype=int
    )

    positive_ms = _compute_tail_scatter_error(
        moment_emu=moment_emu,
        moment_std_err_emu=moment_std_err_emu,
        indices=positive_tail_indices,
        label="positive",
        uncertainty_scale=recipe.uncertainty_scale,
    )
    negative_ms = _compute_tail_scatter_error(
        moment_emu=moment_emu,
        moment_std_err_emu=moment_std_err_emu,
        indices=negative_tail_indices,
        label="negative",
        uncertainty_scale=recipe.uncertainty_scale,
    )
    flags.extend(positive_ms["flags"])
    flags.extend(negative_ms["flags"])

    decreasing_branch = _select_primary_branch(branches, "decreasing")
    increasing_branch = _select_primary_branch(branches, "increasing")

    positive_mr = _compute_zero_intercept_uncertainty(
        field_mT=field_mT,
        moment_emu=moment_emu,
        moment_std_err_emu=moment_std_err_emu,
        branch=decreasing_branch,
        label="positive",
        window_width_mT=recipe.uncertainty_zero_field_window_width_mT,
        min_points=recipe.uncertainty_zero_field_min_points,
        uncertainty_scale=recipe.uncertainty_scale,
    )
    negative_mr = _compute_zero_intercept_uncertainty(
        field_mT=field_mT,
        moment_emu=moment_emu,
        moment_std_err_emu=moment_std_err_emu,
        branch=increasing_branch,
        label="negative",
        window_width_mT=recipe.uncertainty_zero_field_window_width_mT,
        min_points=recipe.uncertainty_zero_field_min_points,
        uncertainty_scale=recipe.uncertainty_scale,
    )
    flags.extend(positive_mr["flags"])
    flags.extend(negative_mr["flags"])

    positive_hc = _compute_zero_crossing_uncertainty(
        field_mT=field_mT,
        moment_emu=moment_emu,
        moment_std_err_emu=moment_std_err_emu,
        branch=increasing_branch,
        label="positive",
        crossing_value_mT=detailed_metrics.get("coercive_field_positive_mT"),
        half_width_mT=recipe.uncertainty_switching_half_width_mT,
        min_points=recipe.uncertainty_switching_min_points,
        min_switching_slope_emu_per_mT=recipe.uncertainty_min_switching_slope_emu_per_mT,
        uncertainty_scale=recipe.uncertainty_scale,
    )
    negative_hc = _compute_zero_crossing_uncertainty(
        field_mT=field_mT,
        moment_emu=moment_emu,
        moment_std_err_emu=moment_std_err_emu,
        branch=decreasing_branch,
        label="negative",
        crossing_value_mT=detailed_metrics.get("coercive_field_negative_mT"),
        half_width_mT=recipe.uncertainty_switching_half_width_mT,
        min_points=recipe.uncertainty_switching_min_points,
        min_switching_slope_emu_per_mT=recipe.uncertainty_min_switching_slope_emu_per_mT,
        uncertainty_scale=recipe.uncertainty_scale,
    )
    flags.extend(positive_hc["flags"])
    flags.extend(negative_hc["flags"])

    ms_error = _pair_mean_error(positive_ms["error"], negative_ms["error"])
    mr_error = _pair_mean_error(positive_mr["error"], negative_mr["error"])
    hc_error = _pair_mean_error(positive_hc["error"], negative_hc["error"])
    hex_error = _pair_mean_error(positive_hc["error"], negative_hc["error"])

    squareness_error = _compute_ratio_error(
        numerator=direct_observables.get("Mr_emu"),
        denominator=direct_observables.get("Ms_emu"),
        numerator_error=mr_error,
        denominator_error=ms_error,
    )

    loop_area = direct_observables.get("loop_area_emu_mT")
    loop_area_error, loop_area_variants, loop_area_flags = _compute_loop_area_error(
        field_mT=field_mT,
        moment_emu=moment_emu,
        branches=branches,
        base_loop_area_emu_mT=loop_area,
        recipe=recipe,
    )
    flags.extend(loop_area_flags)

    uncertainty_flags = list(dict.fromkeys(flags))
    return {
        "ms_std_pos": positive_ms["std"],
        "ms_std_neg": negative_ms["std"],
        "ms_n_points_pos": positive_ms["n_points"],
        "ms_n_points_neg": negative_ms["n_points"],
        "ms_error_pos": positive_ms["error"],
        "ms_error_neg": negative_ms["error"],
        "ms_error": ms_error,
        "mr_error_pos": positive_mr["error"],
        "mr_error_neg": negative_mr["error"],
        "mr_error": mr_error,
        "mr_zero_window_width_mT": recipe.uncertainty_zero_field_window_width_mT,
        "mr_interp_method": "linear_interp_at_zero_with_local_linear_sensitivity",
        "mr_noise_pos_emu": positive_mr["noise"],
        "mr_noise_neg_emu": negative_mr["noise"],
        "mr_interp_sensitivity_pos_emu": positive_mr["interp_sensitivity"],
        "mr_interp_sensitivity_neg_emu": negative_mr["interp_sensitivity"],
        "hc_error_pos": positive_hc["error"],
        "hc_error_neg": negative_hc["error"],
        "hc_error": hc_error,
        "switching_slope_pos": positive_hc["slope"],
        "switching_slope_neg": negative_hc["slope"],
        "hc_noise_pos_emu": positive_hc["noise"],
        "hc_noise_neg_emu": negative_hc["noise"],
        "hex_error": hex_error,
        "squareness_error": squareness_error,
        "loop_area_error": loop_area_error,
        "loop_area_variants_emu_mT": loop_area_variants,
        "uncertainty_flags": uncertainty_flags,
        "instrument_noise_fallback_used": bool(
            positive_ms["used_instrument_fallback"]
            or negative_ms["used_instrument_fallback"]
            or positive_mr["used_instrument_fallback"]
            or negative_mr["used_instrument_fallback"]
            or positive_hc["used_instrument_fallback"]
            or negative_hc["used_instrument_fallback"]
        ),
        "uncertainty_scale_applied": recipe.uncertainty_scale,
    }


def _build_trust_diagnostics(
    *,
    field_mT: np.ndarray,
    moment_emu: np.ndarray,
    branches: list[BranchSegment],
    detailed_metrics: dict[str, Any],
    background_qc: dict[str, Any],
    background_details: dict[str, Any],
    direct_observables: dict[str, Any],
    warnings: list[str],
    uncertainty_flags: list[str],
) -> dict[str, Any]:
    saturation_confidence, saturation_confidence_components = _compute_saturation_confidence(
        detailed_metrics=detailed_metrics,
        background_qc=background_qc,
        background_details=background_details,
    )
    branch_asymmetry, branch_asymmetry_components = _compute_branch_asymmetry(
        field_mT=field_mT,
        moment_emu=moment_emu,
        branches=branches,
        detailed_metrics=detailed_metrics,
    )
    switching_complexity, switching_complexity_label, switching_complexity_components = (
        _compute_switching_complexity(
            field_mT=field_mT,
            moment_emu=moment_emu,
            branches=branches,
        )
    )
    ambiguity_flags = _build_ambiguity_flags(
        detailed_metrics=detailed_metrics,
        direct_observables=direct_observables,
        background_qc=background_qc,
        saturation_confidence=saturation_confidence,
        branch_asymmetry=branch_asymmetry,
        switching_complexity_label=switching_complexity_label,
        warnings=warnings,
        uncertainty_flags=uncertainty_flags,
    )
    return {
        "saturation_confidence": saturation_confidence,
        "saturation_confidence_components": saturation_confidence_components,
        "background_fit_details": {
            "background_qc_passed": background_qc.get("passed"),
            "background_mode": background_details.get("background_mode"),
            "background_subtraction_mode": background_details.get("subtraction_mode"),
            "background_correction_accepted": background_details.get("correction_accepted"),
            "background_decision_reason": background_details.get("decision_reason"),
            "vsm_quality_model": background_details.get("quality_model"),
            "vsm_quality_status": background_details.get("quality_status"),
            "vsm_quality_weight": background_details.get("quality_weight"),
            "vsm_quality_reasons": background_details.get("quality_reasons"),
            "legacy_background_mode": background_details.get("legacy_background_mode"),
            "legacy_background_correction_accepted": background_details.get(
                "legacy_correction_accepted"
            ),
            "legacy_background_decision_reason": background_details.get(
                "legacy_decision_reason"
            ),
            "background_slope_mean_emu_per_mT": background_details.get("slope_emu_per_mT"),
            "background_slope_positive_emu_per_mT": background_details.get(
                "positive_slope_emu_per_mT"
            ),
            "background_slope_negative_emu_per_mT": background_details.get(
                "negative_slope_emu_per_mT"
            ),
            "background_intercept_positive_emu": background_details.get("positive_intercept_emu"),
            "background_intercept_negative_emu": background_details.get("negative_intercept_emu"),
            "positive_tail_flatness_ratio": background_qc.get("positive_tail_flatness_ratio"),
            "negative_tail_flatness_ratio": background_qc.get("negative_tail_flatness_ratio"),
            "raw_tail_slope_disagreement_ratio": background_qc.get(
                "raw_tail_slope_disagreement_ratio"
            ),
            "background_score_raw": background_qc.get("score_raw"),
            "background_score_corrected": background_qc.get("score_corrected"),
            "background_score_delta": background_qc.get("score_delta"),
            "raw_plateau_slope_positive_normalized": background_qc.get("comparison", {}).get(
                "raw_plateau_slope_positive_normalized"
            ),
            "raw_plateau_slope_negative_normalized": background_qc.get("comparison", {}).get(
                "raw_plateau_slope_negative_normalized"
            ),
            "corrected_plateau_slope_positive_normalized": background_qc.get("comparison", {}).get(
                "corrected_plateau_slope_positive_normalized"
            ),
            "corrected_plateau_slope_negative_normalized": background_qc.get("comparison", {}).get(
                "corrected_plateau_slope_negative_normalized"
            ),
            "background_flatness_gain_positive": background_qc.get("comparison", {}).get(
                "background_flatness_gain_positive"
            ),
            "background_flatness_gain_negative": background_qc.get("comparison", {}).get(
                "background_flatness_gain_negative"
            ),
            "background_flatness_gain_score": background_qc.get("comparison", {}).get(
                "background_flatness_gain_score"
            ),
            "background_tail_slope_symmetry_score": background_qc.get("comparison", {}).get(
                "background_tail_slope_symmetry_score"
            ),
            "background_saturation_magnitude_symmetry_score": background_qc.get(
                "comparison", {}
            ).get("background_saturation_magnitude_symmetry_score"),
            "raw_switching_width_mT": background_qc.get("comparison", {}).get(
                "raw_switching_width_mT"
            ),
            "corrected_switching_width_mT": background_qc.get("comparison", {}).get(
                "corrected_switching_width_mT"
            ),
            "background_switching_width_relative_change": background_qc.get("comparison", {}).get(
                "background_switching_width_relative_change"
            ),
            "raw_zero_crossing_candidate_count": background_qc.get("comparison", {}).get(
                "raw_zero_crossing_candidate_count"
            ),
            "corrected_zero_crossing_candidate_count": background_qc.get("comparison", {}).get(
                "corrected_zero_crossing_candidate_count"
            ),
            "positive_tail_point_count": len(
                background_details.get("selected_positive_indices", [])
            ),
            "negative_tail_point_count": len(
                background_details.get("selected_negative_indices", [])
            ),
        },
        "ambiguity_flags": ambiguity_flags,
        "branch_asymmetry": branch_asymmetry,
        "branch_asymmetry_components": branch_asymmetry_components,
        "switching_complexity": switching_complexity,
        "switching_complexity_label": switching_complexity_label,
        "switching_complexity_components": switching_complexity_components,
    }


def _compute_loop_area(
    field_mT: np.ndarray,
    moment_emu: np.ndarray,
    branches: list[BranchSegment],
    *,
    grid_mode: str = "union",
    grid_points: int = 256,
) -> tuple[float | None, float | None, float | None, float | None]:
    increasing_branch = _select_primary_branch(branches, "increasing")
    decreasing_branch = _select_primary_branch(branches, "decreasing")
    if increasing_branch is None or decreasing_branch is None:
        return None, None, None, None

    increasing_field, increasing_moment = _branch_arrays(field_mT, moment_emu, increasing_branch)
    decreasing_field, decreasing_moment = _branch_arrays(field_mT, moment_emu, decreasing_branch)
    increasing_field, increasing_moment = _sorted_unique_xy(increasing_field, increasing_moment)
    decreasing_field, decreasing_moment = _sorted_unique_xy(decreasing_field, decreasing_moment)
    overlap_start = max(float(increasing_field[0]), float(decreasing_field[0]))
    overlap_end = min(float(increasing_field[-1]), float(decreasing_field[-1]))
    if overlap_end <= overlap_start:
        return None, None, None, None

    if grid_mode == "dense":
        overlap_x = np.linspace(overlap_start, overlap_end, max(int(grid_points), 2))
    else:
        overlap_x = np.unique(
            np.concatenate(
                [
                    increasing_field[
                        (increasing_field >= overlap_start) & (increasing_field <= overlap_end)
                    ],
                    decreasing_field[
                        (decreasing_field >= overlap_start) & (decreasing_field <= overlap_end)
                    ],
                    np.asarray([overlap_start, overlap_end], dtype=float),
                ]
            )
        )
    if overlap_x.size < 2:
        return None, None, overlap_start, overlap_end

    increasing_interp = np.interp(overlap_x, increasing_field, increasing_moment)
    decreasing_interp = np.interp(overlap_x, decreasing_field, decreasing_moment)
    signed_area = float(np.trapezoid(decreasing_interp - increasing_interp, overlap_x))
    absolute_area = float(np.trapezoid(np.abs(decreasing_interp - increasing_interp), overlap_x))
    return absolute_area, signed_area, overlap_start, overlap_end


def _compute_saturation_confidence(
    *,
    detailed_metrics: dict[str, Any],
    background_qc: dict[str, Any],
    background_details: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    Ms_positive = detailed_metrics.get("saturation_moment_positive_emu")
    Ms_negative = detailed_metrics.get("saturation_moment_negative_emu")
    positive_available = Ms_positive is not None
    negative_available = Ms_negative is not None
    availability_score = 1.0 if positive_available and negative_available else 0.0
    background_qc_score = 1.0 if background_qc.get("passed") else 0.25

    positive_flatness_ratio = float(background_qc.get("positive_tail_flatness_ratio") or 0.0)
    negative_flatness_ratio = float(background_qc.get("negative_tail_flatness_ratio") or 0.0)
    flatness_tolerance = max(
        float(background_details.get("positive_tail_flatness_ratio_tolerance", 0.08) or 0.08), 1e-18
    )
    flatness_penalty = max(positive_flatness_ratio, negative_flatness_ratio) / flatness_tolerance
    flatness_score = float(np.clip(1.0 - flatness_penalty, 0.0, 1.0))

    slope_disagreement_ratio = float(background_qc.get("raw_tail_slope_disagreement_ratio") or 0.0)
    slope_disagreement_tolerance = max(
        float(background_details.get("raw_tail_slope_disagreement_ratio_tolerance", 0.35) or 0.35),
        1e-18,
    )
    slope_consistency_score = float(
        np.clip(1.0 - slope_disagreement_ratio / slope_disagreement_tolerance, 0.0, 1.0)
    )

    score = float(
        np.mean([availability_score, background_qc_score, flatness_score, slope_consistency_score])
    )
    return score, {
        "availability_score": availability_score,
        "background_qc_score": background_qc_score,
        "flatness_score": flatness_score,
        "slope_consistency_score": slope_consistency_score,
        "positive_tail_flatness_ratio": positive_flatness_ratio,
        "negative_tail_flatness_ratio": negative_flatness_ratio,
        "raw_tail_slope_disagreement_ratio": slope_disagreement_ratio,
    }


def _compute_branch_asymmetry(
    *,
    field_mT: np.ndarray,
    moment_emu: np.ndarray,
    branches: list[BranchSegment],
    detailed_metrics: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    Ms_positive = detailed_metrics.get("saturation_moment_positive_emu")
    Ms_negative = detailed_metrics.get("saturation_moment_negative_emu")
    Mr_positive = detailed_metrics.get("remanence_positive_emu")
    Mr_negative = detailed_metrics.get("remanence_negative_emu")
    Hc_positive = detailed_metrics.get("coercive_field_positive_mT")
    Hc_negative = detailed_metrics.get("coercive_field_negative_mT")

    Ms_component = _relative_difference(
        abs(Ms_positive) if Ms_positive is not None else None,
        abs(Ms_negative) if Ms_negative is not None else None,
    )
    Mr_component = _relative_difference(
        abs(Mr_positive) if Mr_positive is not None else None,
        abs(Mr_negative) if Mr_negative is not None else None,
    )
    Hc_component = _relative_difference(
        abs(Hc_positive) if Hc_positive is not None else None,
        abs(Hc_negative) if Hc_negative is not None else None,
    )
    shape_component = _compute_branch_shape_mismatch(field_mT, moment_emu, branches)
    score = float(np.mean([Ms_component, Mr_component, Hc_component, shape_component]))
    return score, {
        "Ms_component": Ms_component,
        "Mr_component": Mr_component,
        "Hc_component": Hc_component,
        "shape_component": shape_component,
    }


def _compute_switching_complexity(
    *,
    field_mT: np.ndarray,
    moment_emu: np.ndarray,
    branches: list[BranchSegment],
) -> tuple[float, str, dict[str, Any]]:
    increasing_branch = _select_primary_branch(branches, "increasing")
    decreasing_branch = _select_primary_branch(branches, "decreasing")
    branch_count = len(branches)
    expected_branch_count = 3
    extra_branch_component = float(np.clip((branch_count - expected_branch_count) / 3.0, 0.0, 1.0))

    zero_crossing_candidates = 0
    switching_peak_count = 0
    for branch in (increasing_branch, decreasing_branch):
        if branch is None:
            continue
        branch_field, branch_moment = _branch_arrays(field_mT, moment_emu, branch)
        zero_crossing_candidates += _count_zero_crossing_candidates(branch_field, branch_moment)
        switching_peak_count += _count_switching_peaks(branch_field, branch_moment)

    extra_zero_crossings = max(zero_crossing_candidates - 2, 0)
    zero_crossing_component = float(np.clip(extra_zero_crossings / 4.0, 0.0, 1.0))
    extra_switching_peaks = max(switching_peak_count - 2, 0)
    multistep_component = float(np.clip(extra_switching_peaks / 4.0, 0.0, 1.0))
    score = float(np.mean([extra_branch_component, zero_crossing_component, multistep_component]))
    if score < 0.25:
        label = "simple"
    elif score < 0.6:
        label = "moderate"
    else:
        label = "complex"
    return (
        score,
        label,
        {
            "branch_count": branch_count,
            "extra_branch_component": extra_branch_component,
            "zero_crossing_candidates": zero_crossing_candidates,
            "zero_crossing_component": zero_crossing_component,
            "switching_peak_count": switching_peak_count,
            "multistep_component": multistep_component,
        },
    )


def _build_ambiguity_flags(
    *,
    detailed_metrics: dict[str, Any],
    direct_observables: dict[str, Any],
    background_qc: dict[str, Any],
    saturation_confidence: float,
    branch_asymmetry: float,
    switching_complexity_label: str,
    warnings: list[str],
    uncertainty_flags: list[str],
) -> list[str]:
    flags: list[str] = list(dict.fromkeys(str(warning) for warning in warnings))
    if direct_observables.get("loop_area_emu_mT") is None:
        flags.append("loop_area_unavailable")
    if (
        direct_observables.get("Ms_emu") is None
        or abs(direct_observables.get("Ms_emu") or 0.0) <= 1e-18
    ):
        flags.append("low_saturation_signal")
    if direct_observables.get("Hc_mT") is None:
        flags.append("unstable_zero_crossing")
    if not background_qc.get("passed", False):
        flags.append("background_qc_failed")
    if saturation_confidence < 0.5:
        flags.append("low_saturation_confidence")
    if branch_asymmetry > 0.5:
        flags.append("asymmetric_branches")
    if switching_complexity_label != "simple":
        flags.append(f"switching_complexity_{switching_complexity_label}")
    flags.extend(uncertainty_flags)
    return list(dict.fromkeys(flags))


def _compute_branch_shape_mismatch(
    field_mT: np.ndarray,
    moment_emu: np.ndarray,
    branches: list[BranchSegment],
) -> float:
    increasing_branch = _select_primary_branch(branches, "increasing")
    decreasing_branch = _select_primary_branch(branches, "decreasing")
    if increasing_branch is None or decreasing_branch is None:
        return 1.0

    increasing_field, increasing_moment = _branch_arrays(field_mT, moment_emu, increasing_branch)
    decreasing_field, decreasing_moment = _branch_arrays(field_mT, moment_emu, decreasing_branch)
    increasing_field, increasing_moment = _sorted_unique_xy(increasing_field, increasing_moment)
    decreasing_field, decreasing_moment = _sorted_unique_xy(decreasing_field, decreasing_moment)
    overlap_start = max(float(increasing_field[0]), float(decreasing_field[0]))
    overlap_end = min(float(increasing_field[-1]), float(decreasing_field[-1]))
    if overlap_end <= overlap_start:
        return 1.0

    overlap_x = np.linspace(overlap_start, overlap_end, 256)
    increasing_interp = np.interp(overlap_x, increasing_field, increasing_moment)
    decreasing_interp = np.interp(overlap_x, decreasing_field, decreasing_moment)
    numerator = float(np.mean(np.abs(increasing_interp + decreasing_interp)))
    denominator = max(float(np.mean(np.abs(decreasing_interp - increasing_interp))), 1e-18)
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _compute_loop_closure_error(
    *,
    field_mT: np.ndarray,
    moment_emu: np.ndarray,
    branches: list[BranchSegment],
    positive_tail_indices: np.ndarray,
    negative_tail_indices: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    increasing_branch = _select_primary_branch(branches, "increasing")
    decreasing_branch = _select_primary_branch(branches, "decreasing")
    moment_span = max(float(np.max(moment_emu) - np.min(moment_emu)), 1e-18)
    if increasing_branch is None or decreasing_branch is None:
        return 1.0, {
            "positive_tail_branch_mismatch": None,
            "negative_tail_branch_mismatch": None,
        }

    positive_tail_branch_mismatch = _tail_branch_mismatch(
        field_mT=field_mT,
        moment_emu=moment_emu,
        increasing_branch=increasing_branch,
        decreasing_branch=decreasing_branch,
        tail_indices=np.asarray(positive_tail_indices, dtype=int),
        moment_span=moment_span,
    )
    negative_tail_branch_mismatch = _tail_branch_mismatch(
        field_mT=field_mT,
        moment_emu=moment_emu,
        increasing_branch=increasing_branch,
        decreasing_branch=decreasing_branch,
        tail_indices=np.asarray(negative_tail_indices, dtype=int),
        moment_span=moment_span,
    )
    return float(np.mean([positive_tail_branch_mismatch, negative_tail_branch_mismatch])), {
        "positive_tail_branch_mismatch": positive_tail_branch_mismatch,
        "negative_tail_branch_mismatch": negative_tail_branch_mismatch,
    }


def _compute_switching_width(
    *,
    positive_coercive_field_mT: float | None,
    negative_coercive_field_mT: float | None,
) -> float | None:
    if positive_coercive_field_mT is None or negative_coercive_field_mT is None:
        return None
    return float(abs(float(positive_coercive_field_mT) - float(negative_coercive_field_mT)))


def _compute_coercive_crossing_ambiguity(
    *,
    field_mT: np.ndarray,
    moment_emu: np.ndarray,
    branches: list[BranchSegment],
    detailed_metrics: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    increasing_branch = _select_primary_branch(branches, "increasing")
    decreasing_branch = _select_primary_branch(branches, "decreasing")

    positive_candidates = 0
    negative_candidates = 0
    if increasing_branch is not None:
        increasing_field, increasing_moment = _branch_arrays(
            field_mT, moment_emu, increasing_branch
        )
        positive_candidates = _count_zero_crossing_candidates(increasing_field, increasing_moment)
    if decreasing_branch is not None:
        decreasing_field, decreasing_moment = _branch_arrays(
            field_mT, moment_emu, decreasing_branch
        )
        negative_candidates = _count_zero_crossing_candidates(decreasing_field, decreasing_moment)

    ambiguity_count = max(positive_candidates - 1, 0) + max(negative_candidates - 1, 0)
    if detailed_metrics.get("coercive_field_positive_mT") is None:
        ambiguity_count += 1
    if detailed_metrics.get("coercive_field_negative_mT") is None:
        ambiguity_count += 1
    return int(ambiguity_count), {
        "positive_branch_zero_crossing_candidates": positive_candidates,
        "negative_branch_zero_crossing_candidates": negative_candidates,
    }


def _count_zero_crossing_candidates(x: np.ndarray, y: np.ndarray) -> int:
    x_sorted, y_sorted = _sorted_unique_xy(x, y)
    if x_sorted.size < 2:
        return 0
    count = 0
    for index in range(x_sorted.size - 1):
        y0 = float(y_sorted[index])
        y1 = float(y_sorted[index + 1])
        if y0 == y1 == 0.0:
            continue
        if y0 == 0.0 or y1 == 0.0 or y0 * y1 < 0.0:
            count += 1
    return count


def _count_switching_peaks(x: np.ndarray, y: np.ndarray) -> int:
    x_sorted, y_sorted = _sorted_unique_xy(x, y)
    if x_sorted.size < 5:
        return 0
    gradient = np.abs(np.gradient(y_sorted, x_sorted))
    peak_threshold = 0.35 * float(np.max(gradient))
    if peak_threshold <= 0.0:
        return 0
    peak_indices: list[int] = []
    for index in range(1, gradient.size - 1):
        if gradient[index] < peak_threshold:
            continue
        if gradient[index] >= gradient[index - 1] and gradient[index] >= gradient[index + 1]:
            if peak_indices and index - peak_indices[-1] < 3:
                if gradient[index] > gradient[peak_indices[-1]]:
                    peak_indices[-1] = index
                continue
            peak_indices.append(index)
    return len(peak_indices)


def _compute_tail_scatter_error(
    *,
    moment_emu: np.ndarray,
    moment_std_err_emu: np.ndarray,
    indices: np.ndarray,
    label: str,
    uncertainty_scale: float,
) -> dict[str, Any]:
    values = np.asarray(moment_emu[indices], dtype=float)
    std_err_values = np.asarray(moment_std_err_emu[indices], dtype=float)
    std = float(np.std(values, ddof=1)) if values.size >= 2 else None
    flags: list[str] = []
    used_instrument_fallback = False
    error = None

    if values.size >= 2 and std is not None:
        error = float(std / np.sqrt(values.size))
    else:
        error, used_instrument_fallback = _estimate_instrument_standard_error(
            std_err_values, values.size
        )
        if error is None:
            flags.append(f"missing_tail_scatter_{label}")

    if error is not None:
        error = float(error * uncertainty_scale)

    return {
        "std": std,
        "n_points": int(values.size),
        "error": error,
        "used_instrument_fallback": used_instrument_fallback,
        "flags": flags,
    }


def _compute_zero_intercept_uncertainty(
    *,
    field_mT: np.ndarray,
    moment_emu: np.ndarray,
    moment_std_err_emu: np.ndarray,
    branch: BranchSegment | None,
    label: str,
    window_width_mT: float,
    min_points: int,
    uncertainty_scale: float,
) -> dict[str, Any]:
    flags: list[str] = []
    if branch is None:
        return {
            "error": None,
            "noise": None,
            "interp_sensitivity": None,
            "used_instrument_fallback": False,
            "flags": [f"missing_primary_{label}_branch_for_mr_uncertainty"],
        }

    branch_field, branch_moment, branch_std = _branch_arrays_with_std(
        field_mT, moment_emu, moment_std_err_emu, branch
    )
    branch_field, branch_moment, branch_std = _sorted_unique_xy_std(
        branch_field, branch_moment, branch_std
    )
    if branch_field.size < 2 or float(branch_field[0]) > 0.0 or float(branch_field[-1]) < 0.0:
        return {
            "error": None,
            "noise": None,
            "interp_sensitivity": None,
            "used_instrument_fallback": False,
            "flags": [f"insufficient_zero_window_points_{label}_mr"],
        }

    local_indices = _select_local_indices(branch_field, 0.0, window_width_mT, min_points)
    noise, used_instrument_fallback = _estimate_local_noise(
        branch_moment[local_indices],
        branch_std[local_indices],
    )
    if noise is None:
        flags.append(f"insufficient_zero_window_points_{label}_mr")
        return {
            "error": None,
            "noise": None,
            "interp_sensitivity": None,
            "used_instrument_fallback": used_instrument_fallback,
            "flags": flags,
        }

    interp_zero = float(np.interp(0.0, branch_field, branch_moment))
    local_x = np.asarray(branch_field[local_indices], dtype=float)
    local_y = np.asarray(branch_moment[local_indices], dtype=float)
    if local_x.size >= 2:
        coefficients = np.polyfit(local_x, local_y, deg=1)
        fit_zero = float(coefficients[1])
    else:
        fit_zero = interp_zero
        flags.append(f"insufficient_zero_window_points_{label}_mr")

    interp_sensitivity = abs(interp_zero - fit_zero)
    error = float(np.sqrt(noise**2 + interp_sensitivity**2) * uncertainty_scale)
    return {
        "error": error,
        "noise": noise,
        "interp_sensitivity": interp_sensitivity,
        "used_instrument_fallback": used_instrument_fallback,
        "flags": flags,
    }


def _compute_zero_crossing_uncertainty(
    *,
    field_mT: np.ndarray,
    moment_emu: np.ndarray,
    moment_std_err_emu: np.ndarray,
    branch: BranchSegment | None,
    label: str,
    crossing_value_mT: float | None,
    half_width_mT: float,
    min_points: int,
    min_switching_slope_emu_per_mT: float,
    uncertainty_scale: float,
) -> dict[str, Any]:
    flags: list[str] = []
    if branch is None or crossing_value_mT is None:
        return {
            "error": None,
            "noise": None,
            "slope": None,
            "used_instrument_fallback": False,
            "flags": [f"unstable_zero_crossing_{label}_hc"],
        }

    branch_field, branch_moment, branch_std = _branch_arrays_with_std(
        field_mT, moment_emu, moment_std_err_emu, branch
    )
    branch_field, branch_moment, branch_std = _sorted_unique_xy_std(
        branch_field, branch_moment, branch_std
    )
    local_indices = _select_local_indices(
        branch_field, crossing_value_mT, half_width_mT, min_points
    )
    local_x = np.asarray(branch_field[local_indices], dtype=float)
    local_y = np.asarray(branch_moment[local_indices], dtype=float)
    local_std = np.asarray(branch_std[local_indices], dtype=float)
    if local_x.size < 2:
        return {
            "error": None,
            "noise": None,
            "slope": None,
            "used_instrument_fallback": False,
            "flags": [f"insufficient_zero_window_points_{label}_hc"],
        }

    noise, used_instrument_fallback = _estimate_local_noise(local_y, local_std)
    if noise is None:
        return {
            "error": None,
            "noise": None,
            "slope": None,
            "used_instrument_fallback": used_instrument_fallback,
            "flags": [f"insufficient_zero_window_points_{label}_hc"],
        }

    slope = float(np.polyfit(local_x, local_y, deg=1)[0])
    if abs(slope) < min_switching_slope_emu_per_mT:
        flags.append(f"low_switching_slope_{label}")
        return {
            "error": None,
            "noise": noise,
            "slope": slope,
            "used_instrument_fallback": used_instrument_fallback,
            "flags": flags,
        }

    error = float((noise / abs(slope)) * uncertainty_scale)
    return {
        "error": error,
        "noise": noise,
        "slope": slope,
        "used_instrument_fallback": used_instrument_fallback,
        "flags": flags,
    }


def _compute_loop_area_error(
    *,
    field_mT: np.ndarray,
    moment_emu: np.ndarray,
    branches: list[BranchSegment],
    base_loop_area_emu_mT: float | None,
    recipe: VsmPreprocessingRecipe,
) -> tuple[float | None, dict[str, float | None], list[str]]:
    variants: dict[str, float | None] = {
        "nominal_union_grid": base_loop_area_emu_mT,
    }
    dense_area, _, _, _ = _compute_loop_area(
        field_mT, moment_emu, branches, grid_mode="dense", grid_points=128
    )
    variants["dense_grid"] = dense_area

    smoothed_moment = _smooth_for_loop_area(moment_emu, recipe)
    if smoothed_moment is None:
        variants["smoothed_union_grid"] = None
    else:
        smoothed_area, _, _, _ = _compute_loop_area(
            field_mT, smoothed_moment, branches, grid_mode="union"
        )
        variants["smoothed_union_grid"] = smoothed_area

    valid_values = [float(value) for value in variants.values() if value is not None]
    if len(valid_values) < 2:
        return None, variants, ["loop_area_stability_unavailable"]

    return float(np.std(valid_values, ddof=1) * recipe.uncertainty_scale), variants, []


def _smooth_for_loop_area(
    moment_emu: np.ndarray, recipe: VsmPreprocessingRecipe
) -> np.ndarray | None:
    window = int(recipe.uncertainty_loop_area_smoothing_window)
    if window % 2 == 0:
        window += 1
    if moment_emu.size < window:
        return None
    return np.asarray(
        savgol_filter(
            np.asarray(moment_emu, dtype=float),
            window_length=window,
            polyorder=recipe.uncertainty_loop_area_smoothing_polyorder,
        ),
        dtype=float,
    )


def _estimate_local_noise(
    values: np.ndarray, std_err_values: np.ndarray
) -> tuple[float | None, bool]:
    values = np.asarray(values, dtype=float)
    if values.size >= 2:
        empirical_noise = float(np.std(values, ddof=1))
        if np.isfinite(empirical_noise):
            return empirical_noise, False
    return _estimate_instrument_standard_error(std_err_values, values.size)


def _estimate_instrument_standard_error(
    std_err_values: np.ndarray, point_count: int
) -> tuple[float | None, bool]:
    finite_std = np.asarray(std_err_values, dtype=float)
    finite_std = finite_std[np.isfinite(finite_std) & (finite_std > 0.0)]
    if finite_std.size == 0 or point_count <= 0:
        return None, False
    return float(np.sqrt(np.sum(finite_std**2)) / point_count), True


def _select_local_indices(
    x: np.ndarray, center: float, half_width: float, min_points: int
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    indices = np.flatnonzero(np.abs(x - center) <= half_width)
    if indices.size >= min_points:
        return np.asarray(indices, dtype=int)
    nearest_order = np.argsort(np.abs(x - center))
    nearest = nearest_order[: min(min_points, x.size)]
    return np.asarray(np.sort(nearest), dtype=int)


def _compute_ratio_error(
    *,
    numerator: float | None,
    denominator: float | None,
    numerator_error: float | None,
    denominator_error: float | None,
) -> float | None:
    if (
        numerator is None
        or denominator is None
        or numerator_error is None
        or denominator_error is None
    ):
        return None
    if abs(denominator) <= 1e-18:
        return None
    return float(
        np.sqrt(
            (numerator_error / denominator) ** 2
            + ((numerator * denominator_error) / (denominator**2)) ** 2
        )
    )


def _relative_difference(first: float | None, second: float | None) -> float:
    if first is None or second is None:
        return 1.0
    denominator = max(abs(first), abs(second), 1e-18)
    return float(np.clip(abs(first - second) / denominator, 0.0, 1.0))


def _tail_branch_mismatch(
    *,
    field_mT: np.ndarray,
    moment_emu: np.ndarray,
    increasing_branch: BranchSegment,
    decreasing_branch: BranchSegment,
    tail_indices: np.ndarray,
    moment_span: float,
) -> float:
    if tail_indices.size == 0:
        return 1.0

    increasing_mask = _indices_in_branch(
        tail_indices,
        start_index=increasing_branch.start_index,
        end_index=increasing_branch.end_index,
    )
    decreasing_mask = _indices_in_branch(
        tail_indices,
        start_index=decreasing_branch.start_index,
        end_index=decreasing_branch.end_index,
    )
    if not np.any(increasing_mask) or not np.any(decreasing_mask):
        return 1.0

    increasing_mean = float(np.mean(moment_emu[tail_indices[increasing_mask]]))
    decreasing_mean = float(np.mean(moment_emu[tail_indices[decreasing_mask]]))
    return float(np.clip(abs(increasing_mean - decreasing_mean) / moment_span, 0.0, 1.0))


def _indices_in_branch(indices: np.ndarray, *, start_index: int, end_index: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    return (indices >= int(start_index)) & (indices <= int(end_index))


def _slope_from_indices(field_mT: np.ndarray, moment_emu: np.ndarray, indices: np.ndarray) -> float:
    indices = np.asarray(indices, dtype=int)
    if indices.size < 2:
        return 0.0
    coefficients = np.polyfit(
        np.asarray(field_mT[indices], dtype=float),
        np.asarray(moment_emu[indices], dtype=float),
        deg=1,
    )
    return float(coefficients[0])
