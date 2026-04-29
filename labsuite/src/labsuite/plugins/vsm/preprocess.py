"""Preprocessing helpers for VSM loop analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from labsuite.core.measurement_models import FitResult
from labsuite.core.recipes import VsmPreprocessingRecipe
from labsuite.plugins.vsm.derived import summarize_loop_quality
from labsuite.plugins.vsm.models import BranchSegment, VsmDataset
from labsuite.plugins.vsm.quality import evaluate_vsm_subtraction_quality, quality_to_dict


def apply_vsm_preprocessing(
    dataset: VsmDataset,
    recipe: VsmPreprocessingRecipe,
) -> tuple[np.ndarray, list[dict[str, Any]], list[str]]:
    """Apply optional smoothing while recording provenance."""

    processed_moment = np.asarray(dataset.moment_emu, dtype=float).copy()
    steps: list[dict[str, Any]] = [
        {
            "step": "unit_standardization",
            "field_unit": "mT",
            "moment_unit": "emu",
        }
    ]
    warnings: list[str] = []

    if not recipe.smoothing_enabled:
        steps.append(
            {
                "step": "smoothing",
                "enabled": False,
                "method": "none",
            }
        )
        return processed_moment, steps, warnings

    window = recipe.smoothing_window
    if window % 2 == 0:
        window += 1
    if processed_moment.size < window:
        warnings.append("smoothing_requested_but_window_exceeds_point_count")
        steps.append(
            {
                "step": "smoothing",
                "enabled": False,
                "method": "savgol",
                "reason": "window_exceeds_point_count",
            }
        )
        return processed_moment, steps, warnings

    processed_moment = np.asarray(
        savgol_filter(processed_moment, window_length=window, polyorder=recipe.smoothing_polyorder),
        dtype=float,
    )
    steps.append(
        {
            "step": "smoothing",
            "enabled": True,
            "method": "savgol",
            "window": window,
            "polyorder": recipe.smoothing_polyorder,
        }
    )
    return processed_moment, steps, warnings


def split_vsm_branches(field_mT: np.ndarray) -> tuple[np.ndarray, list[BranchSegment]]:
    """Split acquisition-order field data into monotonic branches."""

    if field_mT.size < 2:
        branch_ids = np.zeros(field_mT.size, dtype=int)
        branch = BranchSegment(
            branch_id=0,
            direction="flat",
            start_index=0,
            end_index=max(field_mT.size - 1, 0),
            point_count=int(field_mT.size),
            field_start_mT=float(field_mT[0]) if field_mT.size else 0.0,
            field_end_mT=float(field_mT[-1]) if field_mT.size else 0.0,
        )
        return branch_ids, [branch]

    diffs = np.diff(field_mT)
    branch_boundaries: list[tuple[int, int, int]] = []
    branch_start = 0
    current_sign = 0

    for diff_index, diff_value in enumerate(diffs):
        sign = 0 if diff_value == 0.0 else (1 if diff_value > 0.0 else -1)
        if sign == 0:
            continue
        if current_sign == 0:
            current_sign = sign
            continue
        if sign != current_sign:
            branch_boundaries.append((branch_start, diff_index, current_sign))
            branch_start = diff_index + 1
            current_sign = sign

    if current_sign == 0:
        current_sign = 1
    branch_boundaries.append((branch_start, field_mT.size - 1, current_sign))

    branch_ids = np.zeros(field_mT.size, dtype=int)
    branches: list[BranchSegment] = []
    for branch_id, (start_index, end_index, sign) in enumerate(branch_boundaries):
        branch_ids[start_index : end_index + 1] = branch_id
        direction = "increasing" if sign >= 0 else "decreasing"
        branches.append(
            BranchSegment(
                branch_id=branch_id,
                direction=direction,
                start_index=start_index,
                end_index=end_index,
                point_count=end_index - start_index + 1,
                field_start_mT=float(field_mT[start_index]),
                field_end_mT=float(field_mT[end_index]),
            )
        )
    return branch_ids, branches


def fit_background_slope(
    field_mT: np.ndarray,
    moment_emu: np.ndarray,
    moment_std_err_emu: np.ndarray,
    branches: list[BranchSegment],
    recipe: VsmPreprocessingRecipe,
    *,
    temperature_k: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    """Fit split high-field tails, evaluate the correction, and select a final loop."""

    positive_indices, negative_indices, selection_details, selection_warnings = select_tail_indices(
        field_mT,
        moment_emu,
        moment_std_err_emu,
        recipe,
    )
    positive_fit = selection_details["positive_fit"]
    negative_fit = selection_details["negative_fit"]
    positive_indices, negative_indices, positive_fit, negative_fit, rescue_metadata = (
        _apply_soft_warning_tail_window_rescue(
            field_mT=field_mT,
            moment_emu=moment_emu,
            moment_std_err_emu=moment_std_err_emu,
            positive_indices=positive_indices,
            negative_indices=negative_indices,
            positive_fit=positive_fit,
            negative_fit=negative_fit,
            selection_details=selection_details,
            recipe=recipe,
        )
    )
    selection_details["tail_window_metadata"].update(rescue_metadata)

    applied_slope = _mean_of_available(
        positive_fit.parameters.get("slope_emu_per_mT"),
        negative_fit.parameters.get("slope_emu_per_mT"),
    )
    average_intercept = _mean_of_available(
        positive_fit.parameters.get("intercept_emu"),
        negative_fit.parameters.get("intercept_emu"),
    )
    uncorrected_moment = np.asarray(moment_emu, dtype=float)
    slope_corrected_moment = np.asarray(moment_emu - applied_slope * field_mT, dtype=float)

    positive_tail_mask = np.zeros(field_mT.size, dtype=bool)
    negative_tail_mask = np.zeros(field_mT.size, dtype=bool)
    positive_tail_mask[positive_indices] = True
    negative_tail_mask[negative_indices] = True
    combined_tail_mask = positive_tail_mask | negative_tail_mask

    evaluation, qc_warnings = _evaluate_background_mode(
        field_mT=field_mT,
        uncorrected_moment_emu=uncorrected_moment,
        slope_corrected_moment_emu=slope_corrected_moment,
        positive_indices=positive_indices,
        negative_indices=negative_indices,
        branches=branches,
        positive_fit=positive_fit,
        negative_fit=negative_fit,
        recipe=recipe,
        temperature_k=temperature_k,
        tail_selection_metadata=selection_details["tail_window_metadata"],
    )
    legacy_background_mode = str(evaluation["background_mode"])
    quality = evaluate_vsm_subtraction_quality(
        field_mT=field_mT,
        raw_moment_emu=uncorrected_moment,
        corrected_moment_emu=slope_corrected_moment,
        positive_tail_indices=positive_indices,
        negative_tail_indices=negative_indices,
        positive_fit=positive_fit,
        negative_fit=negative_fit,
        corrected_metrics=evaluation["corrected_candidate_metrics"],
        background_slope=applied_slope,
        recipe=recipe,
        method="slope_only",
        hcut_fraction=recipe.background_tail_fraction,
    )
    quality_payload = quality_to_dict(quality)
    meaningful_correction = bool(
        evaluation.get("decision_checks", {}).get("meaningful_slope", False)
    )
    if recipe.vsm_quality_model == "legacy":
        background_mode = legacy_background_mode
        decision_reason = str(evaluation["decision_reason"])
        correction_accepted = bool(evaluation["correction_accepted"])
        qc_passed = bool(evaluation["qc_passed"])
    elif not meaningful_correction:
        background_mode = "none"
        decision_reason = "slope_below_meaningful_threshold"
        correction_accepted = False
        qc_passed = True
    elif quality.status in {"accept", "downweight"}:
        background_mode = "slope_only"
        decision_reason = f"vsm_quality_{quality.status}"
        correction_accepted = True
        qc_passed = True
    else:
        background_mode = "rejected"
        decision_reason = quality.reasons[0] if quality.reasons else "vsm_quality_rejected"
        correction_accepted = False
        qc_passed = False

    evaluation.update(
        {
            "quality_model": recipe.vsm_quality_model,
            "quality": quality_payload,
            "quality_status": quality.status,
            "quality_weight": quality.weight,
            "quality_reasons": list(quality.reasons),
            "legacy_background_mode": legacy_background_mode,
            "legacy_correction_accepted": bool(evaluation["correction_accepted"]),
            "legacy_decision_reason": evaluation["decision_reason"],
            "legacy_qc_passed": bool(evaluation["qc_passed"]),
            "background_mode": background_mode,
            "correction_accepted": correction_accepted,
            "decision_reason": decision_reason,
            "passed": qc_passed,
            "qc_passed": qc_passed,
        }
    )
    final_moment = (
        np.asarray(slope_corrected_moment, dtype=float)
        if background_mode == "slope_only"
        else np.asarray(uncorrected_moment, dtype=float)
    )

    background_fit = {
        "positive_tail_fit": positive_fit,
        "negative_tail_fit": negative_fit,
        "combined_background": {
            "slope_emu_per_mT": applied_slope,
            "intercept_emu": average_intercept,
            "positive_slope_emu_per_mT": positive_fit.parameters.get("slope_emu_per_mT"),
            "negative_slope_emu_per_mT": negative_fit.parameters.get("slope_emu_per_mT"),
            "positive_intercept_emu": positive_fit.parameters.get("intercept_emu"),
            "negative_intercept_emu": negative_fit.parameters.get("intercept_emu"),
            "selected_positive_indices": [int(index) for index in positive_indices],
            "selected_negative_indices": [int(index) for index in negative_indices],
            **selection_details["tail_window_metadata"],
            "background_tail_flatness_ratio_tolerance": (
                recipe.background_tail_flatness_ratio_tolerance
            ),
            "positive_tail_flatness_ratio_tolerance": (
                recipe.background_tail_flatness_ratio_tolerance
            ),
            "raw_tail_slope_disagreement_ratio_tolerance": (
                recipe.background_slope_disagreement_ratio_tolerance
            ),
            "background_mode": background_mode,
            "subtraction_mode": background_mode,
            "used_intercept_in_correction": False,
            "correction_accepted": correction_accepted,
            "decision_reason": decision_reason,
            "decision_checks": evaluation["decision_checks"],
            "qc_passed": qc_passed,
            "quality": quality_payload,
            "quality_model": recipe.vsm_quality_model,
            "quality_status": quality.status,
            "quality_weight": quality.weight,
            "quality_reasons": list(quality.reasons),
            "legacy_background_mode": legacy_background_mode,
            "legacy_correction_accepted": bool(evaluation["legacy_correction_accepted"]),
            "legacy_decision_reason": evaluation["legacy_decision_reason"],
            "legacy_qc_passed": bool(evaluation["legacy_qc_passed"]),
            "qc": evaluation,
        },
    }
    loop_variants = {
        "uncorrected_moment_emu": uncorrected_moment,
        "slope_corrected_moment_emu": slope_corrected_moment,
        "final_moment_emu": final_moment,
        "final_field_mT": np.asarray(field_mT, dtype=float),
    }
    tail_masks = {
        "positive_tail_mask": positive_tail_mask,
        "negative_tail_mask": negative_tail_mask,
        "combined_tail_mask": combined_tail_mask,
    }
    effective_qc_warnings = list(qc_warnings)
    if recipe.vsm_quality_model == "simple" and background_mode != legacy_background_mode:
        effective_qc_warnings = [
            f"legacy_{warning}" if warning.startswith("background_fit_rejected_") else warning
            for warning in effective_qc_warnings
        ]
        if quality.status == "downweight":
            effective_qc_warnings.append("vsm_quality_downweighted")
        elif quality.status == "reject":
            effective_qc_warnings.append("vsm_quality_rejected")
    return background_fit, loop_variants, tail_masks, [*selection_warnings, *effective_qc_warnings]


def select_tail_indices(
    field_mT: np.ndarray,
    moment_emu: np.ndarray,
    moment_std_err_emu: np.ndarray,
    recipe: VsmPreprocessingRecipe,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[str]]:
    """Select symmetric positive and negative high-field tails."""

    max_abs_field = float(np.max(np.abs(field_mT)))
    threshold = max_abs_field * (1.0 - recipe.background_tail_fraction)

    positive_initial_indices = _initial_tail_indices(
        field_mT,
        threshold=threshold,
        min_points=recipe.background_min_points_per_side,
        positive=True,
    )
    negative_initial_indices = _initial_tail_indices(
        field_mT,
        threshold=threshold,
        min_points=recipe.background_min_points_per_side,
        positive=False,
    )
    positive_indices, positive_fit, positive_selection = _select_adaptive_tail_window(
        field_mT=field_mT,
        moment_emu=moment_emu,
        moment_std_err_emu=moment_std_err_emu,
        initial_indices=positive_initial_indices,
        recipe=recipe,
        label="positive_tail",
        positive=True,
    )
    negative_indices, negative_fit, negative_selection = _select_adaptive_tail_window(
        field_mT=field_mT,
        moment_emu=moment_emu,
        moment_std_err_emu=moment_std_err_emu,
        initial_indices=negative_initial_indices,
        recipe=recipe,
        label="negative_tail",
        positive=False,
    )

    warnings: list[str] = []
    if positive_indices.size < recipe.background_min_points_per_side:
        warnings.append("positive_tail_has_insufficient_points")
    if negative_indices.size < recipe.background_min_points_per_side:
        warnings.append("negative_tail_has_insufficient_points")
    if np.any(field_mT[positive_indices] <= 0.0):
        warnings.append("positive_tail_contains_non_positive_fields")
    if np.any(field_mT[negative_indices] >= 0.0):
        warnings.append("negative_tail_contains_non_negative_fields")

    return (
        positive_indices,
        negative_indices,
        {
            "positive_fit": positive_fit,
            "negative_fit": negative_fit,
            "positive_candidate_windows": _build_tail_window_candidates(
                field_mT=field_mT,
                ordered_indices=positive_initial_indices,
                minimum_count=min(
                    recipe.background_min_points_per_side, positive_initial_indices.size
                ),
            ),
            "negative_candidate_windows": _build_tail_window_candidates(
                field_mT=field_mT,
                ordered_indices=negative_initial_indices,
                minimum_count=min(
                    recipe.background_min_points_per_side, negative_initial_indices.size
                ),
            ),
            "tail_window_metadata": {
                "tail_window_selection_mode": "iterative_shrink",
                "positive_tail_window_initial_point_count": positive_selection[
                    "initial_point_count"
                ],
                "negative_tail_window_initial_point_count": negative_selection[
                    "initial_point_count"
                ],
                "positive_tail_window_selected_point_count": positive_selection[
                    "selected_point_count"
                ],
                "negative_tail_window_selected_point_count": negative_selection[
                    "selected_point_count"
                ],
                "positive_tail_window_initial_field_min_mT": positive_selection[
                    "initial_field_min_mT"
                ],
                "positive_tail_window_initial_field_max_mT": positive_selection[
                    "initial_field_max_mT"
                ],
                "negative_tail_window_initial_field_min_mT": negative_selection[
                    "initial_field_min_mT"
                ],
                "negative_tail_window_initial_field_max_mT": negative_selection[
                    "initial_field_max_mT"
                ],
                "positive_tail_window_selected_field_min_mT": positive_selection[
                    "selected_field_min_mT"
                ],
                "positive_tail_window_selected_field_max_mT": positive_selection[
                    "selected_field_max_mT"
                ],
                "negative_tail_window_selected_field_min_mT": negative_selection[
                    "selected_field_min_mT"
                ],
                "negative_tail_window_selected_field_max_mT": negative_selection[
                    "selected_field_max_mT"
                ],
                "positive_tail_window_soft_r_squared_rescue_attempted": False,
                "negative_tail_window_soft_r_squared_rescue_attempted": False,
                "positive_tail_window_rescue_changed_selection": False,
                "negative_tail_window_rescue_changed_selection": False,
            },
        },
        warnings,
    )


def _initial_tail_indices(
    field_mT: np.ndarray,
    *,
    threshold: float,
    min_points: int,
    positive: bool,
) -> np.ndarray:
    if positive:
        indices = np.flatnonzero(field_mT >= threshold)
        indices = indices[field_mT[indices] > 0.0]
        if indices.size < min_points:
            indices = np.flatnonzero(field_mT > 0.0)
        ordering = np.argsort(field_mT[indices])[::-1]
    else:
        indices = np.flatnonzero(field_mT <= -threshold)
        indices = indices[field_mT[indices] < 0.0]
        if indices.size < min_points:
            indices = np.flatnonzero(field_mT < 0.0)
        ordering = np.argsort(field_mT[indices])
    return np.asarray(indices[ordering], dtype=int)


def _select_adaptive_tail_window(
    *,
    field_mT: np.ndarray,
    moment_emu: np.ndarray,
    moment_std_err_emu: np.ndarray,
    initial_indices: np.ndarray,
    recipe: VsmPreprocessingRecipe,
    label: str,
    positive: bool,
) -> tuple[np.ndarray, FitResult, dict[str, float | int | None]]:
    ordered_indices = np.asarray(initial_indices, dtype=int)
    if ordered_indices.size == 0:
        fit = _fit_tail_line(
            field_mT,
            moment_emu,
            moment_std_err_emu,
            ordered_indices,
            recipe=recipe,
            label=label,
        )
        return (
            ordered_indices,
            fit,
            _build_tail_window_metadata(ordered_indices, ordered_indices, field_mT),
        )

    minimum_count = min(recipe.background_min_points_per_side, ordered_indices.size)
    candidate_windows = _build_tail_window_candidates(
        field_mT=field_mT,
        ordered_indices=ordered_indices,
        minimum_count=minimum_count,
    )
    best_rank: tuple[float, float, int] | None = None
    best_fit: FitResult | None = None
    best_indices = candidate_windows[0]
    for candidate_indices in candidate_windows:
        candidate_fit = _fit_tail_line(
            field_mT,
            moment_emu,
            moment_std_err_emu,
            candidate_indices,
            recipe=recipe,
            label=label,
        )
        candidate_rank = (
            _fit_r_squared_value(candidate_fit),
            -_normalized_tail_slope_magnitude(
                candidate_fit, field_mT, moment_emu, candidate_indices
            ),
            int(candidate_indices.size),
        )
        if best_rank is None or candidate_rank > best_rank:
            best_rank = candidate_rank
            best_fit = candidate_fit
            best_indices = candidate_indices

    assert best_fit is not None
    if positive:
        best_indices = np.asarray(best_indices[field_mT[best_indices] > 0.0], dtype=int)
    else:
        best_indices = np.asarray(best_indices[field_mT[best_indices] < 0.0], dtype=int)
    return (
        best_indices,
        best_fit,
        _build_tail_window_metadata(ordered_indices, best_indices, field_mT),
    )


def _build_tail_window_candidates(
    *,
    field_mT: np.ndarray,
    ordered_indices: np.ndarray,
    minimum_count: int,
) -> list[np.ndarray]:
    if ordered_indices.size == 0 or minimum_count <= 0:
        return [np.asarray([], dtype=int)]
    return [
        _sort_indices_by_field(field_mT, np.asarray(ordered_indices[:point_count], dtype=int))
        for point_count in range(minimum_count, ordered_indices.size + 1)
    ]


def _apply_soft_warning_tail_window_rescue(
    *,
    field_mT: np.ndarray,
    moment_emu: np.ndarray,
    moment_std_err_emu: np.ndarray,
    positive_indices: np.ndarray,
    negative_indices: np.ndarray,
    positive_fit: FitResult,
    negative_fit: FitResult,
    selection_details: dict[str, Any],
    recipe: VsmPreprocessingRecipe,
) -> tuple[np.ndarray, np.ndarray, FitResult, FitResult, dict[str, Any]]:
    rescue_metadata = {
        "positive_tail_window_soft_r_squared_rescue_attempted": False,
        "negative_tail_window_soft_r_squared_rescue_attempted": False,
        "positive_tail_window_rescue_changed_selection": False,
        "negative_tail_window_rescue_changed_selection": False,
    }
    if _is_soft_warning_r_squared(positive_fit.metrics.get("r_squared"), recipe):
        rescue_metadata["positive_tail_window_soft_r_squared_rescue_attempted"] = True
        rescued_indices, rescued_fit = _select_soft_warning_tail_window(
            field_mT=field_mT,
            moment_emu=moment_emu,
            moment_std_err_emu=moment_std_err_emu,
            candidate_windows=selection_details["positive_candidate_windows"],
            current_fit=positive_fit,
            other_fit=negative_fit,
            recipe=recipe,
            label="positive_tail",
        )
        rescue_metadata["positive_tail_window_rescue_changed_selection"] = not np.array_equal(
            rescued_indices,
            np.asarray(positive_indices, dtype=int),
        )
        positive_indices = rescued_indices
        positive_fit = rescued_fit
    if _is_soft_warning_r_squared(negative_fit.metrics.get("r_squared"), recipe):
        rescue_metadata["negative_tail_window_soft_r_squared_rescue_attempted"] = True
        rescued_indices, rescued_fit = _select_soft_warning_tail_window(
            field_mT=field_mT,
            moment_emu=moment_emu,
            moment_std_err_emu=moment_std_err_emu,
            candidate_windows=selection_details["negative_candidate_windows"],
            current_fit=negative_fit,
            other_fit=positive_fit,
            recipe=recipe,
            label="negative_tail",
        )
        rescue_metadata["negative_tail_window_rescue_changed_selection"] = not np.array_equal(
            rescued_indices,
            np.asarray(negative_indices, dtype=int),
        )
        negative_indices = rescued_indices
        negative_fit = rescued_fit
    rescue_metadata.update(
        {
            "positive_tail_window_selected_point_count": int(
                np.asarray(positive_indices, dtype=int).size
            ),
            "negative_tail_window_selected_point_count": int(
                np.asarray(negative_indices, dtype=int).size
            ),
            "positive_tail_window_selected_field_min_mT": float(np.min(field_mT[positive_indices]))
            if np.asarray(positive_indices, dtype=int).size
            else None,
            "positive_tail_window_selected_field_max_mT": float(np.max(field_mT[positive_indices]))
            if np.asarray(positive_indices, dtype=int).size
            else None,
            "negative_tail_window_selected_field_min_mT": float(np.min(field_mT[negative_indices]))
            if np.asarray(negative_indices, dtype=int).size
            else None,
            "negative_tail_window_selected_field_max_mT": float(np.max(field_mT[negative_indices]))
            if np.asarray(negative_indices, dtype=int).size
            else None,
        }
    )
    return positive_indices, negative_indices, positive_fit, negative_fit, rescue_metadata


def _select_soft_warning_tail_window(
    *,
    field_mT: np.ndarray,
    moment_emu: np.ndarray,
    moment_std_err_emu: np.ndarray,
    candidate_windows: list[np.ndarray],
    current_fit: FitResult,
    other_fit: FitResult,
    recipe: VsmPreprocessingRecipe,
    label: str,
) -> tuple[np.ndarray, FitResult]:
    best_rank: tuple[float, float, int] | None = None
    best_indices = np.asarray(current_fit.selected_indices, dtype=int)
    best_fit = current_fit
    for candidate_indices in candidate_windows:
        candidate_fit = _fit_tail_line(
            field_mT,
            moment_emu,
            moment_std_err_emu,
            candidate_indices,
            recipe=recipe,
            label=label,
        )
        candidate_applied_slope = _mean_of_available(
            candidate_fit.parameters.get("slope_emu_per_mT"),
            other_fit.parameters.get("slope_emu_per_mT"),
        )
        candidate_corrected_moment = np.asarray(
            moment_emu - candidate_applied_slope * field_mT, dtype=float
        )
        candidate_gain = _candidate_tail_flatness_gain(
            field_mT=field_mT,
            raw_moment_emu=moment_emu,
            corrected_moment_emu=candidate_corrected_moment,
            moment_std_err_emu=moment_std_err_emu,
            indices=candidate_indices,
            recipe=recipe,
            label=label,
        )
        candidate_rank = (
            np.clip(candidate_gain, 0.0, 1.0),
            _fit_r_squared_value(candidate_fit),
            int(candidate_indices.size),
        )
        if best_rank is None or candidate_rank > best_rank:
            best_rank = candidate_rank
            best_indices = np.asarray(candidate_indices, dtype=int)
            best_fit = candidate_fit
    return best_indices, best_fit


def _candidate_tail_flatness_gain(
    *,
    field_mT: np.ndarray,
    raw_moment_emu: np.ndarray,
    corrected_moment_emu: np.ndarray,
    moment_std_err_emu: np.ndarray,
    indices: np.ndarray,
    recipe: VsmPreprocessingRecipe,
    label: str,
) -> float:
    raw_fit = _fit_tail_line(
        field_mT,
        raw_moment_emu,
        moment_std_err_emu,
        indices,
        recipe=recipe,
        label=f"{label}_raw_eval",
    )
    corrected_fit = _fit_tail_line(
        field_mT,
        corrected_moment_emu,
        moment_std_err_emu,
        indices,
        recipe=recipe,
        label=f"{label}_corrected_eval",
    )
    raw_normalized = _normalized_tail_slope_magnitude(raw_fit, field_mT, raw_moment_emu, indices)
    corrected_normalized = _normalized_tail_slope_magnitude(
        corrected_fit, field_mT, corrected_moment_emu, indices
    )
    return _flatness_gain(raw_normalized, corrected_normalized)


def _build_tail_window_metadata(
    initial_indices: np.ndarray,
    selected_indices: np.ndarray,
    field_mT: np.ndarray,
) -> dict[str, float | int | None]:
    initial_fields = (
        np.asarray(field_mT[initial_indices], dtype=float)
        if initial_indices.size
        else np.asarray([], dtype=float)
    )
    selected_fields = (
        np.asarray(field_mT[selected_indices], dtype=float)
        if selected_indices.size
        else np.asarray([], dtype=float)
    )
    return {
        "initial_point_count": int(initial_indices.size),
        "selected_point_count": int(selected_indices.size),
        "initial_field_min_mT": float(np.min(initial_fields)) if initial_fields.size else None,
        "initial_field_max_mT": float(np.max(initial_fields)) if initial_fields.size else None,
        "selected_field_min_mT": float(np.min(selected_fields)) if selected_fields.size else None,
        "selected_field_max_mT": float(np.max(selected_fields)) if selected_fields.size else None,
    }


def _sort_indices_by_field(field_mT: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        return np.asarray(indices, dtype=int)
    return np.asarray(indices[np.argsort(field_mT[indices])], dtype=int)


def _fit_r_squared_value(fit: FitResult) -> float:
    r_squared = fit.metrics.get("r_squared")
    return float(r_squared) if r_squared is not None else float("-inf")


def _normalized_tail_slope_magnitude(
    fit: FitResult,
    field_mT: np.ndarray,
    moment_emu: np.ndarray,
    indices: np.ndarray,
) -> float:
    if indices.size == 0:
        return float("inf")
    x_selected = np.asarray(field_mT[indices], dtype=float)
    y_selected = np.asarray(moment_emu[indices], dtype=float)
    slope = float(fit.parameters.get("slope_emu_per_mT", 0.0))
    scale = max(
        float(np.mean(np.abs(y_selected))) / max(float(np.max(np.abs(x_selected))), 1e-18), 1e-18
    )
    return float(abs(slope) / scale)


def _fit_tail_line(
    field_mT: np.ndarray,
    moment_emu: np.ndarray,
    moment_std_err_emu: np.ndarray,
    indices: np.ndarray,
    *,
    recipe: VsmPreprocessingRecipe,
    label: str,
) -> FitResult:
    """Fit one saturation tail to a straight line."""

    if indices.size < 2:
        return FitResult(
            model_name="linear_tail_fit",
            parameters={
                "slope_emu_per_mT": 0.0,
                "intercept_emu": float(np.mean(moment_emu[indices])) if indices.size else 0.0,
            },
            metrics={
                "r_squared": None,
                "rmse_emu": None,
                "selected_point_count": float(indices.size),
            },
            success=False,
            message=f"{label}_insufficient_points",
            selected_indices=[int(index) for index in indices],
            fitted_x=[float(value) for value in field_mT[indices]],
            fitted_y=[float(value) for value in moment_emu[indices]],
            residual_y=[0.0 for _ in range(indices.size)],
        )

    x_selected = np.asarray(field_mT[indices], dtype=float)
    y_selected = np.asarray(moment_emu[indices], dtype=float)
    weights = _build_fit_weights(moment_std_err_emu[indices], recipe)
    coefficients = np.polyfit(x_selected, y_selected, deg=1, w=weights)
    slope, intercept = float(coefficients[0]), float(coefficients[1])
    fitted_y = slope * x_selected + intercept
    residual = y_selected - fitted_y
    centered = y_selected - float(np.mean(y_selected))
    rss = float(np.sum(residual**2))
    tss = float(np.sum(centered**2))

    return FitResult(
        model_name="linear_tail_fit",
        parameters={
            "slope_emu_per_mT": slope,
            "intercept_emu": intercept,
        },
        metrics={
            "r_squared": None if tss <= 0.0 else 1.0 - rss / tss,
            "rmse_emu": float(np.sqrt(np.mean(residual**2))),
            "selected_point_count": float(indices.size),
        },
        success=True,
        message=f"{label}_fit",
        selected_indices=[int(index) for index in indices],
        fitted_x=[float(value) for value in x_selected],
        fitted_y=[float(value) for value in fitted_y],
        residual_y=[float(value) for value in residual],
    )


def _build_fit_weights(
    moment_std_err_emu: np.ndarray, recipe: VsmPreprocessingRecipe
) -> np.ndarray | None:
    finite_std_err = np.asarray(moment_std_err_emu, dtype=float)
    valid_weights = np.isfinite(finite_std_err) & (finite_std_err > 0.0)
    if not np.any(valid_weights):
        return None
    scaled = np.maximum(finite_std_err[valid_weights] * recipe.uncertainty_scale, 1e-18)
    weights = np.ones(finite_std_err.size, dtype=float)
    weights[valid_weights] = 1.0 / scaled
    return weights


def _evaluate_background_mode(
    field_mT: np.ndarray,
    *,
    uncorrected_moment_emu: np.ndarray,
    slope_corrected_moment_emu: np.ndarray,
    positive_indices: np.ndarray,
    negative_indices: np.ndarray,
    branches: list[BranchSegment],
    positive_fit: FitResult,
    negative_fit: FitResult,
    recipe: VsmPreprocessingRecipe,
    temperature_k: np.ndarray | None,
    tail_selection_metadata: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Evaluate split-tail background correction quality and select a mode."""

    raw_metrics = summarize_loop_quality(
        field_mT=field_mT,
        moment_emu=uncorrected_moment_emu,
        branches=branches,
        positive_tail_indices=positive_indices,
        negative_tail_indices=negative_indices,
        temperature_k=temperature_k,
    )
    corrected_metrics = summarize_loop_quality(
        field_mT=field_mT,
        moment_emu=slope_corrected_moment_emu,
        branches=branches,
        positive_tail_indices=positive_indices,
        negative_tail_indices=negative_indices,
        temperature_k=temperature_k,
    )
    raw_slope_disagreement_ratio = _relative_difference(
        positive_fit.parameters.get("slope_emu_per_mT", 0.0),
        negative_fit.parameters.get("slope_emu_per_mT", 0.0),
    )
    positive_corrected_slope = float(corrected_metrics["plateau_slope_positive_emu_per_mT"])
    negative_corrected_slope = float(corrected_metrics["plateau_slope_negative_emu_per_mT"])
    corrected_tail_slope_abs_mismatch = abs(positive_corrected_slope - negative_corrected_slope)
    positive_flatness_gain = _flatness_gain(
        float(raw_metrics["plateau_slope_positive_normalized"]),
        float(corrected_metrics["plateau_slope_positive_normalized"]),
    )
    negative_flatness_gain = _flatness_gain(
        float(raw_metrics["plateau_slope_negative_normalized"]),
        float(corrected_metrics["plateau_slope_negative_normalized"]),
    )
    flatness_gain_score = float(
        np.mean(
            [
                np.clip(positive_flatness_gain, 0.0, 1.0),
                np.clip(negative_flatness_gain, 0.0, 1.0),
            ]
        )
    )
    flatness_gain_balance_score = float(
        np.clip(
            1.0
            - _relative_difference(
                np.clip(positive_flatness_gain, 0.0, 1.0),
                np.clip(negative_flatness_gain, 0.0, 1.0),
            ),
            0.0,
            1.0,
        )
    )
    switching_width_relative_change = _relative_change(
        raw_metrics.get("switching_width_mT"),
        corrected_metrics.get("switching_width_mT"),
    )
    zero_crossing_candidate_increase = int(
        corrected_metrics["zero_crossing_candidate_count"]
        - raw_metrics["zero_crossing_candidate_count"]
    )
    coercive_ambiguity_worsening = int(
        corrected_metrics["coercive_ambiguity_count"] - raw_metrics["coercive_ambiguity_count"]
    )
    switching_asymmetry_increase = float(
        corrected_metrics["switching_asymmetry_ratio"] - raw_metrics["switching_asymmetry_ratio"]
    )
    raw_switching_integrity_score = _compute_switching_integrity_quality(
        metrics=raw_metrics,
        baseline_metrics=raw_metrics,
        switching_width_relative_change=0.0,
        zero_crossing_candidate_increase=0,
        coercive_ambiguity_worsening=0,
        switching_asymmetry_increase=0.0,
    )
    corrected_switching_integrity_score = _compute_switching_integrity_quality(
        metrics=corrected_metrics,
        baseline_metrics=raw_metrics,
        switching_width_relative_change=switching_width_relative_change,
        zero_crossing_candidate_increase=zero_crossing_candidate_increase,
        coercive_ambiguity_worsening=coercive_ambiguity_worsening,
        switching_asymmetry_increase=switching_asymmetry_increase,
    )
    score_raw, score_components_raw = _compute_background_quality_score(raw_metrics, recipe)
    score_corrected, score_components_corrected = _compute_background_quality_score(
        corrected_metrics,
        recipe,
        flatness_gain_score=flatness_gain_score,
        flatness_gain_balance_score=flatness_gain_balance_score,
        switching_integrity_score=corrected_switching_integrity_score,
    )
    score_delta = float(score_corrected - score_raw)

    positive_fit_r_squared = positive_fit.metrics.get("r_squared")
    negative_fit_r_squared = negative_fit.metrics.get("r_squared")
    applied_slope = _mean_of_available(
        positive_fit.parameters.get("slope_emu_per_mT"),
        negative_fit.parameters.get("slope_emu_per_mT"),
    )

    decision_checks = {
        "meaningful_slope": bool(
            abs(applied_slope) >= recipe.background_min_meaningful_slope_emu_per_mT
        ),
        "positive_tail_fit_success": bool(positive_fit.success),
        "negative_tail_fit_success": bool(negative_fit.success),
        "positive_tail_fit_r_squared_soft_ok": bool(
            positive_fit_r_squared is not None
            and positive_fit_r_squared >= recipe.background_tail_fit_min_r_squared
        ),
        "negative_tail_fit_r_squared_soft_ok": bool(
            negative_fit_r_squared is not None
            and negative_fit_r_squared >= recipe.background_tail_fit_min_r_squared
        ),
        "positive_tail_fit_r_squared_catastrophic_ok": bool(
            positive_fit_r_squared is not None
            and positive_fit_r_squared >= recipe.background_tail_fit_catastrophic_r_squared
        ),
        "negative_tail_fit_r_squared_catastrophic_ok": bool(
            negative_fit_r_squared is not None
            and negative_fit_r_squared >= recipe.background_tail_fit_catastrophic_r_squared
        ),
        "tail_slope_disagreement_ok": bool(
            raw_slope_disagreement_ratio <= recipe.background_slope_disagreement_ratio_tolerance
        ),
        "positive_tail_flatness_regression_ok": bool(
            float(corrected_metrics["plateau_slope_positive_normalized"])
            - float(raw_metrics["plateau_slope_positive_normalized"])
            <= recipe.background_max_tail_flatness_regression
        ),
        "negative_tail_flatness_regression_ok": bool(
            float(corrected_metrics["plateau_slope_negative_normalized"])
            - float(raw_metrics["plateau_slope_negative_normalized"])
            <= recipe.background_max_tail_flatness_regression
        ),
        "flatness_gain_score_ok": bool(
            flatness_gain_score >= recipe.background_min_flatness_gain_score
        ),
        "positive_tail_flatness_gain_override_ok": bool(
            np.clip(positive_flatness_gain, 0.0, 1.0)
            >= recipe.background_tail_fit_override_min_flatness_gain_per_tail
        ),
        "negative_tail_flatness_gain_override_ok": bool(
            np.clip(negative_flatness_gain, 0.0, 1.0)
            >= recipe.background_tail_fit_override_min_flatness_gain_per_tail
        ),
        "flatness_gain_balance_ok": bool(
            flatness_gain_balance_score
            >= recipe.background_tail_fit_override_min_gain_balance_score
        ),
        "corrected_zero_crossing_increase_ok": bool(
            zero_crossing_candidate_increase <= recipe.background_max_zero_crossing_increase
        ),
        "corrected_switching_width_available_or_not_needed": bool(
            raw_metrics.get("switching_width_mT") is None
            or corrected_metrics.get("switching_width_mT") is not None
        ),
        "corrected_switching_width_change_ok": bool(
            switching_width_relative_change is not None
            and switching_width_relative_change
            <= recipe.background_max_switching_width_relative_change
        ),
        "corrected_coercive_ambiguity_worsening_ok": bool(
            coercive_ambiguity_worsening <= recipe.background_max_coercive_ambiguity_worsening
        ),
        "score_improved": bool(score_delta >= recipe.background_min_score_improvement),
        "soft_tail_fit_r_squared_override_flatness_gain_ok": bool(
            flatness_gain_score >= recipe.background_tail_fit_override_min_flatness_gain_score
        ),
        "soft_tail_fit_r_squared_override_switching_integrity_ok": bool(
            corrected_switching_integrity_score
            >= recipe.background_tail_fit_override_min_switching_integrity_score
        ),
    }
    positive_tail_fit_r_squared_soft_warning = bool(
        positive_fit_r_squared is not None
        and recipe.background_tail_fit_catastrophic_r_squared
        <= positive_fit_r_squared
        < recipe.background_tail_fit_min_r_squared
    )
    negative_tail_fit_r_squared_soft_warning = bool(
        negative_fit_r_squared is not None
        and recipe.background_tail_fit_catastrophic_r_squared
        <= negative_fit_r_squared
        < recipe.background_tail_fit_min_r_squared
    )
    positive_tail_fit_r_squared_catastrophic = not decision_checks[
        "positive_tail_fit_r_squared_catastrophic_ok"
    ]
    negative_tail_fit_r_squared_catastrophic = not decision_checks[
        "negative_tail_fit_r_squared_catastrophic_ok"
    ]
    soft_tail_fit_r_squared_warning_present = bool(
        positive_tail_fit_r_squared_soft_warning or negative_tail_fit_r_squared_soft_warning
    )
    decision_checks["positive_tail_fit_r_squared_soft_warning"] = (
        positive_tail_fit_r_squared_soft_warning
    )
    decision_checks["negative_tail_fit_r_squared_soft_warning"] = (
        negative_tail_fit_r_squared_soft_warning
    )
    decision_checks["soft_tail_fit_r_squared_warning_present"] = (
        soft_tail_fit_r_squared_warning_present
    )
    soft_override_passed = bool(
        soft_tail_fit_r_squared_warning_present
        and decision_checks["positive_tail_fit_success"]
        and decision_checks["negative_tail_fit_success"]
        and decision_checks["positive_tail_fit_r_squared_catastrophic_ok"]
        and decision_checks["negative_tail_fit_r_squared_catastrophic_ok"]
        and decision_checks["positive_tail_flatness_regression_ok"]
        and decision_checks["negative_tail_flatness_regression_ok"]
        and decision_checks["positive_tail_flatness_gain_override_ok"]
        and decision_checks["negative_tail_flatness_gain_override_ok"]
        and decision_checks["soft_tail_fit_r_squared_override_flatness_gain_ok"]
        and decision_checks["flatness_gain_balance_ok"]
        and decision_checks["soft_tail_fit_r_squared_override_switching_integrity_ok"]
    )
    decision_checks["soft_tail_fit_r_squared_override_passed"] = soft_override_passed

    fit_gate_reasons = {
        "positive_tail_fit_success": "positive_tail_fit_failed",
        "negative_tail_fit_success": "negative_tail_fit_failed",
        "positive_tail_fit_r_squared_catastrophic_ok": (
            "positive_tail_fit_r_squared_catastrophically_low"
        ),
        "negative_tail_fit_r_squared_catastrophic_ok": (
            "negative_tail_fit_r_squared_catastrophically_low"
        ),
    }
    switching_gate_reasons = {
        "corrected_zero_crossing_increase_ok": "corrected_zero_crossings_increased",
        "corrected_switching_width_available_or_not_needed": (
            "corrected_switching_width_unavailable"
        ),
        "corrected_switching_width_change_ok": "corrected_switching_width_distorted",
        "corrected_coercive_ambiguity_worsening_ok": "corrected_coercive_ambiguity_worsened",
    }

    warnings: list[str] = []
    if positive_indices.size and np.any(field_mT[positive_indices] <= 0.0):
        warnings.append("background_fit_failed_positive_tail_mixed_sign")
    if negative_indices.size and np.any(field_mT[negative_indices] >= 0.0):
        warnings.append("background_fit_failed_negative_tail_mixed_sign")

    decision_reason = "score_improved"
    background_mode = "slope_only"
    fit_failures = [
        label
        for label, passed in decision_checks.items()
        if label in fit_gate_reasons and not passed
    ]
    switching_gate_failures = [
        label
        for label, passed in decision_checks.items()
        if label in switching_gate_reasons and not passed
    ]
    if warnings:
        background_mode = "rejected"
        decision_reason = warnings[0]
    elif not decision_checks["meaningful_slope"]:
        background_mode = "none"
        decision_reason = "slope_below_meaningful_threshold"
    elif fit_failures:
        background_mode = "rejected"
        decision_reason = fit_gate_reasons[fit_failures[0]]
    elif switching_gate_failures:
        background_mode = "rejected"
        decision_reason = switching_gate_reasons[switching_gate_failures[0]]
    elif not decision_checks["positive_tail_flatness_regression_ok"]:
        background_mode = "none"
        decision_reason = "positive_tail_flatness_regressed"
    elif not decision_checks["negative_tail_flatness_regression_ok"]:
        background_mode = "none"
        decision_reason = "negative_tail_flatness_regressed"
    elif soft_tail_fit_r_squared_warning_present:
        if soft_override_passed:
            background_mode = "slope_only"
            decision_reason = "soft_tail_fit_r_squared_overridden"
        else:
            background_mode = "none"
            if not decision_checks["positive_tail_flatness_gain_override_ok"]:
                decision_reason = "soft_tail_fit_r_squared_not_overridden_low_positive_gain"
            elif not decision_checks["negative_tail_flatness_gain_override_ok"]:
                decision_reason = "soft_tail_fit_r_squared_not_overridden_low_negative_gain"
            elif not decision_checks["soft_tail_fit_r_squared_override_flatness_gain_ok"]:
                decision_reason = "soft_tail_fit_r_squared_not_overridden_low_combined_gain"
            elif not decision_checks["flatness_gain_balance_ok"]:
                decision_reason = "soft_tail_fit_r_squared_not_overridden_unbalanced_tail_gains"
            elif not decision_checks["soft_tail_fit_r_squared_override_switching_integrity_ok"]:
                decision_reason = "soft_tail_fit_r_squared_not_overridden_switching_integrity"
            else:
                decision_reason = "soft_tail_fit_r_squared_not_overridden"
    elif not decision_checks["flatness_gain_score_ok"]:
        background_mode = "none"
        decision_reason = "flatness_gain_below_threshold"
    elif not decision_checks["score_improved"]:
        background_mode = "none"
        decision_reason = "score_improvement_below_threshold"

    if positive_tail_fit_r_squared_soft_warning:
        warnings.append("background_fit_warning_positive_tail_fit_r_squared_below_soft_threshold")
    if negative_tail_fit_r_squared_soft_warning:
        warnings.append("background_fit_warning_negative_tail_fit_r_squared_below_soft_threshold")
    if not decision_checks["tail_slope_disagreement_ok"]:
        warnings.append("background_fit_warning_tail_slope_disagreement_above_tolerance")
    if background_mode == "rejected":
        warnings.extend(
            f"background_fit_rejected_{fit_gate_reasons[label]}" for label in fit_failures
        )
        warnings.extend(
            f"background_fit_rejected_{switching_gate_reasons[label]}"
            for label in switching_gate_failures
        )
    elif background_mode == "none":
        warnings.append(f"background_fit_optional_{decision_reason}")
    elif soft_tail_fit_r_squared_warning_present:
        warnings.append("background_fit_warning_soft_tail_fit_r_squared_overridden")

    qc_metrics = {
        "passed": background_mode != "rejected",
        "qc_passed": background_mode != "rejected",
        "background_mode": background_mode,
        "correction_accepted": background_mode == "slope_only",
        "decision_reason": decision_reason,
        "decision_checks": decision_checks,
        "corrected_positive_tail_slope_emu_per_mT": positive_corrected_slope,
        "corrected_negative_tail_slope_emu_per_mT": negative_corrected_slope,
        "corrected_tail_slope_abs_mismatch_emu_per_mT": corrected_tail_slope_abs_mismatch,
        "positive_tail_flatness_ratio": float(corrected_metrics["plateau_flatness_ratio_positive"]),
        "negative_tail_flatness_ratio": float(corrected_metrics["plateau_flatness_ratio_negative"]),
        "raw_tail_slope_disagreement_ratio": raw_slope_disagreement_ratio,
        "raw_switching_integrity_score": raw_switching_integrity_score,
        "corrected_switching_integrity_score": corrected_switching_integrity_score,
        "positive_tail_fit_r_squared": positive_fit_r_squared,
        "negative_tail_fit_r_squared": negative_fit_r_squared,
        "positive_tail_fit_r_squared_soft_warning": positive_tail_fit_r_squared_soft_warning,
        "negative_tail_fit_r_squared_soft_warning": negative_tail_fit_r_squared_soft_warning,
        "positive_tail_fit_r_squared_catastrophic": positive_tail_fit_r_squared_catastrophic,
        "negative_tail_fit_r_squared_catastrophic": negative_tail_fit_r_squared_catastrophic,
        "background_flatness_gain_balance_score": flatness_gain_balance_score,
        "background_flatness_gain_balance_ok": decision_checks["flatness_gain_balance_ok"],
        "background_soft_override_passed": soft_override_passed,
        **tail_selection_metadata,
        "slope_emu_per_mT": applied_slope,
        "raw_metrics": raw_metrics,
        "corrected_candidate_metrics": corrected_metrics,
        "comparison": {
            "raw_plateau_slope_positive_normalized": float(
                raw_metrics["plateau_slope_positive_normalized"]
            ),
            "raw_plateau_slope_negative_normalized": float(
                raw_metrics["plateau_slope_negative_normalized"]
            ),
            "corrected_plateau_slope_positive_normalized": float(
                corrected_metrics["plateau_slope_positive_normalized"]
            ),
            "corrected_plateau_slope_negative_normalized": float(
                corrected_metrics["plateau_slope_negative_normalized"]
            ),
            "background_flatness_gain_positive": positive_flatness_gain,
            "background_flatness_gain_negative": negative_flatness_gain,
            "background_flatness_gain_score": flatness_gain_score,
            "background_flatness_gain_balance_score": flatness_gain_balance_score,
            "background_flatness_gain_balance_ok": decision_checks["flatness_gain_balance_ok"],
            "background_soft_override_passed": soft_override_passed,
            "background_tail_slope_symmetry_score": float(
                corrected_metrics["tail_slope_symmetry_score"]
            ),
            "background_saturation_magnitude_symmetry_score": float(
                corrected_metrics["saturation_magnitude_symmetry_score"]
            ),
            "raw_switching_width_mT": raw_metrics.get("switching_width_mT"),
            "corrected_switching_width_mT": corrected_metrics.get("switching_width_mT"),
            "background_switching_width_relative_change": switching_width_relative_change,
            "raw_zero_crossing_candidate_count": int(raw_metrics["zero_crossing_candidate_count"]),
            "corrected_zero_crossing_candidate_count": int(
                corrected_metrics["zero_crossing_candidate_count"]
            ),
            "background_zero_crossing_candidate_increase": zero_crossing_candidate_increase,
            "background_switching_asymmetry_increase": switching_asymmetry_increase,
            "plateau_flatness_ratio_delta": float(corrected_metrics["plateau_flatness_ratio"])
            - float(raw_metrics["plateau_flatness_ratio"]),
            "saturation_consistency_ratio_delta": float(
                corrected_metrics["saturation_consistency_ratio"]
            )
            - float(raw_metrics["saturation_consistency_ratio"]),
            "branch_asymmetry_delta": float(corrected_metrics["branch_asymmetry"])
            - float(raw_metrics["branch_asymmetry"]),
            "loop_closure_error_delta": float(corrected_metrics["loop_closure_error"])
            - float(raw_metrics["loop_closure_error"]),
            "coercive_ambiguity_count_delta": coercive_ambiguity_worsening,
        },
        "score_raw": score_raw,
        "score_corrected": score_corrected,
        "score_delta": score_delta,
        "score_components_raw": score_components_raw,
        "score_components_corrected": score_components_corrected,
    }
    return qc_metrics, warnings


def _compute_background_quality_score(
    metrics: dict[str, Any],
    recipe: VsmPreprocessingRecipe,
    *,
    flatness_gain_score: float | None = None,
    flatness_gain_balance_score: float | None = None,
    switching_integrity_score: float | None = None,
) -> tuple[float, dict[str, float]]:
    if flatness_gain_score is None:
        positive_quality = 1.0 / (
            1.0 + max(float(metrics["plateau_slope_positive_normalized"]), 0.0)
        )
        negative_quality = 1.0 / (
            1.0 + max(float(metrics["plateau_slope_negative_normalized"]), 0.0)
        )
        flatness_gain_score = float(np.mean([positive_quality, negative_quality]))
    if switching_integrity_score is None:
        switching_integrity_score = _compute_switching_integrity_quality(
            metrics=metrics,
            baseline_metrics=metrics,
            switching_width_relative_change=0.0,
            zero_crossing_candidate_increase=0,
            coercive_ambiguity_worsening=0,
            switching_asymmetry_increase=0.0,
        )
    if flatness_gain_balance_score is None:
        flatness_gain_balance_score = float(metrics["tail_slope_symmetry_score"])
    score = (
        recipe.background_score_weight_flatness_gain * float(flatness_gain_score)
        + recipe.background_score_weight_tail_slope_symmetry * float(flatness_gain_balance_score)
        + recipe.background_score_weight_saturation_magnitude_symmetry
        * float(metrics["saturation_magnitude_symmetry_score"])
        + recipe.background_score_weight_switching_integrity * float(switching_integrity_score)
    )
    return float(score), {
        "flatness_gain_score": float(flatness_gain_score),
        "flatness_gain_balance_score": float(flatness_gain_balance_score),
        "tail_slope_symmetry_score": float(metrics["tail_slope_symmetry_score"]),
        "saturation_magnitude_symmetry_score": float(
            metrics["saturation_magnitude_symmetry_score"]
        ),
        "switching_integrity_score": float(switching_integrity_score),
    }


def _compute_switching_integrity_quality(
    *,
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    switching_width_relative_change: float | None,
    zero_crossing_candidate_increase: int,
    coercive_ambiguity_worsening: int,
    switching_asymmetry_increase: float,
) -> float:
    zero_crossing_score = 1.0 / (1.0 + max(int(zero_crossing_candidate_increase), 0))
    coercive_ambiguity_score = 1.0 / (1.0 + max(int(coercive_ambiguity_worsening), 0))
    if switching_width_relative_change is None:
        switching_width_score = 0.0
    else:
        switching_width_score = float(np.clip(1.0 - switching_width_relative_change, 0.0, 1.0))
    switching_asymmetry_score = float(
        np.clip(1.0 - max(float(switching_asymmetry_increase), 0.0), 0.0, 1.0)
    )
    return float(
        np.mean(
            [
                zero_crossing_score,
                coercive_ambiguity_score,
                switching_width_score,
                switching_asymmetry_score,
            ]
        )
    )


def _flatness_gain(raw_normalized_slope: float, corrected_normalized_slope: float) -> float:
    denominator = max(abs(float(raw_normalized_slope)), 1e-18)
    return float((float(raw_normalized_slope) - float(corrected_normalized_slope)) / denominator)


def _is_soft_warning_r_squared(r_squared: float | None, recipe: VsmPreprocessingRecipe) -> bool:
    return bool(
        r_squared is not None
        and recipe.background_tail_fit_catastrophic_r_squared
        <= float(r_squared)
        < recipe.background_tail_fit_min_r_squared
    )


def _relative_change(first: float | None, second: float | None) -> float | None:
    if first is None:
        return 0.0 if second is None else None
    if second is None:
        return None
    denominator = max(abs(float(first)), 1e-18)
    return float(abs(float(second) - float(first)) / denominator)


def _mean_of_available(first: float | None, second: float | None) -> float:
    values = [float(value) for value in (first, second) if value is not None]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _relative_difference(first: float, second: float) -> float:
    denominator = max(abs(first), abs(second), 1e-18)
    return abs(first - second) / denominator
