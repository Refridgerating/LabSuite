"""Preprocessing helpers for VSM loop analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from labsuite.core.measurement_models import FitResult
from labsuite.core.recipes import VsmPreprocessingRecipe
from labsuite.plugins.vsm.models import BranchSegment, VsmDataset


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
    recipe: VsmPreprocessingRecipe,
) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray], list[str]]:
    """Fit split high-field tails and subtract only the mean slope term."""

    positive_indices, negative_indices, selection_warnings = select_tail_indices(field_mT, recipe)
    positive_fit = _fit_tail_line(
        field_mT,
        moment_emu,
        moment_std_err_emu,
        positive_indices,
        recipe=recipe,
        label="positive_tail",
    )
    negative_fit = _fit_tail_line(
        field_mT,
        moment_emu,
        moment_std_err_emu,
        negative_indices,
        recipe=recipe,
        label="negative_tail",
    )

    applied_slope = _mean_of_available(
        positive_fit.parameters.get("slope_emu_per_mT"),
        negative_fit.parameters.get("slope_emu_per_mT"),
    )
    average_intercept = _mean_of_available(
        positive_fit.parameters.get("intercept_emu"),
        negative_fit.parameters.get("intercept_emu"),
    )
    corrected_moment = np.asarray(moment_emu - applied_slope * field_mT, dtype=float)

    positive_tail_mask = np.zeros(field_mT.size, dtype=bool)
    negative_tail_mask = np.zeros(field_mT.size, dtype=bool)
    positive_tail_mask[positive_indices] = True
    negative_tail_mask[negative_indices] = True
    combined_tail_mask = positive_tail_mask | negative_tail_mask

    qc_metrics, qc_warnings = _evaluate_background_qc(
        field_mT,
        corrected_moment,
        positive_indices,
        negative_indices,
        positive_fit=positive_fit,
        negative_fit=negative_fit,
        recipe=recipe,
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
            "subtraction_mode": "slope_only_split_tails",
            "used_intercept_in_correction": False,
            "qc": qc_metrics,
        },
    }
    tail_masks = {
        "positive_tail_mask": positive_tail_mask,
        "negative_tail_mask": negative_tail_mask,
        "combined_tail_mask": combined_tail_mask,
    }
    return background_fit, corrected_moment, tail_masks, [*selection_warnings, *qc_warnings]


def select_tail_indices(
    field_mT: np.ndarray,
    recipe: VsmPreprocessingRecipe,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Select symmetric positive and negative high-field tails."""

    max_abs_field = float(np.max(np.abs(field_mT)))
    threshold = max_abs_field * (1.0 - recipe.background_tail_fraction)

    positive_indices = np.flatnonzero(field_mT >= threshold)
    negative_indices = np.flatnonzero(field_mT <= -threshold)

    if positive_indices.size < recipe.background_min_points_per_side:
        positive_indices = np.flatnonzero(field_mT > 0.0)
        if positive_indices.size > recipe.background_min_points_per_side:
            positive_indices = positive_indices[np.argsort(field_mT[positive_indices])][-recipe.background_min_points_per_side :]
    if negative_indices.size < recipe.background_min_points_per_side:
        negative_indices = np.flatnonzero(field_mT < 0.0)
        if negative_indices.size > recipe.background_min_points_per_side:
            negative_indices = negative_indices[np.argsort(field_mT[negative_indices])][: recipe.background_min_points_per_side]

    positive_indices = np.asarray(np.sort(positive_indices[field_mT[positive_indices] > 0.0]), dtype=int)
    negative_indices = np.asarray(np.sort(negative_indices[field_mT[negative_indices] < 0.0]), dtype=int)

    warnings: list[str] = []
    if positive_indices.size < recipe.background_min_points_per_side:
        warnings.append("positive_tail_has_insufficient_points")
    if negative_indices.size < recipe.background_min_points_per_side:
        warnings.append("negative_tail_has_insufficient_points")
    if np.any(field_mT[positive_indices] <= 0.0):
        warnings.append("positive_tail_contains_non_positive_fields")
    if np.any(field_mT[negative_indices] >= 0.0):
        warnings.append("negative_tail_contains_non_negative_fields")

    return positive_indices, negative_indices, warnings


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


def _build_fit_weights(moment_std_err_emu: np.ndarray, recipe: VsmPreprocessingRecipe) -> np.ndarray | None:
    finite_std_err = np.asarray(moment_std_err_emu, dtype=float)
    valid_weights = np.isfinite(finite_std_err) & (finite_std_err > 0.0)
    if not np.any(valid_weights):
        return None
    scaled = np.maximum(finite_std_err[valid_weights] * recipe.uncertainty_scale, 1e-18)
    weights = np.ones(finite_std_err.size, dtype=float)
    weights[valid_weights] = 1.0 / scaled
    return weights


def _evaluate_background_qc(
    field_mT: np.ndarray,
    corrected_moment_emu: np.ndarray,
    positive_indices: np.ndarray,
    negative_indices: np.ndarray,
    *,
    positive_fit: FitResult,
    negative_fit: FitResult,
    recipe: VsmPreprocessingRecipe,
) -> tuple[dict[str, Any], list[str]]:
    """Evaluate split-tail background correction quality."""

    full_moment_span = max(float(np.max(corrected_moment_emu) - np.min(corrected_moment_emu)), 1e-18)
    positive_corrected_slope = _slope_from_indices(field_mT, corrected_moment_emu, positive_indices)
    negative_corrected_slope = _slope_from_indices(field_mT, corrected_moment_emu, negative_indices)
    positive_flatness_ratio = abs(positive_corrected_slope) * max(float(np.ptp(field_mT[positive_indices])) if positive_indices.size else 0.0, 1.0) / full_moment_span
    negative_flatness_ratio = abs(negative_corrected_slope) * max(float(np.ptp(field_mT[negative_indices])) if negative_indices.size else 0.0, 1.0) / full_moment_span
    raw_slope_disagreement_ratio = _relative_difference(
        positive_fit.parameters.get("slope_emu_per_mT", 0.0),
        negative_fit.parameters.get("slope_emu_per_mT", 0.0),
    )
    corrected_tail_slope_abs_mismatch = abs(positive_corrected_slope - negative_corrected_slope)

    warnings: list[str] = []
    if positive_indices.size and np.any(field_mT[positive_indices] <= 0.0):
        warnings.append("background_fit_qc_failed_positive_tail_mixed_sign")
    if negative_indices.size and np.any(field_mT[negative_indices] >= 0.0):
        warnings.append("background_fit_qc_failed_negative_tail_mixed_sign")
    if positive_flatness_ratio > recipe.background_tail_flatness_ratio_tolerance:
        warnings.append("background_fit_qc_failed_positive_tail_not_flat")
    if negative_flatness_ratio > recipe.background_tail_flatness_ratio_tolerance:
        warnings.append("background_fit_qc_failed_negative_tail_not_flat")
    if raw_slope_disagreement_ratio > recipe.background_slope_disagreement_ratio_tolerance:
        warnings.append("background_fit_qc_failed_tail_slope_disagreement")

    qc_passed = (
        positive_fit.success
        and negative_fit.success
        and not warnings
    )
    qc_metrics = {
        "passed": qc_passed,
        "corrected_positive_tail_slope_emu_per_mT": positive_corrected_slope,
        "corrected_negative_tail_slope_emu_per_mT": negative_corrected_slope,
        "corrected_tail_slope_abs_mismatch_emu_per_mT": corrected_tail_slope_abs_mismatch,
        "positive_tail_flatness_ratio": positive_flatness_ratio,
        "negative_tail_flatness_ratio": negative_flatness_ratio,
        "raw_tail_slope_disagreement_ratio": raw_slope_disagreement_ratio,
    }
    return qc_metrics, warnings


def _slope_from_indices(field_mT: np.ndarray, moment_emu: np.ndarray, indices: np.ndarray) -> float:
    if indices.size < 2:
        return 0.0
    coefficients = np.polyfit(np.asarray(field_mT[indices], dtype=float), np.asarray(moment_emu[indices], dtype=float), deg=1)
    return float(coefficients[0])


def _mean_of_available(first: float | None, second: float | None) -> float:
    values = [float(value) for value in (first, second) if value is not None]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _relative_difference(first: float, second: float) -> float:
    denominator = max(abs(first), abs(second), 1e-18)
    return abs(first - second) / denominator
