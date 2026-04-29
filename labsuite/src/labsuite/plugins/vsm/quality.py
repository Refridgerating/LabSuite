"""Simple transparent quality scoring for VSM background subtraction."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict
from typing import Any

import numpy as np

from labsuite.core.measurement_models import FitResult
from labsuite.core.recipes import VsmPreprocessingRecipe
from labsuite.plugins.vsm.models import VSMSubtractionQuality


def evaluate_vsm_subtraction_quality(
    *,
    field_mT: np.ndarray,
    raw_moment_emu: np.ndarray,
    corrected_moment_emu: np.ndarray,
    positive_tail_indices: np.ndarray,
    negative_tail_indices: np.ndarray,
    positive_fit: FitResult,
    negative_fit: FitResult,
    corrected_metrics: dict[str, Any],
    background_slope: float | None,
    recipe: VsmPreprocessingRecipe,
    method: str = "slope_only",
    hcut_fraction: float | None = None,
) -> VSMSubtractionQuality:
    """Score one slope-corrected VSM loop without treating slope sign as a failure."""

    field = np.asarray(field_mT, dtype=float)
    raw_moment = np.asarray(raw_moment_emu, dtype=float)
    corrected_moment = np.asarray(corrected_moment_emu, dtype=float)
    positive_indices = np.asarray(positive_tail_indices, dtype=int)
    negative_indices = np.asarray(negative_tail_indices, dtype=int)

    reasons: list[str] = []
    if positive_indices.size < recipe.background_min_points_per_side:
        reasons.append("insufficient_tail_points")
    if negative_indices.size < recipe.background_min_points_per_side:
        reasons.append("insufficient_tail_points")
    if not positive_fit.success or not negative_fit.success:
        reasons.append("fit_failed")

    residual_slope_pos, residual_intercept_pos, residual_rmse_pos = _fit_line(
        field, corrected_moment, positive_indices
    )
    residual_slope_neg, residual_intercept_neg, residual_rmse_neg = _fit_line(
        field, corrected_moment, negative_indices
    )
    del residual_intercept_pos, residual_intercept_neg

    ms_pos = _mean_or_none(corrected_moment[positive_indices])
    ms_neg = _mean_or_none(corrected_moment[negative_indices])
    ms_emu = _metric_float(corrected_metrics, "Ms_emu")
    if ms_emu is None:
        ms_emu = _metric_float(corrected_metrics, "saturation_moment_mean_abs_emu")
    if ms_emu is None and ms_pos is not None and ms_neg is not None:
        ms_emu = 0.5 * (abs(ms_pos) + abs(ms_neg))

    if ms_emu is None or not np.isfinite(ms_emu):
        reasons.append("nonfinite_ms")
    elif ms_emu <= recipe.vsm_quality_near_zero_ms_emu:
        reasons.append("near_zero_ms")

    max_abs_field = max(float(np.nanmax(np.abs(field))) if field.size else 0.0, 1e-18)
    signal_scale = _signal_scale(raw_moment, corrected_moment, ms_emu)
    slope_scale = max((abs(ms_emu) if ms_emu is not None else signal_scale) / max_abs_field, 1e-18)
    slope_ratio = _max_finite_abs(residual_slope_pos, residual_slope_neg) / slope_scale
    slope_score = score_residual_slope(slope_ratio, recipe)
    if slope_ratio >= recipe.vsm_quality_slope_extreme_ratio:
        reasons.append("extreme_residual_slope")
    elif slope_ratio >= recipe.vsm_quality_slope_downweight_ratio:
        reasons.append("downweighted_moderate_slope_error")
    else:
        reasons.append("accepted_low_slope_error")

    symmetry_error = _branch_symmetry_error(ms_pos, ms_neg)
    symmetry_score = score_branch_symmetry(symmetry_error, recipe)
    if (
        symmetry_error is not None
        and symmetry_error >= recipe.vsm_quality_symmetry_catastrophic_error
    ):
        reasons.append("poor_branch_symmetry")

    cutoff_cv = _cutoff_ms_cv(
        field_mT=field,
        raw_moment_emu=raw_moment,
        fractions=recipe.vsm_hcut_fractions,
        min_points=recipe.background_min_points_per_side,
    )
    stability_score = score_cutoff_stability(cutoff_cv, recipe)
    if cutoff_cv is not None and cutoff_cv >= recipe.vsm_quality_cutoff_cv_downweight:
        reasons.append("unstable_cutoff")

    tail_rmse = _mean_finite(
        positive_fit.metrics.get("rmse_emu"),
        negative_fit.metrics.get("rmse_emu"),
        residual_rmse_pos,
        residual_rmse_neg,
    )
    rmse_ratio = None if tail_rmse is None else float(tail_rmse) / signal_scale
    rmse_score = score_tail_rmse(rmse_ratio, recipe)
    if rmse_ratio is not None and rmse_ratio >= recipe.vsm_quality_tail_rmse_downweight_ratio:
        reasons.append("high_tail_rmse")

    weight = _clamp01(
        0.35 * stability_score
        + 0.30 * slope_score
        + 0.20 * symmetry_score
        + 0.15 * rmse_score
    )
    status = classify_vsm_quality(weight, reasons, recipe)
    return VSMSubtractionQuality(
        method=method,
        hcut_fraction=hcut_fraction,
        ms_emu=None if ms_emu is None else float(ms_emu),
        background_slope=None if background_slope is None else float(background_slope),
        residual_slope_pos=residual_slope_pos,
        residual_slope_neg=residual_slope_neg,
        tail_rmse=tail_rmse,
        symmetry_error=symmetry_error,
        cutoff_cv=cutoff_cv,
        slope_score=slope_score,
        symmetry_score=symmetry_score,
        stability_score=stability_score,
        rmse_score=rmse_score,
        weight=weight,
        status=status,
        reasons=list(dict.fromkeys(reasons)),
    )


def score_residual_slope(
    normalized_abs_slope: float | None, recipe: VsmPreprocessingRecipe
) -> float:
    if normalized_abs_slope is None or not np.isfinite(normalized_abs_slope):
        return 0.0
    return _linear_score(float(normalized_abs_slope), recipe.vsm_quality_slope_extreme_ratio)


def score_branch_symmetry(
    symmetry_error: float | None, recipe: VsmPreprocessingRecipe
) -> float:
    if symmetry_error is None or not np.isfinite(symmetry_error):
        return 0.0
    return _linear_score(float(symmetry_error), recipe.vsm_quality_symmetry_catastrophic_error)


def score_cutoff_stability(cutoff_cv: float | None, recipe: VsmPreprocessingRecipe) -> float:
    if cutoff_cv is None or not np.isfinite(cutoff_cv):
        return 0.0
    return _linear_score(float(cutoff_cv), recipe.vsm_quality_cutoff_cv_extreme)


def score_tail_rmse(rmse_ratio: float | None, recipe: VsmPreprocessingRecipe) -> float:
    if rmse_ratio is None or not np.isfinite(rmse_ratio):
        return 0.0
    return _linear_score(float(rmse_ratio), recipe.vsm_quality_tail_rmse_extreme_ratio)


def classify_vsm_quality(
    weight: float,
    reasons: Sequence[str],
    recipe: VsmPreprocessingRecipe,
) -> str:
    hard_reject_reasons = {
        "fit_failed",
        "nonfinite_ms",
        "near_zero_ms",
        "extreme_residual_slope",
        "poor_branch_symmetry",
        "insufficient_tail_points",
    }
    if hard_reject_reasons.intersection(reasons):
        return "reject"
    if weight >= 0.75:
        return "accept"
    if weight >= recipe.vsm_min_weight:
        return "downweight"
    return "reject"


def weighted_ms_summary(
    rows: Iterable[dict[str, Any]],
    *,
    accept_downweighted: bool = True,
    min_weight: float = 0.45,
) -> dict[str, Any]:
    """Compute transparent unweighted and weighted Ms statistics for one group."""

    materialized = list(rows)
    accepted_count = sum(1 for row in materialized if row.get("status") == "accept")
    downweighted_count = sum(1 for row in materialized if row.get("status") == "downweight")
    rejected_count = sum(1 for row in materialized if row.get("status") == "reject")
    included: list[tuple[float, float]] = []
    unweighted_values: list[float] = []
    for row in materialized:
        ms = _float_or_none(row.get("ms_emu"))
        if ms is None:
            continue
        status = str(row.get("status") or "")
        weight = _float_or_none(row.get("weight"))
        weight = 0.0 if weight is None else max(float(weight), 0.0)
        if (
            status == "accept" or (accept_downweighted and status == "downweight")
        ) and weight >= min_weight:
            included.append((ms, weight))
            unweighted_values.append(ms)

    weighted_mean = None
    weighted_std = None
    if included:
        values = np.asarray([item[0] for item in included], dtype=float)
        weights = np.asarray([item[1] for item in included], dtype=float)
        weight_sum = float(np.sum(weights))
        if weight_sum > 0.0:
            weighted_mean = float(np.sum(values * weights) / weight_sum)
            weighted_std = float(
                np.sqrt(np.sum(weights * (values - weighted_mean) ** 2) / weight_sum)
            )

    return {
        "item_count": len(materialized),
        "included_count": len(included),
        "accepted_count": accepted_count,
        "downweighted_count": downweighted_count,
        "rejected_count": rejected_count,
        "unweighted_mean_ms_emu": None
        if not unweighted_values
        else float(np.mean(np.asarray(unweighted_values, dtype=float))),
        "weighted_mean_ms_emu": weighted_mean,
        "weighted_std_ms_emu": weighted_std,
    }


def quality_to_dict(quality: VSMSubtractionQuality) -> dict[str, Any]:
    return asdict(quality)


def _linear_score(value: float, zero_at: float) -> float:
    return _clamp01(1.0 - max(float(value), 0.0) / max(float(zero_at), 1e-18))


def _clamp01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _fit_line(
    field_mT: np.ndarray, moment_emu: np.ndarray, indices: np.ndarray
) -> tuple[float | None, float | None, float | None]:
    if indices.size < 2:
        return None, None, None
    x = np.asarray(field_mT[indices], dtype=float)
    y = np.asarray(moment_emu[indices], dtype=float)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return None, None, None
    slope, intercept = np.polyfit(x, y, deg=1)
    residual = y - (slope * x + intercept)
    return float(slope), float(intercept), float(np.sqrt(np.mean(residual**2)))


def _branch_symmetry_error(ms_pos: float | None, ms_neg: float | None) -> float | None:
    if ms_pos is None or ms_neg is None:
        return None
    if not np.isfinite(ms_pos) or not np.isfinite(ms_neg):
        return None
    return float(abs(ms_pos + ms_neg) / max(abs(ms_pos - ms_neg), 1e-18))


def _cutoff_ms_cv(
    *,
    field_mT: np.ndarray,
    raw_moment_emu: np.ndarray,
    fractions: Sequence[float],
    min_points: int,
) -> float | None:
    values: list[float] = []
    max_abs_field = float(np.nanmax(np.abs(field_mT))) if field_mT.size else 0.0
    if max_abs_field <= 0.0:
        return None
    for fraction in fractions:
        threshold = max_abs_field * (1.0 - float(fraction))
        positive = np.flatnonzero(field_mT >= threshold)
        negative = np.flatnonzero(field_mT <= -threshold)
        if positive.size < min_points or negative.size < min_points:
            continue
        pos_slope, _pos_intercept, _pos_rmse = _fit_line(field_mT, raw_moment_emu, positive)
        neg_slope, _neg_intercept, _neg_rmse = _fit_line(field_mT, raw_moment_emu, negative)
        if pos_slope is None or neg_slope is None:
            continue
        applied_slope = 0.5 * (pos_slope + neg_slope)
        corrected = raw_moment_emu - applied_slope * field_mT
        ms_pos = _mean_or_none(corrected[positive])
        ms_neg = _mean_or_none(corrected[negative])
        if ms_pos is None or ms_neg is None:
            continue
        values.append(0.5 * (abs(ms_pos) + abs(ms_neg)))
    if len(values) < 2:
        return None
    values_array = np.asarray(values, dtype=float)
    mean_value = float(np.mean(np.abs(values_array)))
    if mean_value <= 1e-18:
        return None
    return float(np.std(values_array) / mean_value)


def _signal_scale(
    raw_moment_emu: np.ndarray, corrected_moment_emu: np.ndarray, ms_emu: float | None
) -> float:
    candidates = []
    if ms_emu is not None and np.isfinite(ms_emu):
        candidates.append(abs(float(ms_emu)))
    for values in (raw_moment_emu, corrected_moment_emu):
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            candidates.append(float(np.ptp(finite)))
    return max(max(candidates) if candidates else 1e-18, 1e-18)


def _metric_float(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return _float_or_none(value)


def _float_or_none(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _mean_or_none(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _mean_finite(*values: Any) -> float | None:
    numeric = [_float_or_none(value) for value in values]
    numeric = [value for value in numeric if value is not None]
    if not numeric:
        return None
    return float(np.mean(np.asarray(numeric, dtype=float)))


def _max_finite_abs(*values: Any) -> float:
    numeric = [_float_or_none(value) for value in values]
    numeric = [abs(value) for value in numeric if value is not None]
    if not numeric:
        return float("inf")
    return float(max(numeric))
