from __future__ import annotations

import numpy as np

from labsuite.core.measurement_models import FitResult
from labsuite.core.recipes import load_vsm_recipe
from labsuite.plugins.vsm.preprocess import (
    _evaluate_background_mode,
    fit_background_slope,
    split_vsm_branches,
)

TAIL_SELECTION_METADATA = {
    "tail_window_selection_mode": "iterative_shrink",
    "positive_tail_window_initial_point_count": 24,
    "negative_tail_window_initial_point_count": 24,
    "positive_tail_window_selected_point_count": 24,
    "negative_tail_window_selected_point_count": 24,
    "positive_tail_window_initial_field_min_mT": 80.0,
    "positive_tail_window_initial_field_max_mT": 100.0,
    "negative_tail_window_initial_field_min_mT": -100.0,
    "negative_tail_window_initial_field_max_mT": -80.0,
    "positive_tail_window_selected_field_min_mT": 80.0,
    "positive_tail_window_selected_field_max_mT": 100.0,
    "negative_tail_window_selected_field_min_mT": -100.0,
    "negative_tail_window_selected_field_max_mT": -80.0,
    "positive_tail_window_soft_r_squared_rescue_attempted": False,
    "negative_tail_window_soft_r_squared_rescue_attempted": False,
    "positive_tail_window_rescue_changed_selection": False,
    "negative_tail_window_rescue_changed_selection": False,
}


def _loop_quality_stub(
    *,
    slope_pos: float,
    slope_neg: float,
    slope_pos_norm: float,
    slope_neg_norm: float,
    Ms_pos: float,
    Ms_neg: float,
    tail_slope_symmetry_score: float,
    saturation_magnitude_symmetry_score: float,
    switching_width_mT: float | None,
    switching_asymmetry_ratio: float,
    zero_crossing_candidate_count: int,
    coercive_ambiguity_count: int,
    branch_asymmetry: float,
    loop_closure_error: float,
    plateau_flatness_ratio: float,
    plateau_flatness_ratio_positive: float,
    plateau_flatness_ratio_negative: float,
    saturation_consistency_ratio: float,
) -> dict[str, float | int | list[str] | None]:
    return {
        "plateau_slope_positive_emu_per_mT": slope_pos,
        "plateau_slope_negative_emu_per_mT": slope_neg,
        "plateau_slope_positive_normalized": slope_pos_norm,
        "plateau_slope_negative_normalized": slope_neg_norm,
        "saturation_moment_positive_emu": Ms_pos,
        "saturation_moment_negative_emu": Ms_neg,
        "tail_slope_symmetry_score": tail_slope_symmetry_score,
        "saturation_magnitude_symmetry_score": saturation_magnitude_symmetry_score,
        "switching_width_mT": switching_width_mT,
        "switching_asymmetry_ratio": switching_asymmetry_ratio,
        "zero_crossing_candidate_count": zero_crossing_candidate_count,
        "coercive_ambiguity_count": coercive_ambiguity_count,
        "branch_asymmetry": branch_asymmetry,
        "loop_closure_error": loop_closure_error,
        "plateau_flatness_ratio": plateau_flatness_ratio,
        "plateau_flatness_ratio_positive": plateau_flatness_ratio_positive,
        "plateau_flatness_ratio_negative": plateau_flatness_ratio_negative,
        "saturation_consistency_ratio": saturation_consistency_ratio,
        "warnings": [],
    }


def _fit_stub(slope: float, r_squared: float, point_count: int) -> FitResult:
    return FitResult(
        model_name="linear_tail_fit",
        parameters={"slope_emu_per_mT": slope, "intercept_emu": 0.0},
        metrics={
            "r_squared": r_squared,
            "rmse_emu": 1.0e-7,
            "selected_point_count": float(point_count),
        },
        success=True,
        message="ok",
        selected_indices=list(range(point_count)),
        fitted_x=[],
        fitted_y=[],
        residual_y=[],
    )


def test_fit_background_slope_unit_accepts_clean_linear_background(
    project_root, build_vsm_loop_arrays
) -> None:
    recipe = load_vsm_recipe(project_root / "recipes" / "vsm" / "default.yaml")
    loop = build_vsm_loop_arrays(background_slope_emu_per_mT=2.0e-7)
    branch_ids, branches = split_vsm_branches(loop["field_mT"])

    background_fit, loop_variants, tail_masks, warnings = fit_background_slope(
        loop["field_mT"],
        loop["moment_emu"],
        loop["moment_std_err_emu"],
        branches,
        recipe,
        temperature_k=loop["temperature_k"],
    )

    qc = background_fit["combined_background"]["qc"]
    assert branch_ids.size == loop["field_mT"].size
    assert tail_masks["combined_tail_mask"].any()
    assert background_fit["combined_background"]["background_mode"] == "slope_only"
    assert background_fit["combined_background"]["correction_accepted"] is True
    assert (
        qc["comparison"]["background_flatness_gain_score"]
        >= recipe.background_min_flatness_gain_score
    )
    assert qc["tail_window_selection_mode"] == "iterative_shrink"
    assert np.allclose(
        loop_variants["final_moment_emu"], loop_variants["slope_corrected_moment_emu"]
    )
    assert not warnings


def test_fit_background_slope_unit_adaptively_shrinks_curved_tail_window(
    project_root, build_vsm_loop_arrays
) -> None:
    recipe = load_vsm_recipe(project_root / "recipes" / "vsm" / "default.yaml")
    loop = build_vsm_loop_arrays(
        background_slope_emu_per_mT=2.0e-7,
        positive_tail_curvature_emu=1.5e-5,
        negative_tail_curvature_emu=-1.5e-5,
    )
    _branch_ids, branches = split_vsm_branches(loop["field_mT"])

    background_fit, loop_variants, _tail_masks, warnings = fit_background_slope(
        loop["field_mT"],
        loop["moment_emu"],
        loop["moment_std_err_emu"],
        branches,
        recipe,
        temperature_k=loop["temperature_k"],
    )

    qc = background_fit["combined_background"]["qc"]
    assert background_fit["combined_background"]["background_mode"] == "slope_only"
    assert (
        qc["positive_tail_window_selected_point_count"]
        < qc["positive_tail_window_initial_point_count"]
    )
    assert (
        qc["negative_tail_window_selected_point_count"]
        < qc["negative_tail_window_initial_point_count"]
    )
    assert qc["positive_tail_fit_r_squared"] >= recipe.background_tail_fit_min_r_squared
    assert qc["negative_tail_fit_r_squared"] >= recipe.background_tail_fit_min_r_squared
    assert np.allclose(
        loop_variants["final_moment_emu"], loop_variants["slope_corrected_moment_emu"]
    )
    assert not warnings


def test_fit_background_slope_unit_returns_none_for_negligible_slope(
    project_root, build_vsm_loop_arrays
) -> None:
    recipe = load_vsm_recipe(project_root / "recipes" / "vsm" / "default.yaml")
    loop = build_vsm_loop_arrays(background_slope_emu_per_mT=0.0)
    _branch_ids, branches = split_vsm_branches(loop["field_mT"])

    background_fit, loop_variants, _tail_masks, warnings = fit_background_slope(
        loop["field_mT"],
        loop["moment_emu"],
        loop["moment_std_err_emu"],
        branches,
        recipe,
        temperature_k=loop["temperature_k"],
    )

    assert background_fit["combined_background"]["background_mode"] == "none"
    assert (
        background_fit["combined_background"]["decision_reason"]
        == "slope_below_meaningful_threshold"
    )
    assert np.allclose(loop_variants["final_moment_emu"], loop_variants["uncorrected_moment_emu"])
    assert "background_fit_optional_slope_below_meaningful_threshold" in warnings


def test_evaluate_background_mode_returns_none_for_one_sided_plateau_regression(
    monkeypatch, project_root
) -> None:
    recipe = load_vsm_recipe(project_root / "recipes" / "vsm" / "default.yaml")
    field_mT = np.linspace(-100.0, 100.0, 81, dtype=float)
    positive_indices = np.flatnonzero(field_mT > 80.0)
    negative_indices = np.flatnonzero(field_mT < -80.0)

    raw_metrics = _loop_quality_stub(
        slope_pos=2.0e-7,
        slope_neg=2.1e-7,
        slope_pos_norm=0.55,
        slope_neg_norm=0.52,
        Ms_pos=3.4e-5,
        Ms_neg=-3.5e-5,
        tail_slope_symmetry_score=0.95,
        saturation_magnitude_symmetry_score=0.97,
        switching_width_mT=4.0,
        switching_asymmetry_ratio=0.08,
        zero_crossing_candidate_count=2,
        coercive_ambiguity_count=0,
        branch_asymmetry=0.10,
        loop_closure_error=0.02,
        plateau_flatness_ratio=0.030,
        plateau_flatness_ratio_positive=0.030,
        plateau_flatness_ratio_negative=0.028,
        saturation_consistency_ratio=0.04,
    )
    corrected_metrics = _loop_quality_stub(
        slope_pos=0.3e-7,
        slope_neg=3.4e-7,
        slope_pos_norm=0.08,
        slope_neg_norm=0.63,
        Ms_pos=3.45e-5,
        Ms_neg=-3.48e-5,
        tail_slope_symmetry_score=0.65,
        saturation_magnitude_symmetry_score=0.98,
        switching_width_mT=4.0,
        switching_asymmetry_ratio=0.09,
        zero_crossing_candidate_count=2,
        coercive_ambiguity_count=0,
        branch_asymmetry=0.11,
        loop_closure_error=0.02,
        plateau_flatness_ratio=0.040,
        plateau_flatness_ratio_positive=0.008,
        plateau_flatness_ratio_negative=0.040,
        saturation_consistency_ratio=0.03,
    )

    calls = iter([raw_metrics, corrected_metrics])
    monkeypatch.setattr(
        "labsuite.plugins.vsm.preprocess.summarize_loop_quality", lambda **_kwargs: next(calls)
    )

    fit = _fit_stub(2.0e-7, 0.98, positive_indices.size)
    evaluation, warnings = _evaluate_background_mode(
        field_mT=field_mT,
        uncorrected_moment_emu=np.zeros_like(field_mT),
        slope_corrected_moment_emu=np.zeros_like(field_mT),
        positive_indices=positive_indices,
        negative_indices=negative_indices,
        branches=[],
        positive_fit=fit,
        negative_fit=fit,
        recipe=recipe,
        temperature_k=np.full(field_mT.shape, 300.0, dtype=float),
        tail_selection_metadata=TAIL_SELECTION_METADATA,
    )

    assert evaluation["background_mode"] == "none"
    assert evaluation["decision_reason"] == "negative_tail_flatness_regressed"
    assert evaluation["correction_accepted"] is False
    assert "background_fit_optional_negative_tail_flatness_regressed" in warnings


def test_evaluate_background_mode_rejects_extra_zero_crossings(monkeypatch, project_root) -> None:
    recipe = load_vsm_recipe(project_root / "recipes" / "vsm" / "default.yaml")
    field_mT = np.linspace(-100.0, 100.0, 81, dtype=float)
    positive_indices = np.flatnonzero(field_mT > 80.0)
    negative_indices = np.flatnonzero(field_mT < -80.0)

    raw_metrics = _loop_quality_stub(
        slope_pos=2.0e-7,
        slope_neg=2.1e-7,
        slope_pos_norm=0.55,
        slope_neg_norm=0.52,
        Ms_pos=3.4e-5,
        Ms_neg=-3.5e-5,
        tail_slope_symmetry_score=0.95,
        saturation_magnitude_symmetry_score=0.97,
        switching_width_mT=4.0,
        switching_asymmetry_ratio=0.08,
        zero_crossing_candidate_count=2,
        coercive_ambiguity_count=0,
        branch_asymmetry=0.10,
        loop_closure_error=0.02,
        plateau_flatness_ratio=0.030,
        plateau_flatness_ratio_positive=0.030,
        plateau_flatness_ratio_negative=0.028,
        saturation_consistency_ratio=0.04,
    )
    corrected_metrics = _loop_quality_stub(
        slope_pos=0.2e-7,
        slope_neg=0.2e-7,
        slope_pos_norm=0.06,
        slope_neg_norm=0.06,
        Ms_pos=3.45e-5,
        Ms_neg=-3.45e-5,
        tail_slope_symmetry_score=0.99,
        saturation_magnitude_symmetry_score=1.0,
        switching_width_mT=4.1,
        switching_asymmetry_ratio=0.10,
        zero_crossing_candidate_count=4,
        coercive_ambiguity_count=1,
        branch_asymmetry=0.12,
        loop_closure_error=0.03,
        plateau_flatness_ratio=0.010,
        plateau_flatness_ratio_positive=0.010,
        plateau_flatness_ratio_negative=0.010,
        saturation_consistency_ratio=0.0,
    )

    calls = iter([raw_metrics, corrected_metrics])
    monkeypatch.setattr(
        "labsuite.plugins.vsm.preprocess.summarize_loop_quality", lambda **_kwargs: next(calls)
    )

    fit = _fit_stub(2.0e-7, 0.98, positive_indices.size)
    evaluation, warnings = _evaluate_background_mode(
        field_mT=field_mT,
        uncorrected_moment_emu=np.zeros_like(field_mT),
        slope_corrected_moment_emu=np.zeros_like(field_mT),
        positive_indices=positive_indices,
        negative_indices=negative_indices,
        branches=[],
        positive_fit=fit,
        negative_fit=fit,
        recipe=recipe,
        temperature_k=np.full(field_mT.shape, 300.0, dtype=float),
        tail_selection_metadata=TAIL_SELECTION_METADATA,
    )

    assert evaluation["background_mode"] == "rejected"
    assert evaluation["decision_reason"] == "corrected_zero_crossings_increased"
    assert evaluation["correction_accepted"] is False
    assert "background_fit_rejected_corrected_zero_crossings_increased" in warnings


def test_evaluate_background_mode_poor_r_squared_but_negligible_slope_is_none(
    monkeypatch, project_root
) -> None:
    recipe = load_vsm_recipe(project_root / "recipes" / "vsm" / "default.yaml")
    field_mT = np.linspace(-100.0, 100.0, 81, dtype=float)
    positive_indices = np.flatnonzero(field_mT > 80.0)
    negative_indices = np.flatnonzero(field_mT < -80.0)

    raw_metrics = _loop_quality_stub(
        slope_pos=1.0e-8,
        slope_neg=1.1e-8,
        slope_pos_norm=0.03,
        slope_neg_norm=0.03,
        Ms_pos=3.4e-5,
        Ms_neg=-3.5e-5,
        tail_slope_symmetry_score=0.99,
        saturation_magnitude_symmetry_score=0.97,
        switching_width_mT=4.0,
        switching_asymmetry_ratio=0.08,
        zero_crossing_candidate_count=2,
        coercive_ambiguity_count=0,
        branch_asymmetry=0.10,
        loop_closure_error=0.02,
        plateau_flatness_ratio=0.005,
        plateau_flatness_ratio_positive=0.005,
        plateau_flatness_ratio_negative=0.005,
        saturation_consistency_ratio=0.04,
    )
    calls = iter([raw_metrics, raw_metrics])
    monkeypatch.setattr(
        "labsuite.plugins.vsm.preprocess.summarize_loop_quality", lambda **_kwargs: next(calls)
    )

    poor_fit = _fit_stub(1.0e-8, 0.1, positive_indices.size)
    evaluation, _warnings = _evaluate_background_mode(
        field_mT=field_mT,
        uncorrected_moment_emu=np.zeros_like(field_mT),
        slope_corrected_moment_emu=np.zeros_like(field_mT),
        positive_indices=positive_indices,
        negative_indices=negative_indices,
        branches=[],
        positive_fit=poor_fit,
        negative_fit=poor_fit,
        recipe=recipe,
        temperature_k=np.full(field_mT.shape, 300.0, dtype=float),
        tail_selection_metadata=TAIL_SELECTION_METADATA,
    )

    assert evaluation["background_mode"] == "none"
    assert evaluation["decision_reason"] == "slope_below_meaningful_threshold"


def test_evaluate_background_mode_soft_r_squared_warning_can_be_overridden(
    monkeypatch, project_root
) -> None:
    recipe = load_vsm_recipe(project_root / "recipes" / "vsm" / "default.yaml")
    field_mT = np.linspace(-100.0, 100.0, 81, dtype=float)
    positive_indices = np.flatnonzero(field_mT > 80.0)
    negative_indices = np.flatnonzero(field_mT < -80.0)

    raw_metrics = _loop_quality_stub(
        slope_pos=2.0e-7,
        slope_neg=2.2e-7,
        slope_pos_norm=0.65,
        slope_neg_norm=0.60,
        Ms_pos=3.4e-5,
        Ms_neg=-3.5e-5,
        tail_slope_symmetry_score=0.94,
        saturation_magnitude_symmetry_score=0.96,
        switching_width_mT=4.0,
        switching_asymmetry_ratio=0.08,
        zero_crossing_candidate_count=2,
        coercive_ambiguity_count=0,
        branch_asymmetry=0.10,
        loop_closure_error=0.02,
        plateau_flatness_ratio=0.030,
        plateau_flatness_ratio_positive=0.030,
        plateau_flatness_ratio_negative=0.028,
        saturation_consistency_ratio=0.04,
    )
    corrected_metrics = _loop_quality_stub(
        slope_pos=0.15e-7,
        slope_neg=0.18e-7,
        slope_pos_norm=0.02,
        slope_neg_norm=0.03,
        Ms_pos=3.42e-5,
        Ms_neg=-3.44e-5,
        tail_slope_symmetry_score=0.97,
        saturation_magnitude_symmetry_score=0.99,
        switching_width_mT=4.05,
        switching_asymmetry_ratio=0.08,
        zero_crossing_candidate_count=2,
        coercive_ambiguity_count=0,
        branch_asymmetry=0.10,
        loop_closure_error=0.02,
        plateau_flatness_ratio=0.006,
        plateau_flatness_ratio_positive=0.006,
        plateau_flatness_ratio_negative=0.007,
        saturation_consistency_ratio=0.02,
    )
    calls = iter([raw_metrics, corrected_metrics])
    monkeypatch.setattr(
        "labsuite.plugins.vsm.preprocess.summarize_loop_quality", lambda **_kwargs: next(calls)
    )

    borderline_fit = _fit_stub(2.0e-7, 0.4, positive_indices.size)
    evaluation, warnings = _evaluate_background_mode(
        field_mT=field_mT,
        uncorrected_moment_emu=np.zeros_like(field_mT),
        slope_corrected_moment_emu=np.zeros_like(field_mT),
        positive_indices=positive_indices,
        negative_indices=negative_indices,
        branches=[],
        positive_fit=borderline_fit,
        negative_fit=borderline_fit,
        recipe=recipe,
        temperature_k=np.full(field_mT.shape, 300.0, dtype=float),
        tail_selection_metadata=TAIL_SELECTION_METADATA,
    )

    assert evaluation["background_mode"] == "slope_only"
    assert evaluation["decision_reason"] == "soft_tail_fit_r_squared_overridden"
    assert evaluation["positive_tail_fit_r_squared_soft_warning"] is True
    assert evaluation["negative_tail_fit_r_squared_soft_warning"] is True
    assert evaluation["background_flatness_gain_balance_ok"] is True
    assert evaluation["background_soft_override_passed"] is True
    assert "background_fit_warning_soft_tail_fit_r_squared_overridden" in warnings


def test_evaluate_background_mode_soft_r_squared_warning_with_low_positive_gain_is_none(
    monkeypatch, project_root
) -> None:
    recipe = load_vsm_recipe(project_root / "recipes" / "vsm" / "default.yaml")
    field_mT = np.linspace(-100.0, 100.0, 81, dtype=float)
    positive_indices = np.flatnonzero(field_mT > 80.0)
    negative_indices = np.flatnonzero(field_mT < -80.0)

    raw_metrics = _loop_quality_stub(
        slope_pos=2.0e-7,
        slope_neg=2.1e-7,
        slope_pos_norm=0.55,
        slope_neg_norm=0.52,
        Ms_pos=3.4e-5,
        Ms_neg=-3.5e-5,
        tail_slope_symmetry_score=0.95,
        saturation_magnitude_symmetry_score=0.97,
        switching_width_mT=4.0,
        switching_asymmetry_ratio=0.08,
        zero_crossing_candidate_count=2,
        coercive_ambiguity_count=0,
        branch_asymmetry=0.10,
        loop_closure_error=0.02,
        plateau_flatness_ratio=0.030,
        plateau_flatness_ratio_positive=0.030,
        plateau_flatness_ratio_negative=0.028,
        saturation_consistency_ratio=0.04,
    )
    corrected_metrics = _loop_quality_stub(
        slope_pos=1.3e-7,
        slope_neg=0.1e-7,
        slope_pos_norm=0.46,
        slope_neg_norm=0.01,
        Ms_pos=3.42e-5,
        Ms_neg=-3.45e-5,
        tail_slope_symmetry_score=0.99,
        saturation_magnitude_symmetry_score=0.99,
        switching_width_mT=4.05,
        switching_asymmetry_ratio=0.08,
        zero_crossing_candidate_count=2,
        coercive_ambiguity_count=0,
        branch_asymmetry=0.10,
        loop_closure_error=0.02,
        plateau_flatness_ratio=0.020,
        plateau_flatness_ratio_positive=0.020,
        plateau_flatness_ratio_negative=0.020,
        saturation_consistency_ratio=0.02,
    )
    calls = iter([raw_metrics, corrected_metrics])
    monkeypatch.setattr(
        "labsuite.plugins.vsm.preprocess.summarize_loop_quality", lambda **_kwargs: next(calls)
    )

    borderline_fit = _fit_stub(2.0e-7, 0.4, positive_indices.size)
    evaluation, warnings = _evaluate_background_mode(
        field_mT=field_mT,
        uncorrected_moment_emu=np.zeros_like(field_mT),
        slope_corrected_moment_emu=np.zeros_like(field_mT),
        positive_indices=positive_indices,
        negative_indices=negative_indices,
        branches=[],
        positive_fit=borderline_fit,
        negative_fit=borderline_fit,
        recipe=recipe,
        temperature_k=np.full(field_mT.shape, 300.0, dtype=float),
        tail_selection_metadata=TAIL_SELECTION_METADATA,
    )

    assert evaluation["background_mode"] == "none"
    assert (
        evaluation["decision_reason"] == "soft_tail_fit_r_squared_not_overridden_low_positive_gain"
    )
    assert evaluation["background_soft_override_passed"] is False
    assert (
        "background_fit_optional_soft_tail_fit_r_squared_not_overridden_low_positive_gain"
        in warnings
    )


def test_evaluate_background_mode_soft_r_squared_warning_with_unbalanced_tail_gains_is_none(
    monkeypatch, project_root
) -> None:
    recipe = load_vsm_recipe(project_root / "recipes" / "vsm" / "default.yaml")
    field_mT = np.linspace(-100.0, 100.0, 81, dtype=float)
    positive_indices = np.flatnonzero(field_mT > 80.0)
    negative_indices = np.flatnonzero(field_mT < -80.0)

    raw_metrics = _loop_quality_stub(
        slope_pos=2.0e-7,
        slope_neg=2.1e-7,
        slope_pos_norm=0.55,
        slope_neg_norm=0.52,
        Ms_pos=3.4e-5,
        Ms_neg=-3.5e-5,
        tail_slope_symmetry_score=0.95,
        saturation_magnitude_symmetry_score=0.97,
        switching_width_mT=4.0,
        switching_asymmetry_ratio=0.08,
        zero_crossing_candidate_count=2,
        coercive_ambiguity_count=0,
        branch_asymmetry=0.10,
        loop_closure_error=0.02,
        plateau_flatness_ratio=0.030,
        plateau_flatness_ratio_positive=0.030,
        plateau_flatness_ratio_negative=0.028,
        saturation_consistency_ratio=0.04,
    )
    corrected_metrics = _loop_quality_stub(
        slope_pos=0.02e-7,
        slope_neg=1.0e-7,
        slope_pos_norm=0.01,
        slope_neg_norm=0.30,
        Ms_pos=3.42e-5,
        Ms_neg=-3.45e-5,
        tail_slope_symmetry_score=0.20,
        saturation_magnitude_symmetry_score=0.99,
        switching_width_mT=4.05,
        switching_asymmetry_ratio=0.08,
        zero_crossing_candidate_count=2,
        coercive_ambiguity_count=0,
        branch_asymmetry=0.10,
        loop_closure_error=0.02,
        plateau_flatness_ratio=0.020,
        plateau_flatness_ratio_positive=0.003,
        plateau_flatness_ratio_negative=0.020,
        saturation_consistency_ratio=0.02,
    )
    calls = iter([raw_metrics, corrected_metrics])
    monkeypatch.setattr(
        "labsuite.plugins.vsm.preprocess.summarize_loop_quality", lambda **_kwargs: next(calls)
    )

    borderline_fit = _fit_stub(2.0e-7, 0.4, positive_indices.size)
    evaluation, warnings = _evaluate_background_mode(
        field_mT=field_mT,
        uncorrected_moment_emu=np.zeros_like(field_mT),
        slope_corrected_moment_emu=np.zeros_like(field_mT),
        positive_indices=positive_indices,
        negative_indices=negative_indices,
        branches=[],
        positive_fit=borderline_fit,
        negative_fit=borderline_fit,
        recipe=recipe,
        temperature_k=np.full(field_mT.shape, 300.0, dtype=float),
        tail_selection_metadata=TAIL_SELECTION_METADATA,
    )

    assert evaluation["background_mode"] == "none"
    assert (
        evaluation["decision_reason"]
        == "soft_tail_fit_r_squared_not_overridden_unbalanced_tail_gains"
    )
    assert evaluation["background_flatness_gain_balance_ok"] is False
    assert (
        "background_fit_optional_soft_tail_fit_r_squared_not_overridden_unbalanced_tail_gains"
        in warnings
    )


def test_evaluate_background_mode_catastrophic_r_squared_with_meaningful_slope_is_rejected(
    monkeypatch, project_root
) -> None:
    recipe = load_vsm_recipe(project_root / "recipes" / "vsm" / "default.yaml")
    field_mT = np.linspace(-100.0, 100.0, 81, dtype=float)
    positive_indices = np.flatnonzero(field_mT > 80.0)
    negative_indices = np.flatnonzero(field_mT < -80.0)

    raw_metrics = _loop_quality_stub(
        slope_pos=2.0e-7,
        slope_neg=2.1e-7,
        slope_pos_norm=0.55,
        slope_neg_norm=0.52,
        Ms_pos=3.4e-5,
        Ms_neg=-3.5e-5,
        tail_slope_symmetry_score=0.95,
        saturation_magnitude_symmetry_score=0.97,
        switching_width_mT=4.0,
        switching_asymmetry_ratio=0.08,
        zero_crossing_candidate_count=2,
        coercive_ambiguity_count=0,
        branch_asymmetry=0.10,
        loop_closure_error=0.02,
        plateau_flatness_ratio=0.030,
        plateau_flatness_ratio_positive=0.030,
        plateau_flatness_ratio_negative=0.028,
        saturation_consistency_ratio=0.04,
    )
    calls = iter([raw_metrics, raw_metrics])
    monkeypatch.setattr(
        "labsuite.plugins.vsm.preprocess.summarize_loop_quality", lambda **_kwargs: next(calls)
    )

    catastrophic_fit = _fit_stub(2.0e-7, 0.1, positive_indices.size)
    evaluation, warnings = _evaluate_background_mode(
        field_mT=field_mT,
        uncorrected_moment_emu=np.zeros_like(field_mT),
        slope_corrected_moment_emu=np.zeros_like(field_mT),
        positive_indices=positive_indices,
        negative_indices=negative_indices,
        branches=[],
        positive_fit=catastrophic_fit,
        negative_fit=catastrophic_fit,
        recipe=recipe,
        temperature_k=np.full(field_mT.shape, 300.0, dtype=float),
        tail_selection_metadata=TAIL_SELECTION_METADATA,
    )

    assert evaluation["background_mode"] == "rejected"
    assert evaluation["decision_reason"] == "positive_tail_fit_r_squared_catastrophically_low"
    assert "background_fit_rejected_positive_tail_fit_r_squared_catastrophically_low" in warnings
