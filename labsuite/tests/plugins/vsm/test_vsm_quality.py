from __future__ import annotations

import numpy as np

from labsuite.core.measurement_models import FitResult
from labsuite.core.recipes import load_vsm_recipe
from labsuite.plugins.vsm.quality import (
    evaluate_vsm_subtraction_quality,
    score_branch_symmetry,
    score_cutoff_stability,
    weighted_ms_summary,
)


def _fit_stub(slope: float = 2.0e-7, rmse: float = 1.0e-8, point_count: int = 40) -> FitResult:
    return FitResult(
        model_name="linear_tail_fit",
        parameters={"slope_emu_per_mT": slope, "intercept_emu": 0.0},
        metrics={"r_squared": 0.99, "rmse_emu": rmse, "selected_point_count": float(point_count)},
        success=True,
        message="ok",
        selected_indices=list(range(point_count)),
        fitted_x=[],
        fitted_y=[],
        residual_y=[],
    )


def _quality_case(project_root, *, residual_slope: float = -1.0e-9, neg_scale: float = 1.0):
    recipe = load_vsm_recipe(project_root / "recipes" / "vsm" / "default.yaml")
    field = np.linspace(-100.0, 100.0, 241, dtype=float)
    ms = 3.5e-5
    raw = ms * np.tanh(field / 4.0) + 2.0e-7 * field
    corrected = ms * np.tanh(field / 4.0) + residual_slope * field
    corrected[field < -80.0] *= neg_scale
    pos = np.flatnonzero(field >= 75.0)
    neg = np.flatnonzero(field <= -75.0)
    return evaluate_vsm_subtraction_quality(
        field_mT=field,
        raw_moment_emu=raw,
        corrected_moment_emu=corrected,
        positive_tail_indices=pos,
        negative_tail_indices=neg,
        positive_fit=_fit_stub(point_count=pos.size),
        negative_fit=_fit_stub(point_count=neg.size),
        corrected_metrics={"Ms_emu": ms, "saturation_moment_mean_abs_emu": ms},
        background_slope=2.0e-7,
        recipe=recipe,
        hcut_fraction=recipe.background_tail_fraction,
    )


def test_negative_residual_slope_small_magnitude_is_not_rejected(project_root) -> None:
    quality = _quality_case(project_root, residual_slope=-1.0e-9)

    assert quality.residual_slope_pos < 0.0
    assert quality.status in {"accept", "downweight"}
    assert "extreme_residual_slope" not in quality.reasons
    assert "accepted_low_slope_error" in quality.reasons


def test_large_residual_slope_is_rejected(project_root) -> None:
    quality = _quality_case(project_root, residual_slope=1.0e-6)

    assert quality.status == "reject"
    assert "extreme_residual_slope" in quality.reasons


def test_branch_symmetry_score_tracks_error(project_root) -> None:
    recipe = load_vsm_recipe(project_root / "recipes" / "vsm" / "default.yaml")

    good = score_branch_symmetry(0.02, recipe)
    bad = score_branch_symmetry(0.50, recipe)

    assert good > bad


def test_bad_branch_symmetry_lowers_quality(project_root) -> None:
    quality = _quality_case(project_root, residual_slope=-1.0e-9, neg_scale=0.35)

    assert quality.symmetry_error is not None
    assert quality.symmetry_score < 1.0
    assert quality.weight < 0.95


def test_cutoff_instability_lowers_score(project_root) -> None:
    recipe = load_vsm_recipe(project_root / "recipes" / "vsm" / "default.yaml")

    stable = score_cutoff_stability(0.01, recipe)
    unstable = score_cutoff_stability(0.30, recipe)

    assert stable > unstable


def test_weighted_ms_summary_downweights_low_quality_rows() -> None:
    rows = [
        {"ms_emu": 10.0, "weight": 1.0, "status": "accept"},
        {"ms_emu": 4.0, "weight": 0.5, "status": "downweight"},
        {"ms_emu": 100.0, "weight": 0.1, "status": "reject"},
    ]

    summary = weighted_ms_summary(rows, accept_downweighted=True, min_weight=0.45)

    assert summary["included_count"] == 2
    assert summary["accepted_count"] == 1
    assert summary["downweighted_count"] == 1
    assert summary["rejected_count"] == 1
    assert summary["unweighted_mean_ms_emu"] == 7.0
    assert np.isclose(summary["weighted_mean_ms_emu"], 8.0)
