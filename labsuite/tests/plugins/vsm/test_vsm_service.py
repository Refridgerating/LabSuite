from __future__ import annotations

import numpy as np

from labsuite.plugins.vsm.service import analyze_vsm_file


def test_analyze_vsm_file_extracts_filename_metadata_and_metrics(project_root, vsm_sample_files) -> None:
    source_file = vsm_sample_files[0]
    recipe_path = project_root / "recipes" / "vsm" / "default.yaml"

    result = analyze_vsm_file(source_file, recipe_path)

    assert result.measurement.modality == "vsm"
    assert result.measurement.sample.sample_id == "MTJ-B"
    assert result.measurement.sample.series_id == "MTJ-B"
    assert result.measurement.replicate_id == "R1"
    assert result.summary_metrics["acquisition_index"] == "00001"
    assert result.summary_metrics["branch_count"] >= 3
    assert result.summary_metrics["background_slope_emu_per_mT"] is not None
    assert result.summary_metrics["background_slope_positive_emu_per_mT"] is not None
    assert result.summary_metrics["background_slope_negative_emu_per_mT"] is not None
    assert result.summary_metrics["background_intercept_emu"] is not None
    assert result.summary_metrics["background_mode"] in {"none", "slope_only", "rejected"}
    assert result.summary_metrics["background_subtraction_mode"] == result.summary_metrics["background_mode"]
    assert result.summary_metrics["background_correction_accepted"] == (
        result.summary_metrics["background_mode"] == "slope_only"
    )
    assert result.summary_metrics["background_decision_reason"] is not None
    assert result.summary_metrics["background_flatness_gain_score"] is not None
    assert result.summary_metrics["background_flatness_gain_balance_score"] is not None
    assert result.summary_metrics["background_tail_slope_symmetry_score"] is not None
    assert result.summary_metrics["background_saturation_magnitude_symmetry_score"] is not None
    assert result.summary_metrics["tail_window_selection_mode"] == "iterative_shrink"
    assert result.summary_metrics["positive_tail_window_initial_point_count"] >= result.summary_metrics["positive_tail_window_selected_point_count"]
    assert result.summary_metrics["negative_tail_window_initial_point_count"] >= result.summary_metrics["negative_tail_window_selected_point_count"]
    assert result.summary_metrics["Ms_emu"] == result.analysis_payload["metrics"]["final"]["Ms_emu"]
    assert result.summary_metrics["Mr_emu"] == result.analysis_payload["metrics"]["final"]["Mr_emu"]
    assert result.summary_metrics["Hc_mT"] == result.analysis_payload["metrics"]["final"]["Hc_mT"]
    assert result.summary_metrics["exchange_bias_mT"] == result.summary_metrics["loop_shift_mT"]
    assert result.summary_metrics["loop_area_emu_mT"] is not None
    assert result.summary_metrics["loop_area_emu_mT"] > 0.0
    assert result.summary_metrics["ms_error"] is not None
    assert result.summary_metrics["mr_error"] is not None
    assert result.summary_metrics["hc_error"] is not None
    assert result.summary_metrics["hex_error"] is not None
    assert result.summary_metrics["squareness_error"] is not None
    assert result.summary_metrics["loop_area_error"] is not None
    assert result.summary_metrics["ms_n_points_pos"] > 0
    assert result.summary_metrics["ms_n_points_neg"] > 0
    assert result.summary_metrics["mr_interp_method"] == "linear_interp_at_zero_with_local_linear_sensitivity"
    assert result.summary_metrics["saturation_confidence"] is not None
    assert 0.0 <= result.summary_metrics["saturation_confidence"] <= 1.0
    assert result.summary_metrics["branch_asymmetry"] is not None
    assert result.summary_metrics["switching_complexity"] is not None
    assert result.summary_metrics["switching_complexity_label"] in {"simple", "moderate", "complex"}
    assert isinstance(result.summary_metrics["ambiguity_flags"], list)
    assert result.summary_metrics["centering_applied"] is False
    assert result.summary_metrics["temperature_k"] is not None

    background_fit = result.analysis_payload["background_fit"]
    combined_background = background_fit["combined_background"]
    assert background_fit["positive_tail_fit"].selected_indices
    assert background_fit["negative_tail_fit"].selected_indices
    assert np.all(
        np.asarray(result.analysis_payload["raw_data"]["field_mT"], dtype=float)[background_fit["positive_tail_fit"].selected_indices] > 0.0
    )
    assert np.all(
        np.asarray(result.analysis_payload["raw_data"]["field_mT"], dtype=float)[background_fit["negative_tail_fit"].selected_indices] < 0.0
    )
    expected_mean_slope = (
        result.summary_metrics["background_slope_positive_emu_per_mT"]
        + result.summary_metrics["background_slope_negative_emu_per_mT"]
    ) / 2.0
    assert np.isclose(result.summary_metrics["background_slope_emu_per_mT"], expected_mean_slope)
    processed = np.asarray(result.analysis_payload["processed_data"]["processed_moment_emu"], dtype=float)
    uncorrected = np.asarray(result.analysis_payload["processed_data"]["uncorrected_moment_emu"], dtype=float)
    field = np.asarray(result.analysis_payload["raw_data"]["field_mT"], dtype=float)
    slope_corrected = np.asarray(result.analysis_payload["processed_data"]["slope_corrected_moment_emu"], dtype=float)
    selected = np.asarray(result.analysis_payload["processed_data"]["selected_moment_emu"], dtype=float)
    assert np.allclose(uncorrected, processed)
    assert np.allclose(slope_corrected, processed - result.summary_metrics["background_slope_emu_per_mT"] * field)
    if result.summary_metrics["background_mode"] == "slope_only":
        assert np.allclose(selected, slope_corrected)
    else:
        assert np.allclose(selected, uncorrected)
    assert combined_background["used_intercept_in_correction"] is False
    assert combined_background["qc"]["corrected_positive_tail_slope_emu_per_mT"] is not None
    assert combined_background["qc"]["corrected_negative_tail_slope_emu_per_mT"] is not None
    assert "raw_metrics" in combined_background["qc"]
    assert "corrected_candidate_metrics" in combined_background["qc"]
    assert "raw_plateau_slope_positive_normalized" in combined_background["qc"]["comparison"]
    assert "background_flatness_gain_score" in combined_background["qc"]["comparison"]
    assert "raw_switching_width_mT" in combined_background["qc"]["comparison"]
    assert "corrected_zero_crossing_candidate_count" in combined_background["qc"]["comparison"]
    assert "tail_window_selection_mode" in combined_background["qc"]
    assert "positive_tail_fit_r_squared_soft_warning" in combined_background["qc"]
    assert "positive_tail_window_selected_point_count" in combined_background["qc"]
    assert "background_flatness_gain_balance_score" in combined_background["qc"]["comparison"]
    assert "positive_tail_window_soft_r_squared_rescue_attempted" in combined_background["qc"]
    assert np.isclose(
        result.summary_metrics["squareness"],
        result.summary_metrics["Mr_emu"] / result.summary_metrics["Ms_emu"],
    )
    assert np.isclose(
        result.summary_metrics["hex_error"],
        0.5
        * np.sqrt(
            result.summary_metrics["hc_error_pos"] ** 2
            + result.summary_metrics["hc_error_neg"] ** 2
        ),
    )
    assert "direct_observables" in result.analysis_payload
    assert "trust_diagnostics" in result.analysis_payload
    assert "uncertainty_estimates" in result.analysis_payload
    assert result.analysis_payload["direct_observables"]["loop_area_emu_mT"] == result.summary_metrics["loop_area_emu_mT"]
    assert result.analysis_payload["trust_diagnostics"]["saturation_confidence"] == result.summary_metrics["saturation_confidence"]
    assert result.analysis_payload["uncertainty_estimates"]["ms_error"] == result.summary_metrics["ms_error"]
    assert result.analysis_payload["uncertainty_estimates"]["mr_error"] == result.summary_metrics["mr_error"]


def test_analyze_vsm_file_persists_centering_diagnostics(project_root, vsm_sample_files) -> None:
    source_file = vsm_sample_files[-1]
    recipe_path = project_root / "recipes" / "vsm" / "default.yaml"

    result = analyze_vsm_file(source_file, recipe_path)

    centering = result.analysis_payload["centering"]
    assert "field_offset_mT" in centering
    assert "moment_offset_emu" in centering
    assert centering["applied"] is False
    assert result.plot_manifest is not None
    assert result.summary_metrics["background_qc_passed"] in {True, False}
    assert "background_fit_details" in result.analysis_payload["trust_diagnostics"]
    assert "ambiguity_flags" in result.analysis_payload["trust_diagnostics"]
    assert "uncertainty_flags" in result.analysis_payload["uncertainty_estimates"]
    assert "instrument_noise_fallback_used" in result.analysis_payload["uncertainty_estimates"]


def test_analyze_vsm_file_background_mode_none_for_negligible_slope(tmp_path, project_root, write_vsm_sample) -> None:
    source_file = write_vsm_sample(tmp_path, sample_stem="SyntheticNone-300K-R1_00001", background_slope_emu_per_mT=0.0)
    recipe_path = project_root / "recipes" / "vsm" / "default.yaml"

    result = analyze_vsm_file(source_file, recipe_path)

    assert result.summary_metrics["background_mode"] == "none"
    assert result.summary_metrics["background_correction_accepted"] is False
    assert result.summary_metrics["background_decision_reason"] == "slope_below_meaningful_threshold"
    assert result.summary_metrics["background_flatness_gain_score"] is not None
    assert np.allclose(
        np.asarray(result.analysis_payload["processed_data"]["selected_moment_emu"], dtype=float),
        np.asarray(result.analysis_payload["processed_data"]["uncorrected_moment_emu"], dtype=float),
    )
    assert result.analysis_payload["metrics"]["corrected_candidate"]["Ms_emu"] is not None


def test_analyze_vsm_file_background_mode_slope_only_for_clean_linear_background(
    tmp_path,
    project_root,
    write_vsm_sample,
) -> None:
    source_file = write_vsm_sample(
        tmp_path,
        sample_stem="SyntheticSlope-300K-R1_00001",
        background_slope_emu_per_mT=2.0e-7,
    )
    recipe_path = project_root / "recipes" / "vsm" / "default.yaml"

    result = analyze_vsm_file(source_file, recipe_path)

    assert result.summary_metrics["background_mode"] == "slope_only"
    assert result.summary_metrics["background_correction_accepted"] is True
    assert result.summary_metrics["background_score_delta"] >= 0.0
    assert result.summary_metrics["background_flatness_gain_score"] >= 0.0
    assert result.summary_metrics["background_flatness_gain_balance_score"] >= 0.0
    assert np.allclose(
        np.asarray(result.analysis_payload["processed_data"]["selected_moment_emu"], dtype=float),
        np.asarray(result.analysis_payload["processed_data"]["slope_corrected_moment_emu"], dtype=float),
    )
    assert not np.allclose(
        np.asarray(result.analysis_payload["processed_data"]["selected_moment_emu"], dtype=float),
        np.asarray(result.analysis_payload["processed_data"]["uncorrected_moment_emu"], dtype=float),
    )


def test_analyze_vsm_file_adaptive_tail_window_can_keep_slope_only_on_curved_tails(
    tmp_path,
    project_root,
    write_vsm_sample,
) -> None:
    source_file = write_vsm_sample(
        tmp_path,
        sample_stem="SyntheticAdaptive-300K-R1_00001",
        background_slope_emu_per_mT=2.0e-7,
        positive_tail_curvature_emu=1.5e-5,
        negative_tail_curvature_emu=-1.5e-5,
    )
    recipe_path = project_root / "recipes" / "vsm" / "default.yaml"

    result = analyze_vsm_file(source_file, recipe_path)

    assert result.summary_metrics["background_mode"] == "slope_only"
    assert result.summary_metrics["positive_tail_window_selected_point_count"] < result.summary_metrics["positive_tail_window_initial_point_count"]
    assert result.summary_metrics["negative_tail_window_selected_point_count"] < result.summary_metrics["negative_tail_window_initial_point_count"]
    assert result.summary_metrics["positive_tail_fit_r_squared_soft_warning"] is False
    assert result.summary_metrics["negative_tail_fit_r_squared_soft_warning"] is False


def test_analyze_vsm_file_soft_warning_override_exposes_rescue_diagnostics(
    project_root,
    vsm_sample_files,
) -> None:
    source_file = vsm_sample_files[0]
    recipe_path = project_root / "recipes" / "vsm" / "default.yaml"

    result = analyze_vsm_file(source_file, recipe_path)

    assert result.summary_metrics["background_flatness_gain_balance_score"] is not None
    assert result.summary_metrics["background_soft_override_passed"] in {True, False}
    assert result.summary_metrics["positive_tail_window_soft_r_squared_rescue_attempted"] in {True, False}
    assert result.summary_metrics["negative_tail_window_soft_r_squared_rescue_attempted"] in {True, False}
    assert result.summary_metrics["positive_tail_window_rescue_changed_selection"] in {True, False}
    assert result.summary_metrics["negative_tail_window_rescue_changed_selection"] in {True, False}


def test_analyze_vsm_file_background_mode_rejected_for_bad_tail_fit(tmp_path, project_root, write_vsm_sample) -> None:
    source_file = write_vsm_sample(
        tmp_path,
        sample_stem="SyntheticRejected-300K-R1_00001",
        background_slope_emu_per_mT=2.0e-7,
        positive_tail_curvature_emu=5.0e-5,
        negative_tail_curvature_emu=-5.0e-5,
    )
    recipe_path = project_root / "recipes" / "vsm" / "default.yaml"

    result = analyze_vsm_file(source_file, recipe_path)

    assert result.summary_metrics["background_mode"] == "rejected"
    assert result.summary_metrics["background_correction_accepted"] is False
    assert result.summary_metrics["background_qc_passed"] is False
    assert result.summary_metrics["background_decision_reason"] == "corrected_zero_crossings_increased"
    assert result.summary_metrics["corrected_zero_crossing_candidate_count"] >= result.summary_metrics["raw_zero_crossing_candidate_count"]
    assert np.allclose(
        np.asarray(result.analysis_payload["processed_data"]["selected_moment_emu"], dtype=float),
        np.asarray(result.analysis_payload["processed_data"]["uncorrected_moment_emu"], dtype=float),
    )
    assert result.analysis_payload["metrics"]["corrected_candidate"]["plateau_flatness_ratio"] is not None
    assert result.summary_metrics["positive_tail_fit_r_squared_catastrophic"] is False
