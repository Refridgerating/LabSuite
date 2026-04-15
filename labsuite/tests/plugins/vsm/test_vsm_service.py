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
    assert result.summary_metrics["background_subtraction_mode"] == "slope_only_split_tails"
    assert result.summary_metrics["Ms_emu"] == result.summary_metrics["saturation_moment_mean_abs_emu"]
    assert result.summary_metrics["Mr_emu"] == result.summary_metrics["remanence_mean_abs_emu"]
    assert result.summary_metrics["Hc_mT"] == result.summary_metrics["coercive_field_mean_abs_mT"]
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
    assert np.all(np.asarray(result.analysis_payload["raw_data"]["field_mT"], dtype=float)[background_fit["positive_tail_fit"].selected_indices] > 0.0)
    assert np.all(np.asarray(result.analysis_payload["raw_data"]["field_mT"], dtype=float)[background_fit["negative_tail_fit"].selected_indices] < 0.0)
    expected_mean_slope = (
        result.summary_metrics["background_slope_positive_emu_per_mT"]
        + result.summary_metrics["background_slope_negative_emu_per_mT"]
    ) / 2.0
    assert np.isclose(result.summary_metrics["background_slope_emu_per_mT"], expected_mean_slope)
    assert len(result.analysis_payload["processed_data"]["corrected_moment_emu"]) == result.summary_metrics["point_count"]
    assert np.any(
        np.asarray(result.analysis_payload["processed_data"]["corrected_moment_emu"], dtype=float)
        != np.asarray(result.analysis_payload["raw_data"]["moment_emu"], dtype=float)
    )
    processed = np.asarray(result.analysis_payload["processed_data"]["processed_moment_emu"], dtype=float)
    field = np.asarray(result.analysis_payload["raw_data"]["field_mT"], dtype=float)
    corrected = np.asarray(result.analysis_payload["processed_data"]["corrected_moment_emu"], dtype=float)
    assert np.allclose(corrected, processed - result.summary_metrics["background_slope_emu_per_mT"] * field)
    assert combined_background["used_intercept_in_correction"] is False
    assert combined_background["qc"]["corrected_positive_tail_slope_emu_per_mT"] is not None
    assert combined_background["qc"]["corrected_negative_tail_slope_emu_per_mT"] is not None
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
