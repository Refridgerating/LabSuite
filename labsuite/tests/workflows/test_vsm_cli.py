from __future__ import annotations

import csv
import json

from labsuite.cli.main import main


def test_vsm_single_exports_all_artifacts(tmp_path, project_root, vsm_sample_files) -> None:
    source_file = vsm_sample_files[0]
    output_dir = tmp_path / "vsm_single"
    recipe_path = project_root / "recipes" / "vsm" / "default.yaml"

    exit_code = main(
        [
            "vsm",
            "single",
            "--input",
            str(source_file),
            "--recipe",
            str(recipe_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    json_path = output_dir / f"{source_file.stem}_analysis.json"
    csv_path = output_dir / f"{source_file.stem}_trace.csv"
    summary_path = output_dir / f"{source_file.stem}_summary.csv"
    figure_path = output_dir / f"{source_file.stem}_figure.png"
    assert json_path.exists()
    assert csv_path.exists()
    assert summary_path.exists()
    assert figure_path.exists()
    assert figure_path.stat().st_size > 0

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["measurement"]["modality"] == "vsm"
    assert payload["summary_metrics"]["sample_id"] == "MTJ-B"
    assert payload["analysis_payload"]["background_fit"]["combined_background"]["slope_emu_per_mT"] is not None
    assert payload["analysis_payload"]["background_fit"]["combined_background"]["background_mode"] in {
        "none",
        "slope_only",
        "rejected",
    }
    assert payload["analysis_payload"]["background_fit"]["positive_tail_fit"]["parameters"]["slope_emu_per_mT"] is not None
    assert payload["analysis_payload"]["background_fit"]["negative_tail_fit"]["parameters"]["slope_emu_per_mT"] is not None
    assert payload["analysis_payload"]["centering"]["applied"] is False
    assert len(payload["analysis_payload"]["branches"]) >= 3
    assert payload["plot_manifest"]["figure_type"] == "vsm_loop_diagnostic"
    assert payload["summary_metrics"]["background_mode"] in {"none", "slope_only", "rejected"}
    assert payload["summary_metrics"]["background_subtraction_mode"] == payload["summary_metrics"]["background_mode"]
    assert payload["summary_metrics"]["Ms_emu"] is not None
    assert payload["summary_metrics"]["Mr_emu"] is not None
    assert payload["summary_metrics"]["Hc_mT"] is not None
    assert payload["summary_metrics"]["ms_error"] is not None
    assert payload["summary_metrics"]["mr_error"] is not None
    assert payload["summary_metrics"]["hc_error"] is not None
    assert payload["summary_metrics"]["hex_error"] is not None
    assert payload["summary_metrics"]["squareness_error"] is not None
    assert payload["summary_metrics"]["exchange_bias_mT"] == payload["summary_metrics"]["loop_shift_mT"]
    assert payload["summary_metrics"]["loop_area_emu_mT"] is not None
    assert payload["summary_metrics"]["loop_area_error"] is not None
    assert payload["summary_metrics"]["saturation_confidence"] is not None
    assert "direct_observables" in payload["analysis_payload"]
    assert "trust_diagnostics" in payload["analysis_payload"]
    assert "uncertainty_estimates" in payload["analysis_payload"]
    assert "uncorrected" in payload["analysis_payload"]["metrics"]
    assert "corrected_candidate" in payload["analysis_payload"]["metrics"]
    assert "final" in payload["analysis_payload"]["metrics"]

    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert header == (
        "acquisition_index,raw_field_oe,field_mT,raw_moment_emu,processed_moment_emu,"
        "uncorrected_moment_emu,slope_corrected_moment_emu,background_fit_emu,positive_tail_fit_emu,negative_tail_fit_emu,"
        "branch_id,branch_direction,temperature_k,moment_std_err_emu,selected_moment_emu,final_field_mT,final_moment_emu"
    )
    summary_header = summary_path.read_text(encoding="utf-8").splitlines()[0]
    assert "Ms_emu" in summary_header
    assert "ms_error" in summary_header
    assert "Mr_emu" in summary_header
    assert "mr_error" in summary_header
    assert "Hc_mT" in summary_header
    assert "hc_error" in summary_header
    assert "squareness" in summary_header
    assert "squareness_error" in summary_header
    assert "exchange_bias_mT" in summary_header
    assert "hex_error" in summary_header
    assert "loop_area_emu_mT" in summary_header
    assert "loop_area_error" in summary_header
    assert "saturation_confidence" in summary_header
    assert "branch_asymmetry" in summary_header
    assert "switching_complexity" in summary_header
    assert "ambiguity_flags" in summary_header
    assert "coercive_field_negative_mT" in summary_header
    assert "background_slope_emu_per_mT" in summary_header
    assert "background_slope_positive_emu_per_mT" in summary_header
    assert "background_slope_negative_emu_per_mT" in summary_header
    assert "background_mode" in summary_header
    assert "background_subtraction_mode" in summary_header
    assert "background_correction_accepted" in summary_header
    assert "background_decision_reason" in summary_header
    assert "background_qc_passed" in summary_header
    assert "background_flatness_gain_score" in summary_header
    assert "background_flatness_gain_balance_score" in summary_header
    assert "background_flatness_gain_balance_ok" in summary_header
    assert "background_soft_override_passed" in summary_header
    assert "background_tail_slope_symmetry_score" in summary_header
    assert "background_saturation_magnitude_symmetry_score" in summary_header
    assert "positive_tail_fit_r_squared_soft_warning" in summary_header
    assert "positive_tail_fit_r_squared_catastrophic" in summary_header
    assert "tail_window_selection_mode" in summary_header
    assert "positive_tail_window_selected_point_count" in summary_header
    assert "positive_tail_window_soft_r_squared_rescue_attempted" in summary_header
    assert "positive_tail_window_rescue_changed_selection" in summary_header
    assert "raw_plateau_slope_positive_normalized" in summary_header
    assert "corrected_plateau_slope_negative_normalized" in summary_header
    assert "raw_switching_width_mT" in summary_header
    assert "corrected_zero_crossing_candidate_count" in summary_header
    assert "background_score_delta" in summary_header
    assert "centering_field_offset_mT" in summary_header
    assert "warnings" in summary_header


def test_vsm_batch_export_and_report_workflows(tmp_path, project_root, vsm_sample_files, capsys) -> None:
    source_dir = vsm_sample_files[0].parent
    recipe_path = project_root / "recipes" / "vsm" / "default.yaml"
    batch_dir = tmp_path / "vsm_batch"

    exit_code = main(
        [
            "vsm",
            "batch",
            "--input",
            str(source_dir),
            "--pattern",
            "MTJ-B-*.dat",
            "--recipe",
            str(recipe_path),
            "--output-dir",
            str(batch_dir),
        ]
    )
    assert exit_code == 0
    assert (batch_dir / "batch_summary.csv").exists()
    assert (batch_dir / "batch_manifest.json").exists()
    assert (batch_dir / "batch_hysteresis_overlay.png").exists()
    assert (batch_dir / "batch_hysteresis_overlay.png").stat().st_size > 0
    output = capsys.readouterr().out
    assert f"Batch figure [batch_hysteresis_overlay]: {batch_dir / 'batch_hysteresis_overlay.png'}" in output

    manifest = json.loads((batch_dir / "batch_manifest.json").read_text(encoding="utf-8"))
    assert manifest["succeeded_count"] == len(vsm_sample_files)
    assert manifest["failed_count"] == 0
    assert manifest["batch_figures"] == {
        "batch_hysteresis_overlay": str((batch_dir / "batch_hysteresis_overlay.png").resolve())
    }
    assert manifest["batch_figure_png"] == str((batch_dir / "batch_hysteresis_overlay.png").resolve())

    first_json = batch_dir / vsm_sample_files[0].stem / f"{vsm_sample_files[0].stem}_analysis.json"
    export_dir = tmp_path / "vsm_export"
    exit_code = main(
        [
            "vsm",
            "export",
            "--input",
            str(first_json),
            "--output-dir",
            str(export_dir),
        ]
    )
    assert exit_code == 0
    assert (export_dir / f"{vsm_sample_files[0].stem}_trace.csv").exists()
    assert (export_dir / f"{vsm_sample_files[0].stem}_summary.csv").exists()
    assert (export_dir / f"{vsm_sample_files[0].stem}_figure.png").exists()

    single_report = tmp_path / "single_report.md"
    exit_code = main(
        [
            "vsm",
            "report",
            "--input",
            str(first_json),
            "--output",
            str(single_report),
        ]
    )
    assert exit_code == 0
    assert "VSM Report" in single_report.read_text(encoding="utf-8")
    report_text = single_report.read_text(encoding="utf-8")
    assert "Direct Observables" in report_text
    assert "Trust Diagnostics" in report_text
    assert "Secondary Diagnostics" in report_text
    assert "+/-" in report_text
    assert "Background mode" in report_text
    assert "Tail selection mode" in report_text
    assert "Tail fit R^2" in report_text
    assert "Flatness gain balance" in report_text

    batch_report = tmp_path / "batch_report.md"
    exit_code = main(
        [
            "vsm",
            "report",
            "--input",
            str(batch_dir),
            "--output",
            str(batch_report),
        ]
    )
    assert exit_code == 0
    batch_text = batch_report.read_text(encoding="utf-8")
    assert "VSM Batch Report" in batch_text
    assert "Temperature trend:" in batch_text
    assert "Sat. Conf." in batch_text


def test_esr_modality_single_command_matches_legacy_artifacts(tmp_path, project_root, write_bruker_esr_sample) -> None:
    source_file = write_bruker_esr_sample(tmp_path / "generic_esr_trace.dsc")
    output_dir = tmp_path / "generic_esr"
    recipe_path = project_root / "recipes" / "esr" / "default.yaml"

    exit_code = main(
        [
            "esr",
            "single",
            "--input",
            str(source_file),
            "--recipe",
            str(recipe_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "generic_esr_trace_analysis.json").exists()
    assert (output_dir / "generic_esr_trace_trace.csv").exists()
    assert (output_dir / "generic_esr_trace_summary.csv").exists()
    assert (output_dir / "generic_esr_trace_figure.png").exists()


def test_vsm_batch_summary_contains_grouping_metadata(tmp_path, project_root, vsm_sample_files) -> None:
    source_dir = vsm_sample_files[0].parent
    recipe_path = project_root / "recipes" / "vsm" / "default.yaml"
    batch_dir = tmp_path / "vsm_batch_summary"

    exit_code = main(
        [
            "vsm",
            "batch",
            "--input",
            str(source_dir),
            "--pattern",
            "MTJ-B-*.dat",
            "--recipe",
            str(recipe_path),
            "--output-dir",
            str(batch_dir),
        ]
    )
    assert exit_code == 0

    rows = list(csv.DictReader((batch_dir / "batch_summary.csv").open("r", encoding="utf-8", newline="")))
    assert len(rows) == len(vsm_sample_files)
    assert all(row["sample_id"] == "MTJ-B" for row in rows)
    assert all(row["replicate_id"] == "R1" for row in rows)
    assert all(row["background_mode"] in {"none", "slope_only", "rejected"} for row in rows)
    assert all(row["background_subtraction_mode"] == row["background_mode"] for row in rows)
    assert all(row["Ms_emu"] for row in rows)
    assert all(row["ms_error"] for row in rows)
    assert all(row["saturation_confidence"] for row in rows)
