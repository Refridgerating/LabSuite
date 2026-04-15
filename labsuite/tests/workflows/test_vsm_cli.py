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
    assert payload["analysis_payload"]["background_fit"]["combined_background"]["subtraction_mode"] == "slope_only_split_tails"
    assert payload["analysis_payload"]["background_fit"]["positive_tail_fit"]["parameters"]["slope_emu_per_mT"] is not None
    assert payload["analysis_payload"]["background_fit"]["negative_tail_fit"]["parameters"]["slope_emu_per_mT"] is not None
    assert payload["analysis_payload"]["centering"]["applied"] is False
    assert len(payload["analysis_payload"]["branches"]) >= 3
    assert payload["plot_manifest"]["figure_type"] == "vsm_loop_diagnostic"
    assert payload["summary_metrics"]["background_subtraction_mode"] == "slope_only_split_tails"
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

    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert header == (
        "acquisition_index,raw_field_oe,field_mT,raw_moment_emu,processed_moment_emu,"
        "corrected_moment_emu,background_fit_emu,positive_tail_fit_emu,negative_tail_fit_emu,branch_id,branch_direction,temperature_k,"
        "moment_std_err_emu,final_field_mT,final_moment_emu"
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
    assert "background_subtraction_mode" in summary_header
    assert "background_qc_passed" in summary_header
    assert "centering_field_offset_mT" in summary_header
    assert "warnings" in summary_header


def test_vsm_batch_export_and_report_workflows(tmp_path, project_root, vsm_sample_files) -> None:
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

    manifest = json.loads((batch_dir / "batch_manifest.json").read_text(encoding="utf-8"))
    assert manifest["succeeded_count"] == len(vsm_sample_files)
    assert manifest["failed_count"] == 0

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
    assert "+/-" in report_text

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
    assert all(row["background_subtraction_mode"] == "slope_only_split_tails" for row in rows)
    assert all(row["Ms_emu"] for row in rows)
    assert all(row["ms_error"] for row in rows)
    assert all(row["saturation_confidence"] for row in rows)
