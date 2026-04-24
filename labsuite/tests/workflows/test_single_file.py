from __future__ import annotations

import json

from labsuite.cli.main import main


def test_esr_single_file_cli_exports_all_artifacts(tmp_path, project_root, write_bruker_esr_sample) -> None:
    source_file = write_bruker_esr_sample(tmp_path / "workflow_trace.dsc")
    output_dir = tmp_path / "artifacts"
    recipe_path = project_root / "recipes" / "esr" / "default.yaml"

    exit_code = main(
        [
            "esr-single",
            str(source_file),
            "--recipe",
            str(recipe_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0

    json_path = output_dir / "workflow_trace_analysis.json"
    csv_path = output_dir / "workflow_trace_trace.csv"
    summary_csv_path = output_dir / "workflow_trace_summary.csv"
    figure_path = output_dir / "workflow_trace_figure.png"

    assert json_path.exists()
    assert csv_path.exists()
    assert summary_csv_path.exists()
    assert figure_path.exists()
    assert figure_path.stat().st_size > 0

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["fit_selection"]["single_fit"]["model_name"] == "derivative_lorentzian"
    assert payload["recipe"]["name"] == "esr-default"
    assert payload["metadata"]["parser"] == "bruker_esr_native_v1"
    assert payload["metadata"]["bruker"]["point_count"] == 401
    assert payload["fit_selection"]["selected_mode"] == "single"
    assert "integrated_curves" in payload
    assert "primary_integrated_curves" in payload
    assert "fit_local_integrated_curves" in payload
    assert "local_integrated_curves" in payload
    assert "baseline_summaries" in payload
    assert payload["primary_integrated_curves"]["integration_kind"] == "primary_fit_model"
    assert payload["fit_local_integrated_curves"] is None or payload["fit_local_integrated_curves"]["integration_kind"] == "fit_local_windowed_model"
    assert payload["local_integrated_curves"] is None or payload["local_integrated_curves"]["integration_kind"] == "primary_local_window"
    assert payload["integrated_curves"]["integration_kind"] == "diagnostic_full_span"
    assert payload["integral_summaries"]["total"]["label"] == "total"
    assert payload["integral_summaries"]["total"]["integration_kind"] == "primary_fit_model"
    assert payload["integral_summaries"]["fit_local_total"]["integration_kind"] == "fit_local_windowed_model"
    assert payload["integral_summaries"]["local_total"]["integration_kind"] == "primary_local_window"
    assert payload["integral_summaries"]["diagnostic_total"]["integration_kind"] == "diagnostic_full_span"
    assert "qc" in payload
    assert payload["resonance_metrics"]["config"]["compute_resonance_metrics"] is True
    assert len(payload["resonance_metrics"]["modes"]) == 1
    assert payload["resonance_metrics"]["modes"][0]["hres"] is not None
    assert "single_fit_attempts" in payload["fit_selection"]
    assert len(payload["fit_selection"]["single_fit_attempts"]) == 1
    assert payload["fit_selection"]["single_fit_attempts"][0]["scope"] == "global_full_trace"
    assert payload["fit_selection"]["single_fit_attempts"][0]["selected_for_primary"] is True
    assert payload["fit_selection"]["single_fit"]["residual_summary"]["rmse"] >= 0.0
    assert payload["fit_selection"]["single_fit"]["feature_summary"]["zero_crossing_field_mT"] is not None
    assert payload["fit_selection"]["single_fit"]["parameter_diagnostics"]["center_mT"]["stderr_missing"] in {True, False}
    assert payload["fit_selection"]["single_fit"]["convergence"]["success"] is True

    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert header == (
        "field_mT,raw_derivative_signal,processed_derivative_signal,"
        "primary_absorption_signal,primary_area_signal,"
        "fit_local_absorption_signal,fit_local_area_signal,"
        "local_absorption_signal,local_area_signal,"
        "diagnostic_absorption_signal,diagnostic_area_signal,selected_fit_signal,selected_residual"
    )
    summary_header = summary_csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert "derivative_baseline_slope" in summary_header
    assert "zero_crossing_field_mT" in summary_header
    assert "center_mT_stderr" in summary_header
    assert "convergence_message" in summary_header
    assert "diagnostic_full_span_area_integral" in summary_header
    assert "integral_kind" in summary_header
    assert "integration_start_field_mT" in summary_header
    assert "baseline_polyorder" in summary_header
    assert "fit_scope" in summary_header
    assert "fit_valid" in summary_header
    assert "fit_rejection_reason" in summary_header
    assert "selected_for_primary" in summary_header
    assert "integration_window_clipped_by_detected_window" in summary_header
    assert "fit_local_windowed_intensity_proxy" in summary_header
    assert "local_windowed_intensity_proxy" in summary_header
    assert "fit_local_disagreement_flag" in summary_header
    assert "hres" in summary_header
    assert "asymmetry_ratio" in summary_header


def test_esr_single_file_cli_forced_split_exports_components(tmp_path, project_root, write_bruker_esr_sample) -> None:
    source_file = write_bruker_esr_sample(
        tmp_path / "split_trace.dsc",
        components=[
            {"amplitude": 1.1, "center_mT": 335.0, "gamma_mT": 0.9, "offset": 0.0},
            {"amplitude": 0.95, "center_mT": 345.0, "gamma_mT": 1.0, "offset": 0.0},
        ],
    )
    output_dir = tmp_path / "split_artifacts"
    recipe_path = project_root / "recipes" / "esr" / "default.yaml"

    exit_code = main(
        [
            "esr-single",
            str(source_file),
            "--recipe",
            str(recipe_path),
            "--output-dir",
            str(output_dir),
            "--fit-mode",
            "split",
        ]
    )

    assert exit_code == 0
    payload = json.loads((output_dir / "split_trace_analysis.json").read_text(encoding="utf-8"))
    assert payload["fit_selection"]["selected_mode"] == "split"
    assert len(payload["fit_selection"]["peak_fits"]) == 2
    assert payload["primary_integrated_curves"]["integration_kind"] == "primary_fit_model"
    assert payload["fit_local_integrated_curves"] is None or payload["fit_local_integrated_curves"]["integration_kind"] == "fit_local_windowed_model"
    assert payload["local_integrated_curves"] is None or payload["local_integrated_curves"]["integration_kind"] == "primary_local_window"
    assert payload["fit_selection"]["peak_fits"][0]["fit"]["feature_summary"]["zero_crossing_field_mT"] is not None
    assert payload["fit_selection"]["peak_fits"][0]["fit"]["convergence"]["success"] is True
    assert payload["fit_selection"]["peak_fits"][0]["attempts"][0]["scope"] == "peak_window_local"
    assert payload["fit_selection"]["peak_fits"][0]["attempts"][0]["selected_for_primary"] is True
    header = (output_dir / "split_trace_trace.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "peak_1_component_signal" in header
    assert "peak_2_component_signal" in header
    summary_header = (output_dir / "split_trace_summary.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "window_peak_field_mT" in summary_header
    assert "gamma_mT_relative_stderr" in summary_header
    assert "fit_scope" in summary_header


def test_esr_single_file_cli_exports_split_local_suppression_reason(tmp_path, project_root, write_bruker_esr_sample) -> None:
    source_file = write_bruker_esr_sample(
        tmp_path / "edge_split_trace.dsc",
        components=[
            {"amplitude": 1.15, "center_mT": 335.0, "gamma_mT": 0.95, "offset": 0.0},
            {"amplitude": 0.9, "center_mT": 348.1, "gamma_mT": 1.35, "offset": 0.0},
        ],
    )
    output_dir = tmp_path / "edge_split_artifacts"
    recipe_path = project_root / "recipes" / "esr" / "default.yaml"

    exit_code = main(
        [
            "esr-single",
            str(source_file),
            "--recipe",
            str(recipe_path),
            "--output-dir",
            str(output_dir),
            "--fit-mode",
            "split",
        ]
    )

    assert exit_code == 0
    payload = json.loads((output_dir / "edge_split_trace_analysis.json").read_text(encoding="utf-8"))
    assert payload["fit_selection"]["selected_mode"] == "split"
    assert payload["fit_local_integrated_curves"] is None
    assert payload["local_integrated_curves"] is None
    assert payload["integral_summaries"]["fit_local_total"]["area_integral"] is None
    assert payload["integral_summaries"]["local_total"]["area_integral"] is None
    assert payload["qc"]["fit_local_disagreement_reason"] is not None
    assert "split_local_diagnostic_unavailable" in payload["qc"]["fit_local_disagreement_reason"]
    assert any(
        peak["fit"]["derived"].get("local_diagnostic_reason") is not None
        for peak in payload["fit_selection"]["peak_fits"]
    )


def test_esr_single_file_cli_runs_on_actual_bruker_dataset(tmp_path, project_root, bruker_sample_stem) -> None:
    output_dir = tmp_path / "bruker_artifacts"
    recipe_path = project_root / "recipes" / "esr" / "default.yaml"
    source_file = bruker_sample_stem.with_suffix(".dsc")

    exit_code = main(
        [
            "esr-single",
            str(source_file),
            "--recipe",
            str(recipe_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / f"{source_file.stem}_analysis.json").exists()
    assert (output_dir / f"{source_file.stem}_trace.csv").exists()
    assert (output_dir / f"{source_file.stem}_summary.csv").exists()
    assert (output_dir / f"{source_file.stem}_figure.png").exists()
    payload = json.loads((output_dir / f"{source_file.stem}_analysis.json").read_text(encoding="utf-8"))
    assert payload["fit_selection"]["selected_mode"] == "split"
