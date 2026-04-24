from __future__ import annotations

import csv
import json

from labsuite.cli.main import main


def test_fmr_single_exports_all_artifacts(tmp_path, project_root, write_phasefmr_log) -> None:
    source_file = write_phasefmr_log(
        tmp_path / "Temp2-Co-A-2,5to17GHz-R1.log",
        frequencies_GHz=[8.0, 9.0, 10.0, 11.0],
        secondary_resonance_delta_mT=52.0,
        secondary_linewidth_mT=9.0,
    )
    output_dir = tmp_path / "fmr_single"
    recipe_path = project_root / "recipes" / "fmr" / "default.yaml"

    exit_code = main(
        [
            "fmr",
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
    series_path = output_dir / f"{source_file.stem}_series.csv"
    diagnostics_dir = output_dir / "trace_diagnostics"
    assert json_path.exists()
    assert csv_path.exists()
    assert summary_path.exists()
    assert figure_path.exists()
    assert series_path.exists()
    assert diagnostics_dir.exists()
    assert len(list(diagnostics_dir.glob("*.png"))) == 4

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["measurement"]["modality"] == "fmr"
    assert payload["summary_metrics"]["accepted_trace_count"] >= 3
    assert "trace_fit_results" in payload["analysis_payload"]
    assert "series_collection_result" in payload["analysis_payload"]
    assert "physics_collection_result" in payload["analysis_payload"]
    assert payload["provenance"]["resonance_metrics_config"]["compute_resonance_metrics"] is True
    assert payload["analysis_payload"]["resonance_metrics"]
    assert payload["artifacts"]["trace_diagnostics_dir"] == str(diagnostics_dir)
    assert len(payload["artifacts"]["trace_diagnostic_paths"]) == 4
    assert "mode_1" in payload["analysis_payload"]["series_collection_result"]["series_by_label"]
    assert "mode_2" in payload["analysis_payload"]["series_collection_result"]["series_by_label"]
    assert all(fit["r_squared"] is not None for fit in payload["analysis_payload"]["trace_fit_results"])

    summary_rows = list(csv.DictReader(summary_path.open("r", encoding="utf-8", newline="")))
    assert summary_rows
    assert "r_squared" in summary_rows[0]
    assert "selected_mode" in summary_rows[0]
    assert "component_label" in summary_rows[0]
    assert "component_accepted" in summary_rows[0]
    assert "residual_rmse_fraction" in summary_rows[0]
    assert "amplitude_snr" in summary_rows[0]
    assert "feature_center_mT" in summary_rows[0]
    assert "center_feature_disagreement_mT" in summary_rows[0]
    assert "critical_bound_hit_names" in summary_rows[0]
    assert "acceptance_checks" in summary_rows[0]
    assert "hres" in summary_rows[0]
    assert "asymmetry_ratio" in summary_rows[0]

    series_rows = list(csv.DictReader(series_path.open("r", encoding="utf-8", newline="")))
    assert any(row["series_label"] == "mode_1" for row in series_rows)
    assert any(row["series_label"] == "mode_2" for row in series_rows)


def test_fmr_batch_export_and_report_workflows(tmp_path, project_root, write_phasefmr_log) -> None:
    source_dir = tmp_path / "raw_fmr"
    source_dir.mkdir()
    first_file = write_phasefmr_log(source_dir / "Temp2-Co-A-2,5to17GHz-R1.log", frequencies_GHz=[8.0, 9.0, 10.0, 11.0], secondary_resonance_delta_mT=52.0, secondary_linewidth_mT=9.0)
    write_phasefmr_log(source_dir / "Temp2-Co-A-2,5to17GHz-R2.log", frequencies_GHz=[8.0, 9.0, 10.0, 11.0], secondary_resonance_delta_mT=52.0, secondary_linewidth_mT=9.0)
    recipe_path = project_root / "recipes" / "fmr" / "default.yaml"
    batch_dir = tmp_path / "fmr_batch"

    exit_code = main(
        [
            "fmr",
            "batch",
            "--input",
            str(source_dir),
            "--recipe",
            str(recipe_path),
            "--output-dir",
            str(batch_dir),
        ]
    )
    assert exit_code == 0
    assert (batch_dir / "batch_summary.csv").exists()
    assert (batch_dir / "batch_manifest.json").exists()

    first_json = batch_dir / first_file.stem / f"{first_file.stem}_analysis.json"
    export_dir = tmp_path / "fmr_export"
    exit_code = main(
        [
            "fmr",
            "export",
            "--input",
            str(first_json),
            "--output-dir",
            str(export_dir),
        ]
    )
    assert exit_code == 0
    assert (export_dir / f"{first_file.stem}_trace.csv").exists()
    assert (export_dir / f"{first_file.stem}_summary.csv").exists()
    assert (export_dir / f"{first_file.stem}_figure.png").exists()
    assert (export_dir / "trace_diagnostics").exists()

    single_report = tmp_path / "fmr_single_report.md"
    exit_code = main(
        [
            "fmr",
            "report",
            "--input",
            str(first_json),
            "--output",
            str(single_report),
        ]
    )
    assert exit_code == 0
    single_report_text = single_report.read_text(encoding="utf-8")
    assert "FMR Report" in single_report_text
    assert "Fit Modes" in single_report_text
    assert "Series Buckets" in single_report_text
    assert "Diagnostics folder" in single_report_text

    batch_report = tmp_path / "fmr_batch_report.md"
    exit_code = main(
        [
            "fmr",
            "report",
            "--input",
            str(batch_dir),
            "--output",
            str(batch_report),
        ]
    )
    assert exit_code == 0
    batch_report_text = batch_report.read_text(encoding="utf-8")
    assert "FMR Batch Report" in batch_report_text
    assert "Rejection Reasons" in batch_report_text
    assert "accepted components" in batch_report_text


def test_fmr_batch_can_export_resonance_metrics_csv(tmp_path, project_root, write_phasefmr_log) -> None:
    source_dir = tmp_path / "raw_fmr"
    source_dir.mkdir()
    write_phasefmr_log(source_dir / "Temp2-Co-A-2,5to17GHz-R1.log", frequencies_GHz=[8.0, 9.0, 10.0])
    recipe_path = project_root / "recipes" / "fmr" / "default.yaml"
    batch_dir = tmp_path / "fmr_batch"

    exit_code = main(
        [
            "fmr",
            "batch",
            "--input",
            str(source_dir),
            "--recipe",
            str(recipe_path),
            "--output-dir",
            str(batch_dir),
            "--export-resonance-metrics",
        ]
    )

    assert exit_code == 0
    resonance_csv = batch_dir / "batch_resonance_metrics.csv"
    assert resonance_csv.exists()
    rows = list(csv.DictReader(resonance_csv.open("r", encoding="utf-8", newline="")))
    assert rows
    assert "hres" in rows[0]
    assert "resonance_metrics_success" in rows[0]


def test_fmr_cli_field_polarity_flags_export_pair_diagnostics(
    tmp_path,
    project_root,
    write_phasefmr_log,
) -> None:
    source_file = write_phasefmr_log(
        tmp_path / "Temp2-Co-A-2,5to17GHz-R1.log",
        frequencies_GHz=[6.0, 8.0, 10.0, 12.0],
        field_polarities=["positive", "negative"],
        polarity_field_offsets_mT={"positive": 2.0, "negative": -2.0},
    )
    output_dir = tmp_path / "fmr_polarity_single"
    recipe_path = project_root / "recipes" / "fmr" / "default.yaml"

    exit_code = main(
        [
            "fmr",
            "single",
            "--input",
            str(source_file),
            "--recipe",
            str(recipe_path),
            "--output-dir",
            str(output_dir),
            "--registry",
            str(tmp_path / "missing_registry.yaml"),
            "--field-polarity-correction",
            "gonzalez-fuentes",
            "--polarity-column",
            "Polarity",
            "--compare-polarity-fits",
        ]
    )

    assert exit_code == 0
    json_path = output_dir / f"{source_file.stem}_analysis.json"
    summary_path = output_dir / f"{source_file.stem}_summary.csv"
    series_path = output_dir / f"{source_file.stem}_series.csv"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary_metrics"]["field_polarity_pair_count"] == 4
    assert payload["summary_metrics"]["field_polarity_correction_statuses"] == ["applied"]

    summary_rows = list(csv.DictReader(summary_path.open("r", encoding="utf-8", newline="")))
    assert "field_polarity" in summary_rows[0]
    assert "Hres_avg_mT" in summary_rows[0]
    series_rows = list(csv.DictReader(series_path.open("r", encoding="utf-8", newline="")))
    paired_rows = [row for row in series_rows if row["row_type"] == "series_point"]
    assert paired_rows
    assert all(row["polarity_pair_status"] == "paired" for row in paired_rows)
    assert all(row["fit_field"] == "Hres_avg" for row in paired_rows)
