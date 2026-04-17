from __future__ import annotations

from matplotlib.axes import Axes

from labsuite.plugins.fmr.service import analyze_fmr_file, export_fmr_trace_diagnostic_figures


def test_analyze_fmr_file_builds_single_unassigned_series(tmp_path, project_root, write_phasefmr_log) -> None:
    source_file = write_phasefmr_log(tmp_path / "Temp2-Co-A-2,5to17GHz-R1.log", frequencies_GHz=[4.0, 6.0, 8.0, 10.0])
    recipe_path = project_root / "recipes" / "fmr" / "default.yaml"
    result = analyze_fmr_file(source_file, recipe_path)
    assert result.measurement.modality == "fmr"
    assert result.summary_metrics["sample_id"] == "Temp2-Co-A"
    assert "series_collection_result" in result.analysis_payload
    assert "single_unassigned" in result.analysis_payload["series_collection_result"]["series_by_label"]
    assert result.analysis_payload["physics_collection_result"]["physics_by_label"]["single_unassigned"]["kittel_fit"]["success"] is True
    assert all(fit["r_squared"] is not None for fit in result.analysis_payload["trace_fit_results"])


def test_analyze_fmr_file_builds_mode_1_and_mode_2_series_for_double_traces(tmp_path, project_root, write_phasefmr_log) -> None:
    source_file = write_phasefmr_log(tmp_path / "MTJ-A-03APR2026-R1.log", frequencies_GHz=[8.0, 9.0, 10.0, 11.0], secondary_resonance_delta_mT=52.0, secondary_linewidth_mT=9.0)
    recipe_path = project_root / "recipes" / "fmr" / "default.yaml"
    result = analyze_fmr_file(source_file, recipe_path)
    series_labels = result.analysis_payload["series_collection_result"]["series_by_label"].keys()
    assert "mode_1" in series_labels
    assert "mode_2" in series_labels
    assert "single_unassigned" not in series_labels
    assert result.summary_metrics["mode_counts"]["double"] >= 1


def test_analyze_fmr_file_handles_single_trace_without_crashing_physics(tmp_path, project_root, write_phasefmr_log) -> None:
    source_file = write_phasefmr_log(tmp_path / "NiFeStd1-03APR2026-R1.log", frequencies_GHz=[9.459], include_temp=True, temperature_K=122.2)
    recipe_path = project_root / "recipes" / "fmr" / "default.yaml"
    result = analyze_fmr_file(source_file, recipe_path)
    assert result.summary_metrics["trace_count"] == 1
    assert result.summary_metrics["has_multiple_frequencies"] is False
    assert result.summary_metrics["kittel_success"] is False
    assert result.summary_metrics["linewidth_success"] is False
    warnings = result.analysis_payload["physics_collection_result"]["physics_by_label"]["single_unassigned"]["warnings"]
    assert any("insufficient_points" in warning for warning in warnings)


def test_analyze_fmr_file_preserves_rejected_or_partial_components(tmp_path, project_root, write_phasefmr_log) -> None:
    source_file = write_phasefmr_log(tmp_path / "MTJ-A-03APR2026-R1.log", frequencies_GHz=[9.0], secondary_resonance_delta_mT=52.0, secondary_linewidth_mT=250.0)
    recipe_path = project_root / "recipes" / "fmr" / "default.yaml"
    result = analyze_fmr_file(source_file, recipe_path)
    fits = result.analysis_payload["trace_fit_results"]
    assert len(fits) == 1
    assert fits[0]["selected_mode"] in {"single", "double"}
    assert fits[0]["selected_components"]


def test_trace_diagnostics_annotate_r_squared(tmp_path, project_root, write_phasefmr_log, monkeypatch) -> None:
    source_file = write_phasefmr_log(tmp_path / "Temp2-Co-A-2,5to17GHz-R1.log", frequencies_GHz=[8.0, 10.0])
    recipe_path = project_root / "recipes" / "fmr" / "default.yaml"
    result = analyze_fmr_file(source_file, recipe_path)

    captured: list[str] = []
    original_text = Axes.text

    def _capture_text(self, *args, **kwargs):
        if len(args) >= 3:
            captured.append(str(args[2]))
        return original_text(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "text", _capture_text)
    export_fmr_trace_diagnostic_figures(result, tmp_path / "diagnostics")

    assert captured
    assert any("R^2 =" in item for item in captured)
