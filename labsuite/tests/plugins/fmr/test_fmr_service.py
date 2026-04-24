from __future__ import annotations

from matplotlib.axes import Axes

from labsuite.core.recipes import load_fmr_recipe
from labsuite.plugins.fmr.derived import GONZALEZ_FUENTES_SINGLE_POLARITY_WARNING
from labsuite.plugins.fmr.service import analyze_fmr_file, export_fmr_trace_diagnostic_figures


def _write_polarity_recipe(path):
    path.write_text(
        "\n".join(
            [
                "name: fmr-polarity-test",
                "field_polarity_correction:",
                "  enabled: true",
                "  method: gonzalez_fuentes_average",
                "  polarity_column: Polarity",
                "  on_unpaired: warn_and_keep_raw",
                "  fit_field: Hres_avg",
                "  run_comparison_fits: true",
                "measurement_requirements:",
                "  gonzalez_fuentes_average:",
                "    requires_positive_and_negative_field_sweeps: true",
                "    cannot_be_applied_to_single_polarity_data: true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_fmr_recipe_loads_nested_gonzalez_fuentes_measurement_requirements(tmp_path) -> None:
    recipe_path = tmp_path / "nested_fmr.yaml"
    recipe_path.write_text(
        "\n".join(
            [
                "name: nested-fmr",
                "fmr:",
                "  kittel:",
                "    field_polarity_correction:",
                "      enabled: true",
                "      polarity_column: Polarity",
                "  measurement_requirements:",
                "    gonzalez_fuentes_average:",
                "      requires_positive_and_negative_field_sweeps: true",
                "      cannot_be_applied_to_single_polarity_data: true",
                "",
            ]
        ),
        encoding="utf-8",
    )

    recipe = load_fmr_recipe(recipe_path)

    assert recipe.field_polarity_correction.enabled is True
    assert recipe.field_polarity_correction.polarity_column == "Polarity"
    assert (
        recipe.measurement_requirements.gonzalez_fuentes_average.requires_positive_and_negative_field_sweeps
        is True
    )


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
    assert result.provenance["resonance_metrics_config"]["compute_resonance_metrics"] is True
    assert result.analysis_payload["resonance_metrics"]
    assert all("hres" in item for item in result.analysis_payload["resonance_metrics"])


def test_gonzalez_fuentes_correction_skips_single_polarity_and_keeps_raw_kittel(
    tmp_path,
    write_phasefmr_log,
) -> None:
    source_file = write_phasefmr_log(
        tmp_path / "Temp2-Co-A-2,5to17GHz-R1.log",
        frequencies_GHz=[6.0, 8.0, 10.0, 12.0],
        field_polarities=["positive"],
    )
    recipe_path = _write_polarity_recipe(tmp_path / "fmr_polarity.yaml")

    result = analyze_fmr_file(source_file, recipe_path)

    assert result.summary_metrics["kittel_success"] is True
    assert GONZALEZ_FUENTES_SINGLE_POLARITY_WARNING in result.warnings
    assert result.summary_metrics["field_polarity_correction_statuses"] == ["skipped_single_polarity"]
    series = result.analysis_payload["series_collection_result"]["series_by_label"]["single_unassigned"]
    assert all(point["Hres_avg_mT"] is None for point in series["metadata"]["polarity_points"])
    assert series["metadata"]["field_polarity_correction"]["fit_field"] == "Hres"


def test_gonzalez_fuentes_correction_uses_paired_average_for_kittel(
    tmp_path,
    write_phasefmr_log,
) -> None:
    source_file = write_phasefmr_log(
        tmp_path / "Temp2-Co-A-2,5to17GHz-R1.log",
        frequencies_GHz=[6.0, 8.0, 10.0, 12.0],
        field_polarities=["positive", "negative"],
        polarity_field_offsets_mT={"positive": 2.0, "negative": -2.0},
    )
    recipe_path = _write_polarity_recipe(tmp_path / "fmr_polarity.yaml")

    result = analyze_fmr_file(source_file, recipe_path)

    series = result.analysis_payload["series_collection_result"]["series_by_label"]["single_unassigned"]
    paired_points = [
        point
        for point in series["metadata"]["polarity_points"]
        if point["polarity_pair_status"] == "paired"
    ]
    assert len(paired_points) == 4
    assert result.summary_metrics["field_polarity_pair_count"] == 4
    assert result.summary_metrics["field_polarity_correction_statuses"] == ["applied"]
    assert all(point["Hres_avg_mT"] is not None for point in paired_points)
    assert series["resonance_field_mT"] == [point["Hres_avg_mT"] for point in paired_points]
    physics = result.analysis_payload["physics_collection_result"]["physics_by_label"]["single_unassigned"]
    assert physics["kittel_fit"]["success"] is True
    assert "corrected" in physics["metadata"]["polarity_comparison_fits"]


def test_gonzalez_fuentes_correction_keeps_modes_separate_for_double_traces(
    tmp_path,
    write_phasefmr_log,
) -> None:
    source_file = write_phasefmr_log(
        tmp_path / "MTJ-A-03APR2026-R1.log",
        frequencies_GHz=[8.0, 9.0, 10.0, 11.0],
        secondary_resonance_delta_mT=52.0,
        secondary_linewidth_mT=9.0,
        field_polarities=["positive", "negative"],
    )
    recipe_path = _write_polarity_recipe(tmp_path / "fmr_polarity.yaml")

    result = analyze_fmr_file(source_file, recipe_path)

    collection = result.analysis_payload["series_collection_result"]["series_by_label"]
    assert "mode_1" in collection
    assert "mode_2" in collection
    assert collection["mode_1"]["metadata"]["field_polarity_correction"]["paired_point_count"] >= 3
    assert collection["mode_2"]["metadata"]["field_polarity_correction"]["paired_point_count"] >= 3
    for label in ("mode_1", "mode_2"):
        assert all(
            label in point["component_id"]
            for point in collection[label]["metadata"]["polarity_points"]
            if point["polarity_pair_status"] == "paired"
        )


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


def test_trace_diagnostics_include_absorption_subplot(tmp_path, project_root, write_phasefmr_log, monkeypatch) -> None:
    source_file = write_phasefmr_log(tmp_path / "Temp2-Co-A-2,5to17GHz-R1.log", frequencies_GHz=[8.0, 10.0])
    recipe_path = project_root / "recipes" / "fmr" / "default.yaml"
    result = analyze_fmr_file(source_file, recipe_path)

    saved_axes_counts: list[int] = []

    import matplotlib.pyplot as plt

    original_close = plt.close

    def _capture_close(figure):
        if hasattr(figure, "axes"):
            saved_axes_counts.append(len(figure.axes))
        return original_close(figure)

    monkeypatch.setattr(plt, "close", _capture_close)
    export_fmr_trace_diagnostic_figures(result, tmp_path / "diagnostics")

    assert saved_axes_counts
    assert all(count == 3 for count in saved_axes_counts)


def test_trace_diagnostics_label_hres_markers_and_drop_candidate_center_lines(tmp_path, project_root, write_phasefmr_log, monkeypatch) -> None:
    source_file = write_phasefmr_log(
        tmp_path / "MTJ-A-03APR2026-R1.log",
        frequencies_GHz=[8.0, 10.0],
        secondary_resonance_delta_mT=52.0,
        secondary_linewidth_mT=9.0,
    )
    recipe_path = project_root / "recipes" / "fmr" / "default.yaml"
    result = analyze_fmr_file(source_file, recipe_path)

    captured: list[dict[str, object]] = []
    original_axvline = Axes.axvline

    def _capture_axvline(self, x=0, ymin=0, ymax=1, **kwargs):
        captured.append(
            {
                "x": float(x),
                "linestyle": kwargs.get("linestyle"),
                "label": kwargs.get("label"),
            }
        )
        return original_axvline(self, x, ymin=ymin, ymax=ymax, **kwargs)

    monkeypatch.setattr(Axes, "axvline", _capture_axvline)
    export_fmr_trace_diagnostic_figures(result, tmp_path / "diagnostics")

    expected_labels = {
        f"{component['component_label']} Hres"
        for fit in result.analysis_payload["trace_fit_results"]
        for component in fit.get("selected_components", [])
    }
    expected_positions = {
        float(component["H_res_mT"])
        for fit in result.analysis_payload["trace_fit_results"]
        for component in fit.get("selected_components", [])
    }

    assert captured
    assert all(call["linestyle"] != ":" for call in captured)
    assert expected_labels <= {call["label"] for call in captured if call["label"] is not None}
    assert expected_positions <= {call["x"] for call in captured if call["label"] in expected_labels}
