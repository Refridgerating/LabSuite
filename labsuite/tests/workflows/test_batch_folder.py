from __future__ import annotations

import csv
import json

import pytest

from labsuite.core.exceptions import WorkflowError
from labsuite.workflows.batch_folder import discover_esr_source_files, run_esr_batch_workflow


def test_discover_esr_source_files_accepts_direct_file(write_bruker_esr_sample, tmp_path) -> None:
    source_file = write_bruker_esr_sample(tmp_path / "single_trace.dsc")

    discovered = discover_esr_source_files([source_file])

    assert discovered == [source_file.resolve()]


def test_discover_esr_source_files_filters_directory_by_pattern(
    write_bruker_esr_sample, tmp_path
) -> None:
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    keep = write_bruker_esr_sample(source_dir / "alpha_deg_trace.dsc")
    write_bruker_esr_sample(source_dir / "beta_trace.dsc")
    (source_dir / "alpha_deg_trace.csv").write_text("not a descriptor", encoding="utf-8")

    discovered = discover_esr_source_files([source_dir], pattern="*deg*")

    assert discovered == [keep.resolve()]


def test_discover_esr_source_files_respects_recursive_flag(
    write_bruker_esr_sample, tmp_path
) -> None:
    source_dir = tmp_path / "raw"
    nested_dir = source_dir / "nested"
    nested_dir.mkdir(parents=True)
    nested_file = write_bruker_esr_sample(nested_dir / "nested_trace.dsc")

    with pytest.raises(WorkflowError, match="No ESR descriptor source files were discovered"):
        discover_esr_source_files([source_dir], recursive=False, pattern="*.dsc")
    recursive = discover_esr_source_files([source_dir], recursive=True, pattern="*.dsc")

    assert recursive == [nested_file.resolve()]


def test_discover_esr_source_files_deduplicates_and_sorts(
    write_bruker_esr_sample, tmp_path
) -> None:
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    second = write_bruker_esr_sample(source_dir / "b_trace.dsc")
    first = write_bruker_esr_sample(source_dir / "a_trace.dsc")

    discovered = discover_esr_source_files([second, source_dir, first])

    assert discovered == [first.resolve(), second.resolve()]


def test_discover_esr_source_files_rejects_invalid_direct_input(tmp_path) -> None:
    csv_file = tmp_path / "trace.csv"
    csv_file.write_text("field,signal", encoding="utf-8")

    with pytest.raises(WorkflowError, match="Direct file input must be a ESR descriptor source"):
        discover_esr_source_files([csv_file])


def test_discover_esr_source_files_raises_when_no_matches(tmp_path) -> None:
    source_dir = tmp_path / "empty"
    source_dir.mkdir()

    with pytest.raises(WorkflowError, match="No ESR descriptor source files were discovered"):
        discover_esr_source_files([source_dir])


def test_run_esr_batch_workflow_writes_per_file_outputs_and_aggregate_artifacts(
    tmp_path, project_root, write_bruker_esr_sample
) -> None:
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    first = write_bruker_esr_sample(source_dir / "sample-0deg-R1.dsc")
    second = write_bruker_esr_sample(
        source_dir / "sample-45deg-R1.dsc", center_mT=338.5, gamma_mT=0.9
    )
    output_dir = tmp_path / "processed" / "batch_run"
    recipe_path = project_root / "recipes" / "esr" / "default.yaml"

    result = run_esr_batch_workflow(
        inputs=[source_dir],
        recipe_path=recipe_path,
        output_dir=output_dir,
        pattern="*.dsc",
        recursive=False,
    )

    assert result.output_dir == output_dir
    assert result.discovered_sources == [first.resolve(), second.resolve()]
    assert len(result.succeeded_items) == 2
    assert len(result.failed_items) == 0
    assert result.summary_csv_path.exists()
    assert result.manifest_json_path.exists()
    assert (output_dir / "batch_qc.csv").exists()
    assert result.batch_figure_paths == {
        "batch_angle_overlay_R1": output_dir / "batch_angle_overlay_R1.png",
        "batch_processed_offset_R1": output_dir / "batch_processed_offset_R1.png",
    }
    assert all(path.exists() for path in result.batch_figure_paths.values())
    assert all(path.stat().st_size > 0 for path in result.batch_figure_paths.values())

    for item in result.succeeded_items:
        assert item.output_dir.parent == output_dir
        assert item.json_path is not None and item.json_path.exists()
        assert item.csv_path is not None and item.csv_path.exists()
        assert item.summary_csv_path is not None and item.summary_csv_path.exists()
        assert item.figure_path is not None and item.figure_path.exists()

    rows = list(csv.DictReader(result.summary_csv_path.open("r", encoding="utf-8", newline="")))
    assert [row["source_stem"] for row in rows] == ["sample-0deg-R1", "sample-45deg-R1"]
    assert all(row["status"] == "success" for row in rows)
    assert all(row["analysis_json"] for row in rows)
    assert all(row["summary_csv"] for row in rows)
    assert all(row["selected_as_best"] == "True" for row in rows)
    assert all(row["accepted_for_plot"] == "True" for row in rows)

    payload = json.loads(result.manifest_json_path.read_text(encoding="utf-8"))
    assert payload["scan"]["pattern"] == "*.dsc"
    assert payload["scan"]["recursive"] is False
    assert payload["succeeded_count"] == 2
    assert payload["failed_count"] == 0
    assert payload["batch_figure_png"] is None
    assert payload["batch_figures"] == {
        key: str(path) for key, path in sorted(result.batch_figure_paths.items())
    }
    assert len(payload["items"]) == 2


def test_run_esr_batch_workflow_groups_output_by_replicate(
    tmp_path, project_root, write_bruker_esr_sample
) -> None:
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    first = write_bruker_esr_sample(source_dir / "sample-90deg-R1.dsc")
    second = write_bruker_esr_sample(source_dir / "sample-0deg-R2.dsc")
    third = write_bruker_esr_sample(source_dir / "sample-45deg-R2.dsc")
    output_dir = tmp_path / "processed" / "batch_run"
    recipe_path = project_root / "recipes" / "esr" / "default.yaml"

    result = run_esr_batch_workflow(
        inputs=[source_dir],
        recipe_path=recipe_path,
        output_dir=output_dir,
    )

    assert result.discovered_sources == sorted(
        [first.resolve(), second.resolve(), third.resolve()],
        key=lambda path: str(path).lower(),
    )
    assert result.batch_figure_paths == {
        "batch_angle_overlay_R1": output_dir / "batch_angle_overlay_R1.png",
        "batch_angle_overlay_R2": output_dir / "batch_angle_overlay_R2.png",
        "batch_processed_offset_R1": output_dir / "batch_processed_offset_R1.png",
        "batch_processed_offset_R2": output_dir / "batch_processed_offset_R2.png",
    }


def test_run_esr_batch_workflow_continues_when_one_file_fails(
    tmp_path, project_root, write_bruker_esr_sample
) -> None:
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    valid = write_bruker_esr_sample(source_dir / "valid-15deg-R1.dsc")
    broken = source_dir / "broken-30deg-R1.dsc"
    broken.write_text(
        "\n".join(
            [
                "#DESC\t1.2\t* DESCRIPTOR INFORMATION",
                "IKKF\tREAL",
                "IRFMT\tD",
                "XFMT\tD",
                "XTYP\tIDX",
                "YTYP\tNODATA",
                "ZTYP\tNODATA",
                "XPTS\t10",
                "XMIN\t3300",
                "XWID\t100",
                "XUNI\t'G'",
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "processed" / "batch_run"
    recipe_path = project_root / "recipes" / "esr" / "default.yaml"

    result = run_esr_batch_workflow(
        inputs=[source_dir],
        recipe_path=recipe_path,
        output_dir=output_dir,
    )

    assert result.discovered_sources == [broken.resolve(), valid.resolve()]
    assert len(result.succeeded_items) == 1
    assert len(result.failed_items) == 1
    assert result.failed_items[0].source_path == broken.resolve()
    assert "Missing sibling Bruker data file" in (result.failed_items[0].error_message or "")
    assert result.batch_figure_paths == {
        "batch_angle_overlay_R1": output_dir / "batch_angle_overlay_R1.png",
        "batch_processed_offset_R1": output_dir / "batch_processed_offset_R1.png",
    }
    assert all(path.exists() for path in result.batch_figure_paths.values())

    rows = list(csv.DictReader(result.summary_csv_path.open("r", encoding="utf-8", newline="")))
    broken_row = next(row for row in rows if row["source_stem"] == "broken-30deg-R1")
    valid_row = next(row for row in rows if row["source_stem"] == "valid-15deg-R1")
    assert broken_row["status"] == "failed"
    assert "Missing sibling Bruker data file" in broken_row["error_message"]
    assert valid_row["status"] == "success"

    payload = json.loads(result.manifest_json_path.read_text(encoding="utf-8"))
    assert payload["batch_figure_png"] is None
    assert payload["batch_figures"] == {
        key: str(path) for key, path in sorted(result.batch_figure_paths.items())
    }


def test_run_esr_batch_workflow_groups_missing_replicate_under_ungrouped(
    tmp_path, project_root, write_bruker_esr_sample
) -> None:
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    write_bruker_esr_sample(source_dir / "sample-0deg.dsc")
    write_bruker_esr_sample(source_dir / "sample-45deg.dsc")
    output_dir = tmp_path / "processed" / "batch_run"
    recipe_path = project_root / "recipes" / "esr" / "default.yaml"

    result = run_esr_batch_workflow(
        inputs=[source_dir],
        recipe_path=recipe_path,
        output_dir=output_dir,
    )

    assert result.batch_figure_paths == {
        "batch_angle_overlay_UNGROUPED": output_dir / "batch_angle_overlay_UNGROUPED.png",
        "batch_processed_offset_UNGROUPED": output_dir / "batch_processed_offset_UNGROUPED.png",
    }


def test_run_esr_batch_workflow_records_null_batch_figure_when_no_files_succeed(
    tmp_path, project_root
) -> None:
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    for stem in ("broken_a", "broken_b"):
        (source_dir / f"{stem}.dsc").write_text(
            "\n".join(
                [
                    "#DESC\t1.2\t* DESCRIPTOR INFORMATION",
                    "IKKF\tREAL",
                    "IRFMT\tD",
                    "XFMT\tD",
                    "XTYP\tIDX",
                    "YTYP\tNODATA",
                    "ZTYP\tNODATA",
                    "XPTS\t10",
                    "XMIN\t3300",
                    "XWID\t100",
                    "XUNI\t'G'",
                ]
            ),
            encoding="utf-8",
        )
    output_dir = tmp_path / "processed" / "batch_run"
    recipe_path = project_root / "recipes" / "esr" / "default.yaml"

    result = run_esr_batch_workflow(
        inputs=[source_dir],
        recipe_path=recipe_path,
        output_dir=output_dir,
    )

    assert len(result.succeeded_items) == 0
    assert len(result.failed_items) == 2
    assert result.batch_figure_paths == {}
    assert not (output_dir / "batch_angle_overlay_R1.png").exists()
    assert not (output_dir / "batch_processed_offset_R1.png").exists()

    payload = json.loads(result.manifest_json_path.read_text(encoding="utf-8"))
    assert payload["batch_figures"] == {}
    assert payload["batch_figure_png"] is None


def test_run_esr_batch_workflow_selects_best_duplicate_same_angle(
    tmp_path,
    project_root,
    write_bruker_esr_sample,
) -> None:
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    clipped = write_bruker_esr_sample(
        source_dir / "20260220_120000000_sample-65deg-R1.dsc",
        center_mT=343.7,
        gamma_mT=1.2,
        field_start_mT=340.0,
        field_end_mT=346.0,
    )
    best = write_bruker_esr_sample(
        source_dir / "20260220_130000000_sample-65deg-R1.dsc",
        center_mT=340.0,
        gamma_mT=1.0,
        field_start_mT=330.0,
        field_end_mT=350.0,
    )
    other_angle = write_bruker_esr_sample(source_dir / "20260220_140000000_sample-70deg-R1.dsc")
    output_dir = tmp_path / "processed" / "batch_run"
    recipe_path = project_root / "recipes" / "esr" / "default.yaml"

    result = run_esr_batch_workflow(
        inputs=[source_dir],
        recipe_path=recipe_path,
        output_dir=output_dir,
    )

    assert result.batch_figure_paths == {
        "batch_angle_overlay_R1": output_dir / "batch_angle_overlay_R1.png",
        "batch_processed_offset_R1": output_dir / "batch_processed_offset_R1.png",
    }

    qc_rows = list(
        csv.DictReader((output_dir / "batch_qc.csv").open("r", encoding="utf-8", newline=""))
    )
    clipped_row = next(row for row in qc_rows if row["file"] == str(clipped.resolve()))
    best_row = next(row for row in qc_rows if row["file"] == str(best.resolve()))
    other_row = next(row for row in qc_rows if row["file"] == str(other_angle.resolve()))

    assert clipped_row["selected_as_best"] == "False"
    assert clipped_row["accepted_for_plot"] == "False"
    assert clipped_row["reject_reason"] == "edge_truncated"
    assert best_row["selected_as_best"] == "True"
    assert best_row["accepted_for_plot"] == "True"
    assert best_row["reject_reason"] == ""
    assert other_row["selected_as_best"] == "True"
    assert other_row["accepted_for_plot"] == "True"

    summary_rows = list(
        csv.DictReader(result.summary_csv_path.open("r", encoding="utf-8", newline=""))
    )
    best_summary = next(row for row in summary_rows if row["source_file"] == str(best.resolve()))
    clipped_summary = next(
        row for row in summary_rows if row["source_file"] == str(clipped.resolve())
    )
    assert best_summary["selected_as_best"] == "True"
    assert best_summary["accepted_for_plot"] == "True"
    assert clipped_summary["accepted_for_plot"] == "False"
    assert clipped_summary["reject_reason"] == "edge_truncated"
