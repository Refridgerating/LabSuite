from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from labsuite.core.exceptions import WorkflowError
from labsuite.workflows.batch_folder import discover_esr_source_files, run_esr_batch_workflow


def test_discover_esr_source_files_accepts_direct_file(write_bruker_esr_sample, tmp_path) -> None:
    source_file = write_bruker_esr_sample(tmp_path / "single_trace.dsc")

    discovered = discover_esr_source_files([source_file])

    assert discovered == [source_file.resolve()]


def test_discover_esr_source_files_filters_directory_by_pattern(write_bruker_esr_sample, tmp_path) -> None:
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    keep = write_bruker_esr_sample(source_dir / "alpha_deg_trace.dsc")
    write_bruker_esr_sample(source_dir / "beta_trace.dsc")
    (source_dir / "alpha_deg_trace.csv").write_text("not a descriptor", encoding="utf-8")

    discovered = discover_esr_source_files([source_dir], pattern="*deg*")

    assert discovered == [keep.resolve()]


def test_discover_esr_source_files_respects_recursive_flag(write_bruker_esr_sample, tmp_path) -> None:
    source_dir = tmp_path / "raw"
    nested_dir = source_dir / "nested"
    nested_dir.mkdir(parents=True)
    nested_file = write_bruker_esr_sample(nested_dir / "nested_trace.dsc")

    with pytest.raises(WorkflowError, match="No ESR descriptor files were discovered"):
        discover_esr_source_files([source_dir], recursive=False, pattern="*.dsc")
    recursive = discover_esr_source_files([source_dir], recursive=True, pattern="*.dsc")

    assert recursive == [nested_file.resolve()]


def test_discover_esr_source_files_deduplicates_and_sorts(write_bruker_esr_sample, tmp_path) -> None:
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    second = write_bruker_esr_sample(source_dir / "b_trace.dsc")
    first = write_bruker_esr_sample(source_dir / "a_trace.dsc")

    discovered = discover_esr_source_files([second, source_dir, first])

    assert discovered == [first.resolve(), second.resolve()]


def test_discover_esr_source_files_rejects_invalid_direct_input(tmp_path) -> None:
    csv_file = tmp_path / "trace.csv"
    csv_file.write_text("field,signal", encoding="utf-8")

    with pytest.raises(WorkflowError, match="Direct file input must be a Bruker descriptor"):
        discover_esr_source_files([csv_file])


def test_discover_esr_source_files_raises_when_no_matches(tmp_path) -> None:
    source_dir = tmp_path / "empty"
    source_dir.mkdir()

    with pytest.raises(WorkflowError, match="No ESR descriptor files were discovered"):
        discover_esr_source_files([source_dir])


def test_run_esr_batch_workflow_writes_per_file_outputs_and_aggregate_artifacts(
    tmp_path, project_root, write_bruker_esr_sample
) -> None:
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    first = write_bruker_esr_sample(source_dir / "alpha_trace.dsc")
    second = write_bruker_esr_sample(
        source_dir / "beta_trace.dsc",
        components=[
            {"amplitude": 1.0, "center_mT": 335.0, "gamma_mT": 0.8, "offset": 0.0},
            {"amplitude": 0.9, "center_mT": 345.0, "gamma_mT": 0.9, "offset": 0.0},
        ],
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

    for item in result.succeeded_items:
        assert item.output_dir.parent == output_dir
        assert item.json_path is not None and item.json_path.exists()
        assert item.csv_path is not None and item.csv_path.exists()
        assert item.summary_csv_path is not None and item.summary_csv_path.exists()
        assert item.figure_path is not None and item.figure_path.exists()

    rows = list(csv.DictReader(result.summary_csv_path.open("r", encoding="utf-8", newline="")))
    assert [row["source_stem"] for row in rows] == ["alpha_trace", "beta_trace"]
    assert all(row["status"] == "success" for row in rows)
    assert all(row["analysis_json"] for row in rows)
    assert all(row["summary_csv"] for row in rows)

    payload = json.loads(result.manifest_json_path.read_text(encoding="utf-8"))
    assert payload["scan"]["pattern"] == "*.dsc"
    assert payload["scan"]["recursive"] is False
    assert payload["succeeded_count"] == 2
    assert payload["failed_count"] == 0
    assert len(payload["items"]) == 2


def test_run_esr_batch_workflow_continues_when_one_file_fails(tmp_path, project_root, write_bruker_esr_sample) -> None:
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    valid = write_bruker_esr_sample(source_dir / "valid_trace.dsc")
    broken = source_dir / "broken_trace.dsc"
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

    rows = list(csv.DictReader(result.summary_csv_path.open("r", encoding="utf-8", newline="")))
    broken_row = next(row for row in rows if row["source_stem"] == "broken_trace")
    valid_row = next(row for row in rows if row["source_stem"] == "valid_trace")
    assert broken_row["status"] == "failed"
    assert "Missing sibling Bruker data file" in broken_row["error_message"]
    assert valid_row["status"] == "success"
