from __future__ import annotations

import csv
import json

import pytest

from labsuite.cli.main import main


def test_fit_single_accepts_direct_file_input(tmp_path, project_root, write_bruker_esr_sample) -> None:
    source_file = write_bruker_esr_sample(tmp_path / "single_trace.dsc")
    output_dir = tmp_path / "processed" / "single_out"
    recipe_path = project_root / "recipes" / "esr" / "default.yaml"

    exit_code = main(
        [
            "fit-single",
            "--input",
            str(source_file),
            "--recipe",
            str(recipe_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "single_trace_analysis.json").exists()
    assert (output_dir / "single_trace_summary.csv").exists()


def test_fit_single_accepts_folder_input_when_one_match_exists(tmp_path, project_root, write_bruker_esr_sample) -> None:
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    source_file = write_bruker_esr_sample(source_dir / "folder_trace.dsc")
    (source_dir / "folder_trace.csv").write_text("ignore", encoding="utf-8")
    output_dir = tmp_path / "processed" / "single_out"
    recipe_path = project_root / "recipes" / "esr" / "default.yaml"

    exit_code = main(
        [
            "fit-single",
            "--input",
            str(source_dir),
            "--recipe",
            str(recipe_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / f"{source_file.stem}_analysis.json").exists()


def test_fit_single_errors_when_folder_resolves_multiple_matches(tmp_path, project_root, write_bruker_esr_sample) -> None:
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    write_bruker_esr_sample(source_dir / "alpha_trace.dsc")
    write_bruker_esr_sample(source_dir / "beta_trace.dsc")
    recipe_path = project_root / "recipes" / "esr" / "default.yaml"

    with pytest.raises(SystemExit, match=r"fit-single requires exactly one discovered \.dsc file, found 2"):
        main(
            [
                "fit-single",
                "--input",
                str(source_dir),
                "--recipe",
                str(recipe_path),
            ]
        )


def test_fit_batch_accepts_folder_input(tmp_path, project_root, write_bruker_esr_sample) -> None:
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    write_bruker_esr_sample(source_dir / "alpha_trace.dsc")
    write_bruker_esr_sample(source_dir / "beta_trace.dsc")
    output_dir = tmp_path / "processed" / "batch_out"
    recipe_path = project_root / "recipes" / "esr" / "default.yaml"

    exit_code = main(
        [
            "fit-batch",
            "--input",
            str(source_dir),
            "--recipe",
            str(recipe_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "batch_summary.csv").exists()
    assert (output_dir / "batch_manifest.json").exists()
    assert (output_dir / "alpha_trace" / "alpha_trace_analysis.json").exists()
    assert (output_dir / "beta_trace" / "beta_trace_analysis.json").exists()


def test_fit_batch_filters_folder_input_with_pattern(tmp_path, project_root, write_bruker_esr_sample) -> None:
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    write_bruker_esr_sample(source_dir / "alpha_deg_trace.dsc")
    write_bruker_esr_sample(source_dir / "beta_trace.dsc")
    output_dir = tmp_path / "processed" / "batch_out"
    recipe_path = project_root / "recipes" / "esr" / "default.yaml"

    exit_code = main(
        [
            "fit-batch",
            "--input",
            str(source_dir),
            "--pattern",
            "*deg*",
            "--recipe",
            str(recipe_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    rows = list(csv.DictReader((output_dir / "batch_summary.csv").open("r", encoding="utf-8", newline="")))
    assert [row["source_stem"] for row in rows] == ["alpha_deg_trace"]


def test_fit_batch_recursively_discovers_nested_files(tmp_path, project_root, write_bruker_esr_sample) -> None:
    source_dir = tmp_path / "raw"
    nested_dir = source_dir / "nested"
    nested_dir.mkdir(parents=True)
    write_bruker_esr_sample(nested_dir / "nested_trace.dsc")
    output_dir = tmp_path / "processed" / "batch_out"
    recipe_path = project_root / "recipes" / "esr" / "default.yaml"

    exit_code = main(
        [
            "fit-batch",
            "--input",
            str(source_dir),
            "--recursive",
            "--recipe",
            str(recipe_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    payload = json.loads((output_dir / "batch_manifest.json").read_text(encoding="utf-8"))
    assert payload["scan"]["recursive"] is True
    assert [item["source_file"] for item in payload["items"]] == [str((nested_dir / "nested_trace.dsc").resolve())]
