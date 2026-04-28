from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from labsuite.cli.main import main
from labsuite.core.sample_registry import (
    load_measurement_ledger,
    load_processed_ledger,
    load_registry,
)


def test_vsm_single_update_ledger_creates_sample_measurement_and_processed(
    tmp_path: Path,
    project_root: Path,
    write_vsm_sample,
) -> None:
    source = write_vsm_sample(tmp_path / "raw", sample_stem="Sample-300K-R1_00001")
    registry, measurement, processed = _metadata_paths(tmp_path)
    output_dir = tmp_path / "out"

    assert (
        main(
            [
                "vsm",
                "single",
                "--input",
                str(source),
                "--recipe",
                str(project_root / "recipes" / "vsm" / "default.yaml"),
                "--output-dir",
                str(output_dir),
                "--sample-registry",
                str(registry),
                "--measurement-ledger",
                str(measurement),
                "--processed-ledger",
                str(processed),
                "--raw-import-root",
                str(tmp_path / "project_raw"),
                "--update-ledger",
                "--create-sample",
                "--sample-id",
                "S1",
                "--measurement-id",
                "vsm:S1:loop",
                "--mark-canonical",
            ]
        )
        == 0
    )

    assert "S1" in load_registry(registry).samples
    assert load_measurement_ledger(measurement).measurements["vsm:S1:loop"].sample_id == "S1"
    results = load_processed_ledger(processed).processed_results
    assert len(results) == 1
    assert next(iter(results.values())).status == "canonical"


def test_update_ledger_fails_when_sample_missing_without_create(
    tmp_path: Path,
    project_root: Path,
    write_vsm_sample,
) -> None:
    source = write_vsm_sample(tmp_path / "raw", sample_stem="Sample-300K-R1_00001")
    registry, measurement, processed = _metadata_paths(tmp_path)

    with pytest.raises(SystemExit, match="pass --create-sample"):
        main(
            [
                "vsm",
                "single",
                "--input",
                str(source),
                "--recipe",
                str(project_root / "recipes" / "vsm" / "default.yaml"),
                "--output-dir",
                str(tmp_path / "out"),
                "--sample-registry",
                str(registry),
                "--measurement-ledger",
                str(measurement),
                "--processed-ledger",
                str(processed),
                "--raw-import-root",
                str(tmp_path / "project_raw"),
                "--update-ledger",
                "--sample-id",
                "S1",
                "--measurement-id",
                "vsm:S1:loop",
            ]
        )


def test_mark_canonical_conflict_and_replace(
    tmp_path: Path,
    project_root: Path,
    write_vsm_sample,
) -> None:
    source = write_vsm_sample(tmp_path / "raw", sample_stem="Sample-300K-R1_00001")
    registry, measurement, processed = _metadata_paths(tmp_path)
    base_args = [
        "vsm",
        "single",
        "--input",
        str(source),
        "--recipe",
        str(project_root / "recipes" / "vsm" / "default.yaml"),
        "--sample-registry",
        str(registry),
        "--measurement-ledger",
        str(measurement),
        "--processed-ledger",
        str(processed),
        "--raw-import-root",
        str(tmp_path / "project_raw"),
        "--update-ledger",
        "--create-sample",
        "--sample-id",
        "S1",
        "--measurement-id",
        "vsm:S1:loop",
        "--mark-canonical",
    ]
    assert main([*base_args, "--output-dir", str(tmp_path / "out1")]) == 0

    with pytest.raises(SystemExit, match="--replace-canonical"):
        main([*base_args, "--output-dir", str(tmp_path / "out2")])

    assert main([*base_args, "--replace-canonical", "--output-dir", str(tmp_path / "out3")]) == 0
    statuses = [
        result.status for result in load_processed_ledger(processed).processed_results.values()
    ]
    assert statuses.count("canonical") == 1
    assert "superseded" in statuses


def test_batch_metadata_manifest_updates_ledgers_for_multiple_samples(
    tmp_path: Path,
    project_root: Path,
    write_vsm_sample,
) -> None:
    raw_dir = tmp_path / "raw"
    first = write_vsm_sample(raw_dir, sample_stem="A-300K-R1_00001")
    second = write_vsm_sample(raw_dir, sample_stem="B-300K-R1_00001")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "raw_path",
                "sample_id",
                "measurement_id",
                "type",
                "geometry",
                "branch_labels",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "raw_path": str(first),
                "sample_id": "S1",
                "measurement_id": "vsm:S1",
                "type": "vsm",
                "geometry": "unknown",
                "branch_labels": "",
            }
        )
        writer.writerow(
            {
                "raw_path": str(second),
                "sample_id": "S2",
                "measurement_id": "vsm:S2",
                "type": "vsm",
                "geometry": "unknown",
                "branch_labels": "",
            }
        )
    registry, measurement, processed = _metadata_paths(tmp_path)

    assert (
        main(
            [
                "vsm",
                "batch",
                "--input",
                str(raw_dir),
                "--recipe",
                str(project_root / "recipes" / "vsm" / "default.yaml"),
                "--output-dir",
                str(tmp_path / "batch"),
                "--sample-registry",
                str(registry),
                "--measurement-ledger",
                str(measurement),
                "--processed-ledger",
                str(processed),
                "--metadata-manifest",
                str(manifest),
                "--raw-import-root",
                str(tmp_path / "project_raw"),
                "--update-ledger",
                "--create-sample",
                "--mark-canonical",
            ]
        )
        == 0
    )

    assert set(load_registry(registry).samples) == {"S1", "S2"}
    assert set(load_measurement_ledger(measurement).measurements) == {"vsm:S1", "vsm:S2"}


def test_batch_sample_id_without_manifest_warns(
    tmp_path: Path,
    project_root: Path,
    write_vsm_sample,
    capsys,
) -> None:
    raw_dir = tmp_path / "raw"
    write_vsm_sample(raw_dir, sample_stem="A-300K-R1_00001")
    write_vsm_sample(raw_dir, sample_stem="B-300K-R1_00001")
    registry, measurement, processed = _metadata_paths(tmp_path)

    assert (
        main(
            [
                "vsm",
                "batch",
                "--input",
                str(raw_dir),
                "--recipe",
                str(project_root / "recipes" / "vsm" / "default.yaml"),
                "--output-dir",
                str(tmp_path / "batch"),
                "--sample-registry",
                str(registry),
                "--measurement-ledger",
                str(measurement),
                "--processed-ledger",
                str(processed),
                "--raw-import-root",
                str(tmp_path / "project_raw"),
                "--update-ledger",
                "--create-sample",
                "--sample-id",
                "SAME",
            ]
        )
        == 0
    )

    assert "all files are assigned to the same sample" in capsys.readouterr().out
    ledger = load_measurement_ledger(measurement)
    assert len(ledger.measurements) == 2
    assert {record.sample_id for record in ledger.measurements.values()} == {"SAME"}


def test_single_esr_update_ledger_imports_external_raw_and_sidecar(
    tmp_path: Path,
    project_root: Path,
    write_bruker_esr_sample,
) -> None:
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    source = write_bruker_esr_sample(external_dir / "SampleA.dsc")
    registry, measurement, processed = _metadata_paths(tmp_path)
    raw_root = tmp_path / "project_raw"
    output_dir = tmp_path / "out"

    assert (
        main(
            [
                "esr",
                "single",
                "--input",
                str(source),
                "--recipe",
                str(project_root / "recipes" / "esr" / "default.yaml"),
                "--output-dir",
                str(output_dir),
                "--sample-registry",
                str(registry),
                "--measurement-ledger",
                str(measurement),
                "--processed-ledger",
                str(processed),
                "--raw-import-root",
                str(raw_root),
                "--update-ledger",
                "--create-sample",
                "--sample-id",
                "ESR-S1",
                "--measurement-id",
                "esr:ESR-S1:SampleA",
                "--geometry",
                "angular",
                "--mark-canonical",
            ]
        )
        == 0
    )

    imported = raw_root / "ESR" / "SampleA.dsc"
    assert imported.exists()
    assert imported.with_suffix(".DTA").exists()
    record = load_measurement_ledger(measurement).measurements["esr:ESR-S1:SampleA"]
    assert Path(record.raw_path).name == "SampleA.dsc"
    config = yaml.safe_load((output_dir / "analysis_config.yaml").read_text(encoding="utf-8"))
    assert Path(config["source_path"]) == imported.resolve()
    assert Path(config["original_source_path"]) == source.resolve()


def test_single_vsm_import_collision_uses_unique_suffix(
    tmp_path: Path,
    project_root: Path,
    write_vsm_sample,
) -> None:
    source = write_vsm_sample(tmp_path / "external", sample_stem="Collision-300K-R1_00001")
    raw_root = tmp_path / "project_raw"
    existing = raw_root / "VSM" / source.name
    existing.parent.mkdir(parents=True)
    existing.write_text("existing", encoding="utf-8")
    registry, measurement, processed = _metadata_paths(tmp_path)

    assert (
        main(
            [
                "vsm",
                "single",
                "--input",
                str(source),
                "--recipe",
                str(project_root / "recipes" / "vsm" / "default.yaml"),
                "--output-dir",
                str(tmp_path / "out"),
                "--sample-registry",
                str(registry),
                "--measurement-ledger",
                str(measurement),
                "--processed-ledger",
                str(processed),
                "--raw-import-root",
                str(raw_root),
                "--update-ledger",
                "--create-sample",
                "--sample-id",
                "S1",
                "--measurement-id",
                "vsm:S1:collision",
            ]
        )
        == 0
    )

    record = load_measurement_ledger(measurement).measurements["vsm:S1:collision"]
    assert Path(record.raw_path).name == "Collision-300K-R1_00001__2.dat"
    assert existing.read_text(encoding="utf-8") == "existing"


def test_single_in_raw_file_is_not_copied(
    tmp_path: Path,
    project_root: Path,
    write_vsm_sample,
) -> None:
    raw_root = tmp_path / "project_raw"
    source = write_vsm_sample(raw_root / "VSM", sample_stem="AlreadyRaw-300K-R1_00001")
    registry, measurement, processed = _metadata_paths(tmp_path)

    assert (
        main(
            [
                "vsm",
                "single",
                "--input",
                str(source),
                "--recipe",
                str(project_root / "recipes" / "vsm" / "default.yaml"),
                "--output-dir",
                str(tmp_path / "out"),
                "--sample-registry",
                str(registry),
                "--measurement-ledger",
                str(measurement),
                "--processed-ledger",
                str(processed),
                "--raw-import-root",
                str(raw_root),
                "--update-ledger",
                "--create-sample",
                "--sample-id",
                "S1",
                "--measurement-id",
                "vsm:S1:already",
            ]
        )
        == 0
    )

    record = load_measurement_ledger(measurement).measurements["vsm:S1:already"]
    assert Path(record.raw_path).resolve() == source.resolve()
    assert not (raw_root / "VSM" / "AlreadyRaw-300K-R1_00001__2.dat").exists()


def test_interactive_default_measurement_id_uses_imported_stem(
    tmp_path: Path,
    project_root: Path,
    write_vsm_sample,
    monkeypatch,
) -> None:
    source = write_vsm_sample(tmp_path / "external", sample_stem="PromptStem-300K-R1_00001")
    registry, measurement, processed = _metadata_paths(tmp_path)
    answers = iter(["S1", "", "", "y", "n", "n"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    assert (
        main(
            [
                "vsm",
                "single",
                "--input",
                str(source),
                "--recipe",
                str(project_root / "recipes" / "vsm" / "default.yaml"),
                "--output-dir",
                str(tmp_path / "out"),
                "--sample-registry",
                str(registry),
                "--measurement-ledger",
                str(measurement),
                "--processed-ledger",
                str(processed),
                "--raw-import-root",
                str(tmp_path / "project_raw"),
                "--update-ledger",
                "--interactive",
            ]
        )
        == 0
    )

    assert "vsm:S1:PromptStem-300K-R1_00001" in load_measurement_ledger(measurement).measurements


def test_batch_manifest_uses_original_paths_after_import_and_writes_map(
    tmp_path: Path,
    project_root: Path,
    write_vsm_sample,
) -> None:
    raw_dir = tmp_path / "external"
    first = write_vsm_sample(raw_dir, sample_stem="A-300K-R1_00001")
    second = write_vsm_sample(raw_dir, sample_stem="B-300K-R1_00001")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "raw_path",
                "sample_id",
                "measurement_id",
                "type",
                "geometry",
                "branch_labels",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "raw_path": str(first),
                "sample_id": "S1",
                "measurement_id": "vsm:S1:imported",
                "type": "vsm",
                "geometry": "unknown",
                "branch_labels": "",
            }
        )
        writer.writerow(
            {
                "raw_path": str(second),
                "sample_id": "S2",
                "measurement_id": "vsm:S2:imported",
                "type": "vsm",
                "geometry": "unknown",
                "branch_labels": "",
            }
        )
    registry, measurement, processed = _metadata_paths(tmp_path)
    batch_dir = tmp_path / "batch"

    assert (
        main(
            [
                "vsm",
                "batch",
                "--input",
                str(raw_dir),
                "--recipe",
                str(project_root / "recipes" / "vsm" / "default.yaml"),
                "--output-dir",
                str(batch_dir),
                "--sample-registry",
                str(registry),
                "--measurement-ledger",
                str(measurement),
                "--processed-ledger",
                str(processed),
                "--metadata-manifest",
                str(manifest),
                "--raw-import-root",
                str(tmp_path / "project_raw"),
                "--update-ledger",
                "--create-sample",
                "--mark-canonical",
            ]
        )
        == 0
    )

    ledger = load_measurement_ledger(measurement)
    assert set(ledger.measurements) == {"vsm:S1:imported", "vsm:S2:imported"}
    assert all("project_raw" in record.raw_path for record in ledger.measurements.values())
    assert (batch_dir / "raw_import_map.csv").exists()


def _metadata_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    return (
        tmp_path / "metadata" / "sample_registry.yaml",
        tmp_path / "metadata" / "measurement_ledger.yaml",
        tmp_path / "metadata" / "processed_ledger.yaml",
    )
