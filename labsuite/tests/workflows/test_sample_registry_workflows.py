from __future__ import annotations

import csv
import json
from pathlib import Path

from labsuite.cli.main import main
from labsuite.core.sample_registry import SampleRecord, SampleRegistry, register_measurement, save_registry


def test_vsm_single_uses_registry_sample_and_writes_provenance(
    tmp_path: Path,
    project_root: Path,
    write_vsm_sample,
) -> None:
    source = write_vsm_sample(tmp_path / "raw", sample_stem="FilenameOnly-300K-R1_00001")
    registry_path = tmp_path / "metadata" / "sample_registry.yaml"
    registry = SampleRegistry(samples={"REG-SAMPLE": SampleRecord(sample_id="REG-SAMPLE", replicate="R9")})
    register_measurement(
        registry,
        sample_id="REG-SAMPLE",
        path=source,
        measurement_type="vsm",
        registry_base_dir=registry_path.parent,
    )
    save_registry(registry, registry_path)
    output_dir = tmp_path / "vsm_out"

    exit_code = main(
        [
            "vsm",
            "single",
            "--input",
            str(source),
            "--recipe",
            str(project_root / "recipes" / "vsm" / "default.yaml"),
            "--output-dir",
            str(output_dir),
            "--registry",
            str(registry_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads((output_dir / f"{source.stem}_analysis.json").read_text(encoding="utf-8"))
    assert payload["summary_metrics"]["sample_id"] == "REG-SAMPLE"
    assert payload["summary_metrics"]["replicate_id"] == "R9"
    assert (output_dir / "analysis_config.yaml").exists()
    assert (output_dir / "sample_registry_snapshot.yaml").exists()
    assert payload["provenance"]["sample_registry"]["sample_id"] == "REG-SAMPLE"


def test_fmr_registry_run_keeps_meff_and_warns_k_deferred(
    tmp_path: Path,
    project_root: Path,
    write_phasefmr_log,
) -> None:
    source = write_phasefmr_log(
        tmp_path / "FilenameSample-2to10GHz-R1.log",
        frequencies_GHz=[8.0, 9.0, 10.0, 11.0],
    )
    registry_path = tmp_path / "metadata" / "sample_registry.yaml"
    registry = SampleRegistry(samples={"REG-FMR": SampleRecord(sample_id="REG-FMR")})
    register_measurement(
        registry,
        sample_id="REG-FMR",
        path=source,
        measurement_type="fmr",
        geometry="ip",
        registry_base_dir=registry_path.parent,
    )
    save_registry(registry, registry_path)
    output_dir = tmp_path / "fmr_out"

    exit_code = main(
        [
            "fmr",
            "single",
            "--input",
            str(source),
            "--recipe",
            str(project_root / "recipes" / "fmr" / "default.yaml"),
            "--output-dir",
            str(output_dir),
            "--registry",
            str(registry_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads((output_dir / f"{source.stem}_analysis.json").read_text(encoding="utf-8"))
    assert payload["summary_metrics"]["sample_id"] == "REG-FMR"
    assert payload["summary_metrics"]["M_eff_mT"] is not None
    assert any("anisotropy_K_deferred" in warning for warning in payload["summary_metrics"]["warnings"])


def test_batch_writes_unresolved_csv_for_unregistered_files(
    tmp_path: Path,
    project_root: Path,
    write_vsm_sample,
) -> None:
    source_dir = tmp_path / "raw"
    first = write_vsm_sample(source_dir, sample_stem="Registered-300K-R1_00001")
    write_vsm_sample(source_dir, sample_stem="Unregistered-300K-R1_00001")
    registry_path = tmp_path / "metadata" / "sample_registry.yaml"
    registry = SampleRegistry(samples={"REG-VSM": SampleRecord(sample_id="REG-VSM")})
    register_measurement(
        registry,
        sample_id="REG-VSM",
        path=first,
        measurement_type="vsm",
        registry_base_dir=registry_path.parent,
    )
    save_registry(registry, registry_path)
    output_dir = tmp_path / "batch"

    exit_code = main(
        [
            "vsm",
            "batch",
            "--input",
            str(source_dir),
            "--recipe",
            str(project_root / "recipes" / "vsm" / "default.yaml"),
            "--output-dir",
            str(output_dir),
            "--registry",
            str(registry_path),
        ]
    )

    assert exit_code == 0
    unresolved_path = output_dir / "unresolved_files.csv"
    assert unresolved_path.exists()
    rows = list(csv.DictReader(unresolved_path.open("r", encoding="utf-8", newline="")))
    assert len(rows) == 1
    assert rows[0]["source_stem"] == "Unregistered-300K-R1_00001"
    manifest = json.loads((output_dir / "batch_manifest.json").read_text(encoding="utf-8"))
    assert manifest["unresolved_count"] == 1
