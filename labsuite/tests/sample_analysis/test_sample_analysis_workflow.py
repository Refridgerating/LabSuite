from __future__ import annotations

import json
from pathlib import Path

from labsuite.core.sample_registry import (
    MeasurementLedger,
    MeasurementRecord,
    ProcessedLedger,
    ProcessedResultRecord,
    SampleRecord,
    SampleRegistry,
    save_measurement_ledger,
    save_processed_ledger,
    save_registry,
)
from labsuite.sample_analysis.manifest import build_sample_manifest
from labsuite.sample_analysis.service import analyze_sample, build_sample_readiness


def test_sample_analysis_loads_only_canonical_processed_results(
    tmp_path: Path, project_root: Path
) -> None:
    registry_path, measurement_path, processed_path = _metadata_paths(tmp_path)
    raw = tmp_path / "raw_vsm.dat"
    raw.write_text("raw", encoding="utf-8")
    canonical_json = tmp_path / "processed" / "canonical_analysis.json"
    test_json = tmp_path / "processed" / "test_analysis.json"
    _write_vsm_processed(canonical_json, raw, sample_id="S1", measurement_id="vsm:S1")
    _write_vsm_processed(test_json, raw, sample_id="S1", measurement_id="vsm:S1")
    _write_metadata(
        registry_path,
        measurement_path,
        processed_path,
        sample=SampleRecord(sample_id="S1"),
        measurements=[
            MeasurementRecord("vsm:S1", "S1", "vsm", str(raw), "unknown"),
        ],
        results=[
            ProcessedResultRecord(
                "canonical",
                "vsm:S1",
                "S1",
                "vsm",
                str(canonical_json),
                "recipe.yaml",
                status="canonical",
            ),
            ProcessedResultRecord(
                "test", "vsm:S1", "S1", "vsm", str(test_json), "recipe.yaml", status="test"
            ),
        ],
    )

    manifest = build_sample_manifest(
        sample_id="S1",
        registry_path=registry_path,
        measurement_ledger_path=measurement_path,
        processed_ledger_path=processed_path,
    )

    assert len(manifest.processed_inputs) == 1
    assert manifest.processed_inputs[0].processed_json_path == canonical_json


def test_missing_canonical_warns_without_crash(tmp_path: Path, project_root: Path) -> None:
    registry_path, measurement_path, processed_path = _metadata_paths(tmp_path)
    raw = tmp_path / "raw_fmr.log"
    raw.write_text("raw", encoding="utf-8")
    _write_metadata(
        registry_path,
        measurement_path,
        processed_path,
        sample=SampleRecord(sample_id="S1"),
        measurements=[MeasurementRecord("fmr:S1", "S1", "fmr", str(raw), "ip")],
        results=[],
    )

    result = build_sample_readiness(
        sample_id="S1",
        registry_path=registry_path,
        measurement_ledger_path=measurement_path,
        processed_ledger_path=processed_path,
        recipe_path=project_root / "recipes" / "sample_analysis" / "default.yaml",
    )

    assert result["summary"]["readiness"] == "INSUFFICIENT_DATA"
    assert any(item["code"] == "MISSING_CANONICAL_PROCESSED_RESULT" for item in result["warnings"])


def test_sample_analysis_does_not_read_raw_or_scan_processed_root(
    tmp_path: Path, project_root: Path, monkeypatch
) -> None:
    registry_path, measurement_path, processed_path = _metadata_paths(tmp_path)
    raw = tmp_path / "raw_vsm.dat"
    raw.write_text("raw", encoding="utf-8")
    processed_json = tmp_path / "processed" / "raw_vsm_analysis.json"
    _write_vsm_processed(processed_json, raw, sample_id="S1", measurement_id="vsm:S1")
    _write_metadata(
        registry_path,
        measurement_path,
        processed_path,
        sample=SampleRecord(sample_id="S1"),
        measurements=[MeasurementRecord("vsm:S1", "S1", "vsm", str(raw), "unknown")],
        results=[
            ProcessedResultRecord(
                "r1", "vsm:S1", "S1", "vsm", str(processed_json), "recipe.yaml", status="canonical"
            )
        ],
    )
    original_read_text = Path.read_text

    def guarded_read_text(self: Path, *args, **kwargs):
        if self == raw:
            raise AssertionError("sample analysis must not read raw files")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(
        Path,
        "rglob",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("must not scan processed root")
        ),
    )

    result = build_sample_readiness(
        sample_id="S1",
        registry_path=registry_path,
        measurement_ledger_path=measurement_path,
        processed_ledger_path=processed_path,
        recipe_path=project_root / "recipes" / "sample_analysis" / "default.yaml",
    )

    assert result["summary"]["usable_processed_inputs"] == 1


def test_anisotropy_missing_ms_warns(tmp_path: Path, project_root: Path) -> None:
    registry_path, measurement_path, processed_path = _metadata_paths(tmp_path)
    raw = tmp_path / "raw_fmr.log"
    raw.write_text("raw", encoding="utf-8")
    processed_json = tmp_path / "processed" / "raw_fmr_analysis.json"
    _write_fmr_processed(
        processed_json, raw, sample_id="S1", measurement_id="fmr:S1", geometry="ip", meff_mT=800.0
    )
    _write_metadata(
        registry_path,
        measurement_path,
        processed_path,
        sample=SampleRecord(sample_id="S1"),
        measurements=[MeasurementRecord("fmr:S1", "S1", "fmr", str(raw), "ip")],
        results=[
            ProcessedResultRecord(
                "fmr",
                "fmr:S1",
                "S1",
                "fmr",
                str(processed_json),
                "recipe.yaml",
                status="canonical",
                summary={"geometry": "ip"},
            )
        ],
    )

    result = build_sample_readiness(
        sample_id="S1",
        registry_path=registry_path,
        measurement_ledger_path=measurement_path,
        processed_ledger_path=processed_path,
        recipe_path=project_root / "recipes" / "sample_analysis" / "default.yaml",
    )

    assert any(item["code"] == "MISSING_MS" for item in result["warnings"])


def test_k4_possible_and_ip_damping_warning(tmp_path: Path, project_root: Path) -> None:
    registry_path, measurement_path, processed_path = _metadata_paths(tmp_path)
    raw_ip = tmp_path / "ip.log"
    raw_oop = tmp_path / "oop.log"
    raw_ip.write_text("raw", encoding="utf-8")
    raw_oop.write_text("raw", encoding="utf-8")
    ip_json = tmp_path / "processed" / "ip_analysis.json"
    oop_json = tmp_path / "processed" / "oop_analysis.json"
    _write_fmr_processed(
        ip_json,
        raw_ip,
        sample_id="S1",
        measurement_id="fmr:ip",
        geometry="ip",
        meff_mT=500.0,
        alpha=0.01,
    )
    _write_fmr_processed(
        oop_json,
        raw_oop,
        sample_id="S1",
        measurement_id="fmr:oop",
        geometry="oop",
        meff_mT=900.0,
        alpha=None,
    )
    sample = SampleRecord(
        sample_id="S1",
        magnetic_volume_m3=1e-18,
        magnetic_volume_source="manual",
        magnetic_volume_method="test",
    )
    vsm_raw = tmp_path / "vsm.dat"
    vsm_raw.write_text("raw", encoding="utf-8")
    vsm_json = tmp_path / "processed" / "vsm_analysis.json"
    _write_vsm_processed(vsm_json, vsm_raw, sample_id="S1", measurement_id="vsm:S1", ms_emu=1e-6)
    _write_metadata(
        registry_path,
        measurement_path,
        processed_path,
        sample=sample,
        measurements=[
            MeasurementRecord("vsm:S1", "S1", "vsm", str(vsm_raw), "unknown"),
            MeasurementRecord("fmr:ip", "S1", "fmr", str(raw_ip), "ip"),
            MeasurementRecord("fmr:oop", "S1", "fmr", str(raw_oop), "oop"),
        ],
        results=[
            ProcessedResultRecord(
                "vsm", "vsm:S1", "S1", "vsm", str(vsm_json), "recipe.yaml", status="canonical"
            ),
            ProcessedResultRecord(
                "ip",
                "fmr:ip",
                "S1",
                "fmr",
                str(ip_json),
                "recipe.yaml",
                status="canonical",
                summary={"geometry": "ip"},
            ),
            ProcessedResultRecord(
                "oop",
                "fmr:oop",
                "S1",
                "fmr",
                str(oop_json),
                "recipe.yaml",
                status="canonical",
                summary={"geometry": "oop"},
            ),
        ],
    )

    result = build_sample_readiness(
        sample_id="S1",
        registry_path=registry_path,
        measurement_ledger_path=measurement_path,
        processed_ledger_path=processed_path,
        recipe_path=project_root / "recipes" / "sample_analysis" / "default.yaml",
    )

    codes = {item["code"] for item in result["warnings"]}
    assert "K4_POSSIBLE" in codes
    assert "IP_DAMPING_MAY_INCLUDE_TWO_MAGNON" in codes
    assert result["summary"]["readiness"] == "READY_DAMPING"
    assert result["summary"]["Ms_A_per_m"] == 1e9
    assert result["summary"]["primary_alpha_eff"] == 0.01
    assert result["readiness_matrix"]["READY_ANISOTROPY"] is True


def test_esr_g_effective_warning_and_outputs(tmp_path: Path, project_root: Path) -> None:
    registry_path, measurement_path, processed_path = _metadata_paths(tmp_path)
    raw = tmp_path / "raw.dsc"
    raw.write_text("raw", encoding="utf-8")
    processed_json = tmp_path / "processed" / "raw_esr_analysis.json"
    _write_esr_processed(processed_json, raw, sample_id="S1", measurement_id="esr:S1")
    _write_metadata(
        registry_path,
        measurement_path,
        processed_path,
        sample=SampleRecord(sample_id="S1"),
        measurements=[MeasurementRecord("esr:S1", "S1", "esr", str(raw), "angular")],
        results=[
            ProcessedResultRecord(
                "esr",
                "esr:S1",
                "S1",
                "esr",
                str(processed_json),
                "recipe.yaml",
                status="canonical",
                summary={"geometry": "angular"},
            )
        ],
    )

    run = analyze_sample(
        sample_id="S1",
        registry_path=registry_path,
        measurement_ledger_path=measurement_path,
        processed_ledger_path=processed_path,
        recipe_path=project_root / "recipes" / "sample_analysis" / "default.yaml",
        output_dir=tmp_path / "derived" / "S1",
    )

    assert any(item["code"] == "ESR_G_EFFECTIVE_ONLY" for item in run.result["warnings"])
    assert (run.output_dir / "provenance" / "processed_ledger_snapshot.yaml").exists()


def _metadata_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    return (
        tmp_path / "metadata" / "sample_registry.yaml",
        tmp_path / "metadata" / "measurement_ledger.yaml",
        tmp_path / "metadata" / "processed_ledger.yaml",
    )


def _write_metadata(
    registry_path: Path,
    measurement_path: Path,
    processed_path: Path,
    *,
    sample: SampleRecord,
    measurements: list[MeasurementRecord],
    results: list[ProcessedResultRecord],
) -> None:
    save_registry(SampleRegistry(samples={sample.sample_id: sample}), registry_path)
    save_measurement_ledger(
        MeasurementLedger(measurements={item.measurement_id: item for item in measurements}),
        measurement_path,
    )
    save_processed_ledger(
        ProcessedLedger(processed_results={item.result_id: item for item in results}),
        processed_path,
    )


def _write_vsm_processed(
    path: Path, raw: Path, *, sample_id: str, measurement_id: str, ms_emu: float = 1e-6
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "measurement": {"modality": "vsm", "source_path": str(raw)},
                "summary_metrics": {
                    "sample_id": sample_id,
                    "registry_measurement_id": measurement_id,
                    "Ms_emu": ms_emu,
                    "ms_error": ms_emu * 0.05,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_fmr_processed(
    path: Path,
    raw: Path,
    *,
    sample_id: str,
    measurement_id: str,
    geometry: str,
    meff_mT: float,
    alpha: float | None = 0.01,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    linewidth_fit = (
        None
        if alpha is None
        else {
            "success": True,
            "parameters": {"DeltaH0_mT": 2.0, "slope_mT_per_GHz": 0.2},
            "metrics": {"r_squared": 0.99, "rmse_mT": 0.1},
        }
    )
    derived_parameters = {
        "g": 2.1,
        "M_eff_mT": meff_mT,
        "M_eff_T": meff_mT / 1000.0,
        "gamma_GHz_per_T": 29.0,
        "alpha": alpha,
        "DeltaH0_mT": None if linewidth_fit is None else 2.0,
    }
    path.write_text(
        json.dumps(
            {
                "measurement": {"modality": "fmr", "source_path": str(raw)},
                "summary_metrics": {
                    "sample_id": sample_id,
                    "registry_measurement_id": measurement_id,
                    "g_mode": "float",
                    "field_polarity_pair_count": 0,
                },
                "analysis_payload": {
                    "series_collection_result": {
                        "series_by_label": {
                            "main": {
                                "frequency_GHz": [6, 8, 10, 12, 14],
                                "resonance_field_mT": [100, 150, 200, 250, 300],
                                "linewidth_mT": [3, 4, 5, 6, 7],
                            }
                        }
                    },
                    "physics_collection_result": {
                        "physics_by_label": {
                            "main": {
                                "kittel_fit": {
                                    "success": True,
                                    "parameters": {
                                        "gamma_GHz_per_T": 29.0,
                                        "M_eff_T": meff_mT / 1000.0,
                                    },
                                    "metrics": {"r_squared": 0.99},
                                },
                                "linewidth_fit": linewidth_fit,
                                "derived_parameters": derived_parameters,
                            }
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def _write_esr_processed(path: Path, raw: Path, *, sample_id: str, measurement_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "measurement": {"modality": "esr", "source_path": str(raw)},
                "metadata": {
                    "frequency_GHz": 9.5,
                    "sample_id": sample_id,
                    "measurement_id": measurement_id,
                },
                "fit_selection": {
                    "single_fit": {
                        "success": True,
                        "parameters": {"center_mT": 340.0, "gamma_mT": 1.2},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
