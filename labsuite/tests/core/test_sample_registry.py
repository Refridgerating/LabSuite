from __future__ import annotations

from pathlib import Path

from labsuite.core.sample_registry import (
    AnalysisDefaults,
    MeasurementRecord,
    QuantityMetadata,
    SampleRecord,
    SampleRegistry,
    VolumeMetadata,
    find_measurement_by_path,
    find_sample,
    load_registry,
    register_measurement,
    save_registry,
    validate_registry,
)


def test_sample_registry_round_trip_and_lookup(tmp_path: Path) -> None:
    source = tmp_path / "sample.log"
    source.write_text("raw", encoding="utf-8")
    registry_path = tmp_path / "metadata" / "sample_registry.yaml"
    registry = SampleRegistry(
        samples={
            "S1": SampleRecord(
                sample_id="S1",
                aliases=["alias-one"],
                geometry=VolumeMetadata(
                    area=QuantityMetadata(value=2.0, unit="mm^2", uncertainty=0.1),
                    magnetic_thickness=QuantityMetadata(value=1.5, unit="nm", uncertainty=0.05),
                ),
                defaults=AnalysisDefaults(g_mode="bounded", g_value=2.1, ms_source="vsm:S1"),
                measurements=[
                    MeasurementRecord(
                        measurement_id="fmr:sample",
                        sample_id="S1",
                        type="fmr",
                        path=str(source),
                        geometry="ip",
                    )
                ],
            )
        }
    )

    save_registry(registry, registry_path)
    loaded = load_registry(registry_path)

    assert find_sample(loaded, "S1").sample_id == "S1"
    assert find_sample(loaded, "alias-one").sample_id == "S1"
    matched = find_measurement_by_path(loaded, source, registry_base_dir=registry_path.parent)
    assert matched is not None
    assert matched[1].measurement_id == "fmr:sample"


def test_register_measurement_and_validation_warnings(tmp_path: Path) -> None:
    source = tmp_path / "trace.dsc"
    source.write_text("raw", encoding="utf-8")
    registry = SampleRegistry(
        samples={
            "S1": SampleRecord(sample_id="S1", aliases=["dup"]),
            "S2": SampleRecord(sample_id="S2", aliases=["dup"]),
        }
    )

    record = register_measurement(
        registry,
        sample_id="S1",
        path=source,
        measurement_type="esr",
        registry_base_dir=tmp_path,
    )

    assert record.sample_id == "S1"
    messages = validate_registry(registry, registry_base_dir=tmp_path)
    codes = {message.code for message in messages}
    assert "duplicate_alias" in codes
    assert "missing_measurement_geometry" in codes
    assert "volume_metadata_incomplete" in codes
