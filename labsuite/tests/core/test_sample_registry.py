from __future__ import annotations

from pathlib import Path

import pytest

from labsuite.core.magnetic_volume import MagneticVolumeError
from labsuite.core.sample_registry import (
    CanonicalResultExistsError,
    DirectVolumeMetadata,
    MeasurementLedger,
    MetadataSchemaError,
    ProcessedLedger,
    ProcessedResultRecord,
    QuantityMetadata,
    SampleRecord,
    SampleRegistry,
    VolumeMetadata,
    add_processed_result_record,
    load_measurement_ledger,
    load_processed_ledger,
    load_registry,
    resolve_sample_magnetic_volume,
    save_measurement_ledger,
    save_processed_ledger,
    save_registry,
    upsert_measurement_record,
    validate_registry,
)


def test_sample_registry_v2_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "metadata" / "sample_registry.yaml"
    registry = SampleRegistry(
        samples={
            "S1": SampleRecord(
                sample_id="S1",
                aliases=["alias-one"],
                geometry=VolumeMetadata(area=QuantityMetadata(value=2.0, unit="mm^2")),
            )
        }
    )

    save_registry(registry, path)
    loaded = load_registry(path)

    assert loaded.schema_version == 2
    assert loaded.samples["S1"].aliases == ["alias-one"]


def test_sample_registry_preserves_unknown_sample_fields(tmp_path: Path) -> None:
    path = tmp_path / "sample_registry.yaml"
    path.write_text(
        """
schema_version: 2
samples:
  S1:
    sample_id: S1
    custom_field:
      keep: true
    geometry:
      area:
        value: null
        unit: null
        uncertainty: null
      magnetic_thickness:
        value: null
        unit: null
        uncertainty: null
      vmag:
        value: null
        unit: null
        uncertainty: null
        method: null
      custom_geometry_field: abc
    defaults:
      g_mode: float
      g_value: null
      ms_source: null
""",
        encoding="utf-8",
    )

    registry = load_registry(path)
    save_registry(registry, path)
    loaded = load_registry(path)

    assert loaded.samples["S1"].extra["custom_field"] == {"keep": True}
    assert loaded.samples["S1"].geometry.extra["custom_geometry_field"] == "abc"


def test_measurement_and_processed_ledgers_round_trip(tmp_path: Path) -> None:
    raw = tmp_path / "raw.dat"
    raw.write_text("raw", encoding="utf-8")
    measurement_path = tmp_path / "metadata" / "measurement_ledger.yaml"
    processed_path = tmp_path / "metadata" / "processed_ledger.yaml"
    measurement_ledger = MeasurementLedger()
    upsert_measurement_record(
        measurement_ledger,
        measurement_id="vsm:S1:raw",
        sample_id="S1",
        measurement_type="vsm",
        raw_path=raw,
        base_dir=measurement_path.parent,
    )
    save_measurement_ledger(measurement_ledger, measurement_path)
    processed_ledger = ProcessedLedger(
        processed_results={
            "result-1": ProcessedResultRecord(
                result_id="result-1",
                measurement_id="vsm:S1:raw",
                sample_id="S1",
                type="vsm",
                processed_path="processed/raw_analysis.json",
                recipe_path="recipes/vsm/default.yaml",
                status="canonical",
            )
        }
    )
    save_processed_ledger(processed_ledger, processed_path)

    assert load_measurement_ledger(measurement_path).measurements["vsm:S1:raw"].sample_id == "S1"
    assert load_processed_ledger(processed_path).processed_results["result-1"].status == "canonical"


def test_missing_magnetic_volume_resolves_unavailable() -> None:
    resolution = resolve_sample_magnetic_volume(SampleRecord(sample_id="S1"))

    assert not resolution.is_available
    assert "Magnetic volume is unavailable" in resolution.warnings[0]


def test_validate_registry_accepts_canonical_magnetic_volume() -> None:
    registry = SampleRegistry(
        samples={
            "S1": SampleRecord(
                sample_id="S1",
                magnetic_volume_m3=1.23e-13,
                magnetic_volume_source="estimated",
            )
        }
    )

    codes = {message.code for message in validate_registry(registry)}

    assert "volume_metadata_incomplete" not in codes


def test_validate_registry_accepts_resolvable_layer_stack_volume() -> None:
    registry = SampleRegistry(
        samples={
            "S1": SampleRecord(
                sample_id="S1",
                geometry=VolumeMetadata(
                    shape="square",
                    dimensions={"side": 4.0, "side_unit": "mm"},
                ),
                layer_stack=[
                    {"material": "Ta", "thickness": 2.0, "thickness_unit": "nm", "magnetic": False},
                    {"material": "Co", "thickness": 5.0, "thickness_unit": "nm", "magnetic": True},
                ],
            )
        }
    )

    codes = {message.code for message in validate_registry(registry)}

    assert "volume_metadata_incomplete" not in codes


def test_validate_registry_warns_when_volume_unavailable() -> None:
    registry = SampleRegistry(samples={"S1": SampleRecord(sample_id="S1")})

    codes = {message.code for message in validate_registry(registry)}

    assert "volume_metadata_incomplete" in codes


def test_manual_magnetic_volume_resolves_and_overrides_estimate() -> None:
    sample = SampleRecord(
        sample_id="S1",
        geometry=VolumeMetadata(
            shape="rectangle",
            dimensions={"length": 5.0, "width": 5.0, "unit": "mm"},
        ),
        layer_stack=[
            {"material": "Co", "thickness": 5.0, "thickness_unit": "nm", "magnetic": True}
        ],
        magnetic_volume_m3=1.23e-13,
    )

    resolution = resolve_sample_magnetic_volume(sample)

    assert resolution.is_available
    assert resolution.source == "manual"
    assert resolution.magnetic_volume_m3 == pytest.approx(1.23e-13)
    assert any("overrides" in warning for warning in resolution.warnings)


def test_invalid_manual_magnetic_volume_warns_or_raises() -> None:
    sample = SampleRecord(sample_id="S1", magnetic_volume_m3=0.0)

    assert not resolve_sample_magnetic_volume(sample).is_available
    with pytest.raises(MagneticVolumeError, match="greater than zero"):
        resolve_sample_magnetic_volume(sample, strict=True)


def test_registry_resolves_rectangle_layer_stack_volume() -> None:
    sample = SampleRecord(
        sample_id="S1",
        geometry=VolumeMetadata(
            shape="rectangle",
            dimensions={"length": 5.0, "width": 5.0, "unit": "mm"},
        ),
        layer_stack=[
            {"material": "Ta", "thickness": 2.0, "thickness_unit": "nm", "magnetic": False},
            {"material": "Co", "thickness": 5.0, "thickness_unit": "nm", "magnetic": True},
            {"material": "NiFe", "thickness": 7.0, "thickness_unit": "nm", "magnetic": True},
        ],
    )

    resolution = resolve_sample_magnetic_volume(sample)

    assert resolution.is_available
    assert resolution.is_estimated
    assert resolution.magnetic_volume_m3 == pytest.approx(25e-6 * 12e-9)
    assert [layer["material"] for layer in resolution.included_layers] == ["Co", "NiFe"]
    assert [layer["material"] for layer in resolution.excluded_layers] == ["Ta"]


def test_registry_resolves_circle_and_custom_area_volumes() -> None:
    circle = SampleRecord(
        sample_id="circle",
        geometry=VolumeMetadata(shape="circle", dimensions={"radius": 1.0, "unit": "mm"}),
        layer_stack=[
            {"material": "NiFe", "thickness": 10.0, "thickness_unit": "angstrom", "magnetic": True}
        ],
    )
    custom = SampleRecord(
        sample_id="custom",
        geometry=VolumeMetadata(
            shape="custom_area",
            dimensions={"area": 4.0, "area_unit": "mm^2"},
        ),
        layer_stack=[
            {"material": "NiFe", "thickness": 2.0, "thickness_unit": "nm", "magnetic": True}
        ],
    )

    assert resolve_sample_magnetic_volume(circle).magnetic_volume_m3 == pytest.approx(
        3.141592653589793e-6 * 1e-9
    )
    assert resolve_sample_magnetic_volume(custom).magnetic_volume_m3 == pytest.approx(8e-15)


def test_registry_resolver_no_magnetic_layers_is_unavailable() -> None:
    sample = SampleRecord(
        sample_id="S1",
        geometry=VolumeMetadata(shape="square", dimensions={"side": 1.0, "unit": "mm"}),
        layer_stack=[
            {"material": "Ta", "thickness": 2.0, "thickness_unit": "nm", "magnetic": False}
        ],
    )

    resolution = resolve_sample_magnetic_volume(sample)

    assert not resolution.is_available
    assert "No magnetic layers" in resolution.warnings[0]


def test_legacy_direct_vmag_resolves_as_manual() -> None:
    sample = SampleRecord(
        sample_id="S1",
        geometry=VolumeMetadata(
            vmag=DirectVolumeMetadata(value=2.0, unit="cm^3", method="operator")
        ),
    )

    resolution = resolve_sample_magnetic_volume(sample)

    assert resolution.source == "manual"
    assert resolution.magnetic_volume_m3 == pytest.approx(2e-6)


def test_old_registry_shape_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "sample_registry.yaml"
    path.write_text(
        """
schema_version: 1
samples:
  S1:
    sample_id: S1
    measurements: []
""",
        encoding="utf-8",
    )

    with pytest.raises(MetadataSchemaError, match="schema_version: 2 is required"):
        load_registry(path)


def test_replace_canonical_demotes_existing_result() -> None:
    ledger = ProcessedLedger(
        processed_results={
            "old": ProcessedResultRecord(
                result_id="old",
                measurement_id="fmr:S1:raw",
                sample_id="S1",
                type="fmr",
                processed_path="old.json",
                recipe_path="recipe.yaml",
                status="canonical",
            )
        }
    )
    new = ProcessedResultRecord(
        result_id="new",
        measurement_id="fmr:S1:raw",
        sample_id="S1",
        type="fmr",
        processed_path="new.json",
        recipe_path="recipe.yaml",
    )

    with pytest.raises(CanonicalResultExistsError):
        add_processed_result_record(
            ledger,
            result=new,
            mark_canonical=True,
            replace_canonical=False,
        )

    add_processed_result_record(ledger, result=new, mark_canonical=True, replace_canonical=True)

    assert ledger.processed_results["old"].status == "superseded"
    assert ledger.processed_results["new"].status == "canonical"
